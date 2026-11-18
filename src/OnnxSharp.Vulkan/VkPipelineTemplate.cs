// 通用 compute pipeline 模板（P1-5）：N 个 storage buffer + 一段 push constant。
//
// 两个设计选择，都是为了少出错而不是为了通用：
//  1. 从 SPIR-V 反射 local_size 与 binding，host 侧的组数计算不再靠人抄常量；
//  2. 有 VK_KHR_push_descriptor 就默认走它——不存在 descriptor set，每次派发只写命令流，
//     省掉「分配 set + UpdateDescriptorSets + BindDescriptorSets」这一串 CPU 开销。
using System.Text;

namespace OnnxSharp.Vulkan;

internal sealed unsafe class VkPipelineTemplate : IDisposable
{
    public VkDevice Dev { get; }
    public string Shader { get; }
    public int Bindings { get; }
    public int PushConstBytes { get; }
    public VkPipeline Pipeline { get; }
    public bool PushLayout { get; }

    public uint LocalSizeX { get; }
    public uint LocalSizeY { get; }
    public uint LocalSizeZ { get; }
    public uint LocalSizeTotal => LocalSizeX * LocalSizeY * LocalSizeZ;

    private IntPtr _module;
    /// <summary>非 push-descriptor 路径下复用的 set；push 路径恒为 0。</summary>
    private IntPtr _set;

    private VkPipelineTemplate(VkDevice dev, string shader, int bindings, int pushConstBytes)
    {
        Dev = dev;
        Shader = shader;
        Bindings = bindings;
        PushConstBytes = pushConstBytes;
        PushLayout = dev.PushDescriptors;

        byte[] spirv = VkShaders.Load(shader);
        SpirvInfo info = SpirvReflect.Parse(spirv, shader);

        if (info.LocalSizeX == 0)
            throw new InvalidOperationException(
                $"shader \"{shader}\" 没有声明 local_size，不能作为 compute kernel 使用");

        ValidateBindings(info, shader, bindings);

        LocalSizeX = info.LocalSizeX;
        LocalSizeY = info.LocalSizeY;
        LocalSizeZ = info.LocalSizeZ;

        _module = dev.NewShaderModule(spirv);
        try
        {
            Pipeline = dev.NewPipeline(_module, bindings, pushConstBytes, PushLayout);
            Pipeline.Name = shader;
        }
        catch
        {
            dev.DestroyShaderModule(_module);
            _module = IntPtr.Zero;
            throw;
        }
    }

    /// <summary>取（或首次创建）模板。同一设备上相同「shader + binding 数 + push constant 大小」复用。</summary>
    public static VkPipelineTemplate Get(VkDevice dev, string shader, int bindings, int pushConstBytes)
    {
        ArgumentException.ThrowIfNullOrEmpty(shader);
        if (bindings <= 0) throw new ArgumentOutOfRangeException(nameof(bindings));

        string key = string.Create(System.Globalization.CultureInfo.InvariantCulture,
            $"{shader}|{bindings}|{pushConstBytes}");
        lock (dev.Templates)
        {
            if (dev.Templates.TryGetValue(key, out VkPipelineTemplate? cached)) return cached;
            var t = new VkPipelineTemplate(dev, shader, bindings, pushConstBytes);
            dev.Templates[key] = t;
            return t;
        }
    }

    /// <summary>1D 派发需要多少个 workgroup（按 local_size 的 x*y*z 折算）。</summary>
    public uint GroupCountFor(ulong elements)
    {
        ulong groups = (elements + LocalSizeTotal - 1) / LocalSizeTotal;
        if (groups == 0) groups = 1; // 空张量也走一次，避免调用方到处写 if
        if (groups > uint.MaxValue)
            throw new ArgumentOutOfRangeException(nameof(elements),
                $"元素数 {elements} 需要 {groups} 个 workgroup，超过 uint 上限；请按维度拆分");
        return (uint)groups;
    }

    /// <summary>记录一次完整派发：bind pipeline → 绑 N 个 buffer → push constant → dispatch。</summary>
    public void Record(IntPtr cmd, ReadOnlySpan<VkBuffer> buffers, ReadOnlySpan<byte> pushConstants = default)
    {
        if (buffers.Length != Bindings)
            throw new ArgumentException(
                $"模板 \"{Shader}\" 声明 {Bindings} 个 binding，实际传入 {buffers.Length} 个", nameof(buffers));
        CheckPushConstSize(pushConstants.Length);

        Dev.CmdBindPipelineOnly(cmd, Pipeline);

        if (PushLayout)
        {
            Dev.CmdPushDescriptorSet(cmd, Pipeline, buffers);
        }
        else
        {
            // 复用同一个 set：P1 阶段是「一次提交 → 等待」的串行模型，不存在重叠使用
            if (_set == IntPtr.Zero) _set = Dev.NewDescriptorSet(Pipeline);
            for (int i = 0; i < buffers.Length; i++)
                Dev.UpdateStorageBuffer(_set, (uint)i, buffers[i]);
            Dev.CmdBindDescriptorSet(cmd, Pipeline, _set);
        }

        PushPc(cmd, pushConstants);
    }

    /// <summary>
    /// arena 切片版：所有 binding 指向同一个大 buffer 的不同区间（P1-6 的主路径）。
    /// 省掉「每个 tensor 一个 VkBuffer」的对象与内存开销，绑定只剩 offset/range 的差异。
    /// <para>
    /// 注意：非 push 路径复用同一个 descriptor set，而 set 是「记录时更新、执行时读取」的。
    /// 若同一个命令缓冲里该模板出现多次且区间不同，后一次更新会覆盖前一次——静默错误。
    /// 批量场景必须走 push descriptor（<see cref="VkSegment"/> 在构造时已强制）。
    /// </para>
    /// </summary>
    public void RecordSlices(IntPtr cmd, VkBuffer arena, ReadOnlySpan<ulong> offsets, ReadOnlySpan<ulong> sizes,
        ReadOnlySpan<byte> pushConstants = default)
    {
        if (offsets.Length != Bindings || sizes.Length != Bindings)
            throw new ArgumentException(
                $"模板 \"{Shader}\" 声明 {Bindings} 个 binding，实际传入 offsets={offsets.Length}/sizes={sizes.Length}");

        var ranges = new VkDevice.BufferRange[Bindings];
        for (int i = 0; i < Bindings; i++)
            ranges[i] = new VkDevice.BufferRange(arena, offsets[i], sizes[i]);
        RecordRanges(cmd, ranges, pushConstants);
    }

    /// <summary>
    /// 每个 binding 指向各自的 (buffer, offset, size)。激活在段 arena、权重在常驻 buffer 时用它，
    /// 也是 <see cref="VkSegment"/> 唯一使用的绑定入口。
    /// </summary>
    public void RecordRanges(IntPtr cmd, ReadOnlySpan<VkDevice.BufferRange> ranges,
        ReadOnlySpan<byte> pushConstants = default)
    {
        if (ranges.Length != Bindings)
            throw new ArgumentException(
                $"模板 \"{Shader}\" 声明 {Bindings} 个 binding，实际传入 {ranges.Length} 个", nameof(ranges));
        CheckPushConstSize(pushConstants.Length);

        Dev.CmdBindPipelineOnly(cmd, Pipeline);

        if (PushLayout)
        {
            Dev.CmdPushDescriptorSetRanges(cmd, Pipeline, ranges);
        }
        else
        {
            if (_set == IntPtr.Zero) _set = Dev.NewDescriptorSet(Pipeline);
            for (int i = 0; i < Bindings; i++)
                Dev.UpdateStorageBuffer(_set, (uint)i, ranges[i].Buffer, ranges[i].Offset, ranges[i].Size);
            Dev.CmdBindDescriptorSet(cmd, Pipeline, _set);
        }

        PushPc(cmd, pushConstants);
    }

    private void CheckPushConstSize(int actual)
    {
        if (actual != PushConstBytes)
            throw new ArgumentException(
                $"模板 \"{Shader}\" 的 push constant 是 {PushConstBytes} B，实际传入 {actual} B");
    }

    private void PushPc(IntPtr cmd, ReadOnlySpan<byte> pushConstants)
    {
        if (PushConstBytes == 0) return;
        fixed (byte* p = pushConstants)
            Dev.CmdPushConstants(cmd, Pipeline, p, (uint)pushConstants.Length);
    }

    /// <summary>按元素个数派发（组数由反射到的 local_size 算出）。</summary>
    public void Dispatch1D(IntPtr cmd, ulong elements)
    {
        uint groups = GroupCountFor(elements);
        if (Profiling) { TotalGroupsDispatched += groups; TotalDispatches++; }
        Dev.CmdDispatch(cmd, groups);
    }

    /// <summary>归因用：累计派发的工作组数与 dispatch 数，用来核对缩容是否覆盖整张图。</summary>
    public static long TotalGroupsDispatched, TotalDispatches;

    /// <summary>与 <see cref="TotalGroupsDispatched"/> 同网关：只在置位时累计，避免每帧多两次原子写。</summary>
    public static bool Profiling;

    private static void ValidateBindings(SpirvInfo info, string shader, int declared)
    {
        if (info.Descriptors.Length == 0)
        {
            if (declared > 0)
                throw new InvalidOperationException(
                    $"shader \"{shader}\" 没有引用任何 descriptor，但模板声明了 {declared} 个 storage buffer");
            return;
        }

        var bad = new StringBuilder();
        uint maxBinding = 0;
        foreach ((uint set, uint binding) in info.Descriptors)
        {
            if (set != 0)
                bad.Append($" set{set}:binding{binding}");
            if (binding >= declared)
                bad.Append($" (binding{binding} >= {declared})");
            if (binding > maxBinding) maxBinding = binding;
        }

        // 模板只建 set 0，且 binding 0..N-1 全是 storage buffer。对不上就直接报错：
        // 这类错误驱动不会拒绝 pipeline 创建，只会在 dispatch 时读到垃圾或越界。
        if (bad.Length > 0)
            throw new InvalidOperationException(
                $"shader \"{shader}\" 的 descriptor 布局与模板不符（模板: set0 的 0..{declared - 1} 号 storage buffer）:" +
                bad);

        if ((int)maxBinding + 1 != declared)
            throw new InvalidOperationException(
                $"shader \"{shader}\" 实际用到 binding 0..{maxBinding}，模板声明了 {declared} 个；" +
                "多出来的 binding 在 GLSL 里没有对应声明，检查 layout(binding=N) 是否连续");
    }

    public void Dispose()
    {
        if (Pipeline is not null) Dev.DestroyPipeline(Pipeline);
        if (_module != IntPtr.Zero)
        {
            Dev.DestroyShaderModule(_module);
            _module = IntPtr.Zero;
        }
        _set = IntPtr.Zero; // set 随 descriptor pool 一起销毁
    }
}
