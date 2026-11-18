// 一个「段」= 一次提交里连续跑完的一组 compute dispatch，外加它自己的 stage in/out。
//
// 为什么是批量提交：P1-5 实测单个 dispatch 的 GPU 时间是 2.4 us，而 submit→fence 的墙钟
// 是 96 us —— 95% 是提交税。P1-6 实测 16 个 dispatch 合成一次提交快 10.5×（N 卡）/ 4.3×（A 卡）。
//
// P1-7 在 P1-6 上补两件事：
//   1. 混合绑定：激活在段 arena、权重在进程级常驻 buffer，一个 dispatch 的 binding 可以跨 buffer；
//   2. stage in/out 也进同一次提交：host→staging→arena→compute→staging→host 全程只 fence 一次。
//      逐张量 Upload/Download 是「一个张量一次 submit」，16 输入的段光搬运就 1.6 ms。
using System.Runtime.InteropServices;

namespace OnnxSharp.Vulkan;

/// <summary>
/// 一个 binding 指向什么：段内 arena 的槽位，或一块外部 buffer 的切片（权重常驻）。
/// 外部切片一律视为只读，不参与段内依赖判定。
/// </summary>
internal readonly struct VkBind
{
    /// <summary>段内槽位 id；-1 表示外部 buffer。</summary>
    public readonly int Slot;
    public readonly VkBuffer? Buffer;
    public readonly ulong Offset;
    public readonly ulong Size;

    /// <summary>
    /// shader 会不会写这个 binding。只影响段内屏障的插入位置，不影响绑定本身。
    /// <para>
    /// 不能按「最后一个 binding 是输出」来猜：<c>VkConv</c> 是 <c>[in, weight, out, bias]</c>，
    /// 输出在中间。猜错的后果是漏掉屏障——比多插屏障糟得多，而且不会报错。
    /// </para>
    /// </summary>
    public readonly bool Write;

    private VkBind(int slot, VkBuffer? buffer, ulong offset, ulong size, bool write)
    {
        Slot = slot;
        Buffer = buffer;
        Offset = offset;
        Size = size;
        Write = write;
    }

    public bool IsLocal => Slot >= 0;

    public static VkBind Local(int slot) => new(slot, null, 0, 0, false);

    /// <summary>把已有绑定标记为「会被写」。输出槽位都要过这一道。</summary>
    public VkBind AsWrite() => new(Slot, Buffer, Offset, Size, true);

    /// <summary>段内槽位可以裸写 id：<c>AddDispatch(tpl, [inSlot, outSlot], ...)</c>。</summary>
    public static implicit operator VkBind(int slot) => Local(slot);

    public static VkBind External(VkBuffer buffer, ulong offset, ulong size)
    {
        ArgumentNullException.ThrowIfNull(buffer);
        if (size == 0) throw new ArgumentOutOfRangeException(nameof(size), "外部切片大小必须 > 0");
        if ((offset & (VkDevice.DescriptorOffsetAlign - 1)) != 0)
            throw new ArgumentException("外部切片的 offset 未按 minStorageBufferOffsetAlignment 对齐", nameof(offset));
        return new VkBind(-1, buffer, offset, size, false);
    }
}

internal sealed unsafe class VkSegment : IDisposable
{
    /// <summary>槽位对齐，取 minStorageBufferOffsetAlignment 的规范上限。</summary>
    public const ulong SlotAlign = VkDevice.DescriptorOffsetAlign;

    /// <summary>
    /// host 侧三段计时（纳秒累计）：摊平输入 / 提交并等待 / 拷回输出。
    /// <para>
    /// 为什么不用 GPU 时间戳：<c>vkCmdWriteTimestamp</c> 在这些核上会给出 0.7 us 这种
    /// 「超过峰值算力」的数（数据却是对的），没法用来判断瓶颈。而 host 侧这三个量
    /// 加起来就是端到端时间，且能直接指出「是搬运还是算」。
    /// </para>
    /// </summary>
    public static long ProfHostInNs;
    /// <summary>
    /// 见 <see cref="ProfHostInNs"/>：录制+提交+等 fence 的累计时间（≈ GPU 侧）。
    /// <para>
    /// 单位必须是**纳秒**。`Stopwatch.GetTimestamp()` 返回的是 tick，不是 ns：
    /// 直接拿 tick 当 ns 用（Windows 上 1 tick = 100 ns）会把每一段都少报 100 倍，
    /// 于是「段内 143 us」与逐层 15.5 ms 并存——两个数都自洽，只是差 100 倍。
    /// </para>
    /// </summary>
    public static long ProfSubmitNs;

    /// <summary>
    /// <see cref="ProfSubmitNs"/> 中「CPU 打命令流」的那部分（不含提交与等 fence）。
    /// 两者的差就是 GPU 真正执行的时间。152 次 dispatch 的图里，这个比例决定了
    /// 该去减 dispatch 数（录制占比高）还是该去优化核（等待占比高）。
    /// </summary>
    public static long ProfRecordNs;

    /// <summary>tick → ns（`Stopwatch.Frequency` 是运行时给的，不能假设 10 MHz）。</summary>
    private static readonly double NsPerTick = 1e9 / System.Diagnostics.Stopwatch.Frequency;

    private static long TicksToNs(long ticks) => (long)(ticks * NsPerTick);
    /// <summary>见 <see cref="ProfHostInNs"/>：回读 staging 并拷到 host 的累计时间。</summary>
    public static long ProfHostOutNs;

    /// <summary>是否累计上面的计时（由 <c>ONNXSHARP_VK_PROFILE=1</c> 打开，默认关）。</summary>
    public static bool Profiling { get; } =
        Environment.GetEnvironmentVariable("ONNXSHARP_VK_PROFILE") == "1";

    /// <summary>
    /// 按段名累计 <c>[host-in, submit+wait, host-out]</c>（纳秒）。
    /// 全局总量只能说明「段内不慢」，要定位到具体哪一段还得按名字分开看。
    /// </summary>
    public static readonly Dictionary<string, long[]> ProfBySegment = [];

    private readonly VkDevice _dev;
    private readonly string _name;
    private readonly List<Slot> _slots = [];
    private readonly List<GpuDispatch> _dispatches = [];

    /// <summary>自上次屏障以来被写过的槽位（见 <see cref="NeedsBarrier"/>）。</summary>
    private readonly HashSet<int> _written = [];

    /// <summary>自上次屏障以来被读过的槽位。只用来判 WAR，不单独构成依赖。</summary>
    private readonly HashSet<int> _read = [];

    /// <summary>构建期统计：实际插入的屏障数。段是多线程建的吗？不是——都是发射期单线程。</summary>
    public static long Barriers;

    /// <summary>构建期统计：按旧规则（读写不分）会插、但实际省掉的 RR 屏障数。</summary>
    public static long BarriersSavedRR;

    private ulong _cursor;
    private VkBuffer? _arena;
    private VkBuffer? _stageIn, _stageOut;
    private IntPtr _cmd, _fence;
    private VkDevice.BufferRange[] _ranges = [];
    private bool _sealed, _disposed;

    /// <summary>
    /// 归因：=false 时退回「每帧重录」，用来 A/B 出命令流复用的净收益。
    /// 默认 true。放静态字段而不是每帧读环境变量，是因为它影响的是录制路径的取舍，
    /// 而不是某一帧的内容。
    /// </summary>
    public static readonly bool ReuseCommand =
        Environment.GetEnvironmentVariable("ONNXSHARP_VK_NORECORD") != "1";

    /// <summary>当前 _cmd 里已经录好一份可直接重提交的命令流（见 Execute 里的 reusable）。</summary>
    private bool _recorded;

    /// <summary>录制时的归因开关状态：任何一项变了，命令流就不再等价，必须重录。</summary>
    private ulong _recClamp;
    private int _recInject;
    private int _recIn, _recOut;
    private bool _recSkipAll, _recNoBarrier, _recTimer;

    private readonly record struct Slot(ulong Offset, ulong Size);

    private sealed class GpuDispatch
    {
        public required VkPipelineTemplate Tpl;
        public required VkBind[] Binds;
        public required byte[] PushConstants;
        public required ulong Elements;
    }

    /// <summary>一块 host 数据要写进的段内槽位。</summary>
    public readonly struct StageIn
    {
        public readonly int Slot;
        public readonly void* Src;
        public readonly ulong Bytes;

        public StageIn(int slot, void* src, ulong bytes)
        {
            Slot = slot;
            Src = src;
            Bytes = bytes;
        }
    }

    /// <summary>一个要读回 host 的段内槽位。<paramref name="Bytes"/>=0 表示整个槽位。</summary>
    public readonly struct StageOut
    {
        public readonly int Slot;
        public readonly void* Dst;
        /// <summary>要拷回的字节数；槽位尾部为对齐补齐时用它避免多拷。</summary>
        public readonly ulong Bytes;

        public StageOut(int slot, void* dst, ulong bytes = 0)
        {
            Slot = slot;
            Dst = dst;
            Bytes = bytes;
        }
    }

    public VkSegment(VkDevice dev, string name = "segment")
    {
        // 只能走 push descriptor：普通 descriptor set 的内容是「记录时更新、执行时读取」的，
        // 段内多个 dispatch 若复用同一个 set，后一次的 vkUpdateDescriptorSets 会把前一次的绑定
        // 一起改掉，结果**静默错误**（数值错但不报任何错）。要支持 fallback 设备，得给每个
        // dispatch 分配独立的 set（descriptor 环），放在 P1-9。
        if (!dev.PushDescriptors)
            throw new PlatformNotSupportedException(
                "VkSegment 需要 VK_KHR_push_descriptor：本设备不支持，批量提交会产出错误的绑定");

        _dev = dev;
        _name = name;
    }

    public string Name => _name;
    public int SlotCount => _slots.Count;
    public int DispatchCount => _dispatches.Count;

    /// <summary>
    /// 逐 dispatch 的 GPU 毫秒数（长度 = <see cref="DispatchCount"/>），只有绑定的
    /// timer 槽位 ≥ dispatch 数 + 1 时才会填；否则为 null。
    /// <para>
    /// 为什么不在每个 dispatch 前后各打两点：那样同一段要 2N 个槽，而「dispatch 后打点」
    /// 天然等于上一段终点，N+1 个槽就够，且相邻差分正好是「屏障 + 该 dispatch」。
    /// </para>
    /// </summary>
    public double[]? DispatchMs { get; private set; }

    /// <summary>每个 dispatch 的标识（模板名 + 线程数），与 <see cref="DispatchMs"/> 同序。</summary>
    public string[] DispatchLabels
    {
        get
        {
            var labels = new string[_dispatches.Count];
            for (int i = 0; i < labels.Length; i++)
                labels[i] = $"{_dispatches[i].Tpl.Shader}/{_dispatches[i].Elements}";
            return labels;
        }
    }

    /// <summary>arena 实际字节数（含对齐填充）；未 Seal 时为预算值。</summary>
    public ulong ArenaBytes => _cursor;

    public VkBuffer Arena => _arena ?? throw new InvalidOperationException($"段 \"{_name}\" 还没 Seal");

    public ulong SlotOffset(int slot) => _slots[CheckSlot(slot)].Offset;
    public ulong SlotSize(int slot) => _slots[CheckSlot(slot)].Size;

    /// <summary>登记一个 tensor 槽位，返回其 id。偏移在登记时就定下来，Seal 时一次性分配物理内存。</summary>
    public int NewSlot(ulong bytes)
    {
        ThrowIfSealed();
        if (bytes == 0) throw new ArgumentOutOfRangeException(nameof(bytes), "槽位大小必须 > 0");

        ulong offset = Align(_cursor);
        _slots.Add(new Slot(offset, bytes));
        _cursor = offset + bytes;
        return _slots.Count - 1;
    }

    /// <summary>
    /// 追加一个 dispatch，binding i ← <paramref name="binds"/>[i]（段内槽位或外部权重切片）。
    /// 段内槽位可以直接写 id（<see cref="VkBind"/> 有隐式转换）。
    /// </summary>
    public void AddDispatch(VkPipelineTemplate tpl, ReadOnlySpan<VkBind> binds,
        ReadOnlySpan<byte> pushConstants, ulong elements)
    {
        ThrowIfSealed();
        ArgumentNullException.ThrowIfNull(tpl);
        if (!ReferenceEquals(tpl.Dev, _dev))
            throw new ArgumentException($"模板 \"{tpl.Shader}\" 属于另一个设备");
        if (binds.Length != tpl.Bindings)
            throw new ArgumentException(
                $"模板 \"{tpl.Shader}\" 有 {tpl.Bindings} 个 binding，实际给了 {binds.Length} 个", nameof(binds));
        if (pushConstants.Length != tpl.PushConstBytes)
            throw new ArgumentException(
                $"模板 \"{tpl.Shader}\" 的 push constant 是 {tpl.PushConstBytes} B，实际 {pushConstants.Length} B",
                nameof(pushConstants));
        foreach (VkBind b in binds)
            if (b.IsLocal) CheckSlot(b.Slot);

        _dispatches.Add(new GpuDispatch
        {
            Tpl = tpl,
            Binds = binds.ToArray(),
            PushConstants = pushConstants.ToArray(),
            Elements = elements,
        });
    }

    /// <summary>分配 arena。幂等；分配后不能再加槽位或 dispatch（偏移已固化进 descriptor）。</summary>
    public void Seal()
    {
        if (_sealed) return;
        if (_dispatches.Count == 0)
        {
            // 归因开关（SkipCopy/SkipElementwise）会把整段算子都跳掉，这时容忍空段：
            // 只打标记，Execute 直接返回。生产路径上不会走到这里——建段的前提就是至少 1 个 dispatch，
            // 所以这条分支不会掩盖真正的「发射器什么都没发」错误。
            if (VkZero.SkipCopy || VkZero.SkipElementwise || VkZero.SkipAll)
            {
                _sealed = true;
                return;
            }
            throw new InvalidOperationException($"段 \"{_name}\" 没有任何 dispatch");
        }

        // 空槽位（登记了但没被任何 dispatch 引用）仍然占内存。P1-9 接 BufferAllocator
        // 做生命周期着色后，这里会改成只分配真正用到的区间。
        //
        // 走池：段重建（换 shape、换设备）时 1.2 MB 的 vkAllocateMemory 要 646 us，
        // 而整段重建才 500 us —— 命中池后这条路径几乎归零。
        _arena = _dev.NewBuffer(Math.Max(Align(_cursor), SlotAlign), VkDevice.MemoryKind.DeviceLocal,
            pool: true);
        _recorded = false;   // 新 arena = 新句柄 = 旧命令流作废

        int maxBindings = 1;
        foreach (GpuDispatch d in _dispatches)
            if (d.Binds.Length > maxBindings) maxBindings = d.Binds.Length;
        _ranges = new VkDevice.BufferRange[maxBindings];

        _sealed = true;
    }

    /// <summary>只跑 dispatch（宿主自己用 <see cref="UploadSlot"/> 准备数据）。</summary>
    public void Run(VkTimer? timer = null) => Execute(default, default, timer);

    /// <summary>
    /// stage in + 整段 dispatch + stage out，全程一条命令缓冲、一次提交、一次 fence。
    /// </summary>
    public void Execute(ReadOnlySpan<StageIn> ins, ReadOnlySpan<StageOut> outs, VkTimer? timer = null)
    {
        ThrowIfDisposed();
        Seal();
        // 空段只可能来自归因开关（见 Seal）；没有 arena 可用，也就没有任何工作要提交。
        if (_dispatches.Count == 0) return;

        VkBuffer arena = _arena!;

        // ---- staging 布局：按 4 B 紧凑排（vkCmdCopyBuffer 的硬性要求）
        ulong inBytes = 0, outBytes = 0;
        foreach (StageIn it in ins)
        {
            CheckSlot(it.Slot);
            if ((it.Bytes & 3) != 0)
                throw new ArgumentException($"槽位 {it.Slot} 的 stage in 大小 {it.Bytes} 不是 4 的倍数");
            if (it.Bytes > _slots[it.Slot].Size)
                throw new ArgumentOutOfRangeException(nameof(ins),
                    $"槽位 {it.Slot} 只有 {_slots[it.Slot].Size} B，写入 {it.Bytes} B");
            inBytes += Align4(it.Bytes);
        }
        foreach (StageOut it in outs)
        {
            CheckSlot(it.Slot);
            ulong bytes = StageOutBytes(it);
            if ((bytes & 3) != 0)
                throw new ArgumentException($"槽位 {it.Slot} 的 stage out 大小 {bytes} 不是 4 的倍数");
            outBytes += Align4(bytes);
        }

        VkBuffer? sin = EnsureStaging(ref _stageIn, inBytes, VkDevice.MemoryKind.HostUpload);
        VkBuffer? sout = EnsureStaging(ref _stageOut, outBytes, VkDevice.MemoryKind.HostReadback);

        long t0 = Profiling ? System.Diagnostics.Stopwatch.GetTimestamp() : 0;

        // ---- host → staging，同时算出 copy region
        Span<VkDevice.CopyRegion> inRegions = ins.Length > 0 ? stackalloc VkDevice.CopyRegion[ins.Length] : default;
        Span<VkDevice.CopyRegion> outRegions = outs.Length > 0 ? stackalloc VkDevice.CopyRegion[outs.Length] : default;

        if (ins.Length > 0)
        {
            void* p = sin!.Map();
            ulong off = 0;
            for (int i = 0; i < ins.Length; i++)
            {
                StageIn it = ins[i];
                System.Buffer.MemoryCopy(it.Src, (byte*)p + off, (long)(sin.Size - off), (long)it.Bytes);
                inRegions[i] = new VkDevice.CopyRegion(off, _slots[it.Slot].Offset, it.Bytes);
                off += Align4(it.Bytes);
            }
            sin.Flush(0, inBytes);
            sin.Unmap();
        }

        if (outs.Length > 0)
        {
            ulong off = 0;
            for (int i = 0; i < outs.Length; i++)
            {
                ulong size = StageOutBytes(outs[i]);
                outRegions[i] = new VkDevice.CopyRegion(_slots[outs[i].Slot].Offset, off, size);
                off += Align4(size);
            }
        }

        // ---- 录制：copy in → dispatch 链 → copy out，一次提交
        //
        // 命令流里没有任何随帧变化的东西：槽位偏移在 Seal 时就定了，权重切片是常驻的，
        // push constant 也是发射期算好的常量。所以「录一次、之后每帧只提交」与
        // 「每帧重录」在 GPU 侧完全等价，省下的是纯 CPU 时间——
        // 162 个 dispatch + 158 次屏障重录一次实测 3.3 ms，比整帧计算还贵。
        //
        // 会改变命令流本身的开关（ClampElements / InjectPerDispatch / SkipAll / NoBarrier）
        // 一变就作废重录；staging buffer 换过也一样（命令流里存的是它的句柄）。
        bool reusable = ReuseCommand
            && _recorded
            && _recClamp == VkZero.ClampElements
            && _recInject == VkZero.InjectPerDispatch
            && _recSkipAll == VkZero.SkipAll
            && _recNoBarrier == VkZero.NoBarrier
            && _recTimer == (timer is not null)
            // 进出槽位的个数决定 copy 的条数。同一段的 ins/outs 由分区决定、不会变，
            // 但这里白纸黑字比一次，胜过依赖「不会变」这个约定。
            && _recIn == ins.Length
            && _recOut == outs.Length;

        long t1 = Profiling ? System.Diagnostics.Stopwatch.GetTimestamp() : 0;
        IntPtr cmd;
        long tRec;

        if (reusable)
        {
            cmd = _cmd;
            tRec = t1;
        }
        else
        {
            cmd = BeginCommand();
            timer?.Reset(cmd);
            if (ins.Length > 0) _dev.CmdCopyBuffer(cmd, sin!, arena, inRegions);

            // 段内依赖屏障与首尾的 transfer↔compute 屏障都在这里
            RecordDispatchList(cmd, timer);

            // 录制与「提交+等 fence」必须分开计量：前者是 CPU 在驱动里打命令流，
            // 后者才轮到 GPU。只看 submit+wait 一项无法判断该优化核还是该减 dispatch 数。
            tRec = Profiling ? System.Diagnostics.Stopwatch.GetTimestamp() : 0;

            if (outs.Length > 0) _dev.CmdCopyBuffer(cmd, arena, sout!, outRegions);
            Vk.Check(Vk.vkEndCommandBuffer(cmd), "vkEndCommandBuffer");

            _recorded = true;
            _recClamp = VkZero.ClampElements;
            _recInject = VkZero.InjectPerDispatch;
            _recSkipAll = VkZero.SkipAll;
            _recNoBarrier = VkZero.NoBarrier;
            _recTimer = timer is not null;
            _recIn = ins.Length;
            _recOut = outs.Length;
        }

        SubmitAndWait(cmd);

        // 逐 dispatch 时间：槽位够就一次读回、按相邻槽差分（必须 submit+wait 之后读）
        if (timer is not null && timer.Slots >= (uint)(_dispatches.Count + 1))
            DispatchMs = timer.IntervalMs();

        long t2 = Profiling ? System.Diagnostics.Stopwatch.GetTimestamp() : 0;

        // ---- staging → host
        if (outs.Length > 0)
        {
            void* p = sout!.Map();
            sout.Invalidate(0, outBytes);
            ulong off = 0;
            for (int i = 0; i < outs.Length; i++)
            {
                ulong size = StageOutBytes(outs[i]);
                System.Buffer.MemoryCopy((byte*)p + off, outs[i].Dst, (long)size, (long)size);
                off += Align4(size);
            }
            sout.Unmap();
        }

        if (Profiling)
        {
            long t3 = System.Diagnostics.Stopwatch.GetTimestamp();
            long tin = TicksToNs(t1 - t0), tsub = TicksToNs(t2 - t1), tout = TicksToNs(t3 - t2);
            ProfHostInNs += tin;          // host 摊平 + MemCopy 到 staging + flush
            ProfSubmitNs += tsub;         // 录制 + 提交 + 等 fence ≈ GPU 侧时间
            ProfHostOutNs += tout;        // invalidate + 拷回 host
            ProfRecordNs += TicksToNs(tRec - t1);

            if (!ProfBySegment.TryGetValue(_name, out long[]? acc))
                ProfBySegment[_name] = acc = new long[3];
            acc[0] += tin;
            acc[1] += tsub;
            acc[2] += tout;
        }
    }

    /// <summary>把 host 数据写进槽位（单独一次提交，测试/调试用；连续多个请用 <see cref="Execute"/>）。</summary>
    public void UploadSlot(int slot, void* src, ulong bytes)
    {
        Seal();
        int id = CheckSlot(slot);
        if (bytes > _slots[id].Size)
            throw new ArgumentOutOfRangeException(nameof(bytes),
                $"槽位 {slot} 只有 {_slots[id].Size} B，写入 {bytes} B");
        _dev.Upload(Arena, src, bytes, _slots[id].Offset);
    }

    /// <summary>把槽位内容读回 host（单独一次提交）。</summary>
    public void DownloadSlot(int slot, void* dst)
    {
        Seal();
        int id = CheckSlot(slot);
        _dev.Download(Arena, dst, _slots[id].Size, _slots[id].Offset);
    }

    private void RecordDispatchList(IntPtr cmd, VkTimer? timer)
    {
        // stage in 的 copy 必须先对 compute 可见
        _dev.CmdTransferToComputeBarrier(cmd);

        // 槽位够就逐 dispatch 打点（N+1 个点），否则只打首尾两个点。
        bool perDispatch = timer is not null && timer.Slots >= (uint)(_dispatches.Count + 1);
        timer?.Mark(cmd, 0);

        int marks = 0;
        _written.Clear();
        _read.Clear();
        VkBuffer arena = _arena!;
        foreach (GpuDispatch d in _dispatches)
        {
            if (VkZero.SkipAll) break;   // 归因：只保留 stage in/out 与一次提交，测段的地板
            // 依赖判定见 NeedsBarrier：按读写方向区分，读-读不插屏障。
            // 外部（权重）切片只读，不参与判定。
            if (!VkZero.NoBarrier && NeedsBarrier(d.Binds))
            {
                _dev.CmdComputeBarrier(cmd);
                _written.Clear();
                _read.Clear();
            }

            for (int i = 0; i < d.Binds.Length; i++)
            {
                VkBind b = d.Binds[i];
                if (b.IsLocal)
                {
                    Slot s = _slots[b.Slot];
                    _ranges[i] = new VkDevice.BufferRange(arena, s.Offset, s.Size);
                    if (b.Write) _written.Add(b.Slot);
                    else _read.Add(b.Slot);
                }
                else
                {
                    _ranges[i] = new VkDevice.BufferRange(b.Buffer!, b.Offset, b.Size);
                }
            }

            d.Tpl.RecordRanges(cmd, _ranges.AsSpan(0, d.Binds.Length), d.PushConstants);
            // ClampElements 是归因开关（默认 0 = 不生效），在这里而不是发射期读：
            // 发射只发生一次，而每帧都要重录命令流，所以只有这里能在同一个 Net 上做 A/B。
            d.Tpl.Dispatch1D(cmd, VkZero.ClampElements > 0
                ? Math.Min(d.Elements, VkZero.ClampElements)
                : d.Elements);

            // 归因：紧跟一个同模板、同绑定的空 dispatch，测「真实图里多一个 dispatch」的边际成本
            for (int inj = 0; inj < VkZero.InjectPerDispatch; inj++)
                d.Tpl.Dispatch1D(cmd, 64);

            if (perDispatch) timer!.Mark(cmd, (uint)++marks);
        }

        if (timer is not null && !perDispatch) timer.Mark(cmd, 1);
        // 段尾的写入要对 stage out 的 copy 可见
        _dev.CmdComputeBarrier(cmd);
    }

    private IntPtr BeginCommand()
    {
        IntPtr cmd = _cmd;
        if (cmd == IntPtr.Zero)
        {
            cmd = _dev.NewCommandBuffer();
            _cmd = cmd;
        }
        else
        {
            Vk.Check(Vk.vkResetCommandBuffer(cmd, 0), "vkResetCommandBuffer");
        }

        Vk.VkCommandBufferBeginInfo begin = new()
        {
            SType = VkConst.StCommandBufferBeginInfo,
            Flags = VkConst.CmdBufferUsageOneTimeSubmit,
        };
        Vk.Check(Vk.vkBeginCommandBuffer(cmd, &begin), "vkBeginCommandBuffer");
        return cmd;
    }

    private void SubmitAndWait(IntPtr cmd)
    {
        // fence 复用：P1-5 的 SubmitAndWait 每次都建/删 fence，属于可省的开销
        if (_fence == IntPtr.Zero)
        {
            _fence = _dev.NewFence();
        }
        else
        {
            IntPtr f = _fence;
            Vk.Check(Vk.vkResetFences(_dev.Device, 1, &f), "vkResetFences");
        }
        _dev.Submit(cmd, _fence);
        _dev.WaitFence(_fence);
    }

    private VkBuffer? EnsureStaging(ref VkBuffer? buf, ulong need, VkDevice.MemoryKind kind)
    {
        if (need == 0) return buf;
        ulong want = Align4(need);
        if (buf is not null && buf.Size >= want) return buf;

        buf?.Dispose();
        // 用纯 host 类型：staging 不该占显存。下次需要更大再换，不预留翻倍空间免得白占。
        // 进出两个方向的内存类型不同：回读要 HOST_CACHED，否则 CPU 读 WC 内存慢 10 倍。
        // 走池：staging 是池命中率最高的一类（尺寸随形状重复出现）。
        buf = _dev.NewBuffer(want, kind, pool: true);
        _recorded = false;   // 命令流里存的是旧 buffer 的句柄，换了就不能再复用
        return buf;
    }

    /// <summary>
    /// 本 dispatch 之前是否需要一次 compute 屏障。
    /// <para>
    /// 依赖分三类，代价差别很大——compute 屏障一次约 21–27 us（实测 158 次共 4.3 ms），
    /// 相当于把 GPU 整条流水线排空一遍，不能多插：
    /// </para>
    /// <list type="bullet">
    /// <item>RAW：前驱写过、本 dispatch 要读 —— 必须插。</item>
    /// <item>WAW：前驱写过、本 dispatch 也要写 —— 必须插（写出顺序不保证）。</item>
    /// <item>WAR：前驱读过、本 dispatch 要写 —— 必须插（否则覆盖正在读的数据）。</item>
    /// <item>RR：前驱读过、本 dispatch 也只读 —— <b>不需要</b>。</item>
    /// </list>
    /// <para>
    /// 旧实现把「读过」和「写过」混在一个集合里，于是 RR 也被判成依赖。
    /// 一个张量被多个后继消费（Split 扇出、Reshape 别名被两处读）时每次都要插一次，
    /// 这类屏障纯属浪费。
    /// </para>
    /// </summary>
    private bool NeedsBarrier(VkBind[] binds)
    {
        if (VkZero.OldBarrier)
        {
            // 归因分支：读写不分。保留它是为了让 A/B 能在同一个 Net、同一份命令流上跑，
            // 跨进程比较会被 DVFS 和后台负载淹掉（见 perf 输出的 8.6 / 11.8 抖动）。
            foreach (VkBind b in binds)
                if (b.IsLocal && (_written.Contains(b.Slot) || _read.Contains(b.Slot)))
                {
                    Barriers++;
                    return true;
                }
            return false;
        }

        bool need = false;
        bool oldRule = false;

        foreach (VkBind b in binds)
        {
            if (!b.IsLocal) continue;                 // 外部权重切片只读，不参与判定
            bool written = _written.Contains(b.Slot);
            bool read = _read.Contains(b.Slot);

            if (written || read) oldRule = true;
            if (b.Write ? (written || read) : written) need = true;
        }

        if (need) Barriers++;
        else if (oldRule) BarriersSavedRR++;
        return need;
    }

    /// <summary>stage out 的实际字节数：<see cref="StageOut.Bytes"/>=0 表示整个槽位。</summary>
    private ulong StageOutBytes(StageOut it)
    {
        Slot s = _slots[CheckSlot(it.Slot)];
        ulong bytes = it.Bytes == 0 ? s.Size : it.Bytes;
        if (bytes > s.Size)
            throw new ArgumentOutOfRangeException(nameof(it),
                $"槽位 {it.Slot} 只有 {s.Size} B，请求读回 {bytes} B");
        return bytes;
    }

    private int CheckSlot(int slot)
    {
        if ((uint)slot >= (uint)_slots.Count)
            throw new ArgumentOutOfRangeException(nameof(slot), $"槽位 {slot} 不存在（共 {_slots.Count} 个）");
        return slot;
    }

    private void ThrowIfSealed()
    {
        if (_sealed) throw new InvalidOperationException($"段 \"{_name}\" 已 Seal，不能再改结构");
    }

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(nameof(VkSegment));
    }

    private static ulong Align(ulong v) => (v + SlotAlign - 1) & ~(SlotAlign - 1);
    private static ulong Align4(ulong v) => (v + 3) & ~3UL;

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        if (_cmd != IntPtr.Zero) { _dev.FreeCommandBuffer(_cmd); _cmd = IntPtr.Zero; }
        if (_fence != IntPtr.Zero) { _dev.DestroyFence(_fence); _fence = IntPtr.Zero; }
        _stageIn?.Dispose();
        _stageOut?.Dispose();
        _stageIn = _stageOut = null;
        _arena?.Dispose();
        _arena = null;
    }
}
