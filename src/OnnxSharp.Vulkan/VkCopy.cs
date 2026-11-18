// Concat / Split 的发射器：图里最「便宜」的两类算子，却是段数的最大杀手。
//
// yolo11n 上 162 个算子被 23 个 Concat、11 个 Split 切成了 21 段——它们本身只搬数据，
// 但一段里既有 Conv 又有 Concat，段就能一路连下去；反过来，Concat 一旦不能发射，
// 它前面的 Conv 只能自己成段（1 个 dispatch），被成本门判「不划算」丢回 CPU，
// 于是「Conv 在 GPU / Concat 在 CPU」的往返把整条流水线钉死。
//
// 两个算子共用一个内核（copy_axis.comp），因为它们的动作完全对称：
//   * Concat：多个源 → 目标的若干轴区间（inner/outer 相同，只有轴偏移在变）
//   * Split ：一个源的若干轴区间 → 多个目标
// 差别只在宿主侧「谁发 dispatch、偏移怎么累加」，所以这里只有两份很薄的适配。
//
// 只支持「除拼接轴外维度完全一致」+ F32：这正是 ONNX 对 Concat/Split 的语义要求，
// 也是「一行 memcpy」能成立的前提。别的情形一律不发射，交给 CPU。
using System.Runtime.InteropServices;
using OnnxSharp.Core;
using OnnxSharp.Tensors;

namespace OnnxSharp.Vulkan;

/// <summary><c>copy_axis.comp</c> 的 push constant。字段顺序必须与 shader 完全一致（32 B）。</summary>
[StructLayout(LayoutKind.Sequential)]
internal struct CopyAxisParams
{
    public uint N;
    public uint Outer;
    public uint Inner;
    public uint Dim;
    public uint SrcAxis;
    public uint SrcOff;
    public uint DstAxis;
    public uint DstOff;
}

/// <summary><c>copy_concat4.comp</c> 的 push constant（44 B）。字段顺序必须与 shader 完全一致。</summary>
[StructLayout(LayoutKind.Sequential)]
internal struct CopyConcatParams
{
    public uint N;
    public uint Outer;
    public uint Inner;
    public uint NSrc;
    public uint DstAxis;
    public uint DstOff;
    public uint DimSum;
    public uint Dim0;
    public uint Dim1;
    public uint Dim2;
    public uint Dim3;
}

/// <summary><c>copy_axis.comp</c> 的宿主侧公共部分：轴规范化 + 内/外层求积 + 一次 dispatch。</summary>
internal static class VkCopyAxis
{
    /// <summary>shader 名（对应 Shaders/copy_axis.comp → copy_axis.spv）。</summary>
    public const string Shader = "copy_axis";

    /// <summary>
    /// 向量化版本（每线程 4 个 float）。只在 <c>Inner % 4 == 0</c> 时使用：
    /// 那一维是「每一段的连续长度」，为 4 的倍数时整段都能按 float4 切，没有残块。
    /// </summary>
    public const string Shader4 = "copy_axis4";

    /// <summary>push constant 字节数，必须与 <c>CopyAxisParams</c> 一致。</summary>
    public const int PushConstBytes = 32;

    /// <summary>
    /// 把 ONNX 的（可能为负的）axis 规范到 <c>[0, rank)</c>。
    /// 属性缺失或越界返回 false——规划期不能猜 Axis，猜错就是静默错位。
    /// </summary>
    public static bool TryAxis(LayerInfo op, int rank, out int axis)
    {
        axis = 0;
        if (op.Parameters is null || !op.Parameters.Has("axis")) return false;
        long a = op.Parameters.GetInt("axis", 0);
        if (a < 0) a += rank;
        if (a < 0 || a >= rank) return false;
        axis = (int)a;
        return true;
    }

    /// <summary>第 <paramref name="from"/> 维到第 <paramref name="to"/> 维（不含）的乘积。</summary>
    public static long Product(TensorShape shape, int from, int to)
    {
        long p = 1;
        for (int d = from; d < to; d++)
        {
            if (shape[d] <= 0) return 0;
            p *= shape[d];
        }
        return p;
    }

    /// <summary>
    /// 发一次「轴区间拷贝」。<paramref name="n"/> 是本次要传的元素数（≤ uint.MaxValue）。
    /// <paramref name="srcAxis"/>/<paramref name="dstAxis"/> 是两个张量在该轴上的总长：
    /// 有了它们，跨行跳跃才能算对——这正是「拷贝连续块」和「拷贝整个张量」的区别。
    /// </summary>
    public static unsafe void Dispatch(VkEmitContext ctx, Arg src, Arg dst, long n,
        long outer, long inner, long dim, long srcAxis, long srcOff, long dstAxis, long dstOff)
    {
        LayerInfo op = ctx.Op;
        if (VkZero.SkipCopy) return;   // 归因开关：测「砍掉这类 dispatch 能省多少」，数值会错
        if (n <= 0 || n > uint.MaxValue)
            throw new NotSupportedException($"'{op.Name}' 的拷贝量 {n} 不支持");
        if (outer <= 0 || outer > uint.MaxValue || inner <= 0 || inner > uint.MaxValue
            || dim <= 0 || dim > uint.MaxValue)
            throw new NotSupportedException(
                $"'{op.Name}' 的轴分解 outer={outer} dim={dim} inner={inner} 超出 uint 范围");

        var pc = new CopyAxisParams
        {
            N = (uint)n,
            Outer = (uint)outer,
            Inner = (uint)inner,
            Dim = (uint)dim,
            SrcAxis = (uint)srcAxis,
            SrcOff = (uint)srcOff,
            DstAxis = (uint)dstAxis,
            DstOff = (uint)dstOff,
        };

        ReadOnlySpan<byte> pcb = new(&pc, PushConstBytes);

        // Inner % 4 == 0 ⇒ 每段连续长度能被 float4 整除，整块都能按 16 B 切，
        // 既没有首尾残块，段内偏移也必然是 4 的倍数（对齐由 256 B 的槽位对齐兜底）。
        bool vec4 = VkZero.Vec4Copy && (inner & 3) == 0;
        string shader = vec4 ? Shader4 : Shader;
        ulong threads = vec4 ? (ulong)(n >> 2) : (ulong)n;

        // 归因用：ONNXSHARP_VK_CONVDUMP=1 时打出每次拷贝的算子类型与轴分解。
        // 目的只有一个——分清「纯别名就能省掉的 Reshape」与「真的得搬数据的 Concat/Split」。
        if (VkConvDump.On)
            Console.WriteLine($"    [copy] {op.Type} {op.Name}: n={n} outer={outer} dim={dim} " +
                              $"inner={inner} {shader}");

        ctx.Layer.AddDispatch(shader, 2, PushConstBytes,
            [VkArgs.Bind(ctx, src), ctx.Out(dst)], pcb, threads);
    }

    /// <summary>多源合并版的 shader 名（对应 Shaders/copy_concat4.comp）。</summary>
    public const string ConcatShader = "copy_concat4";

    /// <summary>多源合并版的 push constant 字节数。</summary>
    public const int ConcatPushConstBytes = 44;

    /// <summary>最多能合并的源数（受 shader 里 4 个只读 binding 限制）。</summary>
    public const int MaxConcatSrc = 4;

    /// <summary>
    /// 一次 dispatch 把多个源沿轴拼进目标，代替「每源一次 copy_axis」。
    /// <para>
    /// 收益不在搬运本身，而在同步：每少一个 dispatch 就少一次约 21 us 的 compute 屏障
    /// （实测 190 次屏障共 4.08 ms，占端到端 23%）。yolo11n 的 23 个 Concat 原本拆成 57 次拷贝，
    /// 合并后是 23 次。访存总量与数值语义完全不变。
    /// </para>
    /// </summary>
    /// <param name="dims">各源在拼接轴上占据的长度，顺序与 <paramref name="srcs"/> 一致。</param>
    public static unsafe void DispatchConcat(VkEmitContext ctx, Arg dst, IReadOnlyList<Arg> srcs,
        long outer, long inner, ReadOnlySpan<long> dims, long dstAxis, long dstOff)
    {
        LayerInfo op = ctx.Op;
        if (VkZero.SkipCopy) return;   // 归因开关：测「砍掉这类 dispatch 能省多少」，数值会错
        if (srcs.Count is < 2 or > MaxConcatSrc)
            throw new NotSupportedException($"'{op.Name}': 合并拷贝的源数 {srcs.Count} 超出 [2, {MaxConcatSrc}]");
        if (dims.Length != srcs.Count)
            throw new ArgumentException("dims 与 srcs 长度不一致", nameof(dims));

        long dimSum = 0;
        for (int i = 0; i < dims.Length; i++) dimSum += dims[i];
        long n = outer * dimSum * inner;
        if (n <= 0 || n > uint.MaxValue)
            throw new NotSupportedException($"'{op.Name}' 的合并拷贝量 {n} 不支持");

        var pc = new CopyConcatParams
        {
            N = (uint)n,
            Outer = (uint)outer,
            Inner = (uint)inner,
            NSrc = (uint)srcs.Count,
            DstAxis = (uint)dstAxis,
            DstOff = (uint)dstOff,
            DimSum = (uint)dimSum,
        };
        for (int i = 0; i < dims.Length; i++)
        {
            if (dims[i] <= 0 || dims[i] > uint.MaxValue)
                throw new NotSupportedException($"'{op.Name}': 第 {i} 个源的轴长 {dims[i]} 不支持");
            uint d = (uint)dims[i];
            switch (i)
            {
                case 0: pc.Dim0 = d; break;
                case 1: pc.Dim1 = d; break;
                case 2: pc.Dim2 = d; break;
                default: pc.Dim3 = d; break;
            }
        }

        ReadOnlySpan<byte> pcb = new(&pc, ConcatPushConstBytes);

        // 4 个只读 binding + 1 个写入 binding。源不足 4 个时用第 0 个源占位：
        // 那些 binding 在 shader 里按 NSrc 判断后不会被读到，但描述符必须有效。
        VkBind[] binds = new VkBind[MaxConcatSrc + 1];
        for (int i = 0; i < MaxConcatSrc; i++)
            binds[i] = VkArgs.Bind(ctx, srcs[Math.Min(i, srcs.Count - 1)]);
        binds[MaxConcatSrc] = ctx.Out(dst);

        if (VkConvDump.On)
            Console.WriteLine($"    [copy] {op.Type} {op.Name}: 合并 {srcs.Count} 源 " +
                              $"n={n} outer={outer} inner={inner} {ConcatShader}");

        ctx.Layer.AddDispatch(ConcatShader, MaxConcatSrc + 1, ConcatPushConstBytes,
            binds, pcb, (ulong)(n >> 2));
    }
}

/// <summary>
/// Reshape / Flatten / Squeeze / Unsqueeze / Identity：元素数不变的纯形状变换。
/// <para>
/// 它们在 GPU 上的正确做法是「同一块显存换个形状」（零拷贝别名），但那要求槽位共享、
/// 且 CheckIo 只比字节数不比形状——改动面大。这里退一步发一次整块 memcpy：
/// 数据多走一遍显存（yolo 上每次 1~3 MB，约 0.02 ms），换来的是段不被它们切断。
/// 切断的代价是一次 submit + 一次 host 往返（~120 us 起），比 memcpy 贵一个数量级。
/// </para>
/// </summary>
internal sealed class VkReshapeEmitter : IVkOpEmitter
{
    private static readonly string[] ShapeOnly =
        ["Reshape", "Flatten", "Squeeze", "Unsqueeze", "Identity"];

    public static void Register()
    {
        // 注册表按 Type 索引，这几种算子共用同一份实现，所以要逐个登记。
        foreach (string t in ShapeOnly) VkOpRegistry.Register(t, new VkReshapeEmitter());
    }

    public bool CanEmit(LayerInfo op)
    {
        ArgumentNullException.ThrowIfNull(op);
        if (Array.IndexOf(ShapeOnly, op.Type) < 0) return false;
        // 第 2 个输入是形状/轴常量（int64）：只用来决定输出形状，不进拷贝
        if (op.Inputs.Count is < 1 or > 2 || op.Outputs.Count is not 1) return false;
        if (op.Inputs[0] == op.Outputs[0]) return false;   // 原地变换：读写同槽，不能这样拷

        return VkArgs.IsF32(op, op.Inputs[0]) && VkArgs.IsF32(op, op.Outputs[0]);
    }

    public void Emit(VkEmitContext ctx)
    {
        LayerInfo op = ctx.Op;
        long n = ctx.ShapeOf(op.Outputs[0]).Total();
        long ni = ctx.ShapeOf(op.Inputs[0]).Total();
        if (n != ni)
            throw new OnnxSharpException(
                $"'{op.Name}'({op.Type}): 元素数 {ni} → {n} 变了，不是纯形状变换");

        // 整块连续，等价于一次 memcpy。轴分解可以任意重排（srcAxis=dstAxis=n、偏移 0，
        // 索引式与分解无关），所以这里故意取 inner=4 而不是 inner=1：
        // Inner % 4 == 0 是走 copy_axis4 的条件，留 inner=1 会让 reshape 永远落在标量核上
        // （yolo11n 的 5 次 reshape 合计 1.35M 元素，白搬 11 MB）。
        if ((n & 3) == 0)
            VkCopyAxis.Dispatch(ctx, op.Inputs[0], op.Outputs[0], n, 1, 4, n / 4, n, 0, n, 0);
        else
            VkCopyAxis.Dispatch(ctx, op.Inputs[0], op.Outputs[0], n, 1, 1, n, n, 0, n, 0);
    }
}

/// <summary>
/// Slice：单轴、step=1 的区间拷贝。
/// <para>
/// 它和 Concat/Split 是同一个动作——「源的一段轴区间 → 目标的一段轴区间」，
/// 所以直接用 <c>copy_axis</c>，一个 shader 都不用加。差别只在参数怎么算：
/// Slice 的源区间是 <c>[start, start+len)</c>，目标是紧凑的 <c>[0, len)</c>。
/// </para>
/// <para>
/// 只支持单轴 + step=1。多轴可以拆成多次调用，但那样又回到「一个算子多次 dispatch」，
/// 而它在 yolo11n 里是每秒 30 帧的检测头路径上、每次只切 2 个通道——不值得为它把
/// 参数空间撑开。不满足条件就不发射（回落 CPU），不猜。
/// </para>
/// </summary>
internal sealed class VkSliceEmitter : IVkOpEmitter
{
    public static void Register() => VkOpRegistry.Register("Slice", new VkSliceEmitter());

    public bool CanEmit(LayerInfo op)
    {
        ArgumentNullException.ThrowIfNull(op);
        if (op.Type != "Slice" || op.Outputs.Count != 1) return false;
        if (op.Inputs.Count is < 3 or > 5) return false;
        if (!VkArgs.IsF32(op, op.Inputs[0]) || !VkArgs.IsF32(op, op.Outputs[0])) return false;

        // starts/ends 必须是常量。若来自运行时张量，输出形状要到 Forward 才算得出来，
        // 而段内槽位必须在 Finalize 时就开好——这种情形只能留给 CPU。
        if (!TryConstInts(op, 1, out int[] starts) || !TryConstInts(op, 2, out int[] ends)) return false;
        if (starts.Length != 1 || ends.Length != 1) return false;   // 单轴

        if (!OptionalConstInts(op, 3, out int[]? axes)) return false;
        if (axes is not null && axes.Length != 1) return false;      // 多轴不发射

        if (!OptionalConstInts(op, 4, out int[]? steps)) return false;
        if (steps is not null && (steps.Length != 1 || steps[0] != 1)) return false;  // 负步长/跨步不发射

        return true;
    }

    public void Emit(VkEmitContext ctx)
    {
        LayerInfo op = ctx.Op;
        TensorShape inShape = ctx.ShapeOf(op.Inputs[0]);
        TensorShape outShape = ctx.ShapeOf(op.Outputs[0]);
        int rank = inShape.Rank;
        if (outShape.Rank != rank)
            throw new NotSupportedException(
                $"'{op.Name}': 输出 rank={outShape.Rank} 与输入 {rank} 不一致");

        TryConstInts(op, 1, out int[] starts);
        OptionalConstInts(op, 3, out int[]? axes);

        int axis = axes is { Length: 1 } ? axes[0] : 0;
        if (axis < 0) axis += rank;
        if (axis < 0 || axis >= rank)
            throw new NotSupportedException($"'{op.Name}': axis={axis} 超出 rank={rank}");

        // 除切片轴外每一维都必须原样保留——这是「一次轴区间拷贝」成立的前提。
        for (int d = 0; d < rank; d++)
        {
            if (d == axis) continue;
            if (inShape[d] != outShape[d])
                throw new NotSupportedException(
                    $"'{op.Name}': 第 {d} 维 {inShape[d]} → {outShape[d]} 变了（只支持沿单轴切片）");
        }

        long srcAxis = inShape[axis];
        long len = outShape[axis];
        if (len <= 0) return;   // 空切片：没有要搬的，也不该开 dispatch

        long start = starts[0];
        if (start < 0) start += srcAxis;
        start = Math.Min(Math.Max(start, 0), srcAxis);

        long outer = VkCopyAxis.Product(inShape, 0, axis);
        long inner = VkCopyAxis.Product(inShape, axis + 1, rank);

        VkCopyAxis.Dispatch(ctx, op.Inputs[0], op.Outputs[0], outer * len * inner,
                            outer, inner, len, srcAxis, start, len, 0);
    }

    /// <summary>读第 <paramref name="index"/> 个输入常量并展开成 int[]。</summary>
    private static bool TryConstInts(LayerInfo op, int index, out int[] values)
    {
        values = [];
        if (index >= op.Inputs.Count) return false;
        Arg a = op.Inputs[index];
        if (a.IsEmpty || !VkArgs.IsConst(op, a)) return false;
        values = ReadInts(op.Net!.GetArgTensor(a));
        return true;
    }

    /// <summary>可选输入：不存在或为空（Idx==0）都算「用默认值」，但存在就必须是常量。</summary>
    private static bool OptionalConstInts(LayerInfo op, int index, out int[]? values)
    {
        values = null;
        if (index >= op.Inputs.Count) return true;
        Arg a = op.Inputs[index];
        if (a.IsEmpty) return true;
        if (!VkArgs.IsConst(op, a)) return false;
        values = ReadInts(op.Net!.GetArgTensor(a));
        return true;
    }

    /// <summary>
    /// 常量张量 → int[]，越界饱和。ONNX 的 Slice 用 INT64_MAX/MIN 表示「到边界」，
    /// 直接截断成 int 会绕回一个正数或负数，后面再 clamp 就错了。
    /// </summary>
    private static int[] ReadInts(Tensor t)
    {
        int count = checked((int)t.Count);
        var result = new int[count];
        switch (t.DType)
        {
            case DType.I64:
            {
                ReadOnlySpan<long> s = t.AsSpan<long>();
                for (int i = 0; i < count; i++)
                    result[i] = s[i] > int.MaxValue ? int.MaxValue
                              : s[i] < int.MinValue ? int.MinValue
                              : (int)s[i];
                break;
            }
            case DType.I32:
                t.AsSpan<int>().CopyTo(result);
                break;
            case DType.F32:
            {
                ReadOnlySpan<float> s = t.AsSpan<float>();
                for (int i = 0; i < count; i++) result[i] = (int)s[i];
                break;
            }
            default:
                throw new NotSupportedException($"Slice 参数的 dtype {t.DType} 不支持");
        }
        return result;
    }
}

/// <summary>Concat：多个输入沿 axis 依次写进输出的连续轴区间。</summary>
internal sealed class VkConcatEmitter : IVkOpEmitter
{
    public static void Register() => VkOpRegistry.Register("Concat", new VkConcatEmitter());

    public bool CanEmit(LayerInfo op)
    {
        ArgumentNullException.ThrowIfNull(op);
        if (op.Type != "Concat" || op.Outputs.Count != 1) return false;
        if (op.Inputs.Count < 1) return false;

        // 属性与 dtype 在规划期就已知；形状**不**是——规划发生在形状推断之前，
        // 用形状做门会把所有 Concat 都挡掉（这条注释是防止以后有人「顺手加个 rank 检查」）。
        if (op.Parameters is null || !op.Parameters.Has("axis")) return false;

        foreach (Arg a in op.Inputs)
            if (!VkArgs.IsF32(op, a)) return false;
        return VkArgs.IsF32(op, op.Outputs[0]);
    }

    public void Emit(VkEmitContext ctx)
    {
        LayerInfo op = ctx.Op;
        TensorShape outShape = ctx.ShapeOf(op.Outputs[0]);
        int rank = outShape.Rank;

        if (!VkCopyAxis.TryAxis(op, rank, out int axis))
            throw new NotSupportedException($"'{op.Name}': axis 属性缺失或超出 rank={rank}");

        long outer = VkCopyAxis.Product(outShape, 0, axis);
        long inner = VkCopyAxis.Product(outShape, axis + 1, rank);
        long outAxis = outShape[axis];

        Arg dst = op.Outputs[0];

        // 先把每个源的轴长算齐并校验，再决定「合并」还是「逐源」：
        // 两条路径共用同一份 dims，形状校验只写一遍，避免合并路径漏检。
        Span<long> dims = op.Inputs.Count <= 16 ? stackalloc long[op.Inputs.Count] : new long[op.Inputs.Count];
        long sum = 0;
        for (int k = 0; k < op.Inputs.Count; k++)
        {
            TensorShape s = VkArgs.ShapeOf(ctx, op.Inputs[k]);
            if (s.Rank != rank)
                throw new NotSupportedException(
                    $"'{op.Name}': 输入 rank={s.Rank} 与输出 rank={rank} 不一致");

            // 除拼接轴外必须逐维相等：ONNX 的语义要求，也是「一段连续区间」成立的前提
            for (int d = 0; d < rank; d++)
            {
                if (d == axis) continue;
                if (s[d] != outShape[d])
                    throw new NotSupportedException(
                        $"'{op.Name}': 第 {d} 维 {s[d]} 与输出 {outShape[d]} 不一致（只支持沿单轴拼接）");
            }

            dims[k] = s[axis];
            sum += dims[k];
        }

        if (sum != outAxis)
            throw new OnnxSharpException(
                $"'{op.Name}': 各输入轴长之和 {sum} 与输出轴长 {outAxis} 不一致（模型或形状推断有误）");

        // 合并路径：源数 2..4 且 inner 能被 float4 整除时，一次 dispatch 拼完。
        // 单源没有合并收益（多出 3 个空 binding），退回 copy_axis 更省。
        // 否则退回逐源 copy_axis——正确性一样，只是多几个同步点。
        if (op.Inputs.Count is >= 2 and <= VkCopyAxis.MaxConcatSrc && (inner & 3) == 0 && VkZero.Vec4Copy)
        {
            VkCopyAxis.DispatchConcat(ctx, dst, op.Inputs, outer, inner, dims, outAxis, 0);
            return;
        }

        long off = 0;
        for (int k = 0; k < op.Inputs.Count; k++)
        {
            long dim = dims[k];
            VkCopyAxis.Dispatch(ctx, op.Inputs[k], dst, outer * dim * inner, outer, inner, dim,
                                dim, 0, outAxis, off);
            off += dim;
        }
    }
}

/// <summary>Split：一个输入沿 axis 切成多个输出，每个输出对应源上的一段轴区间。</summary>
internal sealed class VkSplitEmitter : IVkOpEmitter
{
    public static void Register() => VkOpRegistry.Register("Split", new VkSplitEmitter());

    public bool CanEmit(LayerInfo op)
    {
        ArgumentNullException.ThrowIfNull(op);
        if (op.Type != "Split" || op.Outputs.Count < 1) return false;
        if (op.Inputs.Count is < 1 or > 2) return false;   // 第 2 个输入是 split 长度（常量）

        // 同上：规划期只能看属性与 dtype。split 长度若作为输入给出（opset 13+），
        // 那是常量，形状在段内推断时已经确定，不影响这里的判断。
        if (op.Parameters is null || !op.Parameters.Has("axis")) return false;

        // 第 0 个输入才是要切的数据；第 2 个输入（opset 13+ 的 split 长度）是 int64，
        // 拿 IsF32 去卡它会把**所有** Split 都挡在门外。
        // 它也不必是常量：切点由**输出形状**唯一确定，段内形状推断已经算好了每段多长，
        // 那个张量只是把同样的信息再写一遍（yolo 的 attention 里它就是动态算出来的）。
        if (!VkArgs.IsF32(op, op.Inputs[0])) return false;

        foreach (Arg a in op.Outputs)
            if (!a.IsEmpty && !VkArgs.IsF32(op, a)) return false;
        return true;
    }

    public void Emit(VkEmitContext ctx)
    {
        LayerInfo op = ctx.Op;
        TensorShape inShape = ctx.ShapeOf(op.Inputs[0]);
        int rank = inShape.Rank;

        if (!VkCopyAxis.TryAxis(op, rank, out int axis))
            throw new NotSupportedException($"'{op.Name}': axis 属性缺失或超出 rank={rank}");

        long inner = VkCopyAxis.Product(inShape, axis + 1, rank);
        long inAxis = inShape[axis];

        Arg src = op.Inputs[0];
        long off = 0;
        foreach (Arg a in op.Outputs)
        {
            if (a.IsEmpty) continue;
            TensorShape s = ctx.ShapeOf(a);
            if (s.Rank != rank)
                throw new NotSupportedException(
                    $"'{op.Name}': 输出 rank={s.Rank} 与输入 rank={rank} 不一致");

            for (int d = 0; d < rank; d++)
            {
                if (d == axis) continue;
                if (s[d] != inShape[d])
                    throw new NotSupportedException(
                        $"'{op.Name}': 第 {d} 维 {s[d]} 与输入 {inShape[d]} 不一致（只支持沿单轴切分）");
            }

            long dim = s[axis];
            long outer = VkCopyAxis.Product(s, 0, axis);
            VkCopyAxis.Dispatch(ctx, src, a, s.Total(), outer, inner, dim,
                                inAxis, off, dim, 0);
            off += dim;
        }

        if (off != inAxis)
            throw new OnnxSharpException(
                $"'{op.Name}': 各输出轴长之和 {off} 与输入轴长 {inAxis} 不一致（模型或形状推断有误）");
    }
}
