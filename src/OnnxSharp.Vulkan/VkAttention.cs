// attention 子图的三个发射器：Transpose / MatMul / Softmax。
//
// 它们存在的理由不是算力——这三个算子加起来一帧不到 0.2 ms——而是**段边界**。
// yolo11n 的 C2PSA 用了一个完整的 attention，它把图切成两半：
//
//   vkgpu0(58 op) ─ Transpose ─ MatMul ─ vkgpu1(2 op) ─ Softmax ─ Transpose ─ MatMul ─ vkgpu2(96 op)
//
// 每个断点都要付一次 host 往返（实测每段地板 3 ms 量级：stage in/out + submit + fence）。
// 三个算子都发得出去，这条链就能连成一段。
//
// 三个都只支持 yolo 用到的形态，不追求通用：
//   Transpose —— 任意 perm，rank ≤ 8（这是最通用的一面，实现成本反而最低）
//   MatMul    —— A [.., M, K] × B [.., K, N]，两边 batch 必须相同
//   Softmax   —— 只支持 axis = -1
using System.Runtime.InteropServices;
using OnnxSharp.Core;
using OnnxSharp.Tensors;

namespace OnnxSharp.Vulkan;

/// <summary><c>transpose.comp</c> 的 push constant（72 B）。</summary>
[StructLayout(LayoutKind.Sequential)]
internal unsafe struct TransposeParams
{
    public uint N;
    public uint Rank;
    public fixed uint Od[8];
    public fixed uint Istr[8];
}

/// <summary><c>softmax_last.comp</c> 的 push constant（8 B）。</summary>
[StructLayout(LayoutKind.Sequential)]
internal struct SoftmaxParams
{
    public uint Cols;
    public uint Rows;
}

/// <summary><c>matmul.comp</c> 的 push constant（16 B）。</summary>
[StructLayout(LayoutKind.Sequential)]
internal struct MatMulParams
{
    public uint M;
    public uint N;
    public uint K;
    public uint Batch;
}

/// <summary>任意维置换。输出第 d 维取输入第 perm[d] 维。</summary>
internal sealed class VkTransposeEmitter : IVkOpEmitter
{
    public const string Shader = "transpose";
    public const int PushConstBytes = 72;

    /// <summary>shader 里 od/istr 各有 8 个槽。</summary>
    private const int MaxRank = 8;

    public static void Register() => VkOpRegistry.Register("Transpose", new VkTransposeEmitter());

    public bool CanEmit(LayerInfo op)
    {
        ArgumentNullException.ThrowIfNull(op);
        if (op.Type != "Transpose" || op.Inputs.Count != 1 || op.Outputs.Count != 1) return false;
        return VkArgs.IsF32(op, op.Inputs[0]) && VkArgs.IsF32(op, op.Outputs[0]);
    }

    public unsafe void Emit(VkEmitContext ctx)
    {
        LayerInfo op = ctx.Op;
        TensorShape inShape = ctx.ShapeOf(op.Inputs[0]);
        TensorShape outShape = ctx.ShapeOf(op.Outputs[0]);
        int rank = inShape.Rank;
        if (rank != outShape.Rank)
            throw new NotSupportedException($"'{op.Name}': 输入 rank={rank} 与输出 {outShape.Rank} 不一致");
        if (rank > MaxRank)
            throw new NotSupportedException($"'{op.Name}': rank={rank} 超过 {MaxRank}");

        long[] perm;
        if (op.Parameters is not null && op.Parameters.Has("perm"))
        {
            perm = op.Parameters.GetInts("perm")
                   ?? throw new NotSupportedException($"'{op.Name}': perm 属性无法解析");
        }
        else
        {
            // ONNX 的默认 perm 是维度反序
            perm = new long[rank];
            for (int i = 0; i < rank; i++) perm[i] = rank - 1 - i;
        }

        if (perm.Length != rank)
            throw new NotSupportedException($"'{op.Name}': perm 长度 {perm.Length} ≠ rank {rank}");

        long total = outShape.Total();
        if (total <= 0 || total > uint.MaxValue)
            throw new NotSupportedException($"'{op.Name}': 输出元素数 {total} 不支持");

        var pc = new TransposeParams { N = (uint)total, Rank = (uint)rank };
        for (int d = 0; d < rank; d++)
        {
            int p = (int)perm[d];
            if (p < 0 || p >= rank)
                throw new NotSupportedException($"'{op.Name}': perm[{d}]={p} 超出 rank={rank}");

            // 第 d 个输出维的长度，必须等于输入第 p 维的长度（形状推断已经保证，这里只做断言）
            if (outShape[d] != inShape[p])
                throw new OnnxSharpException(
                    $"'{op.Name}': out[{d}]={outShape[d]} 与 in[perm[{d}]={p}]={inShape[p]} 不一致");

            pc.Od[d] = (uint)outShape[d];

            // 输入在第 p 维上的步幅（元素为单位）
            long stride = 1;
            for (int i = p + 1; i < rank; i++) stride *= inShape[i];
            pc.Istr[d] = (uint)stride;
        }

        ReadOnlySpan<byte> pcb = new(&pc, PushConstBytes);
        ctx.Layer.AddDispatch(Shader, 2, PushConstBytes,
            [VkArgs.Bind(ctx, op.Inputs[0]), ctx.Out(op.Outputs[0])], pcb, (ulong)total);
    }
}

/// <summary>沿最后一维的 Softmax。一个工作组处理一行。</summary>
internal sealed class VkSoftmaxEmitter : IVkOpEmitter
{
    public const string Shader = "softmax_last";

    /// <summary>与 shader 里的 local_size_x 一致；行内归约靠它做分片。</summary>
    private const int ThreadsPerRow = 64;

    public static void Register() => VkOpRegistry.Register("Softmax", new VkSoftmaxEmitter());

    public bool CanEmit(LayerInfo op)
    {
        ArgumentNullException.ThrowIfNull(op);
        if (op.Type != "Softmax" || op.Inputs.Count != 1 || op.Outputs.Count != 1) return false;
        if (!VkArgs.IsF32(op, op.Inputs[0]) || !VkArgs.IsF32(op, op.Outputs[0])) return false;

        // 规划期不知道 rank，所以只认 axis = -1（恒为最后一维）。
        // 非负的 axis 也可能是最后一维，但那要等 Emit 才知道；这里放行会让
        // 「axis=1 但 rank=2」这类情形漏进来，不如直接留给 CPU。
        long axis = op.Parameters is not null ? op.Parameters.GetInt("axis", -1) : -1;
        return axis == -1;
    }

    public unsafe void Emit(VkEmitContext ctx)
    {
        LayerInfo op = ctx.Op;
        TensorShape outShape = ctx.ShapeOf(op.Outputs[0]);
        if (outShape.Rank < 1)
            throw new NotSupportedException($"'{op.Name}': 输出 rank={outShape.Rank}");

        long cols = outShape[outShape.Rank - 1];
        long total = outShape.Total();
        if (cols <= 0 || total <= 0 || total > uint.MaxValue || cols > uint.MaxValue)
            throw new NotSupportedException($"'{op.Name}': 形状 {outShape} 不支持");

        long rows = total / cols;

        var pc = new SoftmaxParams { Cols = (uint)cols, Rows = (uint)rows };
        ReadOnlySpan<byte> pcb = new(&pc, sizeof(SoftmaxParams));

        // Dispatch1D 按 local_size_total 折算组数。这里要的是「每行一个工作组」，
        // 所以喂 rows × 64 个元素，算出来正好是 rows 组。
        ctx.Layer.AddDispatch(Shader, 2, sizeof(SoftmaxParams),
            [VkArgs.Bind(ctx, op.Inputs[0]), ctx.Out(op.Outputs[0])], pcb,
            (ulong)rows * ThreadsPerRow);
    }
}

/// <summary>批量矩阵乘：A [B, M, K] × B [B, K, N] → C [B, M, N]。</summary>
internal sealed class VkMatMulEmitter : IVkOpEmitter
{
    public const string Shader = "matmul";

    /// <summary>与 shader 的 local_size_x × local_size_y 一致（16×16）。</summary>
    private const int Tile = 16;
    private const int ThreadsPerGroup = Tile * Tile;

    public static void Register() => VkOpRegistry.Register("MatMul", new VkMatMulEmitter());

    public bool CanEmit(LayerInfo op)
    {
        ArgumentNullException.ThrowIfNull(op);
        if (op.Type != "MatMul" || op.Inputs.Count != 2 || op.Outputs.Count != 1) return false;
        return VkArgs.IsF32(op, op.Inputs[0]) && VkArgs.IsF32(op, op.Inputs[1])
            && VkArgs.IsF32(op, op.Outputs[0]);
    }

    public unsafe void Emit(VkEmitContext ctx)
    {
        LayerInfo op = ctx.Op;
        TensorShape aShape = ctx.ShapeOf(op.Inputs[0]);
        TensorShape bShape = ctx.ShapeOf(op.Inputs[1]);
        TensorShape cShape = ctx.ShapeOf(op.Outputs[0]);

        if (aShape.Rank < 2 || bShape.Rank < 2 || cShape.Rank < 2)
            throw new NotSupportedException(
                $"'{op.Name}': 只支持 rank ≥ 2（实际 A={aShape.Rank} B={bShape.Rank} C={cShape.Rank}）");

        long m = aShape[aShape.Rank - 2];
        long k = aShape[aShape.Rank - 1];
        long k2 = bShape[bShape.Rank - 2];
        long n = bShape[bShape.Rank - 1];

        if (k != k2)
            throw new OnnxSharpException($"'{op.Name}': A 的 K={k} 与 B 的 K={k2} 不一致");

        long aBatch = aShape.Total() / Math.Max(1, m * k);
        long bBatch = bShape.Total() / Math.Max(1, k * n);
        if (aBatch != bBatch)
            throw new NotSupportedException(
                $"'{op.Name}': 暂不支持 batch 广播（A batch={aBatch}, B batch={bBatch}）");
        if (aBatch <= 0 || m <= 0 || n <= 0 || k <= 0)
            throw new NotSupportedException(
                $"'{op.Name}': 形状 {aShape} × {bShape} → {cShape} 不支持");
        if (aBatch * m * n != cShape.Total())
            throw new OnnxSharpException(
                $"'{op.Name}': 输出 {cShape} 与 {aBatch}×{m}×{n} 不自洽");

        if (aBatch > uint.MaxValue || m > uint.MaxValue || n > uint.MaxValue || k > uint.MaxValue)
            throw new NotSupportedException($"'{op.Name}': 维度超过 uint 上限");

        var pc = new MatMulParams { M = (uint)m, N = (uint)n, K = (uint)k, Batch = (uint)aBatch };
        ReadOnlySpan<byte> pcb = new(&pc, sizeof(MatMulParams));

        long tilesM = (m + Tile - 1) / Tile;
        long tilesN = (n + Tile - 1) / Tile;
        long groups = aBatch * tilesM * tilesN;

        ctx.Layer.AddDispatch(Shader, 3, sizeof(MatMulParams),
            [VkArgs.Bind(ctx, op.Inputs[0]), VkArgs.Bind(ctx, op.Inputs[1]),
             ctx.Out(op.Outputs[0])], pcb, (ulong)groups * ThreadsPerGroup);
    }
}
