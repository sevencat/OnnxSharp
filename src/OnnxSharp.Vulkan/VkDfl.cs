// DFL 解码的 GPU 版：`out[b, s, n] = Σ_k softmax_k(x[b, s*bins + k, n]) · k`。
//
// 为什么它值得单独一个发射器：在 yolo11n 上它是检测头里的一个大断点。
// 它发不出去，前面的 Sigmoid、后面的两个 Slice 就都得回 CPU，整条输出分支被切成
// 两段——多一次 host 往返。而它自己其实是「每个输出元素读 16 个输入」的规整访存，
// 没有任何需要 CPU 参与的理由。
//
// 语义与 CPU 侧 Ops/DflOps.cs 的 DflLayer 完全对应（那边是融合后的等价实现，
// 原始子图是 Reshape → Transpose → Softmax → Conv(arange) → Reshape）。
// 这里按语义重算，不物化 softmax、不转置。
using System.Runtime.InteropServices;
using OnnxSharp.Core;
using OnnxSharp.Tensors;

namespace OnnxSharp.Vulkan;

/// <summary><c>dfl_softmax.comp</c> 的 push constant（16 B）。字段顺序必须与 shader 一致。</summary>
[StructLayout(LayoutKind.Sequential)]
internal struct DflParams
{
    public uint N;
    public uint Bins;
    public uint Sides;
    public uint Pixels;
}

/// <summary><c>DflSoftmax</c> 的发射器：输入 <c>[B, S·bins, N]</c>，输出 <c>[B, S, N]</c>。</summary>
internal sealed class VkDflEmitter : IVkOpEmitter
{
    /// <summary>shader 名（对应 Shaders/dfl_softmax.comp）。</summary>
    public const string Shader = "dfl_softmax";

    /// <summary>push constant 字节数，必须与 <see cref="DflParams"/> 一致。</summary>
    public const int PushConstBytes = 16;

    public static void Register() => VkOpRegistry.Register("DflSoftmax", new VkDflEmitter());

    public bool CanEmit(LayerInfo op)
    {
        ArgumentNullException.ThrowIfNull(op);
        if (op.Type != "DflSoftmax" || op.Inputs.Count != 1 || op.Outputs.Count != 1) return false;
        return VkArgs.IsF32(op, op.Inputs[0]) && VkArgs.IsF32(op, op.Outputs[0]);
    }

    public unsafe void Emit(VkEmitContext ctx)
    {
        LayerInfo op = ctx.Op;
        TensorShape inShape = ctx.ShapeOf(op.Inputs[0]);
        TensorShape outShape = ctx.ShapeOf(op.Outputs[0]);

        long bins = op.Parameters is not null ? op.Parameters.GetInt("bins", 16) : 16;
        if (bins < 2 || bins > uint.MaxValue)
            throw new NotSupportedException($"'{op.Name}': bins={bins} 不支持");

        if (inShape.Rank != 3 || outShape.Rank != 3)
            throw new NotSupportedException(
                $"'{op.Name}': 期望 3 维输入输出，实际 {inShape.Rank} → {outShape.Rank}");

        // 段内形状推断已经用它自己的规则算过一遍；这里再核一次不是因为不信它，
        // 而是因为 push constant 里只有 bins/sides/pixels 三个数，算错的偏移会静默读到别的张量。
        if (inShape[0] != outShape[0] || inShape[1] != outShape[1] * bins || inShape[2] != outShape[2])
            throw new OnnxSharpException(
                $"'{op.Name}': 形状 {inShape} → {outShape} 与 bins={bins} 不自洽");

        long total = outShape.Total();
        if (total <= 0 || total > uint.MaxValue)
            throw new NotSupportedException($"'{op.Name}': 输出元素数 {total} 不支持");

        long pixels = outShape[2];
        if (pixels <= 0 || pixels > uint.MaxValue)
            throw new NotSupportedException($"'{op.Name}': 像素数 {pixels} 不支持");

        var pc = new DflParams
        {
            N = (uint)total,
            Bins = (uint)bins,
            Sides = (uint)outShape[1],
            Pixels = (uint)pixels,
        };

        ReadOnlySpan<byte> pcb = new(&pc, PushConstBytes);
        ctx.Layer.AddDispatch(Shader, 2, PushConstBytes,
            [VkArgs.Bind(ctx, op.Inputs[0]), ctx.Out(op.Outputs[0])], pcb, (ulong)total);
    }
}
