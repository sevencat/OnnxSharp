// MaxPool 的发射器。
//
// 为什么优先补它：yolo11n 的 SPPF 是 `Conv → MaxPool(5×5,s1,p2) ×3 → Concat → Conv`，
// 三个池化夹在大段中间，每一个都独立成 1-op 段被成本门丢回 CPU，于是主干被切成四截。
// 它的算力可以忽略（20×20 上三个 5×5 窗口），价值全在「别把段切断」。
//
// 只做 NCHW + F32 + 无 dilation + 单输出：SPPF 就是这形态，而通用的
// ceil_mode/dilations/indices 输出在真实检测模型里极少出现，写全了也没人跑。
// 其余情形一律不发射，让 CPU 接管，行为与现在一致。
using System.Runtime.InteropServices;
using OnnxSharp.Core;
using OnnxSharp.Tensors;

namespace OnnxSharp.Vulkan;

/// <summary><c>maxpool.comp</c> 的 push constant。字段顺序必须与 shader 完全一致（48 B）。</summary>
[StructLayout(LayoutKind.Sequential)]
internal struct MaxPoolParams
{
    public uint Total;   // 输出元素数（N*C*OH*OW）
    public uint C;       // N*C（通道与 batch 合并：池化不跨这两维，合并后索引更短）
    public uint H;
    public uint W;
    public uint OH;
    public uint OW;
    public uint KH;
    public uint KW;
    public uint SH;
    public uint SW;
    public uint PH;
    public uint PW;
}

/// <summary>MaxPool（NCHW、F32、单输出、无 dilation）的发射器。</summary>
internal sealed class VkMaxPoolEmitter : IVkOpEmitter
{
    public const string Shader = "maxpool";
    public const int PushConstBytes = 48;

    public static void Register() => VkOpRegistry.Register("MaxPool", new VkMaxPoolEmitter());

    public bool CanEmit(LayerInfo op)
    {
        ArgumentNullException.ThrowIfNull(op);
        if (op.Type != "MaxPool" || op.Inputs.Count < 1 || op.Outputs.Count != 1) return false;
        if (op.Parameters is null) return false;

        // kernel 在规划期就能读（属性而已，不依赖形状推断）
        long[] kernel = op.Parameters.GetInts("kernel_shape") ?? [];
        if (kernel.Length != 2 || kernel[0] <= 0 || kernel[1] <= 0) return false;

        // 带 dilation 或 ceil_mode 的窗口边界算法不同，不做半吊子支持
        long[] dil = op.Parameters.GetInts("dilations") ?? [];
        foreach (long d in dil) if (d != 1) return false;
        if (op.Parameters.GetInt("ceil_mode", 0) != 0) return false;

        return VkArgs.IsF32(op, op.Inputs[0]) && VkArgs.IsF32(op, op.Outputs[0]);
    }

    public void Emit(VkEmitContext ctx)
    {
        LayerInfo op = ctx.Op;
        TensorShape inShape = ctx.ShapeOf(op.Inputs[0]);
        TensorShape outShape = ctx.ShapeOf(op.Outputs[0]);
        if (inShape.Rank != 4 || outShape.Rank != 4)
            throw new NotSupportedException(
                $"'{op.Name}': MaxPool 只支持 4D 输入，实际 {inShape.Rank}D");

        long[] kernel = op.Parameters!.GetInts("kernel_shape") ?? [];
        long[] strides = op.Parameters.GetInts("strides") ?? [1, 1];
        long[] pads = op.Parameters.GetInts("pads") ?? [0, 0, 0, 0];

        long n = inShape[0], c = inShape[1], h = inShape[2], w = inShape[3];
        long oh = outShape[2], ow = outShape[3];

        var pc = new MaxPoolParams
        {
            Total = checked((uint)(n * c * oh * ow)),
            C = checked((uint)(n * c)),
            H = (uint)h,
            W = (uint)w,
            OH = (uint)oh,
            OW = (uint)ow,
            KH = (uint)kernel[0],
            KW = (uint)kernel[1],
            SH = (uint)strides[0],
            SW = (uint)strides[1],
            PH = (uint)pads[0],
            PW = (uint)pads[1],
        };

        unsafe
        {
            ReadOnlySpan<byte> pcb = new(&pc, PushConstBytes);
            ctx.Layer.AddDispatch(Shader, 2, PushConstBytes,
                [VkArgs.Bind(ctx, op.Inputs[0]), ctx.Out(op.Outputs[0])],
                pcb, pc.Total);
        }
    }
}
