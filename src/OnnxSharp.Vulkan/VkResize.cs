// Resize 的发射器（只做 nearest + asymmetric + floor，即 yolo 的 2× 上采样）。
//
// 为什么值得单独写：FPN 的两次上采样正好卡在 neck 的主干上，左边一路 Conv 段、
// 右边一路 Concat，它一旦不能发射，整条 neck 就被切成两截，多两次 host 往返。
// 它的算力是零（每个输出元素一次索引计算），价值全在连通性。
//
// 为什么只支持一种模式：Resize 的语义由 coordinate_transformation_mode × nearest_mode
// 两个字符串决定，组合有十几种，彼此结果不同且**都不报错**。与其写一个「大概对」的
// 通用核，不如只认 yolo 实际导出的那一种（nearest/asymmetric/floor），
// 其余原样留给 CPU——行为与不支持时完全一致，不会有静默错位。
using System.Runtime.InteropServices;
using OnnxSharp.Core;
using OnnxSharp.Tensors;

namespace OnnxSharp.Vulkan;

/// <summary><c>resize_near.comp</c> 的 push constant。字段顺序必须与 shader 完全一致（24 B）。</summary>
[StructLayout(LayoutKind.Sequential)]
internal struct ResizeNearParams
{
    public uint Total;
    public uint CN;
    public uint H;
    public uint W;
    public uint OH;
    public uint OW;
}

/// <summary>Resize（NCHW、nearest、asymmetric、floor）的发射器。</summary>
internal sealed class VkResizeEmitter : IVkOpEmitter
{
    public const string Shader = "resize_near";
    public const int PushConstBytes = 24;

    public static void Register() => VkOpRegistry.Register("Resize", new VkResizeEmitter());

    public bool CanEmit(LayerInfo op)
    {
        ArgumentNullException.ThrowIfNull(op);
        if (op.Type != "Resize" || op.Inputs.Count < 1 || op.Outputs.Count != 1) return false;
        if (op.Parameters is null) return false;

        // 缺省值与 CPU 侧 ResizeLayer 的缺省值一致：half_pixel / round_prefer_floor。
        // 也就是说「属性缺失」不算 asymmetric，必须显式写明——默认值恰好不在支持范围内。
        if (op.Parameters.GetString("mode", "nearest") != "nearest") return false;
        if (op.Parameters.GetString("coordinate_transformation_mode", "half_pixel") != "asymmetric")
            return false;
        if (op.Parameters.GetString("nearest_mode", "round_prefer_floor") != "floor") return false;

        return VkArgs.IsF32(op, op.Inputs[0]) && VkArgs.IsF32(op, op.Outputs[0]);
    }

    public void Emit(VkEmitContext ctx)
    {
        LayerInfo op = ctx.Op;
        TensorShape inShape = ctx.ShapeOf(op.Inputs[0]);
        TensorShape outShape = ctx.ShapeOf(op.Outputs[0]);
        if (inShape.Rank != 4 || outShape.Rank != 4)
            throw new NotSupportedException(
                $"'{op.Name}': Resize 只支持 4D（NCHW），实际 {inShape.Rank}D");

        long nc = inShape[0] * inShape[1];
        long total = nc * outShape[2] * outShape[3];

        var pc = new ResizeNearParams
        {
            Total = checked((uint)total),
            CN = (uint)nc,
            H = (uint)inShape[2],
            W = (uint)inShape[3],
            OH = (uint)outShape[2],
            OW = (uint)outShape[3],
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
