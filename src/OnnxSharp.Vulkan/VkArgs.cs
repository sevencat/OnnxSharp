// 发射器共用的小工具：arg → 形状 / 绑定 / dtype。
//
// 为什么值得单独一个文件：每个算子都要回答同样的三个问题，而每个问题的答案都有陷阱：
//   * 形状：段内槽位的形状来自段内推断（可能晚于规划），常量输入的形状来自加载期；
//   * 绑定：常量走权重切片（常驻），激活走段内槽位（每帧 stage）；
//   * dtype：规划期只能靠 importer 填的初始值，未绑定网络时不能猜。
// 三处各写一遍，迟早会出现「Conv 用槽位、Add 用权重」这种不一致。
using OnnxSharp.Core;
using OnnxSharp.Tensors;

namespace OnnxSharp.Vulkan;

internal static class VkArgs
{
    /// <summary>该 arg 是否为常量（initializer）。未绑定网络时一律 false。</summary>
    public static bool IsConst(LayerInfo op, Arg arg) =>
        !arg.IsEmpty && op.Net is not null && op.Net.GetArgKind(arg) == ArgKind.Const;

    /// <summary>
    /// dtype 检查。未绑定网络时拒绝——规划期要能挡住 int64/FP16，
    /// 就得靠 importer 填好的 <c>ArgData.DType</c>；查不到就别发射，不要猜。
    /// </summary>
    public static bool IsF32(LayerInfo op, Arg arg)
    {
        if (arg.IsEmpty || op.Net is null) return false;
        return op.Net.GetArgData(arg).DType == DType.F32;
    }

    /// <summary>
    /// 规划期试读常量形状（不抛异常）：规划发生在形状推断之前，只有常量带静态形状。
    /// 用它做「是不是 2-D 卷积」这类判断，才不会因为激活形状未知而漏发射。
    /// </summary>
    public static bool TryConstShape(LayerInfo op, Arg arg, out TensorShape shape)
    {
        shape = default;
        if (!IsConst(op, arg)) return false;
        ArgData data = op.Net!.GetArgData(arg);
        if (!data.HasShape) return false;
        shape = data.Shape;
        return true;
    }

    /// <summary>重量级形状查询：常量读加载期的静态形状，激活读段内推断结果。</summary>
    public static TensorShape ShapeOf(VkEmitContext ctx, Arg arg)
    {
        LayerInfo op = ctx.Op;
        if (!IsConst(op, arg)) return ctx.ShapeOf(arg);

        if (op.Net is null)
            throw new OnnxSharpException($"'{op.Name}' 还没绑定网络，读不到常量形状");
        ArgData data = op.Net.GetArgData(arg);
        if (!data.HasShape)
            throw new OnnxSharpException($"'{op.Name}' 的常量输入 arg#{arg.Idx} 没有静态形状");
        return data.Shape;
    }

    /// <summary>绑定：常量 → 只读权重切片（常驻显存），激活 → 段内槽位。</summary>
    public static VkBind Bind(VkEmitContext ctx, Arg arg) =>
        IsConst(ctx.Op, arg) ? ctx.WeightOf(arg) : ctx.SlotOf(arg);
}
