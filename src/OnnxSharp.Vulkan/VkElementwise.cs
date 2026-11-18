// P2 的第一个真实 GPU 算子：二元逐元素（Add/Sub/Mul/Div/Max/Min）。
//
// 为什么从这个算子起步：形状规则最简单（输出 = 广播结果），但覆盖面最大——
// 真实模型里 BN 之后的偏置加、残差相加、注意力里的 scale 乘几乎都落在它身上，
// 而且它在段里是最常见的「填充算子」：把 Conv 之间隔着的逐元素算子一并吞进段，
// 段长才够（MinOpsPerSegment=3），否则 Conv 永远凑不成段。
//
// 广播的做法见 shader 注释：右对齐 + 广播维步幅取 0，一份 shader 覆盖
// 同形、[1,C,1,1]、标量三种形态，不需要按形状特化编译。
//
// 常量输入（initializer）走权重绑定，不进 stage in：BN 的 gamma/beta 每帧重传一遍
// 纯属浪费，而权重在模型级常驻区里本来就有一份。
using System.Runtime.InteropServices;
using OnnxSharp.Core;
using OnnxSharp.Tensors;

namespace OnnxSharp.Vulkan;

/// <summary><c>binary_bcast.comp</c> 的 push constant。字段顺序必须与 shader 完全一致。</summary>
[StructLayout(LayoutKind.Sequential)]
internal unsafe struct BinaryBcastParams
{
    public uint N;
    public uint Rank;
    public fixed uint Od[VkElementwiseEmitter.MaxRank];
    public fixed uint Sa[VkElementwiseEmitter.MaxRank];
    public fixed uint Sb[VkElementwiseEmitter.MaxRank];
    public uint Kind;
}

/// <summary>Add/Sub/Mul/Div/Max/Min 的发射器（F32、支持 ONNX 右对齐广播）。</summary>
internal sealed class VkElementwiseEmitter : IVkOpEmitter
{
    /// <summary>rank 上限：push constant 预算 128 B 减去 n/rank/kind 后只够 3×8 个 uint。</summary>
    public const int MaxRank = 8;

    private const int KindAdd = 0;
    private const int KindSub = 1;
    private const int KindMul = 2;
    private const int KindDiv = 3;
    private const int KindMax = 4;
    private const int KindMin = 5;

    private static readonly Dictionary<string, int> Kinds = new(StringComparer.Ordinal)
    {
        ["Add"] = KindAdd,
        ["Sub"] = KindSub,
        ["Mul"] = KindMul,
        ["Div"] = KindDiv,
        ["Max"] = KindMax,
        ["Min"] = KindMin,
    };

    /// <summary>注册全部逐元素算子。P2 起由后端初始化统一调用。</summary>
    public static void Register()
    {
        foreach (string type in Kinds.Keys) VkOpRegistry.Register(type, new VkElementwiseEmitter());
    }

    /// <summary>可发射的算子类型（诊断用）。</summary>
    public static IEnumerable<string> SupportedTypes => Kinds.Keys;

    public bool CanEmit(LayerInfo op)
    {
        ArgumentNullException.ThrowIfNull(op);
        // 只认两个输入一个输出：多输入变体（Sum 的 N 元版）留给 P3
        if (op.Inputs.Count != 2 || op.Outputs.Count != 1) return false;
        if (!Kinds.ContainsKey(op.Type)) return false;

        // 规划发生在形状推断之前，但 dtype 在加载期就已知（ONNX 张量自带元素类型），
        // 所以这里能挡住 int64/FP16 的逐元素算子，不必等到建段才炸。
        return VkArgs.IsF32(op, op.Inputs[0]) && VkArgs.IsF32(op, op.Inputs[1])
               && VkArgs.IsF32(op, op.Outputs[0]);
    }

    public unsafe void Emit(VkEmitContext ctx)
    {
        LayerInfo op = ctx.Op;
        if (VkZero.SkipElementwise) return;   // 归因开关：测「砍掉逐元素 dispatch 能省多少」，数值会错
        TensorShape outShape = ctx.ShapeOf(op.Outputs[0]);
        if (outShape.Rank > MaxRank)
            throw new NotSupportedException(
                $"'{op.Name}' 的输出 rank={outShape.Rank} 超过 {MaxRank}（push constant 装不下）");

        long total = outShape.Total();
        if (total <= 0 || total > uint.MaxValue)
            throw new NotSupportedException($"'{op.Name}' 的输出元素数 {total} 不支持");

        int kind = Kinds[op.Type];

        var pc = new BinaryBcastParams { N = (uint)total, Rank = (uint)outShape.Rank, Kind = (uint)kind };
        FillStrides(ctx, op.Inputs[0], outShape, pc.Sa);
        FillStrides(ctx, op.Inputs[1], outShape, pc.Sb);
        for (int d = 0; d < outShape.Rank; d++) pc.Od[d] = (uint)outShape[d];

        ReadOnlySpan<byte> pcb = new(&pc, sizeof(BinaryBcastParams));
        ctx.Layer.AddDispatch("binary_bcast", 3, sizeof(BinaryBcastParams),
            [VkArgs.Bind(ctx, op.Inputs[0]), VkArgs.Bind(ctx, op.Inputs[1]), ctx.Out(op.Outputs[0])],
            pcb, (ulong)total);
    }

    /// <summary>
    /// 把输入形状右对齐到输出形状，算出每个输出维上的步幅：广播维取 0。
    /// 「维为 1」是 ONNX 广播的唯一情形，所以步幅要么 0 要么就是紧凑布局里的正常步幅。
    /// </summary>
    internal static unsafe void FillStrides(VkEmitContext ctx, Arg arg, TensorShape outShape, uint* dst)
    {
        LayerInfo op = ctx.Op;
        TensorShape inShape = VkArgs.ShapeOf(ctx, arg);
        int rank = outShape.Rank;
        int offset = rank - inShape.Rank; // 右对齐：左边补 1
        if (offset < 0)
            throw new NotSupportedException(
                $"'{op.Name}' 的输入 rank={inShape.Rank} 大于输出 rank={rank}");

        uint run = 1;
        for (int d = rank - 1; d >= 0; d--)
        {
            int dim = d - offset >= 0 ? inShape[d - offset] : 1;
            int outDim = outShape[d];

            if (dim != 1 && dim != outDim)
                throw new NotSupportedException(
                    $"'{op.Name}' 的输入形状 {inShape} 与输出 {outShape} 在第 {d} 维不满足广播规则");

            // dim == 1 时该维步幅 0：广播正是靠这一步退化成「重复取同一个值」
            dst[d] = dim == 1 ? 0u : run;
            run *= (uint)dim;
        }
    }

}

/// <summary>
/// 一元逐元素（当前只有 Sigmoid）：复用 <c>binary_bcast.comp</c> 的第 6 号 kind。
/// <para>
/// 为什么不单开一个 shader：形态与二元完全一致（一个元素进、一个元素出，广播规则同样适用），
/// 差别只是不读第二个输入。单开一份核意味着多一份 push constant 布局、多一个 .comp/.spv、
/// 多一处「改了忘记重编」。而这里只是 shader 里多一个分支、宿主侧把同一个槽位绑两次。
/// </para>
/// <para>
/// 它存在的唯一理由是**段边界**：yolo11n 的检测头里 Sigmoid 夹在两个大段之间，
/// 它一发不出去，后面的 8 个算子（DFL 之后的框解码）就只能各自成段，多两次 host 往返。
/// </para>
/// </summary>
internal sealed class VkUnaryEmitter : IVkOpEmitter
{
    private const int KindSigmoid = 6;

    private static readonly Dictionary<string, int> Kinds = new(StringComparer.Ordinal)
    {
        ["Sigmoid"] = KindSigmoid,
    };

    public static void Register()
    {
        foreach (string type in Kinds.Keys) VkOpRegistry.Register(type, new VkUnaryEmitter());
    }

    public bool CanEmit(LayerInfo op)
    {
        ArgumentNullException.ThrowIfNull(op);
        if (op.Inputs.Count != 1 || op.Outputs.Count != 1) return false;
        if (!Kinds.ContainsKey(op.Type)) return false;
        return VkArgs.IsF32(op, op.Inputs[0]) && VkArgs.IsF32(op, op.Outputs[0]);
    }

    public unsafe void Emit(VkEmitContext ctx)
    {
        LayerInfo op = ctx.Op;
        TensorShape outShape = ctx.ShapeOf(op.Outputs[0]);
        if (outShape.Rank > VkElementwiseEmitter.MaxRank)
            throw new NotSupportedException(
                $"'{op.Name}' 的输出 rank={outShape.Rank} 超过 {VkElementwiseEmitter.MaxRank}");

        long total = outShape.Total();
        if (total <= 0 || total > uint.MaxValue)
            throw new NotSupportedException($"'{op.Name}' 的输出元素数 {total} 不支持");

        var pc = new BinaryBcastParams
        {
            N = (uint)total,
            Rank = (uint)outShape.Rank,
            Kind = (uint)Kinds[op.Type],
        };
        VkElementwiseEmitter.FillStrides(ctx, op.Inputs[0], outShape, pc.Sa);
        for (int d = 0; d < outShape.Rank; d++) pc.Od[d] = (uint)outShape[d];

        ReadOnlySpan<byte> pcb = new(&pc, sizeof(BinaryBcastParams));

        // 第二个 binding 必须有效（描述符不能是空的），但 shader 在 kind==6 时直接返回，
        // 不会读它。绑同一个槽位最省事，也保证即使被 hoist 了读的也是合法地址。
        VkBind input = VkArgs.Bind(ctx, op.Inputs[0]);
        ctx.Layer.AddDispatch("binary_bcast", 3, sizeof(BinaryBcastParams),
            [input, input, ctx.Out(op.Outputs[0])], pcb, (ulong)total);
    }
}
