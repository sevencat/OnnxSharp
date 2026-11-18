// P2 的重算子：2-D 卷积（NCHW / F32 / 支持 groups、strides、dilations、pads、auto_pad）。
//
// 为什么第二站是 Conv 而不是激活/Pool：yolo 里 95% 的时间在卷积上，段能不能回本全看它。
// 逐元素算子单独上 GPU 是净亏（实测 16K 元素 CPU 143 us / GPU 323 us）——
// 它只有在「和 Conv 同段、中间结果不出显存」时才有价值。
//
// 这一版是朴素直接卷积，先求正确：
//   * 一个线程一个输出元素，内层 c → kh → kw；
//   * 权重按 ONNX 原样 [M, Cg, KH, KW] 绑定，不做任何重排；
//   * NCHW 进 NCHW 出，不做 NHWC 重排（P2-3）。
// 性能留到后面：1×1 点卷积占 yolo 绝大多数 FLOPs，值得单独特化（P2-2）；
// 3×3 在 NCHW 下的 gather 无法合并，要靠 NHWC + 向量载入（P2-3）。
//
// 三个容易写错的地方都在 shader 里显式注释了：pad 只从起始侧减、组号用输出通道算、
// 越界 continue 而不是 clamp。
using System.Runtime.InteropServices;
using OnnxSharp.Core;
using OnnxSharp.Tensors;

namespace OnnxSharp.Vulkan;

/// <summary><c>conv2d.comp</c> 的 push constant。字段顺序必须与 shader 完全一致（76 B）。</summary>
[StructLayout(LayoutKind.Sequential)]
internal struct Conv2DParams
{
    public uint Total;
    public uint Cg;
    public uint M;
    public uint H;
    public uint W;
    public uint OH;
    public uint OW;
    public uint KH;
    public uint KW;
    public uint SH;
    public uint SW;
    public uint DH;
    public uint DW;
    public uint PH;
    public uint PW;
    public uint Group;
    public uint OutPerGroup;
    public uint HasBias;

    /// <summary>
    /// 融合进卷积的逐元素激活：0 无 / 1 Relu / 2 Sigmoid / 3 SiLU / 4 Tanh。
    /// yolo 的每个卷积后面都跟着 SiLU，图融合后它挂在卷积头上；少了它结果会逐层放大。
    /// </summary>
    public uint Act;
}

/// <summary>2-D Conv 的发射器（F32、2-D、支持分组/步长/膨胀/填充）。</summary>
internal sealed unsafe class VkConvEmitter : IVkOpEmitter
{
    private const int Spatial = 2;

    /// <summary>"conv2d_1x1.comp" 每个线程的输出通道块大小，必须与 shader 里的 MT 一致。</summary>
    private const int PointwiseMT = 8;

    /// <summary>
    /// "conv2d_1x1.comp" 每个线程的连续像素块大小，必须与 shader 里的 TP 一致。
    /// 作用是把算术强度从 MT/2（TP=1 时 4.0 FLOP/B）抬到 TP×MT×2/(TP×4+MT×4) = 16.0，
    /// 也就是「一次寻址、TP×MT 次复用」。增大 TP 会线性涨寄存器（TP×MT 个累加器）：
    /// TP=4 时 32 个，是还能保住占用率的上限。
    /// <para>
    /// 实测：端到端 408–426 us vs TP=1 的 435 us（256→256 @32×32），差异在噪声边缘。
    /// 那个规模下端到端由 stage 搬运主导（~200/415 us 是 kernel），所以这个常量的价值
    /// 是「强度更高 + 尾组被覆盖」，不是已证实的加速。别再往上加 TP 换加速，先把搬运解决。
    /// </para>
    /// </summary>
    private const int PointwiseTP = 4;

    /// <summary>"conv3x3.comp" 每个线程的输出通道块大小，必须与 shader 里的 MT 一致。</summary>
    private const int Conv3x3MT = 8;

    /// <summary>
    /// "conv3x3.comp" 每个线程的连续像素块大小，必须与 shader 里的 TP 一致。
    /// 每 (c, kh) 载入 TP+2 个连续输入，被 MT×TP×3 次 MAC 复用；TP 越大复用越高，
    /// 但寄存器按 MT×TP 线性涨（TP=4 → 32 个累加器），这是还能保住占用率的上限。
    /// </summary>
    private const int Conv3x3TP = 4;

    /// <summary>"conv3x3s2.comp" 每个线程的输出通道块大小，必须与 shader 里的 MT 一致。</summary>
    private const int Conv3x3S2MT = 8;

    /// <summary>
    /// "conv3x3s2.comp" 每个线程的连续输出像素块大小，必须与 shader 里的 TP 一致。
    /// 步长 2 时这 TP 个像素共用的输入跨度是 <c>2·TP+1 = 9</c> 而不是 <c>TP+2</c>，
    /// 但「一次载入、MT×TP×3 次复用」的结构没变，所以 MT/TP 的取值理由与 conv3x3 相同。
    /// </summary>
    private const int Conv3x3S2TP = 4;

    /// <summary>
    /// 3×3 专用核开关（<c>ONNXSHARP_VK_CONV3=0</c> 关闭，退回朴素核）。
    /// 留这个口子不只是为了 A/B——特化核一旦在某些形状上算错或变慢，用户要能立刻回退
    /// 而不用改代码重编。
    /// </summary>
    public static bool Conv3x3Fast { get; set; } = ReadConv3Default();

    private static bool ReadConv3Default()
        => Environment.GetEnvironmentVariable("ONNXSHARP_VK_CONV3") is not "0";

    /// <summary>
    /// 这个卷积能不能走 3×3 专用核。**判据（<see cref="VkCostModel"/>）与发射必须共用
    /// 这一个谓词**——判据以为走快路径、实际发射了慢路径，就会把亏的段判成赚的。
    /// </summary>
    public static bool IsConv3x3Fast(int kh, int kw, int sh, int sw, int dh, int dw, long outPerGroup)
        => Conv3x3Fast && kh == 3 && kw == 3 && sh == 1 && sw == 1 && dh == 1 && dw == 1
           && outPerGroup % Conv3x3MT == 0;

    /// <summary>
    /// 这个卷积能不能走 stride-2 的 3×3 专用核（<c>conv3x3s2.comp</c>）。
    /// 与 <see cref="IsConv3x3Fast"/> 同理：判据与发射必须共用这一个谓词。
    /// <para>
    /// 步长 2 时 TP 个输出像素的输入仍是连续区间（跨度 2·TP+1），滑动窗口复用成立；
    /// 步长 ≥3 时跨度变成 3·TP+1 且抽样更稀疏，载入的样本被用到的比例掉到 1/3 以下，
    /// 收益消失——那些继续走朴素核。
    /// </para>
    /// </summary>
    public static bool IsConv3x3S2Fast(int kh, int kw, int sh, int sw, int dh, int dw, long outPerGroup)
        => Conv3x3Fast && kh == 3 && kw == 3 && sh == 2 && sw == 2 && dh == 1 && dw == 1
           && outPerGroup % Conv3x3S2MT == 0;

    public static void Register() => VkOpRegistry.Register("Conv", new VkConvEmitter());

    /// <summary>三个卷积核的 epilogue 都会补上 <see cref="LayerInfo.FusedActivationKind"/>（pc.Act）。</summary>
    public bool HandlesFusedActivation => true;

    public bool CanEmit(LayerInfo op)
    {
        ArgumentNullException.ThrowIfNull(op);
        if (op.Type != "Conv" || op.Outputs.Count != 1) return false;
        if (op.Inputs.Count is < 2 or > 3) return false;

        // 属性是卷积语义的一半（stride/pad/dilation/group 全在属性里）。
        // 拿不到属性就别发射：猜一个 stride=1 会静默算出「形状对、数值错」的结果。
        if (op.Parameters is null) return false;
        string autoPad = op.Parameters.GetString("auto_pad", "NOTSET");
        if (autoPad is not ("NOTSET" or "VALID" or "SAME_UPPER" or "SAME_LOWER")) return false;

        // 用**权重**的 rank 判断是不是 2-D：权重一定是常量，形状在加载期就已知；
        // 输入的 rank 在规划期可能还未知，用它会漏掉真正该发射的卷积。
        if (!VkArgs.TryConstShape(op, op.Inputs[1], out TensorShape ws) || ws.Rank != 4) return false;

        if (!VkArgs.IsF32(op, op.Inputs[0]) || !VkArgs.IsF32(op, op.Inputs[1])
            || !VkArgs.IsF32(op, op.Outputs[0])) return false;
        if (op.Inputs.Count == 3 && !op.Inputs[2].IsEmpty && !VkArgs.IsF32(op, op.Inputs[2])) return false;

        // 融合激活由 shader 的输出端补上。只认 shader 里真的实现了的那几种：
        // 漏一种就会发射一个「少一次激活」的卷积——形状全对、数值逐层放大，
        // 是本次 yolo err=25 的真凶。宁可退回 CPU 也不能静默算错。
        if (op.FusedActivationKind is { } act && ActCode(act) == 0) return false;

        return true;
    }

    /// <summary>
    /// 激活种类 → shader 的 <c>Act</c> 编号；0 表示「不激活 / shader 没实现」。
    /// 必须与 <c>conv2d.comp</c>、<c>conv2d_1x1.comp</c>、<c>conv3x3.comp</c> 里的分支逐一对齐。
    /// </summary>
    private static uint ActCode(UnaryOpKind? kind) => kind switch
    {
        null => 0u,
        UnaryOpKind.Relu => 1u,
        UnaryOpKind.Sigmoid => 2u,
        UnaryOpKind.Silu => 3u,
        UnaryOpKind.Tanh => 4u,
        _ => 0u,
    };

    public void Emit(VkEmitContext ctx)
    {
        LayerInfo op = ctx.Op;
        TensorShape xs = ctx.ShapeOf(op.Inputs[0]);   // [N, C, H, W]
        TensorShape ws = VkArgs.ShapeOf(ctx, op.Inputs[1]); // [M, Cg, KH, KW]
        TensorShape ys = ctx.ShapeOf(op.Outputs[0]);  // [N, M, OH, OW]

        if (xs.Rank != 4 || ws.Rank != 4 || ys.Rank != 4)
            throw new NotSupportedException(
                $"'{op.Name}': 只支持 2-D 卷积，实际 输入 {xs} / 权重 {ws} / 输出 {ys}");

        int n = xs[0], c = xs[1], h = xs[2], w = xs[3];
        int m = ws[0], cg = ws[1], kh = ws[2], kw = ws[3];
        int oh = ys[2], ow = ys[3];

        LayerParams p = op.Parameters ?? throw new OnnxSharpException($"'{op.Name}' 没有属性集合");
        int group = (int)p.GetInt("group", 1);
        if (group <= 0 || c % group != 0 || m % group != 0 || cg != c / group)
            throw new NotSupportedException(
                $"'{op.Name}': 分组不合法（C={c}, M={m}, group={group}, 权重输入通道={cg}）");
        if (ys[0] != n || ys[1] != m)
            throw new OnnxSharpException(
                $"'{op.Name}': 输出形状 {ys} 与输入/权重不符（期望 [{n},{m},{oh},{ow}]）");

        int sh = Attr(p, "strides", 0, 1), sw = Attr(p, "strides", 1, 1);
        int dh = Attr(p, "dilations", 0, 1), dw = Attr(p, "dilations", 1, 1);
        string autoPad = p.GetString("auto_pad", "NOTSET");

        (int ph, int phEnd) = ResolvePads(op, autoPad, p, h, oh, kh, sh, dh, 0);
        (int pw, int pwEnd) = ResolvePads(op, autoPad, p, w, ow, kw, sw, dw, 1);
        // 结束侧 pad 只决定输出尺寸（已经体现在 oh/ow 里），索引式只需要起始侧
        _ = phEnd + pwEnd;

        long total = (long)n * m * oh * ow;
        if (total <= 0 || total > uint.MaxValue)
            throw new NotSupportedException($"'{op.Name}': 输出元素数 {total} 不支持");

        // 1×1 点卷积走专用核：一个线程算 PointwiseTP 个连续像素 × PointwiseMT 个输出通道。
        // 条件是「块内同组」——OutPerGroup 必须整除通道块大小，否则 cbase 随 k 变化，
        // 一个 v 就不能复用（yolo 最后一层 M=255 就属于这种情况，走朴素核）。
        bool pointwise = kh == 1 && kw == 1 && dh == 1 && dw == 1 && ph == 0 && pw == 0
                         && (m / group) % PointwiseMT == 0;

        // 3×3 走专用核：同一套「输出通道块 × 连续像素块」结构，外加滑动窗口复用。
        // **必须 stride=1 且 dilation=1**：stride≠1 时 TP 个输出像素在输入里不再连续，
        // (TP+2) 的滑动窗口不成立，硬用就把「连续载入」这个唯一优势毁掉了。
        bool conv3 = IsConv3x3Fast(kh, kw, sh, sw, dh, dw, m / group);

        // 3×3 stride 2 走另一个特化核：窗口跨度是 2·TP+1，其余结构完全相同。
        bool conv3s2 = IsConv3x3S2Fast(kh, kw, sh, sw, dh, dw, m / group);

        // 线程数口径随核而变：朴素核按输出元素分派；特化核按「像素组 × 通道块」分派。
        // 只按像素分派过（一个线程算完 M 个通道），在 dGPU 上实测反而慢 14%：
        // 256→256 @32×32 只有 1024 个线程，独显的 SM 根本填不满。切开 M 之后并行度回来。
        // 像素组数按 OW 向上取整：一组必须落在同一行内，否则输出地址跨行、合并性归零。
        int tileTP = pointwise ? PointwiseTP : conv3s2 ? Conv3x3S2TP : Conv3x3TP;
        long owGroups = (pointwise || conv3 || conv3s2) ? (ow + tileTP - 1) / tileTP : 0;
        long threads = pointwise ? (long)n * oh * owGroups * (m / PointwiseMT)
                     : conv3 ? (long)n * oh * owGroups * (m / Conv3x3MT)
                     : conv3s2 ? (long)n * oh * owGroups * (m / Conv3x3S2MT)
                     : total;

        var pc = new Conv2DParams
        {
            Total = (uint)threads,
            Cg = (uint)cg,
            M = (uint)m,
            H = (uint)h,
            W = (uint)w,
            OH = (uint)oh,
            OW = (uint)ow,
            KH = (uint)kh,
            KW = (uint)kw,
            SH = (uint)sh,
            SW = (uint)sw,
            DH = (uint)dh,
            DW = (uint)dw,
            PH = (uint)ph,
            PW = (uint)pw,
            Group = (uint)group,
            OutPerGroup = (uint)(m / group),
            HasBias = 0,
            Act = ActCode(op.FusedActivationKind),
        };

        VkBind weight = VkArgs.Bind(ctx, op.Inputs[1]);
        VkBind bias = weight; // 无 bias 时绑到权重上，靠 HasBias=0 在 shader 里忽略
        if (op.Inputs.Count == 3 && !op.Inputs[2].IsEmpty)
        {
            bias = VkArgs.Bind(ctx, op.Inputs[2]);
            pc.HasBias = 1;
        }

        ReadOnlySpan<byte> pcb = new(&pc, sizeof(Conv2DParams));
        string shader = pointwise ? "conv2d_1x1"
                      : conv3 ? "conv3x3"
                      : conv3s2 ? "conv3x3s2"
                      : "conv2d";

        // 归因用：ONNXSHARP_VK_CONVDUMP=1 时把每层 conv 的形状与实际选中的核打出来。
        // 逐 dispatch 的 GPU 时间戳在这台机器上不可信，但「哪个形状走了哪个核」是确定的；
        // 把它和 ConvBench 的单形状速率拼起来，就能算出每类核应该占多少时间。
        if (VkConvDump.On)
            Console.WriteLine($"    [conv] {op.Name}: {c}->{m} k{kh}x{kw} s{sh},{sw} d{dh},{dw} " +
                              $"g{group} @{oh}x{ow} act={pc.Act} {shader} threads={threads}");

        ctx.Layer.AddDispatch(shader, 4, sizeof(Conv2DParams),
            [VkArgs.Bind(ctx, op.Inputs[0]), weight, ctx.Out(op.Outputs[0]), bias],
            pcb, (ulong)threads);
    }

    /// <summary>
    /// 解析某一维的起始 pad，并**反推校验**：用 (in + padB + padE - effK)/stride + 1 应该等于
    /// 形状推断给出的 out。这是整个发射器唯一的自检——pad 是卷积最容易抄错的地方，
    /// 抄错不会报错，只会让边界像素算错。
    /// </summary>
    private static (int Begin, int End) ResolvePads(LayerInfo op, string autoPad, LayerParams p,
        int inSize, int outSize, int kernel, int stride, int dilation, int axis)
    {
        int effK = (kernel - 1) * dilation + 1;

        int begin, end;
        if (autoPad == "VALID")
        {
            begin = end = 0;
        }
        else if (autoPad is "SAME_UPPER" or "SAME_LOWER")
        {
            int totalPad = Math.Max(0, (outSize - 1) * stride + effK - inSize);
            bool upper = autoPad == "SAME_UPPER";
            begin = upper ? totalPad / 2 : totalPad - totalPad / 2;
            end = totalPad - begin;
        }
        else
        {
            begin = Attr(p, "pads", axis, 0);
            end = Attr(p, "pads", Spatial + axis, begin);
        }

        int expect = (inSize + begin + end - effK) / stride + 1;
        if (expect != outSize)
            throw new OnnxSharpException(
                $"'{op.Name}' 轴 {axis}: pad({begin},{end}) + stride={stride} + dilation={dilation} " +
                $"推得输出 {expect}，但形状推断给出 {outSize}（auto_pad={autoPad}）");

        return (begin, end);
    }

    /// <summary>读属性数组的第 <paramref name="index"/> 项；缺项用 <paramref name="fallback"/>。</summary>
    private static int Attr(LayerParams p, string key, int index, int fallback)
    {
        long[]? v = p.GetInts(key);
        return v is not null && v.Length > index ? (int)v[index] : fallback;
    }
}

/// <summary>归因开关：<c>ONNXSHARP_VK_CONVDUMP=1</c> 时逐层打印 conv 形状与实际选中的核。</summary>
internal static class VkConvDump
{
    public static readonly bool On = Environment.GetEnvironmentVariable("ONNXSHARP_VK_CONVDUMP") == "1";
}
