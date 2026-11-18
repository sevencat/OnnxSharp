// 段级成本判据（P2-6 剩下的那一半）：回答「这一段到底值不值得上 GPU」。
//
// 常数全部来自 benchv3.md 的实测，不是猜的：
//
//   t_gpu ≈ 120 us（每段）+ 4 us × dispatch 数 + 跨 PCIe 字节 / 10.9 GBps
//           + 通用卷积 FLOPs / 0.44 T + 点卷积 FLOPs / 1.47 T
//
// 「固定成本按段摊」这一条经过缩放实验（1/2/4/8 个串联算子）确认：斜率 3–5 us，
// 截距 87–121 us。我第一版写成「120 us × dispatch 数」，把成本按 dispatch 摊了，
// 那会把「6 个算子拼成一段」估成 720 us（实为 144 us）——差五倍，方向都反了。
// 真正的教训是：**贵的是断段，不是 dispatch**。
//
// 为什么需要它：`VkPartitionPlanner.Plan` 只按「连续 + 最短段长」切段，那是个与规模无关的
// 粗判据。实测过 1×1 卷积 256→256 @32×32 端到端 415 us、而同形状 CPU 只要 250–380 us：
// 固定成本 + 搬运就占 355 us，真正算的只有 60 us。
using System.Globalization;
using OnnxSharp.Core;
using OnnxSharp.Tensors;

namespace OnnxSharp.Vulkan;

/// <summary>
/// 段级成本估计。只判断「值不值得」，不改变「能不能」——后者是
/// <see cref="VkOpRegistry.CanEmit"/> 与 <see cref="VkDevicePolicy"/> 的职责。
/// </summary>
internal static class VkCostModel
{
    /// <summary>
    /// **每段**的固定成本（us）：一次 submit + 一次 fence 等待 + 两次 stage 拷贝的建置。
    /// <para>
    /// 早期按「1/2/4/8 个串联逐元素算子」的缩放实验拟合出 87–121 us，误差来自那次实验用的是
    /// 合成链：几个 400 KB 槽位、数据全在 L2、CPU 侧也没有真实张量的建置。
    /// </para>
    /// <para>
    /// 在真实图（yolo11n，4 段）上重测：把所有 dispatch 跳掉、只留 stage in/out + submit，
    /// 段内仍有 12.6 ms，摊到 4 段是 3.15 ms/段；其中纯 fence 等待 6.6 ms（1.65 ms/段）
    /// 与数据量无关，是 WDDM 上一次 submit→fence 的往返。取值 1500 就是这个实测量的保守版本。
    /// </para>
    /// <para>
    /// 但**按 1500 实标定后收益接近零**（yolo11n：4 段→3 段，13.13→12.89 ms，只省 0.24 ms）：
    /// 被弃掉的 vkgpu1 只有 Conv+Mul，段成本省下约 2.1 ms，CPU 上算这两步又要 1.9 ms，正好抵消。
    /// 所以 120 保留，它在这张图上的判定结果是「发」，而这个判定本身是对的。
    /// </para>
    /// <para>
    /// 真正的地板在别处：把 dispatch 全跳掉仍有 12.6 ms/帧（stage in/out 5.4 ms + 纯 fence
    /// 等待 6.6 ms）。那两块与段数**不是线性的**（少一段只省 0.2 ms），要动得改提交架构，
    /// 不是调这个常数。留 ONNXSHARP_VK_FIXEDUS 覆盖只是为了后续能复现这次标定。
    /// </para>
    /// </summary>
    private static readonly double FixedPerSegmentUs = EnvOr("ONNXSHARP_VK_FIXEDUS", 120);

    /// <summary>单次 dispatch 的边际成本（us）：录制 + 相邻 dispatch 之间的屏障。实测 3–5 us。</summary>
    private const double PerDispatchUs = 4;

    /// <summary>
    /// host ↔ device 的有效带宽（GB/s）。10.9 是 PCIe 理论口径，实测到不了：真实图上
    /// stage in/out 合计约 27 MB 却花了 5.4 ms（≈5 GB/s），因为还要算上 staging 的
    /// vkCmdCopyBuffer、拷到 Tensor 这一侧的开销。取 4 是偏乐观的（宁可少拒）。
    /// </summary>
    private static readonly double TransferGBps = EnvOr("ONNXSHARP_VK_GBPS", 10.9);

    private static double EnvOr(string name, double fallback) =>
        double.TryParse(Environment.GetEnvironmentVariable(name), NumberStyles.Float,
                        CultureInfo.InvariantCulture, out double v) && v > 0 ? v : fallback;

    /// <summary>
    /// 1×1 点卷积核的实测算力（TFLOP/s），TP=4×MT=8 分块后的边际值。
    /// </summary>
    private const double GpuPointwiseTflops = 1.47;

    /// <summary>
    /// **朴素**卷积核的实测算力（TFLOP/s）。3×3 每输出要做 9 次邻域取值，而朴素核
    /// 每次 MAC 都各读一个 x 和一个 w，输入零复用。
    /// 由单卷积 A/B（32→32 @64×64，扣固定成本与搬运后 210 us / 75.5 MFLOP）拟定。
    /// </summary>
    private const double GpuConvTflops = 0.36;

    /// <summary>
    /// **3×3 专用核**（"conv3x3.comp"）的实测算力。同一份负载下朴素核 210 us 的
    /// 计算降到 130 us，端到端 526 → 446 us。复用结构：每 (c, kh) 载 TP+2 个连续输入，
    /// 被 MT×TP×3 = 96 次 MAC 共用。仍低于点卷积（1.47）——3 行 halo 没有跨线程复用，
    /// 下一步该做的是共享内存分块。
    /// </summary>
    private const double GpuConv3x3Tflops = 0.58;

    /// <summary>CPU 1×1 卷积（GEMM 式分块走到 AVX）：实测 134 MFLOP / 250–380 us。</summary>
    private const double CpuPointwiseGflops = 400;

    /// <summary>
    /// CPU 通用卷积的有效算力。早期记的是 60 GFLOP/s（75 MFLOP / 1246 us），但那次基线
    /// 没热透——同一份计算在预热 80 次后只要 420–500 us。按 yolo 6op 反推约 200 GFLOP/s。
    /// <para>
    /// 这个数错 3 倍会让判据把「CPU 本来很快」的段判成该上 GPU，也就是白扔一次 PCIe 往返。
    /// </para>
    /// </summary>
    private const double CpuConvGflops = 200;

    /// <summary>
    /// CPU 逐元素的有效带宽（GB/s）。实测 16K 元素 × 3 算子共 230 us，
    /// 扣掉每算子开销后只剩约 50 us 走 128 KB ⇒ 2.6 GB/s 量级；取 6 是**故意偏高**的，
    /// 宁可少拒（放过一段会亏的）也不要多拒（把赚钱的段判给 CPU）。
    /// </summary>
    private const double CpuRamGBps = 6;

    /// <summary>
    /// CPU 每个**非卷积**算子的一次性开销（寻址、形状检查、Tensor 复用）。
    /// 由「3 算子 16K 元素 = 230 us」反推：约 60 us/算子。
    /// <para>
    /// 只对非卷积算子加：卷积的开销已经折进上面那个实测算力里（大循环摊掉了），
    /// 再加一遍会把 yolo 6op 估成 705 us——实测只有 420–500 us。
    /// </para>
    /// </summary>
    private const double CpuPerOpOverheadUs = 60;

    /// <summary>
    /// 估算 GPU 端耗时（us）。<paramref name="bytes"/> 是跨 PCIe 的进出总量，
    /// <paramref name="flops"/> 是段内总算力需求，<paramref name="dispatches"/> 是段内算子数。
    /// </summary>
    public static double GpuUs(double bytes, int dispatches,
                               double conv3x3Flops, double convFlops, double pointwiseFlops)
    {
        double t = FixedPerSegmentUs + PerDispatchUs * Math.Max(dispatches, 1);
        t += bytes / (TransferGBps * 1e3);                                  // B / (GB/s) → us
        t += conv3x3Flops / (GpuConv3x3Tflops * 1e6);                       // FLOP / (TFLOP/s) → us
        t += convFlops / (GpuConvTflops * 1e6);
        t += pointwiseFlops / (GpuPointwiseTflops * 1e6);
        return t;
    }

    /// <summary>
    /// 估算同一段在 CPU 上的耗时（us）。按算子类别取速率——1×1 与 3×3 在 CPU 上差一倍
    /// （400 vs 200 GFLOP/s），混用会让判据在 1×1 上失去分辨力。
    /// </summary>
    public static double CpuUs(VkSegmentPlan plan, NetImpl net)
    {
        double us = 0;
        foreach (LayerInfo op in plan.Ops)
        {
            ConvKind kind = Classify(net, op, out double flops);
            if (kind != ConvKind.None)
            {
                // GFLOP/s → FLOP/us 是 ×1e3（1 GFLOP/s = 1e9 FLOP/s = 1e3 FLOP/us）。
                // 卷积不加每算子开销：那个开销已经折进实测算力里了。
                // CPU 侧不区分 3×3 走不走专用核——那是 GPU 的事。
                us += flops / ((kind == ConvKind.Pointwise ? CpuPointwiseGflops : CpuConvGflops) * 1e3);
                continue;
            }

            us += CpuPerOpOverheadUs;

            // 非卷积：按「读一遍输入 + 写一遍输出」的内存口径估
            us += BytesOf(plan, net) / (CpuRamGBps * 1e3);
        }

        return us;
    }

    /// <summary>
    /// 估算一段在两边的耗时。形状未知时 <paramref name="gpuUs"/> 与 <paramref name="cpuUs"/>
    /// 都为 <c>NaN</c>，调用方应视为「放行」——判据是为了止损，不是为了在信息不足时下注。
    /// </summary>
    public static void Estimate(VkSegmentPlan plan, NetImpl net,
                                out double gpuUs, out double cpuUs,
                                out double bytes, out double flops)
    {
        bytes = BytesOf(plan, net);
        flops = 0;
        double conv3Flops = 0, convFlops = 0, pwFlops = 0;
        foreach (LayerInfo op in plan.Ops)
        {
            ConvKind kind = Classify(net, op, out double f);
            if (kind == ConvKind.None) continue;
            flops += f;
            if (kind == ConvKind.Pointwise) pwFlops += f;
            else if (kind == ConvKind.Conv3x3Fast) conv3Flops += f;
            else convFlops += f;
        }

        if (bytes < 0)   // 有 live-in/out 形状未知
        {
            gpuUs = double.NaN;
            cpuUs = double.NaN;
            bytes = 0;
            return;
        }

        gpuUs = GpuUs(bytes, plan.Count, conv3Flops, convFlops, pwFlops);
        cpuUs = CpuUs(plan, net);
    }

    /// <summary>段内 live-in + live-out 的字节数；任一形状未知返回 -1。</summary>
    private static double BytesOf(VkSegmentPlan plan, NetImpl net)
    {
        double bytes = 0;
        foreach (Arg a in plan.LiveIn)
        {
            ArgData d = net.GetArgData(a);
            if (!d.HasShape) return -1;
            bytes += (double)d.Shape.Total() * d.DType.SizeOf();
        }

        foreach (Arg a in plan.LiveOut)
        {
            ArgData d = net.GetArgData(a);
            if (!d.HasShape) return -1;
            bytes += (double)d.Shape.Total() * d.DType.SizeOf();
        }

        return bytes;
    }

    /// <summary>卷积类别：决定 GPU 侧该用哪个实测算力。</summary>
    private enum ConvKind
    {
        /// <summary>不是卷积（或形状不成立）。</summary>
        None,

        /// <summary>1×1 点卷积，走 "conv2d_1x1.comp"。</summary>
        Pointwise,

        /// <summary>3×3、stride/dilation 1，走 "conv3x3.comp"。</summary>
        Conv3x3Fast,

        /// <summary>其余卷积，走朴素核 "conv2d.comp"。</summary>
        Generic,
    }

    /// <summary>
    /// 卷积的 FLOPs（2×MAC）与类别。类别判定**必须与发射侧同一个谓词**
    /// （<see cref="VkConvEmitter.IsConv3x3Fast"/>），否则会拿快路径的速率去估慢路径的活。
    /// </summary>
    private static ConvKind Classify(NetImpl net, LayerInfo op, out double flops)
    {
        flops = 0;

        if (op.Inputs.Count < 2 || op.Outputs.Count == 0) return ConvKind.None;

        // 只认 Conv。Concat / Add / Mul 同样满足「两输入 + 第二个是 4D」，会被误当成
        // [M, C, kh, kw] 权重：yolo11n 的 Concat 第二输入是 [1,128,80,80]，于是算出
        // 2×80×80×128×80×80 ≈ 10.5 GFLOP 的假活，整段被误判成「GPU 赚 1.8 倍」而发出去。
        if (op.Type != "Conv") return ConvKind.None;

        TensorShape w = net.GetArgData(op.Inputs[1]).Shape;
        TensorShape o = net.GetArgData(op.Outputs[0]).Shape;
        if (w.Rank != 4 || o.Rank != 4) return ConvKind.None;

        long m = w[0], cg = w[1], kh = w[2], kw = w[3];
        long n = o[0], oh = o[2], ow = o[3];
        if (m <= 0 || cg <= 0 || kh <= 0 || kw <= 0 || oh <= 0 || ow <= 0) return ConvKind.None;

        flops = 2.0 * n * m * oh * ow * cg * kh * kw;

        if (kh == 1 && kw == 1) return ConvKind.Pointwise;

        // 属性拿不到就按通用核估（保守：通用核更慢，判据更容易拒，不会把亏的判成赚的）
        if (op.Parameters is null || cg <= 0) return ConvKind.Generic;
        LayerParams p = op.Parameters;
        int group = (int)p.GetInt("group", 1);
        if (group <= 0 || m % group != 0) return ConvKind.Generic;

        int sh = Attr(p, "strides", 0), sw = Attr(p, "strides", 1);
        int dh = Attr(p, "dilations", 0), dw = Attr(p, "dilations", 1);

        // stride-2 的 3×3 走 conv3x3s2.comp，载入复用率与 stride-1 版同量级
        // （每 (c,kh) 载入 2·TP+1 个输入换 MT×TP×3 次 MAC），所以按同一档速率估。
        // 判据依然偏保守：这里用的是实测值的 1/7 左右，宁可低估。
        if (VkConvEmitter.IsConv3x3Fast((int)kh, (int)kw, sh, sw, dh, dw, m / group)) return ConvKind.Conv3x3Fast;
        if (VkConvEmitter.IsConv3x3S2Fast((int)kh, (int)kw, sh, sw, dh, dw, m / group)) return ConvKind.Conv3x3Fast;
        return ConvKind.Generic;
    }

    /// <summary>读属性数组的第 <paramref name="index"/> 项；缺项按 1 处理（同 VkConv 口径）。</summary>
    private static int Attr(LayerParams p, string key, int index)
    {
        long[]? v = p.GetInts(key);
        return v is not null && v.Length > index ? (int)v[index] : 1;
    }
}
