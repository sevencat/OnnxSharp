using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using OnnxSharp.Tensors;
using OnnxSharp.Threading;

namespace OnnxSharp.Kernels;

/// <summary>
/// 输入通道很少（<c>ic ≤ 3</c>，典型是 RGB 首层）的 3×3 直接卷积，AVX-512 版。
/// <para>
/// 这类层在通用路径上是死路：<c>K = 3×3×3 = 27</c>、<c>M = 输出通道 16</c>。im2col 要为
/// 102400 个输出像素物化 <c>27×N</c> = 11 MB 的 col 缓冲（写一遍读一遍 = 22 MB 搬运）；
/// 面板式把这 27 个 K 步摊进 4 行微内核，也不省这 22 MB。
/// </para>
/// <para>
/// 这里换一个并行方向：<b>沿输出宽方向 16 个像素为一个 <c>zmm</c></b>，输出通道在外层摊开。
/// 每次只建一遍面板（27 条 × 4 个像素块 = 6.7 KB，常驻 L1），再让所有输出通道组共用；
/// 步长 2 时一条面板 = 一次 32 元素读 + 一条 <c>vpermt2ps</c> 抽偶数位。
/// </para>
/// <para>
/// 微内核是 4×4：<b>4 个输出通道 × 4 个像素块</b>（16 个累加器 + 4 个权重 + 4 条面板 = 24 个
/// <c>zmm</c>）。关键是循环顺序 —— 一次 <c>vbroadcastss</c> 要喂 4 条 FMA，不能像逐块循环那样
/// 1:1 配对（那样一半的指令在广播）。量出来 4 线程 124→131、16 线程 237→308 GMAC/s。
/// </para>
/// <para>
/// 实测（<c>convbench --iters 150</c>，与 im2col 交替计时）：1/4/16 线程 1.44/0.34/0.14 ms，
/// im2col 2.17/0.59/0.30 ms，即 1.5~2.1×。<b>线程扩展接近线性</b>（1→16 线程 10.4×）；
/// 早先量到「16 线程只有 3.8×」是 <c>convbench</c> 先测 im2col 后测本内核造成的降频假象，
/// 现在两个内核轮流跑，读数才可比。
/// </para>
/// </summary>
internal static unsafe class FirstConv
{
    /// <summary>调试开关：<c>0</c> 关闭本路径（<c>ONNXSHARP_FIRSTCONV=0</c>），用于 A/B。</summary>
    public static bool Enabled { get; set; } =
        Environment.GetEnvironmentVariable("ONNXSHARP_FIRSTCONV") != "0";

    /// <summary>本机是否具备 AVX-512（16 个像素正好一个 <c>zmm</c>）。</summary>
    public static bool IsSupported => Avx512F.IsSupported;

    /// <summary>一个向量覆盖的输出像素数。</summary>
    private const int Lane = 16;

    /// <summary>一个 tile 内并排处理的像素块数。</summary>
    private const int Blocks = 4;

    /// <summary>一个 tile 覆盖的像素数。</summary>
    private const int Tile = Blocks * Lane;

    /// <summary>同时推进的输出通道数 = 独立 FMA 链的条数。</summary>
    private const int OcUnroll = 4;

    public static bool TryConv(
        Tensor input,
        Tensor weight,
        Tensor? bias,
        Tensor output,
        int groups,
        int strideH,
        int strideW,
        int dilH,
        int dilW,
        int padT,
        int padL)
    {
        if (!Enabled || !IsSupported)
            return false;

        if (input.Shape.Rank != 4 || weight.Shape.Rank != 4 || output.Shape.Rank != 4)
            return false;

        int batches = input.Shape.Dims[0];
        int channels = input.Shape.Dims[1];
        int inH = input.Shape.Dims[2];
        int inW = input.Shape.Dims[3];

        int outChannels = output.Shape.Dims[1];
        int outH = output.Shape.Dims[2];
        int outW = output.Shape.Dims[3];

        // 单批次、不分组、输入通道装得进「一个 27 条面板的窗口」（1~3 通道 = 9~27 条 zmm）
        if (batches != 1 || groups != 1 || channels < 1 || channels > 3)
            return false;

        if (weight.Shape.Dims[1] != channels || weight.Shape.Dims[2] != 3 || weight.Shape.Dims[3] != 3)
            return false;

        if (outChannels < 1 || outW < Lane || outH < 1)
            return false;

        if (dilH != 1 || dilW != 1 || strideH != strideW || (strideH != 1 && strideH != 2))
            return false;

        // 3×3 + pad 1 的取样窗口是 [-1, 2]；halo 位置由边界分支补零
        if (padT != 1 || padL != 1)
            return false;

        // 输出尺寸必须与 stride 关系自洽（k3s2p1：out = floor(in/2) 或 ceil(in/2)），
        // 否则说明调用方的 pads/strides 与这里假设的不是一回事
        if (strideH == 2
                ? (outH * 2 < inH || outH * 2 > inH + 1 || outW * 2 < inW || outW * 2 > inW + 1)
                : (outH != inH || outW != inW))
            return false;

        var src = input.AsSpan<float>();
        var w = weight.AsSpan<float>();
        var dst = output.AsSpan<float>();
        var b = bias is null ? Span<float>.Empty : bias.AsSpan<float>();

        fixed (float* psrc = src)
        fixed (float* pw = w)
        fixed (float* pdst = dst)
        fixed (float* pb = b)
        {
            Run(psrc, pw, pb, pdst, channels, outChannels, inH, inW, outH, outW, strideH, padT, padL);
        }

        return true;
    }

    private static void Run(
        float* src,
        float* weights,
        float* bias,
        float* dst,
        int ic,
        int oc,
        int inH,
        int inW,
        int outH,
        int outW,
        int stride,
        int padT,
        int padL)
    {
        int kCount = ic * 9;
        long plane = (long)inH * inW;
        long spatial = (long)outH * outW;

        // 面板布局 [k][b*Lane]：同一条 K 步的 4 个像素块相邻，索引跨度为 Tile
        int panelStride = Tile;

        // fixed 指针不能进 lambda：先转 nint，进 lambda 再转回来
        nint srcPtr = (nint)src;
        nint weightPtr = (nint)weights;
        nint biasPtr = (nint)bias;
        nint outPtr = (nint)dst;

        ParallelRange.For(outH, (from, to) =>
        {
            float* s = (float*)srcPtr;
            float* d = (float*)outPtr;
            float* w = (float*)weightPtr;
            float* b = (float*)biasPtr;

            // stackalloc 只保证 16 字节对齐：不手工对齐到 64 的话，512 位的存取会跨缓存行，
            // store→load 转发也不成立，每条面板装载都要等存储排空。
            byte* raw = stackalloc byte[kCount * Tile * sizeof(float) + 64];
            float* xs = (float*)(((nint)raw + 63) & ~63);

            byte* rawTail = stackalloc byte[Lane * sizeof(float) + 64];
            float* tail = (float*)(((nint)rawTail + 63) & ~63);

            for (int oh = from; oh < to; oh++)
            {
                int iy = oh * stride - padT;
                float* rowOut = d + (long)oh * outW;

                for (int ow0 = 0; ow0 < outW; ow0 += Tile)
                {
                    int nb = Math.Min(Blocks, (outW - ow0 + Lane - 1) / Lane);

                    // 面板只建一遍，后面所有输出通道组共用
                    for (int blk = 0; blk < nb; blk++)
                    {
                        int owp = ow0 + blk * Lane;
                        int n = Math.Min(Lane, outW - owp);

                        BuildPanels(
                            s, xs + blk * Lane, panelStride, ic, plane, inH, inW,
                            iy, owp * stride - padL, stride, n);
                    }

                    int o = 0;

                    if (nb == Blocks)
                    {
                        // 满 tile：一条 K 步的 4 个权重广播能喂 4 个像素块的 16 条 FMA，
                        // 权重广播与 FMA 比降到 1:4（逐块循环里是 1:1，一半的指令在广播）。
                        // 16 个累加器 + 4 权重 + 4 面板 = 24 个 zmm，AVX-512 还剩 8 个余量。
                        for (; o + OcUnroll <= oc; o += OcUnroll)
                        {
                            float* wo = w + (long)o * kCount;

                            var b0 = BroadcastLoad(b, o + 0);
                            var b1 = BroadcastLoad(b, o + 1);
                            var b2 = BroadcastLoad(b, o + 2);
                            var b3 = BroadcastLoad(b, o + 3);

                            var a00 = b0; var a01 = b0; var a02 = b0; var a03 = b0;
                            var a10 = b1; var a11 = b1; var a12 = b1; var a13 = b1;
                            var a20 = b2; var a21 = b2; var a22 = b2; var a23 = b2;
                            var a30 = b3; var a31 = b3; var a32 = b3; var a33 = b3;

                            float* p = xs;
                            float* pEnd = xs + (long)kCount * panelStride;

                            for (; p < pEnd; p += panelStride, wo++)
                            {
                                var w0 = Broadcast(wo[0]);
                                var w1 = Broadcast(wo[kCount]);
                                var w2 = Broadcast(wo[2 * kCount]);
                                var w3 = Broadcast(wo[3 * kCount]);

                                var x0 = Avx512F.LoadVector512(p);
                                var x1 = Avx512F.LoadVector512(p + Lane);
                                var x2 = Avx512F.LoadVector512(p + 2 * Lane);
                                var x3 = Avx512F.LoadVector512(p + 3 * Lane);

                                a00 = Fma(w0, x0, a00); a01 = Fma(w0, x1, a01);
                                a02 = Fma(w0, x2, a02); a03 = Fma(w0, x3, a03);
                                a10 = Fma(w1, x0, a10); a11 = Fma(w1, x1, a11);
                                a12 = Fma(w1, x2, a12); a13 = Fma(w1, x3, a13);
                                a20 = Fma(w2, x0, a20); a21 = Fma(w2, x1, a21);
                                a22 = Fma(w2, x2, a22); a23 = Fma(w2, x3, a23);
                                a30 = Fma(w3, x0, a30); a31 = Fma(w3, x1, a31);
                                a32 = Fma(w3, x2, a32); a33 = Fma(w3, x3, a33);
                            }

                            float* out0 = rowOut + (long)o * spatial + ow0;

                            Avx512F.Store(out0, a00);
                            Avx512F.Store(out0 + Lane, a01);
                            Avx512F.Store(out0 + 2 * Lane, a02);
                            Avx512F.Store(out0 + 3 * Lane, a03);

                            out0 += spatial;
                            Avx512F.Store(out0, a10);
                            Avx512F.Store(out0 + Lane, a11);
                            Avx512F.Store(out0 + 2 * Lane, a12);
                            Avx512F.Store(out0 + 3 * Lane, a13);

                            out0 += spatial;
                            Avx512F.Store(out0, a20);
                            Avx512F.Store(out0 + Lane, a21);
                            Avx512F.Store(out0 + 2 * Lane, a22);
                            Avx512F.Store(out0 + 3 * Lane, a23);

                            out0 += spatial;
                            Avx512F.Store(out0, a30);
                            Avx512F.Store(out0 + Lane, a31);
                            Avx512F.Store(out0 + 2 * Lane, a32);
                            Avx512F.Store(out0 + 3 * Lane, a33);
                        }
                    }
                    else
                    {
                        // 行尾不满一个 tile：像素块少，逐块循环的 1:1 广播也摊得薄
                        for (; o + OcUnroll <= oc; o += OcUnroll)
                        {
                            float* wo = w + (long)o * kCount;

                            for (int blk = 0; blk < nb; blk++)
                            {
                                float* xb = xs + blk * Lane;

                                var a0 = BroadcastLoad(b, o + 0);
                                var a1 = BroadcastLoad(b, o + 1);
                                var a2 = BroadcastLoad(b, o + 2);
                                var a3 = BroadcastLoad(b, o + 3);

                                for (int k = 0; k < kCount; k++)
                                {
                                    var xk = Avx512F.LoadVector512(xb + k * panelStride);

                                    a0 = Avx512F.FusedMultiplyAdd(Broadcast(wo[k]), xk, a0);
                                    a1 = Avx512F.FusedMultiplyAdd(Broadcast(wo[kCount + k]), xk, a1);
                                    a2 = Avx512F.FusedMultiplyAdd(Broadcast(wo[2 * kCount + k]), xk, a2);
                                    a3 = Avx512F.FusedMultiplyAdd(Broadcast(wo[3 * kCount + k]), xk, a3);
                                }

                                int n = Math.Min(Lane, outW - ow0 - blk * Lane);
                                float* out0 = rowOut + (long)o * spatial + ow0 + blk * Lane;

                                Store(out0, a0, n, tail);
                                Store(out0 + spatial, a1, n, tail);
                                Store(out0 + 2 * spatial, a2, n, tail);
                                Store(out0 + 3 * spatial, a3, n, tail);
                            }
                        }
                    }

                    // 输出通道余数：单链 27 步，通道数小于 OcUnroll，代价可忽略
                    for (; o < oc; o++)
                    {
                        float* wo = w + (long)o * kCount;

                        for (int blk = 0; blk < nb; blk++)
                        {
                            float* xb = xs + blk * Lane;

                            var acc = BroadcastLoad(b, o);

                            for (int k = 0; k < kCount; k++)
                                acc = Avx512F.FusedMultiplyAdd(
                                    Broadcast(wo[k]), Avx512F.LoadVector512(xb + k * panelStride), acc);

                            int n = Math.Min(Lane, outW - ow0 - blk * Lane);
                            Store(rowOut + (long)o * spatial + ow0 + blk * Lane, acc, n, tail);
                        }
                    }
                }
            }
        });
    }

    /// <summary>按边界情况选快速/边界面板装载。</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void BuildPanels(
        float* src, float* xs, int panelStride, int ic, long plane, int inH, int inW,
        int iy, int ix, int stride, int n)
    {
        // 快速路径：整行取样窗口都在图内（步长 2 时一次读 32 个 float 供 vpermt2ps 抽取）
        if (n == Lane && iy >= 0 && iy + 2 < inH
            && ix >= 0 && ix + Lane * stride + 2 <= inW)
        {
            FillFast(src, xs, panelStride, ic, plane, inW, iy, ix, stride);
        }
        else
        {
            FillSlow(src, xs, panelStride, ic, plane, inH, inW, iy, ix, stride);
        }
    }

    /// <summary>
    /// 快速面板装载：窗口全在图内。
    /// <para>
    /// 步长 1 时一条 <c>vmovups</c> 就是「16 个像素」；步长 2 时要的是 <c>x0+0, x0+2, …</c>，
    /// 一次读 32 个 float 再用 <c>vpermt2ps</c> 抽偶数位 —— 一条指令，比逐元素抽取省 5 倍。
    /// </para>
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void FillFast(
        float* src, float* xs, int panelStride, int ic, long plane, int inW, int iy, int ix, int stride)
    {
        for (int c = 0; c < ic; c++)
        {
            float* channel = src + (long)c * plane;

            for (int ky = 0; ky < 3; ky++)
            {
                float* row = channel + (long)(iy + ky) * inW + ix;

                for (int kx = 0; kx < 3; kx++)
                {
                    float* panel = xs + (c * 9 + ky * 3 + kx) * panelStride;

                    if (stride == 1)
                        Avx512F.Store(panel, Avx512F.LoadVector512(row + kx));
                    else
                        StoreEven16(row + kx, panel);
                }
            }
        }
    }

    /// <summary>从 <paramref name="source"/> 读 32 个 float，抽偶数位写成一个 16 宽向量。</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void StoreEven16(float* source, float* target)
    {
        var v0 = Avx512F.LoadVector512(source);
        var v1 = Avx512F.LoadVector512(source + Lane);

        Avx512F.Store(target, Avx512F.PermuteVar16x32x2(v0, EvenIndex, v1));
    }

    /// <summary>
    /// 边界面板装载：行越界整块清零；否则按 <c>[start, end)</c> 写有效像素、其余补零。
    /// <para>
    /// 逐元素带边界判断的分支版本会让每行最后一个像素块慢 5 倍以上，这里先把有效区间算出来，
    /// 循环体内不再有分支。
    /// </para>
    /// </summary>
    private static void FillSlow(
        float* src, float* xs, int panelStride, int ic, long plane, int inH, int inW, int iy, int ix, int stride)
    {
        for (int c = 0; c < ic; c++)
        {
            float* channel = src + (long)c * plane;

            for (int ky = 0; ky < 3; ky++)
            {
                int y = iy + ky;

                for (int kx = 0; kx < 3; kx++)
                {
                    float* panel = xs + (c * 9 + ky * 3 + kx) * panelStride;
                    int x0 = ix + kx;

                    if ((uint)y >= (uint)inH || x0 >= inW)
                    {
                        new Span<float>(panel, Lane).Clear();
                        continue;
                    }

                    float* row = channel + (long)y * inW;

                    int start = x0 < 0 ? (-x0 + stride - 1) / stride : 0;
                    int end = Math.Min(Lane, (inW - x0 + stride - 1) / stride);

                    for (int j = 0; j < start; j++)
                        panel[j] = 0;

                    for (int j = start; j < end; j++)
                        panel[j] = row[x0 + j * stride];

                    for (int j = end; j < Lane; j++)
                        panel[j] = 0;
                }
            }
        }
    }

    /// <summary>写出一个向量；不足 16 个像素时先落栈再逐元素拷。</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Store(float* target, Vector512<float> value, int n, float* tail)
    {
        if (n == Lane)
        {
            Avx512F.Store(target, value);
            return;
        }

        Avx512F.Store(tail, value);

        for (int j = 0; j < n; j++)
            target[j] = tail[j];
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector512<float> Broadcast(float value) => Vector512.Create(value);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector512<float> Fma(Vector512<float> a, Vector512<float> b, Vector512<float> c)
        => Avx512F.FusedMultiplyAdd(a, b, c);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector512<float> BroadcastLoad(float* bias, int index)
        => bias is null ? Vector512<float>.Zero : Vector512.Create(bias[index]);

    /// <summary><c>vpermt2ps</c> 的偶数位索引：0,2,…,30 横跨两个源向量。</summary>
    private static readonly Vector512<int> EvenIndex =
        Vector512.Create(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30);
}
