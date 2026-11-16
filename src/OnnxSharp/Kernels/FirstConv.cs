using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using OnnxSharp.Core;
using OnnxSharp.Tensors;
using OnnxSharp.Threading;

namespace OnnxSharp.Kernels;

/// <summary>
/// 3 通道输入（RGB/BGR 首层）的 3×3 直接卷积。
/// <para>
/// 这类层在通用路径上是死路：<c>K = 3×3×3 = 27</c>、<c>M = 输出通道 16</c>，
/// 面板式要 pack 27×16 的 K 面（27 只占 K 方向的一小截），im2col 的 GEMM 又是 M=16 的窄矩阵，
/// 两者都只有 50 GMAC/s 左右的效率 —— YuNet 的 <c>Conv_0</c> 一层就占前向的 15%。
/// </para>
/// <para>
/// 这里换一个并行方向：<b>沿输出宽方向 8 像素为一个向量</b>，输出通道做内层展开。
/// 每块只需 27 次输入向量装载（步长 2 时用一次 32 元素读 + 偶数位抽取），
/// 之后 16×27 条 FMA 全部命中寄存器 —— 指令数与 <c>oc·k / 8</c> 成正比，没有 pack 也没有 col 物化。
/// </para>
/// </summary>
internal static unsafe class FirstConv
{
    /// <summary>
    /// 默认<b>关闭</b>：YuNet 的 <c>Conv_0</c>（3→16, 3×3, s2, 320²）实测比面板式更慢
    /// （1.08 ms vs 0.81 ms）。原因是输出通道做内层时，每条 <c>vfmadd</c> 都要配一条
    /// <c>vbroadcastss</c>，指令比 1:1；面板式是 4 行 × 16 列，权重广播能被 16 个输出列复用。
    /// 留着这版是为了下次有人想试时不用重写（<c>ONNXSHARP_FIRSTCONV=1</c> 打开）。
    /// </summary>
    public static readonly bool Enabled =
        Environment.GetEnvironmentVariable("ONNXSHARP_FIRSTCONV") == "1";

    /// <summary>沿输出宽的向量块宽度。</summary>
    private const int Lane = 8;

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
        if (!Avx.IsSupported || !Fma.IsSupported)
            return false;

        if (input.Shape.Rank != 4 || weight.Shape.Rank != 4 || output.Shape.Rank != 4)
            return false;

        int batches = input.Shape.Dims[0];
        int channels = input.Shape.Dims[1];
        int outChannels = output.Shape.Dims[1];

        if (batches != 1 || channels != 3 || groups != 1)
            return false;

        if (weight.Shape.Dims[1] != 3 || weight.Shape.Dims[2] != 3 || weight.Shape.Dims[3] != 3)
            return false;

        if ((outChannels & (Lane - 1)) != 0 || outChannels < Lane || outChannels > 64)
            return false;

        if (dilH != 1 || dilW != 1 || strideH != strideW || strideH > 2)
            return false;

        int inH = input.Shape.Dims[2];
        int inW = input.Shape.Dims[3];
        int outH = output.Shape.Dims[2];
        int outW = output.Shape.Dims[3];

        if (outW < Lane)
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
            Run(psrc, pw, pb, pdst, inH, inW, outH, outW, outChannels, strideH, padT, padL);
        }

        return true;
    }

    private static void Run(
        float* src,
        float* weights,
        float* bias,
        float* dst,
        int inH,
        int inW,
        int outH,
        int outW,
        int outChannels,
        int stride,
        int padT,
        int padL)
    {
        long spatial = (long)outH * outW;

        ParallelRange.For(outH, (from, to) =>
        {
            // 27 个核位置 × 8 像素的输入向量，循环内复用同一块栈
            float* xs = stackalloc float[27 * Lane];
            float* tail = stackalloc float[Lane];

            for (int oh = from; oh < to; oh++)
            {
                int iy = oh * stride - padT;

                for (int ow0 = 0; ow0 < outW; ow0 += Lane)
                {
                    int n = Math.Min(Lane, outW - ow0);
                    int ix = ow0 * stride - padL;

                    bool fast = n == Lane && iy >= 0 && iy + 2 < inH && ix >= 0 && ix + 2 * Lane - 1 < inW;

                    if (fast)
                    {
                        for (int ic = 0; ic < 3; ic++)
                        {
                            float* channel = src + (long)ic * inH * inW;
                            float* panel = xs + ic * (9 * Lane);

                            for (int ky = 0; ky < 3; ky++)
                            {
                                float* row = channel + (long)(iy + ky) * inW + ix;

                                for (int kx = 0; kx < 3; kx++)
                                {
                                    float* target = panel + (ky * 3 + kx) * Lane;

                                    if (stride == 2)
                                        StoreEven8(row + kx, target);
                                    else
                                        Avx.Store(target, Avx.LoadVector256(row + kx));
                                }
                            }
                        }
                    }
                    else
                    {
                        for (int ic = 0; ic < 3; ic++)
                        {
                            float* channel = src + (long)ic * inH * inW;

                            for (int ky = 0; ky < 3; ky++)
                            {
                                int y = iy + ky;

                                for (int kx = 0; kx < 3; kx++)
                                {
                                    float* panel = xs + (ic * 9 + ky * 3 + kx) * Lane;
                                    int x0 = ix + kx;

                                    if ((uint)y >= (uint)inH)
                                    {
                                        for (int j = 0; j < Lane; j++)
                                            panel[j] = 0;
                                        continue;
                                    }

                                    float* row = channel + (long)y * inW;

                                    for (int j = 0; j < Lane; j++)
                                    {
                                        int x = x0 + j * stride;
                                        panel[j] = (uint)x < (uint)inW ? row[x] : 0;
                                    }
                                }
                            }
                        }
                    }

                    for (int oc = 0; oc < outChannels; oc++)
                    {
                        float* wp = weights + (long)oc * 27;
                        var acc = Vector256.Create(bias == null ? 0f : bias[oc]);

                        for (int k = 0; k < 27; k++)
                            acc = Fma.MultiplyAdd(Vector256.Create(wp[k]), Avx.LoadVector256(xs + k * Lane), acc);

                        float* outRow = dst + (long)oc * spatial + (long)oh * outW + ow0;

                        if (n == Lane)
                        {
                            Avx.Store(outRow, acc);
                        }
                        else
                        {
                            Avx.Store(tail, acc);
                            for (int j = 0; j < n; j++)
                                outRow[j] = tail[j];
                        }
                    }
                }
            }
        });
    }

    /// <summary>从 <paramref name="source"/> 读 16 个 float，抽取偶数位写成一个 8 宽向量。</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void StoreEven8(float* source, float* target)
    {
        var even = Vector256.Create(0, 2, 4, 6, 0, 0, 0, 0);

        var a0 = Avx2.PermuteVar8x32(Avx.LoadVector256(source), even);
        var a1 = Avx2.PermuteVar8x32(Avx.LoadVector256(source + 8), even);
        Avx.Store(target, Avx2.Permute2x128(a0, a1, 0x20));
    }
}
