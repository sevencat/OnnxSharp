using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using OnnxSharp.Tensors;
using OnnxSharp.Threading;

namespace OnnxSharp.Kernels;

/// <summary>
/// depthwise 2D 卷积（<c>group == channels</c>，每组 1 个输出通道）的直接实现。
/// <para>
/// <b>为什么需要它</b>：GEMM 类实现（<see cref="ConvKernels.Im2Col"/> /
/// <see cref="PackedConv"/>）在每组只有 1 个输出通道时退化成 <c>K×spatial</c> 的
/// im2col 物化 —— 数据放大 kH·kW 倍、且完全没有通道间复用。MobileNet 一类的网络
/// （YuNet：26 个 depthwise）里这就是主要开销：YuNet 上 18 个 3×3/s1 depthwise 占 56.6%。
/// </para>
/// <para>
/// 做法是每个 (batch, channel) 独立做一次 2D 卷积，按「输出行 → 核 tap」遍历，
/// 内层对输出宽向量化（8 路 FMA）；累加顺序与 im2col+GEMM 的 k 顺序一致
/// （先 ky 后 kx），所以数值与旧路径逐位相同（YuNet 对拍 sees 0 差异）。
/// </para>
/// </summary>
internal static class DepthwiseConv
{
    /// <summary>尝试执行；不适用（非 depthwise / 非 2D / 非常规布局）时返回 false 交回调用方。</summary>
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
        if (input.Shape.Rank != 4 || weight.Shape.Rank != 4 || output.Shape.Rank != 4)
            return false;

        int batches = input.Shape.Dims[0];
        int channels = input.Shape.Dims[1];
        int inH = input.Shape.Dims[2];
        int inW = input.Shape.Dims[3];

        int outChannels = output.Shape.Dims[1];
        int outH = output.Shape.Dims[2];
        int outW = output.Shape.Dims[3];

        int kH = weight.Shape.Dims[2];
        int kW = weight.Shape.Dims[3];

        // depthwise 的定义：group == 输入通道数 == 输出通道数，且每组的权重是 1×kH×kW
        if (groups != channels || outChannels != channels || weight.Shape.Dims[1] != 1)
            return false;
        if (strideH <= 0 || strideW <= 0 || dilH <= 0 || dilW <= 0)
            return false;

        unsafe
        {
            fixed (float* ps = input.AsSpan<float>())
            fixed (float* pw = weight.AsSpan<float>())
            fixed (float* pb = bias is null ? null : bias.AsSpan<float>())
            fixed (float* pt = output.AsSpan<float>())
            {
                // 指针本身不能跨 lambda 捕获（fixed 局部变量），先落到普通局部
                float* sourcePointer = ps;
                float* weightPointer = pw;
                float* biasPointer = pb;
                float* targetPointer = pt;

                ParallelRange.For(batches * channels, (begin, end) =>
                {
                    for (int t = begin; t < end; t++)
                    {
                        int n = t / channels;
                        int c = t - n * channels;

                        ConvOneChannel(
                            sourcePointer + ((long)n * channels + c) * inH * inW,
                            weightPointer + (long)c * kH * kW,
                            biasPointer is null ? null : biasPointer + c,
                            targetPointer + ((long)n * outChannels + c) * outH * outW,
                            inH, inW, outH, outW, kH, kW,
                            strideH, strideW, dilH, dilW, padT, padL);
                    }
                });
            }
        }

        return true;
    }

    /// <summary>单个 (batch, channel) 的 2D 卷积：输出行在寄存器里累加，写完再落内存。</summary>
    private static unsafe void ConvOneChannel(
        float* source,
        float* weights,
        float* bias,
        float* target,
        int inH,
        int inW,
        int outH,
        int outW,
        int kH,
        int kW,
        int strideH,
        int strideW,
        int dilH,
        int dilW,
        int padT,
        int padL)
    {
        float initial = bias is null ? 0f : bias[0];
        bool vectorizable = strideW == 1 && dilW == 1;

        for (int oy = 0; oy < outH; oy++)
        {
            int y0 = oy * strideH - padT;
            float* outRow = target + (long)oy * outW;

            // 整行先铺上 bias（后面按 tap 累加）
            int x = 0;
            if (Avx.IsSupported)
            {
                var vInitial = Vector256.Create(initial);
                for (; x + 8 <= outW; x += 8)
                    vInitial.Store(outRow + x);
            }

            for (; x < outW; x++)
                outRow[x] = initial;

            for (int ky = 0; ky < kH; ky++)
            {
                int y = y0 + ky * dilH;
                if ((uint)y >= (uint)inH)
                    continue;

                float* inRow = source + (long)y * inW;

                for (int kx = 0; kx < kW; kx++)
                {
                    float w = weights[ky * kW + kx];

                    if (vectorizable)
                    {
                        int shift = kx - padL;
                        int lo = Math.Max(0, -shift);
                        int hi = Math.Min(outW, inW - shift);
                        if (hi <= lo)
                            continue;

                        float* src = inRow + (lo + shift);
                        float* dst = outRow + lo;
                        var vw = Vector256.Create(w);

                        int i = 0;
                        if (Avx.IsSupported)
                        {
                            for (; i + 8 <= hi - lo; i += 8)
                            {
                                var value = Fma.MultiplyAdd(vw, Vector256.Load(src + i), Vector256.Load(dst + i));
                                value.Store(dst + i);
                            }
                        }

                        for (; i < hi - lo; i++)
                            dst[i] += w * src[i];

                        continue;
                    }

                    // 一般情形（stride ≠ 1 或膨胀 ≠ 1）：逐输出列算
                    for (int ox = 0; ox < outW; ox++)
                    {
                        int ix = ox * strideW - padL + kx * dilW;
                        if ((uint)ix >= (uint)inW)
                            continue;
                        outRow[ox] += w * inRow[ix];
                    }
                }
            }
        }
    }
}
