using OnnxSharp.Core;
using OnnxSharp.Tensors;
using OnnxSharp.Threading;

namespace OnnxSharp.Kernels;

/// <summary>
/// NCHW 直接卷积（含 group / stride / dilation / padding）。
/// <para>
/// M2 版本以**正确**为第一目标：不做 im2col、不做 SIMD，靠「按输出通道并行」拿到多核。
/// M4 再替换为 im2col+GEMM / Winograd（见 plan.md）。
/// </para>
/// </summary>
internal static class ConvKernels
{
    public static void Direct(
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
            throw new OnnxSharpException("Conv kernel expects NCHW rank-4 tensors");

        int batches = input.Shape.Dims[0];
        int channels = input.Shape.Dims[1];
        int inH = input.Shape.Dims[2];
        int inW = input.Shape.Dims[3];

        int outChannels = output.Shape.Dims[1];
        int outH = output.Shape.Dims[2];
        int outW = output.Shape.Dims[3];

        int kernelH = weight.Shape.Dims[2];
        int kernelW = weight.Shape.Dims[3];

        if (groups <= 0 || outChannels % groups != 0 || channels % groups != 0)
            throw new OnnxSharpException(
                $"Conv: channels {channels} / out {outChannels} not divisible by groups {groups}");

        int inPerGroup = channels / groups;
        int outPerGroup = outChannels / groups;

        if (weight.Shape.Dims[0] != outChannels || weight.Shape.Dims[1] != inPerGroup)
            throw new OnnxSharpException(
                $"Conv: weight {weight.Shape} does not match {outChannels} outputs x {inPerGroup} in-channels");

        // 每个 (n, oc) 一个任务：权重常驻缓存，且天然负载均衡。
        ParallelRange.For(batches * outChannels, (begin, end) =>
        {
            var src = input.AsSpan<float>();
            var w = weight.AsSpan<float>();
            var b = bias is null ? Span<float>.Empty : bias.AsSpan<float>();
            var dst = output.AsSpan<float>();

            for (int task = begin; task < end; task++)
            {
                int n = task / outChannels;
                int oc = task - n * outChannels;
                int g = oc / outPerGroup;

                long srcBatch = (long)n * channels * inH * inW;
                long dstBase = ((long)n * outChannels + oc) * outH * outW;
                long wBase = (long)oc * inPerGroup * kernelH * kernelW;

                float acc0 = b.Length > 0 ? b[oc] : 0f;

                for (int oy = 0; oy < outH; oy++)
                {
                    int y0 = oy * strideH - padT;

                    for (int ox = 0; ox < outW; ox++)
                    {
                        int x0 = ox * strideW - padL;
                        float sum = acc0;

                        for (int ic = 0; ic < inPerGroup; ic++)
                        {
                            long channelBase = srcBatch + (long)(g * inPerGroup + ic) * inH * inW;
                            long wChannel = wBase + (long)ic * kernelH * kernelW;

                            for (int ky = 0; ky < kernelH; ky++)
                            {
                                int y = y0 + ky * dilH;
                                if ((uint)y >= (uint)inH)
                                    continue;

                                long rowBase = channelBase + (long)y * inW;
                                long wRow = wChannel + (long)ky * kernelW;

                                for (int kx = 0; kx < kernelW; kx++)
                                {
                                    int x = x0 + kx * dilW;
                                    if ((uint)x >= (uint)inW)
                                        continue;

                                    sum += w[(int)(wRow + kx)] * src[(int)(rowBase + x)];
                                }
                            }
                        }

                        dst[(int)(dstBase + (long)oy * outW + ox)] = sum;
                    }
                }
            }
        });
    }
}
