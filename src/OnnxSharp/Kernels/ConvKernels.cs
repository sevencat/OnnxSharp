using System.Buffers;
using OnnxSharp.Core;
using OnnxSharp.Tensors;
using OnnxSharp.Threading;

namespace OnnxSharp.Kernels;

/// <summary>
/// NCHW 卷积实现（含 group / stride / dilation / padding）。
/// <list type="bullet">
/// <item><see cref="Im2Col"/>：主路径，按输出行分块做 im2col，再交给
/// <see cref="GemmKernel"/> 做矩阵乘（对照 OpenCV <c>conv_im2col</c>）。</item>
/// <item><see cref="Direct"/>：保留的参考实现，正确但慢，用于对照与兜底。</item>
/// </list>
/// </summary>
internal static class ConvKernels
{
    /// <summary>GEMM 微内核的列向量宽度（AVX2 下 8 个 float）。列块宽度低于它只能走标量内核。</summary>
    private const int ColumnVectorWidth = 8;

    /// <summary>
    /// im2col + GEMM：把卷积化成
    /// <c>W[outPerGroup, K] · col[K, 输出块像素]</c>，其中 <c>K = inPerGroup*kh*kw</c>。
    /// 大矩阵由 GEMM 内核向量化，外层按「batch × group × 输出行块」并行。
    /// </summary>
    public static void Im2Col(
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

        int inPerGroup = channels / groups;
        int outPerGroup = outChannels / groups;
        int kernelSize = inPerGroup * kernelH * kernelW;

        // 1x1 / stride 1 / 无 pad / 无 dilation：卷积退化成 GEMM，输入直接当列矩阵用
        bool isPointwise = kernelH == 1 && kernelW == 1
            && strideH == 1 && strideW == 1 && dilH == 1 && dilW == 1
            && padT == 0 && padL == 0;

        if (isPointwise)
        {
            Pointwise(input, weight, bias, output, groups, batches,
                channels, inPerGroup, outChannels, outPerGroup, outH * outW);
            return;
        }

        // 分块策略：
        // 1) 行块先按列缓冲预算（约 2 MB）取尽量大，任务数不够喂满多核时再减半。
        // 2) 行块一旦缩小，块内像素（rowsPerBlock*outW）必须仍是 8 的倍数，
        //    否则列块宽度不足 8 的尾巴会掉进标量内核 —— 20×20 这种输出宽度上
        //    以前会缩到 1 行（n 只剩 20，其中 4 列走标量），256→256 只有 40 GMAC/s。
        // 3) 切不动（例如 outW=20）就把并行度交给 GEMM 内部按列切分，而不是把 n 切成碎块。
        int threads = ParallelRange.Threads;
        int budgetPixels = Math.Max(threads * 64, (2 << 20) / (4 * Math.Max(1, kernelSize)));
        int rowsPerBlock = Math.Min(outH, Math.Max(1, budgetPixels / Math.Max(1, outW)));

        while (rowsPerBlock > 1)
        {
            int blocks = (outH + rowsPerBlock - 1) / rowsPerBlock;
            if (batches * groups * blocks >= threads * 2)
                break;

            int half = rowsPerBlock / 2;
            if (half < 1 || (long)half * outW % ColumnVectorWidth != 0)
                break;

            rowsPerBlock = half;
        }

        int blockCount = (outH + rowsPerBlock - 1) / rowsPerBlock;
        int baseTasks = batches * groups * blockCount;

        // 行方向仍然不够时再把输出宽度切开（列切分要额外搬一次输出，代价更高）
        int colSplits = 1;
        while (baseTasks * colSplits < threads * 2
               && outW / (colSplits * 2) >= 32
               && outW / (colSplits * 2) % ColumnVectorWidth == 0)
        {
            colSplits *= 2;
        }

        int tileWidth = (outW + colSplits - 1) / colSplits;
        int tasks = baseTasks * colSplits;

        // 任务还不够：外层串行、把并行度交给 GEMM（它按 16 列的列块切分，列宽天然对齐）
        bool innerParallel = tasks < threads;
        if (innerParallel)
        {
            colSplits = 1;
            tileWidth = outW;
            tasks = baseTasks;
        }

        Action<int, int> runTasks = (begin, end) =>
        {
            var src = input.AsSpan<float>();
            var weights = weight.AsSpan<float>();
            var biases = bias is null ? Span<float>.Empty : bias.AsSpan<float>();
            var dst = output.AsSpan<float>();

            int maxPixels = rowsPerBlock * tileWidth;
            float[] pool = ArrayPool<float>.Shared.Rent(kernelSize * maxPixels);
            float[]? outPool = colSplits > 1
                ? ArrayPool<float>.Shared.Rent(outPerGroup * maxPixels)
                : null;

            for (int task = begin; task < end; task++)
            {
                int colSplit = task % colSplits;
                int rest = task / colSplits;
                int block = rest % blockCount;
                int outer = rest / blockCount;
                int n = outer / groups;
                int g = outer - n * groups;

                int oyBegin = block * rowsPerBlock;
                int rows = Math.Min(rowsPerBlock, outH - oyBegin);

                int oxBegin = colSplit * tileWidth;
                int colCount = Math.Min(tileWidth, outW - oxBegin);

                int pixels = rows * colCount;

                FillColumns(src, pool, channels, inH, inW, n, g * inPerGroup, inPerGroup,
                    kernelH, kernelW, strideH, strideW, dilH, dilW, padT, padL,
                    oyBegin, rows, oxBegin, colCount, outW, pixels);

                long dstBase = ((long)n * outChannels + g * outPerGroup) * outH * outW
                             + (long)oyBegin * outW + oxBegin;

                var weightSlice = weights.Slice(g * outPerGroup * kernelSize, outPerGroup * kernelSize);
                var colSlice = pool.AsSpan(0, kernelSize * pixels);

                if (colSplits == 1)
                {
                    // 整行处理：块内像素在 NCHW 里就是连续的，直接写输出
                    var target = dst.Slice((int)dstBase, (outPerGroup - 1) * outH * outW + pixels);

                    GemmKernel.Multiply(weightSlice, colSlice, target,
                        outPerGroup, kernelSize, pixels,
                        ldc: outH * outW, parallel: innerParallel,
                        bias: biases.IsEmpty ? ReadOnlySpan<float>.Empty : biases.Slice(g * outPerGroup, outPerGroup));
                }
                else
                {
                    // 列切分后块内像素不再连续（每行只取 outW 中的一段），
                    // 先在紧凑缓冲里算完再按行搬回输出。
                    var scratch = outPool!.AsSpan(0, outPerGroup * pixels);

                    GemmKernel.Multiply(weightSlice, colSlice, scratch,
                        outPerGroup, kernelSize, pixels,
                        ldc: pixels, parallel: innerParallel,
                        bias: biases.IsEmpty ? ReadOnlySpan<float>.Empty : biases.Slice(g * outPerGroup, outPerGroup));

                    for (int oc = 0; oc < outPerGroup; oc++)
                    {
                        for (int r = 0; r < rows; r++)
                        {
                            scratch.Slice(oc * pixels + r * colCount, colCount)
                                .CopyTo(dst.Slice((int)(dstBase + (long)oc * outH * outW + (long)r * outW), colCount));
                        }
                    }
                }
            }

            ArrayPool<float>.Shared.Return(pool);

            if (outPool is not null)
                ArrayPool<float>.Shared.Return(outPool);
        };

        if (innerParallel)
            runTasks(0, tasks);
        else
            ParallelRange.For(tasks, runTasks);
    }

    /// <summary>1x1 卷积：<c>out[oc, p] = Σ_ic W[oc, ic] · in[ic, p]</c>，零拷贝。</summary>
    private static void Pointwise(
        Tensor input,
        Tensor weight,
        Tensor? bias,
        Tensor output,
        int groups,
        int batches,
        int channels,
        int inPerGroup,
        int outChannels,
        int outPerGroup,
        int pixels)
    {
        int tasks = batches * groups;
        bool innerParallel = tasks <= 1;

        ParallelRange.For(tasks, (begin, end) =>
        {
            var src = input.AsSpan<float>();
            var weights = weight.AsSpan<float>();
            var biases = bias is null ? Span<float>.Empty : bias.AsSpan<float>();
            var dst = output.AsSpan<float>();

            for (int task = begin; task < end; task++)
            {
                int n = task / groups;
                int g = task - n * groups;

                long srcBase = (long)(n * channels + g * inPerGroup) * pixels;
                long dstBase = ((long)n * outChannels + g * outPerGroup) * pixels;

                GemmKernel.Multiply(
                    weights.Slice(g * outPerGroup * inPerGroup, outPerGroup * inPerGroup),
                    src.Slice((int)srcBase, inPerGroup * pixels),
                    dst.Slice((int)dstBase, outPerGroup * pixels),
                    outPerGroup, inPerGroup, pixels,
                    ldc: pixels,
                    parallel: innerParallel,
                    bias: biases.IsEmpty ? ReadOnlySpan<float>.Empty : biases.Slice(g * outPerGroup, outPerGroup));
            }
        });
    }

    /// <summary>
    /// 把一块输出区域的感受野摊平成 <c>col[K, pixels]</c>（越界填 0）。
    /// <para>
    /// stride == 1 时一整个输出行就是一个输入行上的连续区间，只有左右两侧可能越界，
    /// 所以中间用一次 <see cref="Buffer.MemoryCopy"/> 搬完、两侧各清零一次 —— 原来
    /// 每个元素都要算一次 <c>(uint)x &lt; inW</c> 再逐元素写，3×3 conv 的这步占满了
    /// 整层时间。stride &gt; 1 时无法整段搬，保留逐元素路径。
    /// </para>
    /// </summary>
    private static unsafe void FillColumns(
        ReadOnlySpan<float> src,
        float[] col,
        int channels,
        int inH,
        int inW,
        int batch,
        int channelStart,
        int inPerGroup,
        int kernelH,
        int kernelW,
        int strideH,
        int strideW,
        int dilH,
        int dilW,
        int padT,
        int padL,
        int oyBegin,
        int rows,
        int oxBegin,
        int colCount,
        int outW,
        int pixels)
    {
        int rowBytes = colCount * sizeof(float);

        fixed (float* pin = src)
        fixed (float* pcol = col)
        {
            for (int ic = 0; ic < inPerGroup; ic++)
            {
                float* channelBase = pin + (long)(batch * channels + channelStart + ic) * inH * inW;

                for (int ky = 0; ky < kernelH; ky++)
                {
                    for (int kx = 0; kx < kernelW; kx++)
                    {
                        int colRow = (ic * kernelH + ky) * kernelW + kx;
                        float* target = pcol + (long)colRow * pixels;

                        // 本核位置在输入行上的起点，对应输出块的第一个像素
                        int x0 = oxBegin * strideW - padL + kx * dilW;

                        for (int r = 0; r < rows; r++)
                        {
                            int y = (oyBegin + r) * strideH - padT + ky * dilH;
                            float* row = target + (long)r * colCount;

                            if ((uint)y >= (uint)inH)
                            {
                                new Span<float>(row, colCount).Clear();
                                continue;
                            }

                            float* source = channelBase + (long)y * inW;

                            if (strideW == 1)
                            {
                                int first = Math.Max(x0, 0);
                                int last = Math.Min(x0 + colCount, inW);

                                if (last <= first)
                                {
                                    new Span<float>(row, colCount).Clear();
                                    continue;
                                }

                                int head = first - x0;
                                int body = last - first;
                                if (head > 0)
                                    new Span<float>(row, head).Clear();

                                Buffer.MemoryCopy(source + first, row + head, body * sizeof(float), body * sizeof(float));

                                int tail = colCount - head - body;
                                if (tail > 0)
                                    new Span<float>(row + head + body, tail).Clear();
                            }
                            else
                            {
                                for (int i = 0; i < colCount; i++)
                                {
                                    int x = x0 + i * strideW;
                                    row[i] = (uint)x < (uint)inW ? source[x] : 0f;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

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
