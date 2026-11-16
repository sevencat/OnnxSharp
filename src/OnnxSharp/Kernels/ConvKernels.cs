using System.Buffers;
using System.Diagnostics;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using OnnxSharp.Core;
using OnnxSharp.Tensors;
using OnnxSharp.Threading;

namespace OnnxSharp.Kernels;

/// <summary>
/// 卷积分阶段计时（<c>ONNXSHARP_CONV_PROF=1</c>）：把「1×1 走 pointwise」这条路径拆成
/// <list type="bullet">
/// <item><see cref="PointwiseMs"/>：<c>Im2Col</c> 里整段（含 Span 切片与并行派发）</item>
/// <item><see cref="GemmMs"/>：其中真正的 <see cref="GemmKernel.Multiply"/></item>
/// <item><see cref="ActMs"/>：卷积后融合的激活（<c>SiLU</c>）就地一趟</item>
/// </list>
/// 两者之差就是「不在 GEMM 里」的那部分开销 —— 1×1 层耗时反常时先看这个差。
/// </summary>
internal static class ConvProf
{
    public static readonly bool On =
        Environment.GetEnvironmentVariable("ONNXSHARP_CONV_PROF") == "1";

    private static long _pointwise;
    private static long _gemm;
    private static long _act;

    private const int Capacity = 16384;
    private static readonly long[] CallTicks = new long[Capacity];
    private static readonly int[] CallShape = new int[Capacity * 3];
    private static int _calls;
    private static int _dropped;

    // ---- im2col 路径（3×3 等）：把「打包列矩阵」与「GEMM」分开计 ----
    private static long _colTotal;
    private static long _colPack;
    private static long _colGemm;
    private static long _colWall;
    private static int _colTasks;

    private static readonly long[] ColTicks = new long[Capacity];
    private static readonly int[] ColShape = new int[Capacity * 3];
    private static int _colCalls;
    private static int _colDropped;

    // ---- 面板式直接卷积（PackedConv，3×3 的主路径）：打包面板 vs 微内核 ----
    private static long _packedTotal;
    private static long _packedPack;
    private static long _packedMul;
    private static long _packedWall;
    private static int _packedTasks;

    public static void AddPackedWall(long ticks) => Interlocked.Add(ref _packedWall, ticks);

    // ---- Winograd（3×3 s1 pad1、空间小的层）：只看墙钟，内部不再细分 ----
    private static long _winogradWall;
    private static int _winogradCalls;

    public static void AddWinograd(long ticks)
    {
        Interlocked.Add(ref _winogradWall, ticks);
        Interlocked.Increment(ref _winogradCalls);
    }

    public static (double WallMs, double CallsPerFrame) TakeWinograd(int frames)
    {
        double divisor = 1000.0 / Stopwatch.Frequency / Math.Max(1, frames);
        return (Interlocked.Exchange(ref _winogradWall, 0) * divisor,
                Interlocked.Exchange(ref _winogradCalls, 0) / (double)Math.Max(1, frames));
    }

    /// <summary>记一次面板任务：整段 / 打包 <c>[K][16]</c> / 微内核。面板粒度太细，不做逐形状记录。</summary>
    public static void AddPacked(long totalTicks, long packTicks, long mulTicks)
    {
        Interlocked.Add(ref _packedTotal, totalTicks);
        Interlocked.Add(ref _packedPack, packTicks);
        Interlocked.Add(ref _packedMul, mulTicks);
        Interlocked.Increment(ref _packedTasks);
    }

    public static (double WallMs, double TotalMs, double PackMs, double MulMs, double TasksPerFrame) TakePacked(int frames)
    {
        double scale = 1000.0 / Stopwatch.Frequency / Math.Max(1, frames);
        return (Interlocked.Exchange(ref _packedWall, 0) * scale,
                Interlocked.Exchange(ref _packedTotal, 0) * scale,
                Interlocked.Exchange(ref _packedPack, 0) * scale,
                Interlocked.Exchange(ref _packedMul, 0) * scale,
                Interlocked.Exchange(ref _packedTasks, 0) / (double)Math.Max(1, frames));
    }

    public static void AddIm2Col(long totalTicks, long packTicks, long gemmTicks)
    {
        Interlocked.Add(ref _colTotal, totalTicks);
        Interlocked.Add(ref _colPack, packTicks);
        Interlocked.Add(ref _colGemm, gemmTicks);
        Interlocked.Increment(ref _colTasks);
    }

    /// <summary>
    /// 整个 <c>Im2Col</c> 调用的墙钟（含并行派发）。逐任务累加值在多线程下 ≈ 墙钟 × 占用核数，
    /// 两者都报才不会被误读成「这一层花了 9 ms」。
    /// </summary>
    public static void AddIm2ColWall(long ticks) => Interlocked.Add(ref _colWall, ticks);

    /// <summary>记一次 im2col 任务（整段耗时 + GEMM 形状 m/k/n）。</summary>
    public static void AddIm2ColCall(long ticks, int m, int k, int n)
    {
        int slot = Interlocked.Increment(ref _colCalls) - 1;
        if ((uint)slot >= Capacity)
        {
            Interlocked.Increment(ref _colDropped);
            return;
        }

        ColTicks[slot] = ticks;
        ColShape[slot * 3] = m;
        ColShape[slot * 3 + 1] = k;
        ColShape[slot * 3 + 2] = n;
    }

    /// <summary>
    /// 清空累计值。**必须在预热之后、正式计时之前调一次** —— 否则预热的那几帧会混进来，
    /// 让「每帧」的除数和分子对不上（2026-11-16 前就是这样，1×1 因此被虚报 1.4 倍）。
    /// </summary>
    public static void Reset()
    {
        Interlocked.Exchange(ref _pointwise, 0);
        Interlocked.Exchange(ref _gemm, 0);
        Interlocked.Exchange(ref _act, 0);
        Interlocked.Exchange(ref _calls, 0);
        Interlocked.Exchange(ref _dropped, 0);
        Interlocked.Exchange(ref _colTotal, 0);
        Interlocked.Exchange(ref _colPack, 0);
        Interlocked.Exchange(ref _colGemm, 0);
        Interlocked.Exchange(ref _colWall, 0);
        Interlocked.Exchange(ref _colTasks, 0);
        Interlocked.Exchange(ref _colCalls, 0);
        Interlocked.Exchange(ref _colDropped, 0);
        Interlocked.Exchange(ref _packedTotal, 0);
        Interlocked.Exchange(ref _packedPack, 0);
        Interlocked.Exchange(ref _packedMul, 0);
        Interlocked.Exchange(ref _packedWall, 0);
        Interlocked.Exchange(ref _packedTasks, 0);
        Interlocked.Exchange(ref _winogradWall, 0);
        Interlocked.Exchange(ref _winogradCalls, 0);
    }

    public static void AddPointwise(long ticks) => Interlocked.Add(ref _pointwise, ticks);
    public static void AddGemm(long ticks) => Interlocked.Add(ref _gemm, ticks);
    public static void AddAct(long ticks) => Interlocked.Add(ref _act, ticks);

    /// <summary>记一次 1×1 调用（毫秒 + m/k/n），用于逐形状看谁在拖后腿。</summary>
    public static void AddCall(long ticks, int m, int k, int n)
    {
        int slot = Interlocked.Increment(ref _calls) - 1;
        if ((uint)slot >= Capacity)
        {
            // 溢出不能静默吞掉：次数少了，「每帧贡献」会被低估
            Interlocked.Increment(ref _dropped);
            return;
        }

        CallTicks[slot] = ticks;
        CallShape[slot * 3] = m;
        CallShape[slot * 3 + 1] = k;
        CallShape[slot * 3 + 2] = n;
    }

    /// <summary>取走并清零累计值，**除以实测帧数**换成每帧毫秒。</summary>
    public static (double PointwiseMs, double GemmMs, double ActMs) Take(int frames)
    {
        double scale = 1000.0 / Stopwatch.Frequency / Math.Max(1, frames);
        return (Interlocked.Exchange(ref _pointwise, 0) * scale,
                Interlocked.Exchange(ref _gemm, 0) * scale,
                Interlocked.Exchange(ref _act, 0) * scale);
    }

    /// <summary>
    /// 取走并清零逐调用记录，按形状聚合成「每帧贡献」（最小耗时 × 每帧次数），降序。
    /// 用最小值是为了剔掉首帧的 JIT/冷缓存，次数取每帧平均。
    /// <paramref name="dropped"/> 是被容量截断掉的调用数（非 0 说明次数被低估，要调大 <c>Capacity</c>）。
    /// </summary>
    public static List<(double MinMs, int Count, int M, int K, int N)> TakeCalls(int frames, out int dropped)
    {
        int count = Math.Min(Interlocked.Exchange(ref _calls, 0), Capacity);
        dropped = Interlocked.Exchange(ref _dropped, 0);
        return Aggregate(CallTicks, CallShape, count, frames);
    }

    /// <summary>
    /// 取走并清零 im2col 路径的阶段计时：整段 / 其中打包 / 其中 GEMM（毫秒每帧），
    /// 以及每帧的 im2col 任务数（行块 × 列切分 × batch × group）。
    /// </summary>
    public static (double WallMs, double TotalMs, double PackMs, double GemmMs, double TasksPerFrame) TakeIm2Col(int frames)
    {
        double scale = 1000.0 / Stopwatch.Frequency / Math.Max(1, frames);
        return (Interlocked.Exchange(ref _colWall, 0) * scale,
                Interlocked.Exchange(ref _colTotal, 0) * scale,
                Interlocked.Exchange(ref _colPack, 0) * scale,
                Interlocked.Exchange(ref _colGemm, 0) * scale,
                Interlocked.Exchange(ref _colTasks, 0) / (double)Math.Max(1, frames));
    }

    /// <summary>im2col 任务按 GEMM 形状（m=输出通道 · k=inC·kh·kw · n=块内像素）聚合。</summary>
    public static List<(double MinMs, int Count, int M, int K, int N)> TakeIm2ColCalls(int frames, out int dropped)
    {
        int count = Math.Min(Interlocked.Exchange(ref _colCalls, 0), Capacity);
        dropped = Interlocked.Exchange(ref _colDropped, 0);
        return Aggregate(ColTicks, ColShape, count, frames);
    }

    private static List<(double MinMs, int Count, int M, int K, int N)> Aggregate(
        long[] ticks, int[] shape, int count, int frames)
    {
        double scale = 1000.0 / Stopwatch.Frequency;
        var grouped = new Dictionary<(int, int, int), (long Min, int Count)>();

        for (int i = 0; i < count; i++)
        {
            var key = (shape[i * 3], shape[i * 3 + 1], shape[i * 3 + 2]);
            long value = ticks[i];
            ticks[i] = 0;

            if (grouped.TryGetValue(key, out var slot))
                grouped[key] = (Math.Min(slot.Item1, value), slot.Item2 + 1);
            else
                grouped[key] = (value, 1);
        }

        var list = new List<(double, int, int, int, int)>(grouped.Count);

        foreach (var (key, value) in grouped)
            list.Add((value.Item1 * scale, value.Item2 / Math.Max(1, frames), key.Item1, key.Item2, key.Item3));

        list.Sort((x, y) => (y.Item1 * y.Item2).CompareTo(x.Item1 * x.Item2));
        return list;
    }
}

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

    /// <summary>取 8 个 float 里的偶数道（stride 2 的 im2col 抽头）。</summary>
    private static readonly Vector256<int> EvenLanes = Vector256.Create(0, 2, 4, 6, 0, 0, 0, 0);

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
            long t0 = ConvProf.On ? Stopwatch.GetTimestamp() : 0;
            Pointwise(input, weight, bias, output, groups, batches,
                channels, inPerGroup, outChannels, outPerGroup, outH * outW);
            if (ConvProf.On)
                ConvProf.AddPointwise(Stopwatch.GetTimestamp() - t0);
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

                long t0 = ConvProf.On ? Stopwatch.GetTimestamp() : 0;

                FillColumns(src, pool, channels, inH, inW, n, g * inPerGroup, inPerGroup,
                    kernelH, kernelW, strideH, strideW, dilH, dilW, padT, padL,
                    oyBegin, rows, oxBegin, colCount, outW, pixels);

                long tPack = ConvProf.On ? Stopwatch.GetTimestamp() : 0;
                long tGemm = tPack;

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

                    if (ConvProf.On)
                        tGemm = Stopwatch.GetTimestamp();
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

                    if (ConvProf.On)
                        tGemm = Stopwatch.GetTimestamp();

                    for (int oc = 0; oc < outPerGroup; oc++)
                    {
                        for (int r = 0; r < rows; r++)
                        {
                            scratch.Slice(oc * pixels + r * colCount, colCount)
                                .CopyTo(dst.Slice((int)(dstBase + (long)oc * outH * outW + (long)r * outW), colCount));
                        }
                    }
                }

                if (ConvProf.On)
                {
                    long tEnd = Stopwatch.GetTimestamp();
                    ConvProf.AddIm2Col(tEnd - t0, tPack - t0, tGemm - tPack);
                    ConvProf.AddIm2ColCall(tEnd - t0, outPerGroup, kernelSize, pixels);
                }
            }

            ArrayPool<float>.Shared.Return(pool);

            if (outPool is not null)
                ArrayPool<float>.Shared.Return(outPool);
        };

        long tWall = ConvProf.On ? Stopwatch.GetTimestamp() : 0;

        if (innerParallel)
            runTasks(0, tasks);
        else
            ParallelRange.For(tasks, runTasks);

        if (ConvProf.On)
            ConvProf.AddIm2ColWall(Stopwatch.GetTimestamp() - tWall);
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

                long tg = ConvProf.On ? Stopwatch.GetTimestamp() : 0;

                GemmKernel.Multiply(
                    weights.Slice(g * outPerGroup * inPerGroup, outPerGroup * inPerGroup),
                    src.Slice((int)srcBase, inPerGroup * pixels),
                    dst.Slice((int)dstBase, outPerGroup * pixels),
                    outPerGroup, inPerGroup, pixels,
                    ldc: pixels,
                    parallel: innerParallel,
                    bias: biases.IsEmpty ? ReadOnlySpan<float>.Empty : biases.Slice(g * outPerGroup, outPerGroup));

                if (ConvProf.On)
                {
                    long elapsed = Stopwatch.GetTimestamp() - tg;
                    ConvProf.AddGemm(elapsed);
                    ConvProf.AddCall(elapsed, outPerGroup, inPerGroup, pixels);
                }
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
                                // 逐元素算边界太贵（一次比较 + 一次分支预测失败就是好几拍），
                                // 先把「整段都在界内」的 [lo, hi] 求出来，循环里只剩 load + store，
                                // 两端各清一次零。
                                int lo = x0 >= 0 ? 0 : Math.Min(colCount, (-x0 + strideW - 1) / strideW);
                                int hi = colCount - 1;

                                if (x0 + hi * strideW > inW - 1)
                                    hi = Math.Min(hi, (inW - 1 - x0) / strideW);

                                if (hi < lo)
                                {
                                    new Span<float>(row, colCount).Clear();
                                    continue;
                                }

                                if (lo > 0)
                                    new Span<float>(row, lo).Clear();
                                if (hi + 1 < colCount)
                                    new Span<float>(row + hi + 1, colCount - hi - 1).Clear();

                                if (strideW == 2 && Avx2.IsSupported && hi > lo)
                                {
                                    // stride 2：读 16 个连续 float，取偶数道就是 8 个输出像素。
                                    // 最后一条向量会多读 1 个 float，所以起点上限卡到不会越出本行。
                                    int vecHi = Math.Min(hi, (inW - 2 * ColumnVectorWidth - x0) / 2);
                                    int i = lo;

                                    for (; i + ColumnVectorWidth <= hi + 1 && i <= vecHi; i += ColumnVectorWidth)
                                    {
                                        float* p = source + x0 + i * 2;
                                        var e0 = Avx2.PermuteVar8x32(Avx.LoadVector256(p), EvenLanes);
                                        var e1 = Avx2.PermuteVar8x32(Avx.LoadVector256(p + 8), EvenLanes);
                                        Avx.Store(row + i, Avx2.Permute2x128(e0, e1, 0x20));
                                    }

                                    for (; i <= hi; i++)
                                        row[i] = source[x0 + i * 2];
                                }
                                else
                                {
                                    for (int i = lo; i <= hi; i++)
                                        row[i] = source[x0 + i * strideW];
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
