using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using OnnxSharp.Tensors;

namespace OnnxSharp.Kernels;

/// <summary>
/// <c>ArgMax(MatMul(a, b) + bias)</c> 的融合内核：**不物化 logits**，在寄存器里维护行最大值与下标
/// （思路来自 SimdOnnx 的 <c>MatMul.ArgMax.Avx512</c>）。
/// <para>
/// <b>省掉的是什么</b>：普通写法要写一遍 <c>[m,n]</c> 的 logits、再读一遍做 argmax。
/// 逐 token 解码 <c>m=1, n=8000</c> 只有 32 KB，收益有限；但 CTC/强制对齐那种
/// <c>m = T 帧（几百）× n = 词表（几千）</c> 是几 MB~几十 MB，省下来就是实打实的带宽。
/// </para>
/// <para>
/// <b>并行策略</b>：<c>m ≥ 2</c> 按行切；<c>m == 1</c> 按**列**切（每个线程扫一段列，主线程按
/// 「值大者胜、相等取更小下标」合并），与 ONNX <c>ArgMax</c>（<c>select_last_index = 0</c>）语义一致。
/// </para>
/// <para>Span 不能进 lambda，所以这里用 <c>fixed</c> + 指针（与 <see cref="GemmKernel"/> 一致）。</para>
/// </summary>
public static class MatMulArgMaxKernel
{
    private const int Lanes = 8;
    private const int RowsPerBlock = 4;
    private const long ParallelThreshold = 1L << 21;

    /// <summary>logits 元素数不超过这个值时走「GEMM + 扫描」路径（16 MB 的 scratch 上限）。</summary>
    private const long ScratchElements = 1L << 22;

    /// <summary>逐行求 <c>argmax_j (Σ_k a[row,k]·b[k,j] + bias[j])</c>；<paramref name="bias"/> 为空表示无偏置。</summary>
    public static unsafe void Compute(
        ReadOnlySpan<float> a,
        ReadOnlySpan<float> b,
        ReadOnlySpan<float> bias,
        int m,
        int k,
        int n,
        Span<long> argMax,
        Span<float> maxValues,
        bool parallel = true)
    {
        if (argMax.Length < m)
            throw new ArgumentException($"argMax needs {m} slots", nameof(argMax));
        if (!maxValues.IsEmpty && maxValues.Length < m)
            throw new ArgumentException($"maxValues needs {m} slots", nameof(maxValues));

        bool hasBias = bias.Length >= n;
        int threads = Math.Max(1, Threading.ParallelRange.Threads);

        fixed (float* pa = a)
        fixed (float* pb = b)
        fixed (float* pbias = bias)
        fixed (long* parg = argMax)
        fixed (float* pmax = maxValues)
        {
            float* a0 = pa;
            float* b0 = pb;
            float* bias0 = hasBias ? pbias : null;
            long* arg0 = parg;
            float* max0 = pmax;

            // 分层策略：logits 不大的话，直接让高度优化的 GemmKernel 算到池化缓冲、再向量化扫描
            // —— 比手写 GEMV 循环快（那个只有 8 路累加器，打不过寄存器分块 + 打包的 GEMM）；
            // 只有 logits 真的很大（CTC 长语音）才走「不物化」的流式路径，省掉几 MB 的写读。
            long elements = (long)m * n;
            if (elements <= ScratchElements && (parallel || threads == 1))
            {
                float[] scratch = System.Buffers.ArrayPool<float>.Shared.Rent((int)elements);
                try
                {
                    var logits = scratch.AsSpan(0, (int)elements);
                    Tensors.GemmKernel.Multiply(
                        new ReadOnlySpan<float>(a0, m * k), new ReadOnlySpan<float>(b0, k * n),
                        logits, m, k, n, 0, parallel);

                    if (hasBias)
                        AddColumnBias(logits, bias0, m, n);

                    ScanRows(logits, m, n, arg0, max0);
                }
                finally
                {
                    System.Buffers.ArrayPool<float>.Shared.Return(scratch);
                }

                return;
            }

            if (parallel && m >= 2 && (long)m * k * n >= ParallelThreshold)
            {
                int rowBlocks = (m + RowsPerBlock - 1) / RowsPerBlock;

                // 只按行块并行的话，m = 24 只有 6 个任务（16 核闲 10 个）——
                // 再按列切一刀，任务是 rowBlocks × chunks，每个任务只扫自己那段列。
                int chunks = Math.Clamp(n / 256, 1, Math.Max(1, threads));
                int tasks = rowBlocks * chunks;

                var values = new float[tasks * RowsPerBlock];
                var indices = new int[tasks * RowsPerBlock];

                var options = new ParallelOptions { MaxDegreeOfParallelism = threads };
                Parallel.For(0, tasks, options, task =>
                {
                    int block = task / chunks;
                    int chunk = task % chunks;
                    int row0 = block * RowsPerBlock;
                    int rows = Math.Min(RowsPerBlock, m - row0);
                    int jBegin = (int)((long)chunk * n / chunks);
                    int jEnd = (int)((long)(chunk + 1) * n / chunks);

                    ScanBlock(a0 + (long)row0 * k, rows, b0, bias0, hasBias, k, n, jBegin, jEnd,
                        values.AsSpan(task * RowsPerBlock, RowsPerBlock),
                        indices.AsSpan(task * RowsPerBlock, RowsPerBlock));
                });

                // 合并各列块：值大者胜，相等取更小下标（与串行「首次出现」一致）
                for (int row = 0; row < m; row++)
                {
                    int block = row / RowsPerBlock;
                    float best = float.NegativeInfinity;
                    int bestIndex = 0;

                    for (int chunk = 0; chunk < chunks; chunk++)
                    {
                        int slot = (block * chunks + chunk) * RowsPerBlock + (row % RowsPerBlock);
                        if (indices[slot] < n && values[slot] > best)
                        {
                            best = values[slot];
                            bestIndex = indices[slot];
                        }
                    }

                    arg0[row] = bestIndex;
                    if (max0 is not null)
                        max0[row] = best;
                }

                return;
            }

            ComputeRest(a0, b0, bias0, hasBias, m, k, n, arg0, pmax, parallel, threads);
        }
    }

    /// <summary><c>m == 1</c> 按列切分（解码步）与串行兜底。</summary>
    private static unsafe void ComputeRest(
        float* a0,
        float* b0,
        float* bias0,
        bool hasBias,
        int m,
        int k,
        int n,
        long* parg,
        float* pmax,
        bool parallel,
        int threads)
    {
        // m == 1（每个 token 一次）：按列切分并行，否则整个解码 step 是单线程 GEMV
        if (parallel && m == 1 && (long)k * n >= 1L << 20)
        {
            int chunk = (n + threads - 1) / threads;
            var values = new float[threads];
            var indices = new int[threads];

            var options = new ParallelOptions { MaxDegreeOfParallelism = threads };
            Parallel.For(0, threads, options, t =>
            {
                int begin = t * chunk;
                int end = Math.Min(n, begin + chunk);
                if (begin >= end)
                {
                    values[t] = float.NegativeInfinity;
                    indices[t] = n;
                    return;
                }

                var (value, index) = ScanRow(a0, b0, bias0, hasBias, k, n, begin, end);
                values[t] = value;
                indices[t] = index;
            });

            float best = float.NegativeInfinity;
            int bestIndex = 0;
            for (int t = 0; t < threads; t++)
            {
                // 相等时取更小下标，与串行扫描的「首次出现」一致
                if (indices[t] < n && values[t] > best)
                {
                    best = values[t];
                    bestIndex = indices[t];
                }
            }

            parg[0] = bestIndex;
            if (pmax is not null)
                pmax[0] = best;
            return;
        }

        for (int row = 0; row < m; row++)
        {
            var (value, index) = ScanRow(a0 + (long)row * k, b0, bias0, hasBias, k, n, 0, n);
            parg[row] = index;
            if (pmax is not null)
                pmax[row] = value;
        }
    }

    /// <summary>给物化好的 logits 每列加上偏置（<c>[rows, columns] += bias[columns]</c>）。</summary>
    private static unsafe void AddColumnBias(Span<float> logits, float* bias, int rows, int columns)
    {
        for (int row = 0; row < rows; row++)
        {
            var line = logits.Slice(row * columns, columns);
            int j = 0;

            if (Avx.IsSupported)
            {
                for (; j + Lanes <= columns; j += Lanes)
                {
                    var value = Vector256.Add(Vector256.Load(bias + j), Vector256.LoadUnsafe(ref MemoryMarshal.GetReference(line[j..])));
                    value.StoreUnsafe(ref MemoryMarshal.GetReference(line[j..]));
                }
            }

            for (; j < columns; j++)
                line[j] += bias[j];
        }
    }

    /// <summary>对物化好的 logits 逐行取 argmax（首次出现）。</summary>
    private static unsafe void ScanRows(ReadOnlySpan<float> logits, int rows, int columns, long* argMax, float* maxValues)
    {
        Span<float> lanes = stackalloc float[Lanes];

        for (int row = 0; row < rows; row++)
        {
            var line = logits.Slice(row * columns, columns);
            float best = float.NegativeInfinity;
            int j = 0;

            if (Avx.IsSupported)
            {
                var vectorMax = Vector256.Create(float.NegativeInfinity);
                for (; j + Lanes <= columns; j += Lanes)
                    vectorMax = Vector256.Max(vectorMax, Vector256.LoadUnsafe(ref MemoryMarshal.GetReference(line[j..])));

                vectorMax.CopyTo(lanes);
                for (int lane = 0; lane < Lanes; lane++)
                    best = MathF.Max(best, lanes[lane]);
            }

            for (; j < columns; j++)
                best = MathF.Max(best, line[j]);

            int index = 0;
            for (; index < columns; index++)
            {
                if (line[index] == best)
                    break;
            }

            argMax[row] = index < columns ? index : 0;
            if (maxValues is not null)
                maxValues[row] = best;
        }
    }

    /// <summary>
    /// 一次算 <see cref="RowsPerBlock"/> 行的 argmax：B 的一行数据被 4 个累加器复用，
    /// 所以 B 只需读 <c>ceil(m/4)</c> 遍（逐行做的话是 m 遍，CTC 形状下差 4~8 倍）。
    /// </summary>
    private static unsafe void ScanBlock(
        float* a0,
        int rows,
        float* b,
        float* bias,
        bool hasBias,
        int k,
        int n,
        int jBegin,
        int jEnd,
        Span<float> best,
        Span<int> bestIndex)
    {
        for (int r = 0; r < rows; r++)
        {
            best[r] = float.NegativeInfinity;
            bestIndex[r] = jBegin;
        }

        int j = jBegin;

        for (; j + Lanes <= jEnd; j += Lanes)
        {
            var initial = hasBias ? Vector256.Load(bias + j) : Vector256<float>.Zero;

            // rows == 1 时只留一个累加器，避免空转
            var acc0 = initial;
            var acc1 = initial;
            var acc2 = initial;
            var acc3 = initial;

            for (int kk = 0; kk < k; kk++)
            {
                var weights = Vector256.Load(b + (long)kk * n + j);

                acc0 = Avx.Add(acc0, Avx.Multiply(Vector256.Create(a0[kk]), weights));
                if (rows > 1)
                    acc1 = Avx.Add(acc1, Avx.Multiply(Vector256.Create(a0[k + kk]), weights));
                if (rows > 2)
                    acc2 = Avx.Add(acc2, Avx.Multiply(Vector256.Create(a0[2 * k + kk]), weights));
                if (rows > 3)
                    acc3 = Avx.Add(acc3, Avx.Multiply(Vector256.Create(a0[3 * k + kk]), weights));
            }

            Reduce(acc0, j, best, bestIndex, 0);
            if (rows > 1) Reduce(acc1, j, best, bestIndex, 1);
            if (rows > 2) Reduce(acc2, j, best, bestIndex, 2);
            if (rows > 3) Reduce(acc3, j, best, bestIndex, 3);
        }

        // 尾巴：列数不是 8 的倍数
        for (; j < jEnd; j++)
        {
            for (int r = 0; r < rows; r++)
            {
                float* row = a0 + r * k;
                float sum = hasBias ? bias[j] : 0f;
                for (int kk = 0; kk < k; kk++)
                    sum += row[kk] * b[(long)kk * n + j];

                if (sum > best[r])
                {
                    best[r] = sum;
                    bestIndex[r] = j;
                }
            }
        }
    }

    private static void Reduce(Vector256<float> values, int offset, Span<float> best, Span<int> bestIndex, int row)
    {
        Span<float> lanes = stackalloc float[Lanes];
        values.CopyTo(lanes);

        for (int lane = 0; lane < Lanes; lane++)
        {
            float value = lanes[lane];
            if (value > best[row])
            {
                best[row] = value;
                bestIndex[row] = offset + lane;
            }
        }
    }

    /// <summary>扫一行 <c>[jBegin, jEnd)</c> 的列，返回 (最大值, 首次出现的下标)。</summary>
    private static unsafe (float Value, int Index) ScanRow(
        float* aRow,
        float* b,
        float* bias,
        bool hasBias,
        int k,
        int n,
        int jBegin,
        int jEnd)
    {
        float best = float.NegativeInfinity;
        int bestIndex = jBegin;
        int j = jBegin;

        if (Avx.IsSupported)
        {
            Span<float> lanes = stackalloc float[Lanes];

            for (; j + Lanes <= jEnd; j += Lanes)
            {
                var accumulator = hasBias ? Vector256.Load(bias + j) : Vector256<float>.Zero;

                for (int kk = 0; kk < k; kk++)
                    accumulator = Avx.Add(
                        accumulator,
                        Avx.Multiply(Vector256.Create(aRow[kk]), Vector256.Load(b + (long)kk * n + j)));

                accumulator.CopyTo(lanes);

                for (int lane = 0; lane < Lanes; lane++)
                {
                    float value = lanes[lane];
                    if (value > best)
                    {
                        best = value;
                        bestIndex = j + lane;
                    }
                }
            }
        }

        for (; j < jEnd; j++)
        {
            float sum = hasBias ? bias[j] : 0f;
            for (int kk = 0; kk < k; kk++)
                sum += aRow[kk] * b[(long)kk * n + j];

            if (sum > best)
            {
                best = sum;
                bestIndex = j;
            }
        }

        return (best, bestIndex);
    }
}
