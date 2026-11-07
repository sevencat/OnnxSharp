namespace OnnxSharp.Threading;

/// <summary>
/// <c>cv::parallel_for_</c> 的等价物：由调用方（引擎/kernel）决定并行度，
/// 内核本身不再自行开线程。
/// </summary>
public static class ParallelRange
{
    /// <summary>低于该工作量时直接串行执行，避免调度开销。</summary>
    public static long SerialThreshold = 1 << 16;

    private static int _maxDegreeOfParallelism;

    static ParallelRange()
    {
        // 默认跟 ProcessorCount；ONNXSHARP_THREADS 覆盖（笔记本 APU 上 8 线程常比 16 线程快，
        // 因为超线程争用同一个 FP 单元，而这里的 kernel 全是 FP 密集）。
        if (int.TryParse(Environment.GetEnvironmentVariable("ONNXSHARP_THREADS"), out int threads)
            && threads > 0)
        {
            _maxDegreeOfParallelism = threads;
        }
    }

    /// <summary>可用线程数（0 表示自动）。</summary>
    public static int MaxDegreeOfParallelism
    {
        get => _maxDegreeOfParallelism;
        set => _maxDegreeOfParallelism = value;
    }

    /// <summary>
    /// 实际生效的线程数。kernel 做分块决策（块大小、任务数）时应当用这个而不是
    /// <see cref="Environment.ProcessorCount"/>，否则改并行度时分块策略和线程数会脱节。
    /// </summary>
    public static int Threads => _maxDegreeOfParallelism > 0 ? _maxDegreeOfParallelism : Environment.ProcessorCount;

    /// <summary>把 [0, total) 切分成若干连续区间，回调参数为 [begin, end)。</summary>
    public static void For(int total, Action<int, int> body)
    {
        if (total <= 0)
            return;

        int threads = Threads;

        // 任务数不足时直接串行：Parallel.For 一趟就有 7 µs 左右的固定成本，
        // 小任务（Concat 两路、几 KB 的 Reshape）并行只会更慢。
        if (total <= 1 || threads <= 1 || total < threads)
        {
            body(0, total);
            return;
        }

        int chunk = (total + threads - 1) / threads;

        System.Threading.Tasks.Parallel.For(0, threads, t =>
        {
            int begin = t * chunk;
            int end = Math.Min(begin + chunk, total);
            if (begin < end)
                body(begin, end);
        });
    }
}
