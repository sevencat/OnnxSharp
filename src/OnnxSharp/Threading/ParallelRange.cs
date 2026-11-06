namespace OnnxSharp.Threading;

/// <summary>
/// <c>cv::parallel_for_</c> 的等价物：由调用方（引擎/kernel）决定并行度，
/// 内核本身不再自行开线程。
/// </summary>
public static class ParallelRange
{
    /// <summary>低于该工作量时直接串行执行，避免调度开销。</summary>
    public static long SerialThreshold = 1 << 16;

    /// <summary>可用线程数（0 表示自动）。</summary>
    public static int MaxDegreeOfParallelism { get; set; }

    /// <summary>把 [0, total) 切分成若干连续区间，回调参数为 [begin, end)。</summary>
    public static void For(int total, Action<int, int> body)
    {
        if (total <= 0)
            return;

        int threads = MaxDegreeOfParallelism > 0 ? MaxDegreeOfParallelism : Environment.ProcessorCount;
        if (threads <= 1 || total < threads)
        {
            body(0, total);
            return;
        }

        int chunk = (total + threads - 1) / threads;
        Parallel.For(0, threads, t =>
        {
            int begin = t * chunk;
            int end = Math.Min(begin + chunk, total);
            if (begin < end)
                body(begin, end);
        });
    }
}
