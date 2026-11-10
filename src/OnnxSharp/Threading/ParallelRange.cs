using System.Runtime.ExceptionServices;

namespace OnnxSharp.Threading;

/// <summary>
/// <c>cv::parallel_for_</c> 的等价物：由调用方（引擎/kernel）决定并行度，
/// 内核本身不再自行开线程。
/// <para>
/// 内部是一个<b>常驻线程池</b>（对齐 ONNXRuntime 的 intra-op pool）：一次前向有 200+ 个并行区，
/// 每个区只有几十~几百微秒，用 TPL 的 <c>Parallel.For</c> 光调度就要 14.4 µs/次（见 <c>parbench</c>），
/// 累计就是 2~3 ms。这里改成「调用线程参与 + 工作线程自旋领取任务」，热态一次派发 1~2 µs。
/// </para>
/// </summary>
public static class ParallelRange
{
    /// <summary>低于该工作量时直接串行执行，避免调度开销。</summary>
    public static long SerialThreshold = 1 << 16;

    /// <summary>关掉常驻线程池（<c>ONNXSHARP_THREADPOOL=0</c>），回退到 TPL 的 <c>Parallel.For</c>。</summary>
    private static readonly bool UsePool =
        Environment.GetEnvironmentVariable("ONNXSHARP_THREADPOOL") != "0";

    private static int _maxDegreeOfParallelism;

    private static readonly object PoolLock = new();
    private static Pool? _pool;
    private static int _poolThreads;

    /// <summary>当前线程是否正在某个并行区里干活（用于把嵌套并行降级成串行，避免死锁）。</summary>
    [ThreadStatic]
    private static bool _insideRegion;

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

        // 任务数不足时直接串行：一次派发再便宜也顶不上一段几 KB 的拷贝。
        if (total <= 1 || threads <= 1 || total < threads)
        {
            body(0, total);
            return;
        }

        int chunk = (total + threads - 1) / threads;
        int parts = (total + chunk - 1) / chunk;

        Execute(parts, part =>
        {
            int begin = part * chunk;
            int end = Math.Min(begin + chunk, total);
            if (begin < end)
                body(begin, end);
        });
    }

    /// <summary>
    /// 把 [0, <paramref name="count"/>) 当成 <paramref name="count"/> 个独立任务并行跑。
    /// 任务数不必等于线程数：工作线程动态领取（<c>Interlocked</c> 计数），
    /// 适合块大小不一、或块数远多于线程数的场景。
    /// </summary>
    public static void ForEach(int count, Action<int> body)
    {
        if (count <= 0)
            return;

        if (count == 1 || Threads <= 1)
        {
            for (int i = 0; i < count; i++)
                body(i);

            return;
        }

        Execute(count, body);
    }

    private static void Execute(int count, Action<int> body)
    {
        // 已经在并行区里的线程不能再派发（否则线程池会被自己占满 → 死锁）
        if (_insideRegion || Threads <= 1)
        {
            for (int i = 0; i < count; i++)
                body(i);

            return;
        }

        if (!UsePool)
        {
            // 兜底：旧的 TPL 路径（A/B 与排障用）
            System.Threading.Tasks.Parallel.For(0, count, i => body(i));
            return;
        }

        Pool pool = GetPool();
        if (pool.WorkerCount == 0)
        {
            for (int i = 0; i < count; i++)
                body(i);

            return;
        }

        pool.Run(count, body);
    }

    private static Pool GetPool()
    {
        int threads = Threads;

        lock (PoolLock)
        {
            if (_pool is null || _poolThreads != threads)
            {
                _pool = new Pool(Math.Max(0, threads - 1));   // 调用线程自己也算一个工位
                _poolThreads = threads;
            }

            return _pool;
        }
    }

    /// <summary>一个并行区：任务、计数器、错误都挂在这个对象上，一次性发布，不会读到撕裂状态。</summary>
    private sealed class Region
    {
        public readonly Action<int> Task;
        public readonly int Count;
        public int Claimed;
        public int Finished;
        public ExceptionDispatchInfo? Error;

        public Region(Action<int> task, int count)
        {
            Task = task;
            Count = count;
        }
    }

    private sealed class Pool
    {
        private readonly Thread[] _workers;
        private Region? _current;

        public Pool(int workerCount)
        {
            _workers = new Thread[workerCount];

            for (int i = 0; i < workerCount; i++)
            {
                var thread = new Thread(WorkerLoop)
                {
                    IsBackground = true,
                    Name = $"onnxsharp-pool-{i}",
                };

                _workers[i] = thread;
                thread.Start();
            }
        }

        public int WorkerCount => _workers.Length;

        public void Run(int count, Action<int> task)
        {
            var region = new Region(task, count);
            Volatile.Write(ref _current, region);

            _insideRegion = true;
            try
            {
                Drain(region);
            }
            finally
            {
                _insideRegion = false;
            }

            WaitForWorkers(region);

            ExceptionDispatchInfo? error = region.Error;
            if (error is not null)
                error.Throw();
        }

        private static void Drain(Region region)
        {
            while (true)
            {
                int index = Interlocked.Increment(ref region.Claimed) - 1;
                if (index >= region.Count)
                    return;

                try
                {
                    region.Task(index);
                }
                catch (Exception exception)
                {
                    Interlocked.CompareExchange(ref region.Error, ExceptionDispatchInfo.Capture(exception), null);
                }
            }
        }

        private void WorkerLoop()
        {
            Region? last = null;
            long idle = 0;

            while (true)
            {
                Region? region = Volatile.Read(ref _current);
                if (region is not null && !ReferenceEquals(region, last))
                {
                    last = region;

                    _insideRegion = true;
                    try
                    {
                        Drain(region);
                    }
                    finally
                    {
                        _insideRegion = false;
                    }

                    Interlocked.Increment(ref region.Finished);
                    idle = 0;
                    continue;
                }

                idle = WaitOnce(idle);
            }
        }

        /// <summary>
        /// 等所有工作线程报告完成。
        /// <para>
        /// 不能用 <see cref="SpinWait"/>：它 ~10 次之后就开始 <c>Sleep(1)</c>，
        /// 而这里每次前向有 200+ 个并行区、区间间隔只有几十微秒 —— 一睡就是 1 ms，
        /// 实测把每次派发从 1 µs 拖到 1 ms 以上。改成「紧凑自旋 → Yield → 最后才 Sleep」。
        /// </para>
        /// </summary>
        private void WaitForWorkers(Region region)
        {
            long idle = 0;
            while (Volatile.Read(ref region.Finished) < _workers.Length)
                idle = WaitOnce(idle);
        }

        /// <summary>自旋/让出/休眠三档等待，返回累计空闲计数。</summary>
        private static long WaitOnce(long idle)
        {
            if (idle < 2000)
                Thread.SpinWait(20);            // ~50 µs 紧凑自旋
            else if (idle < 40000)
                Thread.Yield();                 // 让出时间片但不睡
            else
                Thread.Sleep(1);                // 真的没活了（前向之间），别烧 CPU

            return idle + 1;
        }
    }
}
