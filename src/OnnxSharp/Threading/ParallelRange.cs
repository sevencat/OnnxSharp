using System.Runtime.ExceptionServices;
using System.Runtime.InteropServices;

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
        else
        {
            // 没指定就取物理核数：超线程对这里的 FP 密集 kernel 没有吞吐收益，
            // 却让自旋线程池的争用翻倍（32 逻辑核的机器上，OCR 端到端中位数从 35 ms 涨到 184 ms）。
            _maxDegreeOfParallelism = Math.Max(1, Math.Min(Environment.ProcessorCount, PhysicalCoreCount()));
        }
    }

    /// <summary>
    /// 物理核数（Windows 走 <c>GetLogicalProcessorInformationEx</c>，其它平台或失败时退回逻辑核数）。
    /// </summary>
    private static int PhysicalCoreCount()
    {
        if (!OperatingSystem.IsWindows())
            return Environment.ProcessorCount;

        try
        {
            int length = 0;
            GetLogicalProcessorInformationEx(0 /* RelationProcessorCore */, IntPtr.Zero, ref length);
            if (length <= 0)
                return Environment.ProcessorCount;

            IntPtr buffer = Marshal.AllocHGlobal(length);
            try
            {
                if (!GetLogicalProcessorInformationEx(0, buffer, ref length))
                    return Environment.ProcessorCount;

                int cores = 0;
                for (int offset = 0; offset + 8 <= length; )
                {
                    // SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX { DWORD Relationship; DWORD Size; ... }
                    int size = Marshal.ReadInt32(buffer, offset + 4);
                    if (size <= 0)
                        break;

                    cores++;
                    offset += size;
                }

                return cores > 0 ? cores : Environment.ProcessorCount;
            }
            finally
            {
                Marshal.FreeHGlobal(buffer);
            }
        }
        catch
        {
            return Environment.ProcessorCount;
        }
    }

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool GetLogicalProcessorInformationEx(int relationshipType, IntPtr buffer, ref int returnedLength);

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

        /// <summary>
        /// 慢路径的阻塞等待。用它而不是 <c>Thread.Yield</c>/<c>Sleep(1)</c> 序列：
        /// 工作线程在**没有**并行区的时候必须真的睡下去，否则会一直烧 CPU。
        /// </summary>
        private readonly object _gate = new();

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

            // 先发布再唤醒。等待者会在锁内重新检查 _current，所以不存在「漏脉冲」：
            // 即便它错过了这次 PulseAll，进入锁后仍然看得到刚发布的 region。
            lock (_gate) Monitor.PulseAll(_gate);

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

                // 快路径：并行区之间的间隔只有几十微秒，短自旋比任何阻塞都快。
                if (idle < 2000)
                {
                    Thread.SpinWait(20);        // ~50 µs
                    idle++;
                    continue;
                }

                // 慢路径：真的没活了，必须睡下去。
                // 不能像以前那样接一段 Thread.Yield()：SwitchToThread 在没有其他就绪线程时
                // 立刻返回，所以那一档看起来「让出」了，实际是把 CPU 烧满。旧的 yield 预算
                // 是 38000 次（约 1 ms），刚好填满帧内的空隙 —— 结果是 GPU 帧里 15 个工作线
                // 程全程空转，实测进程 CPU 时间达墙钟的 15.3 倍，把功耗预算从 GPU 手里抢走。
                lock (_gate)
                {
                    if (ReferenceEquals(Volatile.Read(ref _current), last))
                        Monitor.Wait(_gate);
                }

                idle = 0;                       // 被唤醒后重新给一小段自旋
            }
        }

        /// <summary>
        /// 等所有工作线程报告完成。
        /// <para>
        /// 这里保留「自旋 → Yield」两档而不是直接阻塞：调用线程等的是**已经在跑**的工作线程，
        /// 正常情况几微秒就结束了；一旦睡下去反而要等调度器，把一次派发从 1 µs 拖到 1 ms。
        /// </para>
        /// </summary>
        private void WaitForWorkers(Region region)
        {
            long idle = 0;
            while (Volatile.Read(ref region.Finished) < _workers.Length)
                idle = WaitOnce(idle);
        }

        /// <summary>调用线程的等待：短自旋 → Yield。不睡，因为等待对象正在别的核上跑。</summary>
        private static long WaitOnce(long idle)
        {
            if (idle < 2000)
                Thread.SpinWait(20);            // ~50 µs 紧凑自旋
            else
                Thread.Yield();                 // 让出时间片但不睡

            return idle + 1;
        }
    }
}
