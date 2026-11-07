using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using OnnxSharp.Core;
using OnnxSharp.Onnx;
using OnnxSharp.Tensors;

// 子命令：
//   (无参数) / smoke         M0 冒烟：手工建图 Add + Relu，验证引擎链路
//   info <model.onnx>        M1：解析 ONNX 结构并打印统计（不算子图）
//
// 用法：dotnet run --project tools/OnnxSharp.Cli -- info F:\ai\models\yolo\yolo11n.onnx

if (args.Length == 0 || args[0].Equals("smoke", StringComparison.OrdinalIgnoreCase))
    return Smoke();

return args[0].ToLowerInvariant() switch
{
    "info" => Info(args),
    "dump-graph" => DumpGraph(args),
    "run" => Run(args),
    "bench" => Bench(args),
    "profile" => Profile(args),
    "detect" => Detect(args),
    "gemmbench" => GemmBench(args),
    "parbench" => ParBench(args),
    "flopsbench" => FlopsBench(args),
    "convbench" => ConvBench(args),
    "help" or "-h" or "--help" => Usage(),
    _ => Usage(),
};

// ---------------- 子命令 ----------------

static int Usage()
{
    Console.WriteLine("usage:");
    Console.WriteLine("  OnnxSharp.Cli                       # 冒烟测试");
    Console.WriteLine("  OnnxSharp.Cli info <model.onnx>     # 打印 ONNX 结构统计");
    Console.WriteLine("  OnnxSharp.Cli dump-graph <model.onnx> [--type Conv] [--limit N]");
    Console.WriteLine("                                      # 导入为图并打印每个节点的推断输出形状");
    Console.WriteLine("  OnnxSharp.Cli run <model.onnx> --input <file.bin> [--input name=file.bin]");
    Console.WriteLine("                     [--compare <golden.bin>] [--repeat N]");
    Console.WriteLine("                                      # 跑一次前向，打印输出统计/耗时");
    Console.WriteLine("  OnnxSharp.Cli bench <model.onnx> --input <file.bin> [--iters N] [--warmup N]");
    Console.WriteLine("                                      # 计时迭代");
    Console.WriteLine("  OnnxSharp.Cli profile <model.onnx> --input <file.bin> [--iters N] [--top N]");
    Console.WriteLine("                                      # 逐算子耗时排行（找热点）");
    Console.WriteLine("  OnnxSharp.Cli detect <yolo.onnx> --input <file.bin> [--conf 0.25] [--iou 0.45]");
    Console.WriteLine("                                      # 调用示例：YOLO decode + NMS（不写进产品库）");
    Console.WriteLine("  OnnxSharp.Cli gemmbench [--iters N] [--shape m,k,n]");
    Console.WriteLine("                                      # GEMM 微内核吞吐（GFLOPS），调优用");
    return 2;
}

static int Smoke()
{
    // Input[2,3] --Add(bias[3])--> Relu --> output
    var impl = new NetImpl();

    var x = impl.Declare("x", ArgKind.Input);
    var graph = impl.NewGraph("main", new[] { x }, isMain: true);

    var bias = impl.NewConstArg("bias", Tensor.FromArray(new float[] { 0.5f, 0.5f, -1f }, new TensorShape(3)));

    var add = LayerRegistry.Create("Add", new LayerParams { Type = "Add", Name = "add" });
    add.Inputs.Add(x);
    add.Inputs.Add(bias);
    var y = impl.Declare("y", ArgKind.Temp);
    add.Outputs.Add(y);
    graph.Append(add);

    var relu = LayerRegistry.Create("Relu", new LayerParams { Type = "Relu", Name = "relu" });
    relu.Inputs.Add(y);
    var z = impl.Declare("z", ArgKind.Temp);
    relu.Outputs.Add(z);
    graph.Append(relu);

    graph.SetOutputs(new[] { z });

    impl.SetInput("x", Tensor.FromArray(new float[] { 1, -2, 3, 4, -5, 6 }, new TensorShape(2, 3)));

    var result = impl.Forward();

    // 期望：add 后 [[1.5, -1.5, 2], [4.5, -4.5, 5]]，Relu 后 [[1.5, 0, 2], [4.5, 0, 5]]
    float[] expected = { 1.5f, 0f, 2f, 4.5f, 0f, 5f };
    var actual = result.AsSpan<float>().ToArray();

    Console.WriteLine($"graph      : {graph}");
    Console.WriteLine($"output     : {result}");
    Console.WriteLine($"values     : {string.Join(", ", actual)}");
    Console.WriteLine($"slots      : {impl.SlotCount}    temps: {impl.TempCount}");

    bool ok = actual.Length == expected.Length;
    for (int i = 0; ok && i < expected.Length; i++)
        ok = MathF.Abs(actual[i] - expected[i]) < 1e-6f;

    Console.WriteLine(ok ? "SMOKE TEST: PASS" : "SMOKE TEST: FAIL");
    return ok ? 0 : 1;
}

static int Info(string[] args)
{
    if (args.Length < 2)
    {
        Console.Error.WriteLine("usage: info <model.onnx>");
        return 2;
    }

    var path = args[1];
    if (!File.Exists(path))
    {
        Console.Error.WriteLine($"model not found: {path}");
        return 2;
    }

    var watch = System.Diagnostics.Stopwatch.StartNew();
    var file = OnnxFile.Load(path);
    watch.Stop();

    long initBytes = 0;
    foreach (var tensor in file.Initializers.Values)
        initBytes += tensor.ByteLength;

    Console.WriteLine($"file      : {file.Path} ({new FileInfo(file.Path).Length / 1048576.0:F1} MB)");
    Console.WriteLine($"producer  : {file.Producer}    opset: {file.Opset}");
    Console.WriteLine($"loaded in : {watch.ElapsedMilliseconds} ms");
    Console.WriteLine($"nodes     : {file.Nodes.Count}");
    Console.WriteLine($"inits     : {file.Initializers.Count} ({initBytes / 1048576.0:F1} MB)");
    Console.WriteLine($"inputs    : {string.Join(", ", file.Inputs)}");
    Console.WriteLine($"outputs   : {string.Join(", ", file.Outputs)}");

    Console.WriteLine("op types  :");
    foreach (var group in file.Nodes.GroupBy(n => n.OpType).OrderByDescending(g => g.Count()))
        Console.WriteLine($"  {group.Key,-24}{group.Count()}");

    return 0;
}

/// <summary>
/// dump-graph：导入成可执行图，跑<b>纯形状推断</b>（不执行任何 kernel），
/// 打印每个节点的输出形状。用于和 onnx.shape_inference 对照。
/// </summary>
static int DumpGraph(string[] args)
{
    if (args.Length < 2)
    {
        Console.Error.WriteLine("usage: dump-graph <model.onnx> [--type <OpType>] [--limit N]");
        return 2;
    }

    string path = args[1];
    string? filter = null;
    int limit = int.MaxValue;

    for (int i = 2; i < args.Length - 1; i++)
    {
        if (args[i] == "--type")
            filter = args[++i];
        else if (args[i] == "--limit")
            limit = int.Parse(args[++i]);
    }

    if (!File.Exists(path))
    {
        Console.Error.WriteLine($"model not found: {path}");
        return 2;
    }

    var watch = System.Diagnostics.Stopwatch.StartNew();
    var net = Net.ReadModel(path);
    var graph = net.MainGraph ?? throw new OnnxSharpException("no main graph");
    Interpreter.Run(net.Impl, graph, executeLayers: false);
    watch.Stop();

    Console.WriteLine($"model     : {path}");
    Console.WriteLine($"imported  : {watch.ElapsedMilliseconds} ms    ops: {graph.Prog.Count}    " +
                      $"slots: {net.Impl.SlotCount}");

    Console.WriteLine("inputs    :");
    foreach (var name in net.InputNames)
        Console.WriteLine($"  {name,-40}{Describe(net.Impl.GetArgData(net.Impl.Require(name)))}");

    Console.WriteLine("ops       :");
    int index = 0;
    int printed = 0;
    foreach (var op in graph.Prog)
    {
        index++;
        if (filter is not null && !op.Type.Equals(filter, StringComparison.OrdinalIgnoreCase))
            continue;
        if (printed++ >= limit)
            break;

        var outs = new List<string>(op.Outputs.Count);
        foreach (var arg in op.Outputs)
        {
            // 注意：必须读 ArgData 而不是 GetArgTensor —— 共享缓冲槽会被后续算子覆盖形状。
            outs.Add(arg.Idx == 0 ? "<none>" : Describe(net.Impl.GetArgData(arg)));
        }

        Console.WriteLine($"  [{index,4}] {op.Type,-12} {op.Name,-44} -> {string.Join(", ", outs)}");
    }

    Console.WriteLine("outputs   :");
    foreach (var arg in graph.Outputs)
        Console.WriteLine($"  {net.Impl.GetArgData(arg).Name,-40}{Describe(net.Impl.GetArgData(arg))}");

    return 0;
}

/// <summary>
/// 用 ArgData 的元信息格式化形状：共享缓冲槽会被后续算子复用，
/// 只有 ArgData 保存的是「本 Arg 生产时刻」的形状。
/// </summary>
static string Describe(ArgData data)
    => $"{data.DType} [{string.Join(",", data.Shape.ToArray())}]";

// ---------------- run / bench ----------------

static int Run(string[] args)
{
    if (args.Length < 2)
    {
        Console.Error.WriteLine("usage: run <model.onnx> --input <file.bin> [--compare <golden.bin>] [--repeat N]");
        return 2;
    }

    string path = args[1];
    var options = ParseOptions(args, 2);
    if (!File.Exists(path))
    {
        Console.Error.WriteLine($"model not found: {path}");
        return 2;
    }

    var loadWatch = System.Diagnostics.Stopwatch.StartNew();
    var net = Net.ReadModel(path);
    loadWatch.Stop();

    var feeds = BindInputs(net, options.Inputs, ParseShape(options.GetString("--shape")));
    int repeat = options.GetInt("--repeat", 1);

    Console.WriteLine($"model     : {path}");
    Console.WriteLine($"imported  : {loadWatch.ElapsedMilliseconds} ms    ops: {net.MainGraph?.Prog.Count}");

    var first = (List<Tensor>?)null;
    var watch = System.Diagnostics.Stopwatch.StartNew();
    for (int i = 0; i < repeat; i++)
        first = net.ForwardAll();
    watch.Stop();

    Console.WriteLine($"forward   : {watch.Elapsed.TotalMilliseconds / repeat:F1} ms/iter ({repeat} iter)");

    var outputs = net.OutputNames;
    for (int i = 0; i < first!.Count; i++)
    {
        var tensor = first[i];
        var span = tensor.AsSpan<float>();
        double sum = 0;
        float min = float.MaxValue, max = float.MinValue;

        for (int j = 0; j < span.Length; j++)
        {
            float value = span[j];
            sum += value;
            if (value < min) min = value;
            if (value > max) max = value;
        }

        Console.WriteLine($"output[{i}] : {outputs[i],-12}{Describe(net.Impl.GetArgData(net.Impl.Require(outputs[i])))}  " +
                          $"min={min:F4} max={max:F4} mean={sum / Math.Max(1, span.Length):F6}");

        if (options.Compare is not null && i == 0)
            CompareWithGolden(span, options.Compare);
    }

    _ = feeds;
    return 0;
}

static int Bench(string[] args)
{
    if (args.Length < 2)
    {
        Console.Error.WriteLine("usage: bench <model.onnx> --input <file.bin> [--iters N] [--warmup N]");
        return 2;
    }

    string path = args[1];
    var options = ParseOptions(args, 2);
    if (!File.Exists(path))
    {
        Console.Error.WriteLine($"model not found: {path}");
        return 2;
    }

    var net = Net.ReadModel(path);
    BindInputs(net, options.Inputs, ParseShape(options.GetString("--shape")));

    int warmup = options.GetInt("--warmup", 1);
    int iters = options.GetInt("--iters", 5);

    for (int i = 0; i < warmup; i++)
        net.ForwardAll();

    var samples = new double[iters];
    for (int i = 0; i < iters; i++)
    {
        var watch = System.Diagnostics.Stopwatch.StartNew();
        net.ForwardAll();
        watch.Stop();
        samples[i] = watch.Elapsed.TotalMilliseconds;
    }

    Array.Sort(samples);
    Console.WriteLine($"model     : {path}");
    Console.WriteLine($"iters     : {iters}  (warmup {warmup})");
    Console.WriteLine($"min       : {samples[0]:F1} ms");
    Console.WriteLine($"median    : {samples[iters / 2]:F1} ms");
    Console.WriteLine($"mean      : {samples.Average():F1} ms");
    Console.WriteLine($"max       : {samples[^1]:F1} ms");
    return 0;
}

/// <summary>
/// GEMM 微内核基准：<c>c[m,n] = a[m,k]·b[k,n]</c> 在典型 Conv 形状上的吞吐（GFLOPS）。
/// <para>
/// n 用「一次 im2col 任务内的像素数」而不是整层像素数 —— Conv 主路径是按输出行块
/// 切开后逐块调 GEMM 的，用整层像素会高估微内核的收益。
/// </para>
/// </summary>
static int GemmBench(string[] args)
{
    var options = ParseOptions(args, 1);
    int iters = options.GetInt("--iters", 50);
    int batches = options.GetInt("--batch", 1);

    // 线程数必须在这里设：GemmKernel 用 ParallelRange.Threads 决定分块和并行度，
    // 而默认并行度是 ProcessorCount —— 测扩展性曲线时必须能压下来。
    int threads = options.GetInt("--threads", 0);
    if (threads > 0)
        OnnxSharp.Threading.ParallelRange.MaxDegreeOfParallelism = threads;

    var shapes = new List<(int M, int K, int N)>();
    string? single = options.GetString("--shape");

    if (!string.IsNullOrEmpty(single))
    {
        var parts = single.Split(',', StringSplitOptions.RemoveEmptyEntries);
        shapes.Add((int.Parse(parts[0]), int.Parse(parts[1]), int.Parse(parts[2])));
    }
    else
    {
        // yolo11n/640 上几类典型 Conv 的行块形状
        shapes.Add((16, 27, 2240));     // 3x3  c3->16   80x80
        shapes.Add((32, 144, 1600));    // 3x3  16->32   40x40
        shapes.Add((64, 144, 800));     // 3x3  32->64   40x40(down)
        shapes.Add((64, 576, 400));     // 3x3  64->64   20x20
        shapes.Add((64, 1152, 1600));   // 3x3 128->64   40x40
        shapes.Add((128, 576, 200));    // 3x3  64->128  20x20
        shapes.Add((128, 1152, 200));   // 3x3 128->128  10x10
        shapes.Add((256, 1152, 100));   // 3x3 128->256  10x10
        shapes.Add((64, 64, 6400));     // 1x1  64->64   80x80
        shapes.Add((256, 256, 6400));   // 1x1 256->256  80x80
    }

    batches = Math.Max(1, batches);

    Console.WriteLine($"threads : {OnnxSharp.Threading.ParallelRange.Threads}");
    Console.WriteLine($"iters   : {iters}");
    Console.WriteLine($"batch   : {batches}");
    Console.WriteLine();
    Console.WriteLine("     m       k       n    batch    per-batch ms    GFLOPS");

    foreach (var (m, k, n) in shapes)
    {
        var a = new float[batches * m * k];
        var b = new float[batches * k * n];
        var c = new float[batches * m * n];
        var rng = new Random(12345);
        for (int i = 0; i < a.Length; i++) a[i] = (float)rng.NextDouble();
        for (int i = 0; i < b.Length; i++) b[i] = (float)rng.NextDouble();

        void Run() => RunBatched(a, b, c, batches, m, k, n);

        // 按时间预热：分层编译下「调用次数」比「迭代次数」更重要，固定 3 次预热会把
        // tier-0 的慢样本算进来（实测同形状能差 4 倍以上）。
        var warm = System.Diagnostics.Stopwatch.StartNew();
        do
        {
            Run();
        }
        while (warm.Elapsed.TotalMilliseconds < 400);

        double best = double.MaxValue;
        for (int i = 0; i < iters; i++)
        {
            var watch = System.Diagnostics.Stopwatch.StartNew();
            Run();
            watch.Stop();
            if (watch.Elapsed.TotalMilliseconds < best)
                best = watch.Elapsed.TotalMilliseconds;
        }

        double perBatch = best / batches;
        double gflops = 2.0 * m * k * n / (perBatch * 1e6);
        Console.WriteLine($"{m,6}  {k,6}  {n,6}  {batches,6}   {perBatch,12:F3}  {gflops,8:F1}");
    }

    return 0;

    static void RunBatched(float[] a, float[] b, float[] c, int batches, int m, int k, int n)
    {
        var aSpan = a.AsSpan();
        var bSpan = b.AsSpan();
        var cSpan = c.AsSpan();
        for (int bi = 0; bi < batches; bi++)
            GemmKernel.Multiply(aSpan.Slice(bi * m * k, m * k), bSpan.Slice(bi * k * n, k * n),
                                cSpan.Slice(bi * m * n, m * n), m, k, n);
    }
}

/// <summary>
/// 并行调度开销基准：测「一次并行调用」本身的固定成本（不含实际工作）。
/// 推理一帧有几百次并行调用，这个数字直接决定小算子的下限。
/// </summary>
static int ParBench(string[] args)
{
    var options = ParseOptions(args, 1);
    int iters = options.GetInt("--iters", 2000);
    int threads = OnnxSharp.Threading.ParallelRange.Threads;

    Console.WriteLine($"threads : {threads}");
    Console.WriteLine($"iters   : {iters}");

    var acc = new long[1];

    void Body(int begin, int end)
    {
        for (int t = begin; t < end; t++)
            Interlocked.Add(ref acc[0], t);
    }

    for (int i = 0; i < 200; i++)
        OnnxSharp.Threading.ParallelRange.For(threads * 4, Body);

    var watch = System.Diagnostics.Stopwatch.StartNew();
    for (int i = 0; i < iters; i++)
        OnnxSharp.Threading.ParallelRange.For(threads * 4, Body);
    watch.Stop();

    double us = watch.Elapsed.TotalMilliseconds * 1000 / iters;
    Console.WriteLine($"ParallelRange.For : {watch.Elapsed.TotalMilliseconds,8:F1} ms / {iters} 次 = {us,7:F1} µs/次");

    for (int i = 0; i < 200; i++)
        System.Threading.Tasks.Parallel.For(0, threads * 4, t => Interlocked.Add(ref acc[0], t));

    watch.Restart();
    for (int i = 0; i < iters; i++)
        System.Threading.Tasks.Parallel.For(0, threads * 4, t => Interlocked.Add(ref acc[0], t));
    watch.Stop();

    double us2 = watch.Elapsed.TotalMilliseconds * 1000 / iters;
    Console.WriteLine($"Parallel.For      : {watch.Elapsed.TotalMilliseconds,8:F1} ms / {iters} 次 = {us2,7:F1} µs/次");

    Console.WriteLine($"acc={acc[0]}");
    return 0;
}

/// <summary>
/// 纯 FMA 峰值基准：只有寄存器里的独立累加链，零访存。
/// <para>
/// 它的意义是给 GEMM 一个**上限参照**：GEMM 跑到这个数的百分之多少，
/// 决定了该继续调内核还是该承认是硬件（全核降频 / 后台负载）限制。
/// 没有这个数就没法判断「45% 扩展性」是代码问题还是机器问题。
/// </para>
/// </summary>
static int FlopsBench(string[] args)
{
    var options = ParseOptions(args, 1);
    int iters = options.GetInt("--iters", 10);
    int inner = options.GetInt("--inner", 500_000);
    int threads = options.GetInt("--threads", 0);

    if (threads > 0)
        OnnxSharp.Threading.ParallelRange.MaxDegreeOfParallelism = threads;

    int degree = OnnxSharp.Threading.ParallelRange.Threads;

    Console.WriteLine($"threads : {degree}");
    Console.WriteLine($"inner   : {inner}  x {FmaBench.Accumulators} FMA/轮");
    Console.WriteLine();

    var sink = new float[degree];

    void Run(int t) => sink[t] = FmaBench.Run(inner);

    // 预热走时间：分层编译要几十次调用才升到 tier-1。
    var warm = System.Diagnostics.Stopwatch.StartNew();
    do
    {
        System.Threading.Tasks.Parallel.For(0, degree,
            new System.Threading.Tasks.ParallelOptions { MaxDegreeOfParallelism = degree }, Run);
    }
    while (warm.Elapsed.TotalMilliseconds < 500);

    double best = double.MaxValue;
    for (int i = 0; i < iters; i++)
    {
        var watch = System.Diagnostics.Stopwatch.StartNew();
        System.Threading.Tasks.Parallel.For(0, degree,
            new System.Threading.Tasks.ParallelOptions { MaxDegreeOfParallelism = degree }, Run);
        watch.Stop();
        if (watch.Elapsed.TotalMilliseconds < best)
            best = watch.Elapsed.TotalMilliseconds;
    }

    double flops = (double)degree * inner * FmaBench.Accumulators * 16;   // 每 FMA = 8 lane × 2
    Console.WriteLine($"per-thread : {inner * FmaBench.Accumulators * 16.0 / (best * 1e6),8:F1} GFLOPS");
    Console.WriteLine($"total      : {flops / (best * 1e6),8:F1} GFLOPS   ({best:F1} ms)");
    Console.WriteLine($"sink       : {sink.Sum():F3}");
    return 0;
}

/// <summary>
/// convbench：单层卷积内核对比（im2col / 面板打包 / Winograd）。
/// 形状描述为 <c>ic,oc,h,w,k,s,p</c>，不给就按 yolo11n 640 的层跑一遍。
/// </summary>
static int ConvBench(string[] args)
{
    var options = ParseOptions(args, 0);
    int iters = options.GetInt("--iters", 20);

    var configs = new List<(int Ic, int Oc, int H, int W, int K, int S, int P)>();
    string? single = options.GetString("--shape");

    if (!string.IsNullOrEmpty(single))
    {
        var p = single.Split(',', StringSplitOptions.RemoveEmptyEntries);
        configs.Add((int.Parse(p[0]), int.Parse(p[1]), int.Parse(p[2]), int.Parse(p[3]),
            int.Parse(p[4]), int.Parse(p[5]), int.Parse(p[6])));
    }
    else
    {
        configs.Add((3, 16, 640, 640, 3, 2, 1));
        configs.Add((16, 32, 320, 320, 3, 2, 1));
        configs.Add((32, 64, 160, 160, 3, 2, 1));
        configs.Add((64, 64, 160, 160, 3, 1, 1));
        configs.Add((64, 128, 80, 80, 3, 2, 1));
        configs.Add((128, 128, 80, 80, 3, 1, 1));
        configs.Add((128, 256, 40, 40, 3, 2, 1));
        configs.Add((256, 256, 40, 40, 3, 1, 1));
        configs.Add((256, 256, 20, 20, 3, 1, 1));
        configs.Add((64, 64, 160, 160, 1, 1, 0));
        configs.Add((128, 128, 80, 80, 1, 1, 0));
        configs.Add((256, 256, 40, 40, 1, 1, 0));
        configs.Add((256, 256, 20, 20, 1, 1, 0));
    }

    Console.WriteLine($"threads : {OnnxSharp.Threading.ParallelRange.Threads}");
    Console.WriteLine($"iters   : {iters}");
    Console.WriteLine();

    double Best(Action body)
    {
        for (int i = 0; i < 3; i++) body();

        double best = double.MaxValue;
        var watch = System.Diagnostics.Stopwatch.StartNew();

        for (int i = 0; i < iters; i++)
        {
            watch.Restart();
            body();
            watch.Stop();
            best = Math.Min(best, watch.Elapsed.TotalMilliseconds);
        }

        return best;
    }

    foreach (var (ic, oc, h, w, k, s, p) in configs)
    {
        int oh = (h + 2 * p - k) / s + 1;
        int ow = (w + 2 * p - k) / s + 1;

        var input = Tensor.Allocate(new TensorShape(1, ic, h, w), DType.F32);
        var weight = Tensor.Allocate(new TensorShape(oc, ic, k, k), DType.F32);
        var bias = Tensor.Allocate(new TensorShape(oc), DType.F32);
        var output = Tensor.Allocate(new TensorShape(1, oc, oh, ow), DType.F32);

        var rng = new Random(1234);
        var xs = input.AsSpan<float>();
        var ws = weight.AsSpan<float>();
        var bs = bias.AsSpan<float>();

        for (int i = 0; i < xs.Length; i++) xs[i] = (float)rng.NextDouble();
        for (int i = 0; i < ws.Length; i++) ws[i] = (float)(rng.NextDouble() - 0.5);
        for (int i = 0; i < bs.Length; i++) bs[i] = (float)(rng.NextDouble() - 0.5);

        double macs = (double)oc * ic * k * k * oh * ow;

        double im2col = Best(() => OnnxSharp.Kernels.ConvKernels.Im2Col(
            input, weight, bias, output, 1, s, s, 1, 1, p, p));

        bool packedUsed = false;
        double packed = double.NaN;

        if (OnnxSharp.Kernels.PackedConv.TryConv(input, weight, bias, output, 1, s, s, 1, 1, p, p))
        {
            packedUsed = true;
            packed = Best(() => { OnnxSharp.Kernels.PackedConv.TryConv(input, weight, bias, output, 1, s, s, 1, 1, p, p); });
        }

        string line = $"ic={ic,-4} oc={oc,-4} {h}x{w} k{k}s{s}p{p} -> {oh}x{ow}  "
            + $"im2col {im2col,7:F2} ms ({macs / im2col / 1e6,6:F1} GMAC/s)";

        if (packedUsed)
            line += $"  packed {packed,7:F2} ms ({macs / packed / 1e6,6:F1})";
        else
            line += "  packed     --";

        // Winograd 默认只覆盖小特征图，这里放开上限量一下它在各尺寸上的真实水平
        long saved = OnnxSharp.Kernels.Winograd.MaxSpatialPixels;
        OnnxSharp.Kernels.Winograd.MaxSpatialPixels = long.MaxValue;

        bool winogradAll = OnnxSharp.Kernels.Winograd.TryConv3x3(
            input, weight, bias, output, 1, s, s, 1, 1, p, p);

        if (winogradAll)
        {
            double winograd = Best(() => { OnnxSharp.Kernels.Winograd.TryConv3x3(input, weight, bias, output, 1, s, s, 1, 1, p, p); });
            line += $"  winograd {winograd,7:F2} ms ({macs / winograd / 1e6,6:F1})";
        }
        else
        {
            line += "  winograd     --";
        }

        OnnxSharp.Kernels.Winograd.MaxSpatialPixels = saved;
        Console.WriteLine(line);
    }

    return 0;
}

/// <summary>
/// profile：逐算子耗时排行。第 1 次迭代包含 Finalize/权重打包，所以默认跑 2 次取平均。
/// </summary>
static int Profile(string[] args)
{
    if (args.Length < 2)
    {
        Console.Error.WriteLine("usage: profile <model.onnx> --input <file.bin> [--iters N] [--top N]");
        return 2;
    }

    string path = args[1];
    var options = ParseOptions(args, 2);
    if (!File.Exists(path))
    {
        Console.Error.WriteLine($"model not found: {path}");
        return 2;
    }

    var net = Net.ReadModel(path);
    BindInputs(net, options.Inputs, ParseShape(options.GetString("--shape")));

    int iters = Math.Max(1, options.GetInt("--iters", 2));
    int top = options.GetInt("--top", 20);

    // 预热不计入统计：前几次迭代是 tier-0 代码，会把计算密集算子的份额抬高一大截。
    int warmup = Math.Max(0, options.GetInt("--warmup", 5));

    var watch = System.Diagnostics.Stopwatch.StartNew();
    var timings = net.ProfileAll(iters, warmup);
    watch.Stop();

    double total = timings.Sum(t => t.Milliseconds);
    double totalMin = timings.Sum(t => t.MinMilliseconds);

    Console.WriteLine($"model     : {path}");
    Console.WriteLine($"ops       : {net.MainGraph?.Prog.Count}    iters: {iters}    warmup: {warmup}    {watch.Elapsed.TotalMilliseconds:F0} ms");
    Console.WriteLine($"total     : {total / iters:F1} ms/iter (mean)   {totalMin:F1} ms/iter (sum of min)");
    Console.WriteLine($"note      : share 按 sum-of-min 算；mean 会被后台负载的重尾放大，和 bench 的 median 对不上");
    Console.WriteLine();
    Console.WriteLine($"{"#",4}  {"type",-18}{"min ms",10}{"avg ms",10}{"share",8}  name");

    int index = 0;
    foreach (var timing in timings.Take(top < 1 ? timings.Count : top))
    {
        Console.WriteLine($"{++index,4}  {timing.Type,-18}{timing.MinMilliseconds,10:F2}{timing.Average,10:F2}{timing.MinMilliseconds / totalMin * 100,7:F1}%  {timing.Name}");
    }

    // 按算子类型汇总：热点分散时（几百个 op 各占 1%）看这个更有用
    Console.WriteLine();
    Console.WriteLine($"{"type",-20}{"ops",6}{"min ms",10}{"avg ms",10}{"share",8}");

    var byType = timings
        .GroupBy(t => t.Type)
        .Select(g => new
        {
            Type = g.Key,
            Ops = g.Count(),
            Min = g.Sum(t => t.MinMilliseconds),
            Milliseconds = g.Sum(t => t.Milliseconds) / iters,
        })
        .OrderByDescending(g => g.Min);

    foreach (var group in byType)
        Console.WriteLine($"{group.Type,-20}{group.Ops,6}{group.Min,10:F1}{group.Milliseconds,10:F1}{group.Min / totalMin * 100,7:F1}%");

    return 0;
}

// ---------------- YOLO decode + NMS（调用示例，不属于引擎） ----------------

static int Detect(string[] args)
{
    if (args.Length < 2)
    {
        Console.Error.WriteLine("usage: detect <yolo.onnx> --input <file.bin> [--conf 0.25] [--iou 0.45]");
        return 2;
    }

    string path = args[1];
    var options = ParseOptions(args, 2);
    float conf = options.GetFloat("--conf", 0.25f);
    float iou = options.GetFloat("--iou", 0.45f);

    var net = Net.ReadModel(path);
    BindInputs(net, options.Inputs, ParseShape(options.GetString("--shape")));

    var output = net.Forward();
    var span = output.AsSpan<float>();

    // YOLOv8/v11: output0 = [1, 4 + numClasses, numAnchors]（无 objectness，类别分数已 sigmoid）
    if (output.Shape.Rank != 3 || output.Shape[1] < 5)
        throw new OnnxSharpException($"detect: unexpected output shape {output.Shape}");

    int attributes = output.Shape[1];
    int anchors = output.Shape[2];
    int classes = attributes - 4;

    var candidates = new List<Detection>();

    for (int a = 0; a < anchors; a++)
    {
        int best = -1;
        float bestScore = conf;

        for (int c = 0; c < classes; c++)
        {
            float score = span[(c + 4) * anchors + a];
            if (score > bestScore)
            {
                bestScore = score;
                best = c;
            }
        }

        if (best < 0)
            continue;

        float cx = span[a];
        float cy = span[anchors + a];
        float w = span[2 * anchors + a];
        float h = span[3 * anchors + a];

        candidates.Add(new Detection
        {
            ClassId = best,
            Score = bestScore,
            X = cx - w * 0.5f,
            Y = cy - h * 0.5f,
            W = w,
            H = h,
        });
    }

    var kept = Nms(candidates, iou);

    Console.WriteLine($"model     : {path}");
    Console.WriteLine($"anchors   : {anchors}    classes: {classes}    candidates: {candidates.Count}    kept: {kept.Count}");

    foreach (var detection in kept.Take(20))
    {
        Console.WriteLine($"  class {detection.ClassId,3}  score {detection.Score:F4}  " +
                          $"box [{detection.X:F1}, {detection.Y:F1}, {detection.W:F1}, {detection.H:F1}]");
    }

    return 0;
}

static List<Detection> Nms(List<Detection> candidates, float iouThreshold)
{
    var sorted = new List<Detection>(candidates);
    sorted.Sort((x, y) => y.Score.CompareTo(x.Score));

    var kept = new List<Detection>();
    var suppressed = new bool[sorted.Count];

    for (int i = 0; i < sorted.Count; i++)
    {
        if (suppressed[i])
            continue;

        kept.Add(sorted[i]);

        for (int j = i + 1; j < sorted.Count; j++)
        {
            if (suppressed[j] || sorted[j].ClassId != sorted[i].ClassId)
                continue;
            if (Detection.IoU(sorted[i], sorted[j]) > iouThreshold)
                suppressed[j] = true;
        }
    }

    return kept;
}

// ---------------- 参数与输入绑定 ----------------

static CliOptions ParseOptions(string[] args, int start)
{
    var options = new CliOptions();

    for (int i = start; i < args.Length; i++)
    {
        string arg = args[i];
        if (arg == "--input" && i + 1 < args.Length)
            options.Inputs.Add(args[++i]);
        else if (arg == "--compare" && i + 1 < args.Length)
            options.Compare = args[++i];
        else if (arg.StartsWith("--", StringComparison.Ordinal) && i + 1 < args.Length)
            options.Add(arg, args[++i]);
    }

    return options;
}

/// <summary>
/// 把 <c>--input file.bin</c>（按模型输入顺序）或 <c>--input name=file.bin</c> 绑定到图上。
/// 只支持静态形状的 F32 输入 —— 这是「引擎外的预处理」边界，letterbox 由调用方完成。
/// </summary>
static int[]? ParseShape(string? text)
{
    if (string.IsNullOrWhiteSpace(text))
        return null;

    var parts = text.Split(new[] { 'x', 'X', ',' }, StringSplitOptions.RemoveEmptyEntries);
    var dims = new int[parts.Length];
    for (int i = 0; i < parts.Length; i++)
        dims[i] = int.Parse(parts[i]);

    return dims;
}

static List<Tensor> BindInputs(Net net, List<string> inputs, int[]? shapeOverride = null)
{
    if (inputs.Count == 0)
        throw new OnnxSharpException("no --input given");

    var names = net.InputNames;
    var tensors = new List<Tensor>(inputs.Count);

    for (int i = 0; i < inputs.Count; i++)
    {
        string name;
        string file;

        int separator = inputs[i].IndexOf('=');
        if (separator > 0)
        {
            name = inputs[i][..separator];
            file = inputs[i][(separator + 1)..];
        }
        else
        {
            if (i >= names.Count)
                throw new OnnxSharpException($"model has {names.Count} input(s), got {inputs.Count}");
            name = names[i];
            file = inputs[i];
        }

        var tensor = LoadTensor(net, name, file, shapeOverride);
        net.SetInput(name, tensor);
        tensors.Add(tensor);
    }

    return tensors;
}

static Tensor LoadTensor(Net net, string name, string file, int[]? shapeOverride = null)
{
    var data = net.Impl.GetArgData(net.Impl.Require(name));

    if (data.DType != DType.F32)
        throw new OnnxSharpException($"input '{name}': only F32 is supported for --input (got {data.DType})");

    TensorShape shape;

    if (shapeOverride is { Length: > 0 })
    {
        shape = new TensorShape(shapeOverride);
    }
    else
    {
        for (int i = 0; i < data.Shape.Rank; i++)
        {
            if (data.Shape.Dims[i] > 0)
                continue;

            throw new OnnxSharpException(
                $"input '{name}': shape {data.Shape} has dynamic dims; pass --shape 1,3,H,W");
        }

        shape = data.Shape;
    }

    long count = shape.Total();
    var bytes = File.ReadAllBytes(file);
    long expected = count * sizeof(float);

    if (bytes.Length != expected)
        throw new OnnxSharpException(
            $"input '{name}': {file} is {bytes.Length} B, expected {expected} B for {shape}");

    var tensor = Tensor.Allocate(shape, DType.F32);
    bytes.CopyTo(System.Runtime.InteropServices.MemoryMarshal.AsBytes(tensor.AsSpan<float>()));
    return tensor;
}

static void CompareWithGolden(ReadOnlySpan<float> actual, string goldenPath)
{
    var expected = new float[actual.Length];
    var bytes = File.ReadAllBytes(goldenPath);

    if (bytes.Length != expected.Length * sizeof(float))
        throw new OnnxSharpException(
            $"golden {goldenPath} is {bytes.Length} B, expected {expected.Length * sizeof(float)} B");

    Buffer.BlockCopy(bytes, 0, expected, 0, bytes.Length);

    double maxAbs = 0;
    double sumAbs = 0;

    for (int i = 0; i < expected.Length; i++)
    {
        double diff = Math.Abs((double)expected[i] - actual[i]);
        sumAbs += diff;
        if (diff > maxAbs) maxAbs = diff;
    }

    Console.WriteLine($"compare   : max|diff|={maxAbs:E3}  mean|diff|={sumAbs / expected.Length:E3}  " +
                      $"({Path.GetFileName(goldenPath)})");
}
