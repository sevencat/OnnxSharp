using System.Diagnostics;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Text;
using System.Text.Json;
using OnnxSharp.Core;
using OnnxSharp.Ocr.PaddleOCR;
using OnnxSharp.Onnx;
using OnnxSharp.Sam3;
using OnnxSharp.Tensors;
using OnnxSharp.Vision;
using ImageFacade;
using FacadeDetection = ImageFacade.Pipeline.Detection;
using FacadeDetectionStyle = ImageFacade.Pipeline.DetectionStyle;

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
    "ocr" => Ocr(args),
    "sam3" => Sam3(args),
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
    Console.WriteLine("  OnnxSharp.Cli detect <yolo.onnx> <image> [--conf 0.25] [--iou 0.45] [--out marked.jpg]");
    Console.WriteLine("                     [--labels names.txt] [--json out.json] [--max-det 300]");
    Console.WriteLine("                     [--classes N] [--layout auto|cf|cl|e2e] [--iters N]");
    Console.WriteLine("                                      # 内建 letterbox + decode + NMS，直接返回原图坐标的框");
    Console.WriteLine("  OnnxSharp.Cli detect <yolo.onnx> --input <file.bin> [--shape 1,3,640,640] [...同上]");
    Console.WriteLine("                                      # 输入已预处理好的 F32，跳过 letterbox");
    Console.WriteLine("  OnnxSharp.Cli ocr <image> --det det.onnx --rec rec.onnx --dict keys.txt [--out marked.jpg]");
    Console.WriteLine("                     [--cls cls.onnx] [--limit 960] [--box 0.6] [--unclip 1.6] [--json]");
    Console.WriteLine("                                      # PaddleOCR：DB 检测 → 方向分类 → CTC 识别");
    Console.WriteLine("  （<image> 由 ImageFacade 解码：JPEG / PNG / BMP / PPM 都可以）");
    Console.WriteLine("  OnnxSharp.Cli sam3 <image> --models <fp32 dir> --box x1,y1,x2,y2 [--box ...]");
    Console.WriteLine("                     [--out marked.jpg] [--conf 0.3] [--format cxcywh|xyxy|px] [--json]");
    Console.WriteLine("                                      # SAM3：图像 + 框提示 → 掩码（模型需先 FP16→FP32）");
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
        Console.Error.WriteLine("usage: dump-graph <model.onnx> [--shape 1,3,960,960] [--type <OpType>] [--limit N]");
        Console.Error.WriteLine("       --shape 用于全动态模型（如 PaddleOCR）：先给输入定具体尺寸，导入期才能折叠形状链");
        return 2;
    }

    string path = args[1];
    CliOptions options = ParseOptions(args, 2);
    string? filter = options.GetString("--type");
    int limit = options.GetInt("--limit", int.MaxValue);

    if (!File.Exists(path))
    {
        Console.Error.WriteLine($"model not found: {path}");
        return 2;
    }

    var watch = System.Diagnostics.Stopwatch.StartNew();
    var net = LoadModel(path, options);
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
    var net = LoadModel(path, options);
    loadWatch.Stop();

    var feeds = BindInputs(net, options.Inputs, SingleShape(options));
    int repeat = options.GetInt("--repeat", 1);

    Console.WriteLine($"model     : {path}");
    Console.WriteLine($"imported  : {loadWatch.ElapsedMilliseconds} ms    ops: {net.MainGraph?.Prog.Count}");

    var first = (List<Tensor>?)null;
    bool detectNaN = options.Has("--debug");
    var watch = System.Diagnostics.Stopwatch.StartNew();
    try
    {
        for (int i = 0; i < repeat; i++)
            first = net.ForwardAll(detectNaN);
    }
    catch (OnnxSharpException ex)
    {
        Console.Error.WriteLine($"debug     : {ex.Message}");
        return 1;
    }
    watch.Stop();

    if (options.Has("--debug"))
        ScanForNaN(net);

    Console.WriteLine($"forward   : {watch.Elapsed.TotalMilliseconds / repeat:F1} ms/iter ({repeat} iter)");

    var outputs = net.OutputNames;
    string? dumpDir = options.GetString("--dump");
    if (dumpDir is not null)
        Directory.CreateDirectory(dumpDir);

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

        if (dumpDir is not null)
        {
            var raw = new byte[span.Length * sizeof(float)];
            System.Runtime.InteropServices.MemoryMarshal.AsBytes(span).CopyTo(raw);
            File.WriteAllBytes(Path.Combine(dumpDir, $"out{i}.f32"), raw);
        }

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

    var net = LoadModel(path, options);
    BindInputs(net, options.Inputs, SingleShape(options));

    int warmup = options.GetInt("--warmup", 1);
    int iters = options.GetInt("--iters", 5);

    for (int i = 0; i < warmup; i++)
        net.ForwardAll();

    // 分配量与 GC 次数是解释「单算子计时之和 ≠ 端到端」的关键：
    // 分配多就会频繁触发 gen0/gen1 GC，而 GC 暂停会被算进当时正在跑的那个算子头上。
    long allocBefore = GC.GetTotalAllocatedBytes(precise: true);
    int g0 = GC.CollectionCount(0), g1 = GC.CollectionCount(1), g2 = GC.CollectionCount(2);
    TimeSpan pauseBefore = GC.GetTotalPauseDuration();

    var samples = new double[iters];
    for (int i = 0; i < iters; i++)
    {
        var watch = System.Diagnostics.Stopwatch.StartNew();
        net.ForwardAll();
        watch.Stop();
        samples[i] = watch.Elapsed.TotalMilliseconds;
    }

    long allocPerIter = (GC.GetTotalAllocatedBytes(precise: true) - allocBefore) / Math.Max(1, iters);
    double gcMsPerIter = (GC.GetTotalPauseDuration() - pauseBefore).TotalMilliseconds / Math.Max(1, iters);

    Array.Sort(samples);
    Console.WriteLine($"model     : {path}");
    Console.WriteLine($"iters     : {iters}  (warmup {warmup})");
    Console.WriteLine($"min       : {samples[0]:F1} ms");
    Console.WriteLine($"median    : {samples[iters / 2]:F1} ms");
    Console.WriteLine($"mean      : {samples.Average():F1} ms");
    Console.WriteLine($"max       : {samples[^1]:F1} ms");
    Console.WriteLine($"alloc     : {allocPerIter / 1024.0:F0} KB/iter");
    Console.WriteLine($"gc        : gen0 {GC.CollectionCount(0) - g0}  gen1 {GC.CollectionCount(1) - g1}  gen2 {GC.CollectionCount(2) - g2}  (共 {iters} 次推理)");
    Console.WriteLine($"gc pause  : {gcMsPerIter:F1} ms/iter ({gcMsPerIter / Math.Max(1e-9, samples[iters / 2]) * 100:F0}% of median)");
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

    var net = LoadModel(path, options);
    BindInputs(net, options.Inputs, SingleShape(options));

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

// ---------------- 检测：图进、框出（预处理 / 解码 / NMS 都在 OnnxSharp.Vision） ----------------
// 两种检测头：YOLO（letterbox + 逐锚点 argmax + NMS）与 RF-DETR（拉伸 + 逐类 sigmoid + top-k）。
// 默认按输出个数判族（RF-DETR 是 dets + labels 双输出），也可以用 --family 强制指定。

static int Detect(string[] args)
{
    if (args.Length < 2)
    {
        Console.Error.WriteLine("usage: detect <model.onnx> <image.ppm|bmp> [--family auto|yolo|rfdetr] [--conf X]");
        Console.Error.WriteLine("                     [--labels names.txt] [--json out.json] [--iters N] [--size N]");
        Console.Error.WriteLine("       yolo  : [--iou 0.45] [--max-det 300] [--classes N] [--layout auto|cf|cl|e2e] [--class-agnostic]");
        Console.Error.WriteLine("       rfdetr: [--max-det 300] [--background -1|none] [--norm imagenet|255]");
        Console.Error.WriteLine("       detect <model.onnx> --input <file.bin> [--shape 1,3,640,640]");
        return 2;
    }

    string modelPath = args[1];
    var options = ParseOptions(args, 2);

    string? family = options.GetString("--family")?.ToLowerInvariant();
    if (family is not (null or "auto" or "yolo" or "rfdetr"))
    {
        Console.Error.WriteLine($"detect: unknown --family '{family}' (auto|yolo|rfdetr)");
        return 2;
    }

    // 默认只跑一次：这是「第一帧」的真实延迟（含 JIT 与首次分配）。
    // 要稳态吞吐就加 --iters 40 —— 分层编译约 30 次调用后才升到 tier-1，少于此数会偏高 1.5~2 倍。
    int iterations = Math.Max(1, options.GetInt("--iters", 1));
    int squareSize = options.GetInt("--size", 0);

    string? imagePath = args.Length > 2 && !args[2].StartsWith("--", StringComparison.Ordinal) ? args[2] : null;
    if (imagePath is null && options.Inputs.Count == 0)
    {
        Console.Error.WriteLine("detect: need either an image path or --input <file.bin>");
        return 2;
    }

    // 图像解码交给 ImageFacade（纯托管软后端，JPEG / PNG / BMP 都能读）
    using CliImage? image = imagePath is null ? null : CliImage.Load(imagePath);

    // 模型只读一次：两个检测器都有接收 Net 的构造函数，避免重复解析 100+ MB 的权重。
    var net = Net.ReadModel(modelPath);
    bool rfdetr = family switch
    {
        "yolo" => false,
        "rfdetr" => true,
        _ => net.OutputNames.Count >= 2,
    };

    // 多次迭代只用于取更稳的耗时（第一次含 JIT 与首次分配）。
    // 前向与后处理分开计时：两者差一个数量级时，能立刻看出是模型慢还是后处理慢。
    IReadOnlyList<Detection> detections = Array.Empty<Detection>();
    double preprocessMilliseconds = double.MaxValue;
    double forwardMilliseconds = double.MaxValue;
    double decodeMilliseconds = double.MaxValue;
    float confidence;
    float? iou;
    string inputDescription;

    if (!rfdetr)
    {
        var detectorOptions = new DetectorOptions
        {
            Confidence = options.GetFloat("--conf", 0.25f),
            IouThreshold = options.GetFloat("--iou", 0.45f),
            MaxDetections = options.GetInt("--max-det", 300),
            ClassCount = options.GetInt("--classes", 0),
            ClassAgnosticNms = options.Has("--class-agnostic"),
            Layout = ParseOutputLayout(options.GetString("--layout")),
            // 只对动态形状的模型生效；静态形状由模型自己决定。
            TargetWidth = squareSize,
            TargetHeight = squareSize,
        };

        var detector = new YoloDetector(net, detectorOptions);
        Letterbox transform = default;
        Tensor input = null!;

        if (image is { } source)
        {
            // 预处理同样取多次的最小值：第一次含 JIT 与数组首次分配，能差 3~5 倍。
            for (int iteration = 0; iteration < iterations; iteration++)
            {
                var preprocessing = Stopwatch.StartNew();
                (input, transform) = detector.Preprocess(source.Pixels, source.Width, source.Height);
                preprocessMilliseconds = Math.Min(preprocessMilliseconds, preprocessing.Elapsed.TotalMilliseconds);
            }
        }
        else
        {
            input = BindInputs(net, options.Inputs, ParseShape(options.GetString("--shape")))[0];

            // 已经是 letterbox 之后的张量，框就落在模型坐标系里 —— 恒等变换即可。
            var inputShape = input.Shape;
            transform = new Letterbox(inputShape[3], inputShape[2], inputShape[3], inputShape[2], 1, 1, 0, 0);
        }

        net.SetInput(detector.InputName, input);

        for (int iteration = 0; iteration < iterations; iteration++)
        {
            var forward = Stopwatch.StartNew();
            var output = net.Forward();
            forwardMilliseconds = Math.Min(forwardMilliseconds, forward.Elapsed.TotalMilliseconds);

            var decode = Stopwatch.StartNew();
            detections = detector.Decode(output, transform);
            decodeMilliseconds = Math.Min(decodeMilliseconds, decode.Elapsed.TotalMilliseconds);
        }

        confidence = detectorOptions.Confidence;
        iou = detectorOptions.IouThreshold;
        inputDescription = $"[1, 3, {detector.TargetHeight}, {detector.TargetWidth}]  ({transform})";
    }
    else
    {
        int? background = -1;
        if (options.GetString("--background") is { } backgroundText)
            background = backgroundText.Equals("none", StringComparison.OrdinalIgnoreCase)
                ? null
                : int.Parse(backgroundText);

        var detectorOptions = new RfDetrOptions
        {
            Confidence = options.GetFloat("--conf", 0.3f),
            MaxDetections = options.GetInt("--max-det", 300),
            BackgroundClassId = background,
            Normalization = ParseNormalization(options.GetString("--norm")),
            TargetWidth = squareSize,
            TargetHeight = squareSize,
        };

        var detector = new RfDetrDetector(net, detectorOptions);
        Tensor input = null!;
        int sourceWidth;
        int sourceHeight;

        if (image is { } source)
        {
            sourceWidth = source.Width;
            sourceHeight = source.Height;

            for (int iteration = 0; iteration < iterations; iteration++)
            {
                var preprocessing = Stopwatch.StartNew();
                input = detector.Preprocess(source.Pixels, source.Width, source.Height);
                preprocessMilliseconds = Math.Min(preprocessMilliseconds, preprocessing.Elapsed.TotalMilliseconds);
            }
        }
        else
        {
            input = BindInputs(net, options.Inputs, ParseShape(options.GetString("--shape")))[0];
            sourceWidth = input.Shape[3];
            sourceHeight = input.Shape[2];
        }

        net.SetInput(detector.InputName, input);

        for (int iteration = 0; iteration < iterations; iteration++)
        {
            var forward = Stopwatch.StartNew();
            var outputs = net.ForwardAll();
            forwardMilliseconds = Math.Min(forwardMilliseconds, forward.Elapsed.TotalMilliseconds);

            var decode = Stopwatch.StartNew();
            detections = detector.Decode(outputs, sourceWidth, sourceHeight);
            decodeMilliseconds = Math.Min(decodeMilliseconds, decode.Elapsed.TotalMilliseconds);
        }

        confidence = detectorOptions.Confidence;
        iou = null;
        inputDescription = $"[1, 3, {detector.TargetHeight}, {detector.TargetWidth}]  " +
                           $"({sourceWidth}x{sourceHeight} -> stretch, {detector.BoxesOutputName}/{detector.LogitsOutputName})";
    }

    var labels = LoadLabels(options.GetString("--labels"));

    Console.WriteLine($"model      : {modelPath}");
    if (image is { } sourceImage)
        Console.WriteLine($"image      : {imagePath} {sourceImage.Width}x{sourceImage.Height}");
    Console.WriteLine($"family     : {(rfdetr ? "rfdetr" : "yolo")}");
    Console.WriteLine($"input      : {inputDescription}");
    if (image is not null)
        Console.WriteLine($"preprocess : {preprocessMilliseconds:F2} ms");
    string iterationHint = iterations < 10 ? "; 稳态吞吐加 --iters 40" : string.Empty;
    Console.WriteLine($"forward    : {forwardMilliseconds:F2} ms  (min of {iterations}{iterationHint})");
    Console.WriteLine($"decode     : {decodeMilliseconds:F2} ms  (min of {iterations})");
    Console.WriteLine($"detections : {detections.Count}");

    int index = 0;
    foreach (var detection in detections.Take(30))
    {
        string name = LabelOf(labels, detection.ClassId);
        Console.WriteLine($"  #{index++,-3} {name,-16} {detection.Score:F4}  " +
                          $"[{detection.X:F1}, {detection.Y:F1}, {detection.Width:F1} x {detection.Height:F1}]");
    }

    if (options.GetString("--json") is { } jsonPath)
    {
        WriteDetectJson(jsonPath, modelPath, imagePath, image, rfdetr ? "rfdetr" : "yolo",
            inputDescription, confidence, iou, preprocessMilliseconds, forwardMilliseconds,
            decodeMilliseconds, detections, labels);
        Console.WriteLine($"json       : {jsonPath}");
    }

    if (image is not null && options.GetString("--out") is { } detectOut)
    {
        SaveDetections(image, detections, labels, detectOut);
        Console.WriteLine($"output     : {Path.GetFullPath(detectOut)}");
    }

    return 0;
}

// ---------------- 画框 / 保存（ImageFacade） ----------------

/// <summary>把检测框画到源图上并另存（ImageFacade 自绘 + 内置中文字体，纯托管）。</summary>
static void SaveDetections(CliImage image, IReadOnlyList<Detection> detections, string[]? labels, string path)
{
    var drawn = new FacadeDetection[detections.Count];
    for (int i = 0; i < drawn.Length; i++)
    {
        Detection detection = detections[i];
        drawn[i] = new FacadeDetection(
            new RectF(detection.X, detection.Y, detection.Width, detection.Height),
            LabelOf(labels, detection.ClassId),
            detection.Score)
        {
            ClassId = detection.ClassId,
        };
    }

    ImageFacade.Pipeline.DetectionDrawing.DrawDetections(image.Image, drawn, LabelStyle(image.Width, image.Height));
    image.Image.Save(path, new SaveOptions { Quality = 92 });
}

/// <summary>把 OCR 文本行画到源图上并另存。</summary>
static void SaveOcrLines(CliImage image, IReadOnlyList<OcrLine> lines, string path)
{
    var drawn = new FacadeDetection[lines.Count];
    for (int i = 0; i < drawn.Length; i++)
    {
        OcrLine line = lines[i];
        drawn[i] = new FacadeDetection(new RectF(line.X, line.Y, line.Width, line.Height), line.Text, line.Score);
    }

    ImageFacade.Pipeline.DetectionDrawing.DrawDetections(image.Image, drawn, LabelStyle(image.Width, image.Height));
    image.Image.Save(path, new SaveOptions { Quality = 92 });
}

/// <summary>标签字号 / 线宽按图幅缩放：图越大字越大，不至于在大图上小到看不清。</summary>
static FacadeDetectionStyle LabelStyle(int width, int height)
{
    float minSide = Math.Min(width, height);
    return new FacadeDetectionStyle
    {
        FontSize = Math.Clamp(minSide * 0.02f, 14f, 72f),
        StrokeWidth = Math.Clamp(minSide * 0.002f, 2f, 8f),
    };
}

// ---------------- OCR：PaddleOCR（DB 检测 + 方向分类 + CTC 识别） ----------------

static int Ocr(string[] args)
{
    if (args.Length < 2)
    {
        Console.Error.WriteLine("usage: ocr <image.ppm|bmp> --det det.onnx --rec rec.onnx --dict ppocr_keys.txt");
        Console.Error.WriteLine("                     [--cls cls.onnx] [--limit 960] [--box 0.6] [--unclip 1.6]");
        Console.Error.WriteLine("                     [--rec-width 640] [--json out.json] [--no-cls]");
        return 2;
    }

    string imagePath = args[1];
    CliOptions options = ParseOptions(args, 2);

    string? detPath = options.GetString("--det");
    string? recPath = options.GetString("--rec");
    string? dictPath = options.GetString("--dict");
    string? clsPath = options.GetString("--cls");

    if (detPath is null || recPath is null || dictPath is null)
    {
        Console.Error.WriteLine("ocr 需要 --det / --rec / --dict（PaddleOCR 的 det.onnx / rec.onnx / 字典）");
        return 2;
    }

    if (!File.Exists(imagePath))
    {
        Console.Error.WriteLine($"image not found: {imagePath}");
        return 2;
    }

    var ocrOptions = new PaddleOcrOptions
    {
        DetLimitSide = options.GetInt("--limit", 960),
        DetThreshold = options.GetFloat("--det-conf", 0.3f),
        BoxThreshold = options.GetFloat("--box", 0.6f),
        UnclipRatio = options.GetFloat("--unclip", 1.6f),
        RecMaxWidth = options.GetInt("--rec-width", 640),
        // 没给 --cls 或显式 --no-cls 时不跑方向分类
        UseClassification = clsPath is not null && !options.Has("--no-cls"),
    };

    var loadWatch = System.Diagnostics.Stopwatch.StartNew();
    PaddleOcrModel model = PaddleOcrModel.FromFiles(detPath, recPath, dictPath, clsPath);
    var pipeline = new PaddleOcrPipeline(model, ocrOptions);
    loadWatch.Stop();

    // 解码交给 ImageFacade：JPEG / PNG / BMP 都能直接读
    using CliImage image = CliImage.Load(imagePath);

    var watch = System.Diagnostics.Stopwatch.StartNew();
    IReadOnlyList<OcrLine> lines = pipeline.Run(image.Pixels, image.Width, image.Height, OcrPixelFormat.Rgb24);
    watch.Stop();

    string? outPath = options.GetString("--out");
    if (!string.IsNullOrEmpty(outPath))
    {
        SaveOcrLines(image, lines, outPath);
        Console.WriteLine($"  output : {Path.GetFullPath(outPath)}");
    }

    string? jsonPath = options.GetString("--json");
    if (!string.IsNullOrEmpty(jsonPath))
    {
        using var stream = File.Create(jsonPath);
        using var json = new Utf8JsonWriter(stream, new JsonWriterOptions { Indented = true, Encoder = System.Text.Encodings.Web.JavaScriptEncoder.UnsafeRelaxedJsonEscaping });

        json.WriteStartObject();
        json.WriteString("image", Path.GetFileName(imagePath));
        json.WriteNumber("width", image.Width);
        json.WriteNumber("height", image.Height);
        json.WriteString("model", model.ToString());
        json.WriteNumber("detLimitSide", ocrOptions.DetLimitSide);
        json.WriteNumber("elapsedMs", Math.Round(watch.Elapsed.TotalMilliseconds, 2));
        json.WriteStartArray("lines");
        foreach (OcrLine line in lines)
        {
            json.WriteStartObject();
            json.WriteString("text", line.Text);
            json.WriteNumber("score", Math.Round(line.Score, 4));
            json.WriteStartArray("box");
            json.WriteNumberValue(Math.Round(line.X, 1));
            json.WriteNumberValue(Math.Round(line.Y, 1));
            json.WriteNumberValue(Math.Round(line.Width, 1));
            json.WriteNumberValue(Math.Round(line.Height, 1));
            json.WriteEndArray();
            json.WriteEndObject();
        }
        json.WriteEndArray();
        json.WriteEndObject();
        json.Flush();

        Console.WriteLine($"{Path.GetFileName(imagePath)} {image.Width}x{image.Height}  ->  {Path.GetFullPath(jsonPath)}");
        Console.WriteLine($"  result : {lines.Count} 行，{watch.Elapsed.TotalMilliseconds:F1} ms");
        return 0;
    }

    Console.WriteLine($"{Path.GetFileName(imagePath)} {image.Width}x{image.Height}");
    Console.WriteLine($"  model  : {model}  (det {Path.GetFileName(detPath)}, rec {Path.GetFileName(recPath)})");
    Console.WriteLine($"  load   : {loadWatch.ElapsedMilliseconds} ms（含 det/rec 导入 + 折叠）");
    Console.WriteLine($"  result : {lines.Count} 行，{watch.Elapsed.TotalMilliseconds:F1} ms（limit {ocrOptions.DetLimitSide}）");
    foreach (OcrLine line in lines)
        Console.WriteLine("    " + line);
    return 0;
}

// ---------------- SAM3：图像 + 框提示 → 掩码 ----------------

static int Sam3(string[] args)
{
    if (args.Length < 2)
    {
        Console.Error.WriteLine("usage: sam3 <image> --models <fp32 dir> --box x1,y1,x2,y2 [--box ...]");
        Console.Error.WriteLine("                     [--out marked.jpg] [--conf 0.3] [--format cxcywh|xyxy|px] [--json out.json]");
        Console.Error.WriteLine("       模型目录需要 vision-encoder / geometry-encoder / decoder 的 FP32 版本；");
        Console.Error.WriteLine("       官方导出是 FP16，先用 tools/scripts/onnx_fp16_to_f32.py 转换。");
        return 2;
    }

    string imagePath = args[1];
    CliOptions options = ParseOptions(args, 2);

    string? modelDirectory = options.GetString("--models");
    if (modelDirectory is null)
    {
        Console.Error.WriteLine("sam3 需要 --models <fp32 模型目录>");
        return 2;
    }

    if (!File.Exists(imagePath))
    {
        Console.Error.WriteLine($"image not found: {imagePath}");
        return 2;
    }

    var boxes = new List<Sam3Box>();
    foreach (string text in options.Values("--box"))
    {
        int[]? parsed = ParseShape(text);
        if (parsed is not { Length: 4 })
        {
            Console.Error.WriteLine($"sam3: --box 需要 4 个数（x1,y1,x2,y2），收到 '{text}'");
            return 2;
        }

        boxes.Add(new Sam3Box(parsed[0], parsed[1], parsed[2], parsed[3]));
    }

    if (boxes.Count == 0)
    {
        Console.Error.WriteLine("sam3: 至少给一个 --box（源图像素坐标）");
        return 2;
    }

    Sam3PromptBoxFormat format = options.GetString("--format")?.ToLowerInvariant() switch
    {
        "xyxy" => Sam3PromptBoxFormat.XyxyNormalized,
        "px" or "pixels" => Sam3PromptBoxFormat.XyxyModelPixels,
        _ => Sam3PromptBoxFormat.CxcywhNormalized,
    };

    var sam3Options = new Sam3Options
    {
        ModelDirectory = modelDirectory,
        ScoreThreshold = options.GetFloat("--conf", 0.3f),
        IouThreshold = options.GetFloat("--iou", 0.5f),
        MaxItems = options.GetInt("--max-items", 0),
        BoxFormat = format,
        BoxLabel = options.GetInt("--label", 1),
        Diagnostic = options.Has("--debug") ? line => Console.WriteLine("  [dbg] " + line) : null,
    };

    using CliImage image = CliImage.Load(imagePath);
    var segmenter = new Sam3Segmenter(sam3Options);

    var watch = System.Diagnostics.Stopwatch.StartNew();
    Sam3Result result = segmenter.Segment(image.Pixels, image.Width, image.Height, boxes);
    watch.Stop();

    Console.WriteLine($"{Path.GetFileName(imagePath)} {image.Width}x{image.Height}  ->  " +
                      $"embedded {result.EmbeddedWidth}x{result.EmbeddedHeight} in {result.ImageSize}²");
    Console.WriteLine($"  models  : {Path.GetFullPath(modelDirectory)}");
    Console.WriteLine($"  prompt  : {boxes.Count} box(es), format {format}, label {sam3Options.BoxLabel}");
    Console.WriteLine($"  timing  : {watch.Elapsed.TotalSeconds:F1} s（vision + geometry + decoder）");
    Console.WriteLine($"  presence: {result.PresenceScore:F4}    items: {result.Items.Count}");

    foreach (Sam3Item item in result.Items)
    {
        int covered = 0;
        foreach (float value in item.Mask)
        {
            if (value >= 0.5f)
                covered++;
        }

        Console.WriteLine($"    score {item.Score:F3}  box " +
                          $"[{item.Box.X1:F0},{item.Box.Y1:F0} - {item.Box.X2:F0},{item.Box.Y2:F0}]  " +
                          $"mask {(float)covered / item.Mask.Length:P1}");
    }

    string? outPath = options.GetString("--out");
    if (!string.IsNullOrEmpty(outPath))
    {
        SaveSam3(image, result, outPath);
        Console.WriteLine($"  output  : {Path.GetFullPath(outPath)}");
    }

    if (options.GetString("--json") is { } jsonPath)
    {
        using var stream = File.Create(jsonPath);
        using var json = new Utf8JsonWriter(stream, new JsonWriterOptions
        {
            Indented = true,
            Encoder = System.Text.Encodings.Web.JavaScriptEncoder.UnsafeRelaxedJsonEscaping,
        });

        json.WriteStartObject();
        json.WriteString("image", Path.GetFileName(imagePath));
        json.WriteNumber("width", image.Width);
        json.WriteNumber("height", image.Height);
        json.WriteNumber("presence", Math.Round(result.PresenceScore, 4));
        json.WriteStartArray("items");
        foreach (Sam3Item item in result.Items)
        {
            json.WriteStartObject();
            json.WriteNumber("score", Math.Round(item.Score, 4));
            json.WriteStartArray("box");
            json.WriteNumberValue(Math.Round(item.Box.X1, 1));
            json.WriteNumberValue(Math.Round(item.Box.Y1, 1));
            json.WriteNumberValue(Math.Round(item.Box.X2, 1));
            json.WriteNumberValue(Math.Round(item.Box.Y2, 1));
            json.WriteEndArray();
            json.WriteEndObject();
        }
        json.WriteEndArray();
        json.WriteEndObject();
        json.Flush();
        Console.WriteLine($"  json    : {Path.GetFullPath(jsonPath)}");
    }

    return 0;
}

// ---------------- 画掩码（ImageFacade） ----------------

/// <summary>把 288² 掩码按最近邻上采样、50% 绿色叠加到源图，再画框与分数。</summary>
static void SaveSam3(CliImage image, Sam3Result result, string path)
{
    int width = image.Width;
    int height = image.Height;
    byte[] rgba = image.Image.CopyPixels(PixelFormat.Rgba32);

    foreach (Sam3Item item in result.Items)
    {
        int maskSize = item.MaskWidth;
        for (int my = 0; my < maskSize; my++)
        {
            int y0 = (int)(my * result.MaskScaleY);
            int y1 = Math.Min(height, (int)((my + 1) * result.MaskScaleY) + 1);

            for (int mx = 0; mx < maskSize; mx++)
            {
                if (item.Mask[my * maskSize + mx] < 0.5f)
                    continue;

                int x0 = (int)(mx * result.MaskScaleX);
                int x1 = Math.Min(width, (int)((mx + 1) * result.MaskScaleX) + 1);

                for (int y = y0; y < y1; y++)
                {
                    int row = y * width;
                    for (int x = x0; x < x1; x++)
                    {
                        int index = (row + x) * 4;
                        rgba[index + 0] = (byte)((rgba[index + 0] + 40) / 2);      // R
                        rgba[index + 1] = (byte)((rgba[index + 1] + 255) / 2);     // G
                        rgba[index + 2] = (byte)((rgba[index + 2] + 60) / 2);      // B
                    }
                }
            }
        }
    }

    using FacadeImage maskImage = ImageHost.Engine.FromPixels(rgba, width, height, width * 4, PixelFormat.Rgba32);

    var drawn = new FacadeDetection[result.Items.Count];
    for (int i = 0; i < drawn.Length; i++)
    {
        Sam3Box box = result.Items[i].Box;
        drawn[i] = new FacadeDetection(
            new RectF(box.X1, box.Y1, Math.Max(1, box.Width), Math.Max(1, box.Height)),
            $"mask {result.Items[i].Score:F2}",
            result.Items[i].Score);
    }

    ImageFacade.Pipeline.DetectionDrawing.DrawDetections(maskImage, drawn, LabelStyle(width, height));
    maskImage.Save(path, new SaveOptions { Quality = 92 });
}

/// <summary>
/// 扫描所有运行时 Arg，报告含 NaN 的张量（用于定位「输出 NaN 但不知道是哪个算子」）。
/// 注意共享缓冲槽会被后来的算子覆盖，所以这里看到的是「某个时刻写过 NaN」的槽，
/// 结合 Arg 名字仍能定位到第一个出问题的算子。
/// </summary>
static void ScanForNaN(Net net)
{
    int hits = 0;
    var lines = new List<string>();

    for (int i = 1; i < net.Impl.ArgCount; i++)
    {
        var data = net.Impl.GetArgData(new Arg(i));
        if (data.Kind is ArgKind.Const or ArgKind.Empty || data.DType != DType.F32 || !data.HasShape)
            continue;

        Tensor tensor;
        try
        {
            tensor = net.Impl.GetArgTensor(new Arg(i));
        }
        catch (Exception)
        {
            continue;
        }

        if (tensor.DType != DType.F32 || tensor.Count == 0)
            continue;

        var span = tensor.AsSpan<float>();
        int nan = 0;
        foreach (float value in span)
        {
            if (float.IsNaN(value))
                nan++;
        }

        if (nan == 0)
            continue;

        hits++;
        if (lines.Count < 20)
            lines.Add($"{data.Name,-60} {data.Shape}  nan {nan}/{span.Length}");
    }

    Console.WriteLine($"nan scan  : {hits} arg(s) contain NaN");
    foreach (string line in lines)
        Console.WriteLine("  " + line);
}

static int[]? SingleShape(CliOptions options)
{
    // 命名形状（name=dims）用于多输入模型，不能当单输入的形状覆盖用。
    for (int i = 0; i < options.Shapes.Count; i++)
    {
        if (options.Shapes[i].Contains('='))
            return null;
    }

    return options.Shapes.Count == 0 ? null : ParseShape(options.Shapes[^1]);
}

static ChannelNormalization ParseNormalization(string? text) => text?.ToLowerInvariant() switch
{
    "255" or "unit" or "div255" => ChannelNormalization.Divide255,
    _ => ChannelNormalization.ImageNet,
};

static OutputLayout ParseOutputLayout(string? text) => text?.ToLowerInvariant() switch
{
    "cf" or "channelsfirst" => OutputLayout.ChannelsFirst,
    "cl" or "channelslast" => OutputLayout.ChannelsLast,
    "e2e" or "endtoend" => OutputLayout.EndToEnd,
    _ => OutputLayout.Auto,
};

static string[]? LoadLabels(string? path)
    => path is null ? null : File.ReadAllLines(path).Select(line => line.Trim()).ToArray();

static string LabelOf(string[]? labels, int classId)
    => labels is not null && classId >= 0 && classId < labels.Length ? labels[classId] : $"class{classId}";

static void WriteDetectJson(
    string path,
    string modelPath,
    string? imagePath,
    CliImage? image,
    string family,
    string inputDescription,
    float confidence,
    float? iou,
    double preprocessMilliseconds,
    double forwardMilliseconds,
    double decodeMilliseconds,
    IReadOnlyList<Detection> detections,
    string[]? labels)
{
    using var stream = File.Create(path);
    using var json = new Utf8JsonWriter(stream, new JsonWriterOptions { Indented = true });

    json.WriteStartObject();
    json.WriteString("model", Path.GetFileName(modelPath));
    json.WriteString("family", family);
    if (imagePath is not null && image is { } source)
    {
        json.WriteString("image", Path.GetFileName(imagePath));
        json.WriteNumber("width", source.Width);
        json.WriteNumber("height", source.Height);
    }

    json.WriteString("input", inputDescription);
    json.WriteNumber("confidence", confidence);
    if (iou is { } iouThreshold)
        json.WriteNumber("iou", iouThreshold);
    json.WriteNumber("preprocessMs", Math.Round(preprocessMilliseconds, 3));
    json.WriteNumber("inferenceMs", Math.Round(forwardMilliseconds, 3));
    json.WriteNumber("decodeMs", Math.Round(decodeMilliseconds, 3));

    json.WriteStartArray("detections");
    foreach (var detection in detections)
    {
        json.WriteStartObject();
        json.WriteNumber("classId", detection.ClassId);
        if (labels is not null && detection.ClassId >= 0 && detection.ClassId < labels.Length)
            json.WriteString("name", labels[detection.ClassId]);
        json.WriteNumber("score", Math.Round(detection.Score, 5));
        json.WriteNumber("x", Math.Round(detection.X, 2));
        json.WriteNumber("y", Math.Round(detection.Y, 2));
        json.WriteNumber("width", Math.Round(detection.Width, 2));
        json.WriteNumber("height", Math.Round(detection.Height, 2));
        json.WriteEndObject();
    }

    json.WriteEndArray();
    json.WriteEndObject();
    json.Flush();
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
        else if (arg == "--shape" && i + 1 < args.Length)
            options.Shapes.Add(args[++i]);
        else if (arg == "--compare" && i + 1 < args.Length)
            options.Compare = args[++i];
        else if (arg.StartsWith("--", StringComparison.Ordinal))
        {
            // --class-agnostic 这类纯开关后面不跟值：下一个 token 又是 --xxx 时就记成空值。
            if (i + 1 < args.Length && !args[i + 1].StartsWith("--", StringComparison.Ordinal))
                options.Add(arg, args[++i]);
            else
                options.Add(arg, string.Empty);
        }
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

/// <summary>
/// 读模型。给了 <c>--shape</c> 就先把输入尺寸固定下来再折叠 ——
/// 全动态模型（PaddleOCR 的 det/rec、SAM3 的四个子模型）不给尺寸时形状链折不出常量，导入就会失败。
/// <para>
/// 支持两种写法：<c>--shape 1,3,640,640</c>（单输入模型），
/// <c>--shape fpn_feat_0=1,256,288,288 --shape ...</c>（多输入模型，可重复）。
/// </para>
/// </summary>
static Net LoadModel(string path, CliOptions? options)
{
    if (options is null || options.Shapes.Count == 0)
        return Net.ReadModel(path);

    var named = new Dictionary<string, int[]>(StringComparer.Ordinal);
    int[]? single = null;

    foreach (string text in options.Shapes)
    {
        int separator = text.IndexOf('=');
        if (separator > 0)
        {
            string name = text[..separator];
            if (ParseShape(text[(separator + 1)..]) is { Length: > 0 } shape)
                named[name] = shape;
        }
        else
        {
            single = ParseShape(text);
        }
    }

    if (named.Count > 0)
        return Net.ReadModel(path, named);

    return single is { Length: > 0 } ? Net.ReadModel(path, single) : Net.ReadModel(path);
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

    if (data.DType is not (DType.F32 or DType.I64 or DType.I32 or DType.Bool or DType.U8))
        throw new OnnxSharpException($"input '{name}': {data.DType} is not supported for --input");

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
    long expected = count * data.DType.SizeOf();

    if (bytes.Length != expected)
        throw new OnnxSharpException(
            $"input '{name}': {file} is {bytes.Length} B, expected {expected} B for {shape} {data.DType}");

    var tensor = Tensor.Allocate(shape, data.DType);
    bytes.CopyTo(tensor.Buffer.AsSpan(0, (int)expected));
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
