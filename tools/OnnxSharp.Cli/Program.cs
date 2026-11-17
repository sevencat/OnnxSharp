using System.Diagnostics;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Text;
using System.Text.Json;
using OnnxSharp.Core;
using OnnxSharp.Ocr.PaddleOCR;
using OnnxSharp.Onnx;
using OnnxSharp.Ops;
using OnnxSharp.Sam3;
using OnnxSharp.Speech;
using OnnxSharp.Speech.Paraformer;
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
    "ab" => Ab(args),
    "profile" => Profile(args),
    "detect" => Detect(args),
    "face" => Face(args),
    "features" => Features(args),
    "asr" => Asr(args),
    "ocr" => Ocr(args),
    "plate" => Plate(args),
    "sam3" => Sam3(args),
    "gemmbench" => GemmBench(args),
    "parbench" => ParBench(args),
    "flopsbench" => FlopsBench(args),
    "convbench" => ConvBench(args),
    "eltwisebench" => EltwiseBench.Run(ParseOptions(args, 1)),
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
    Console.WriteLine("  OnnxSharp.Cli ab <model.onnx> --input <file.bin> --a K=V[,K=V] --b K=V[,...]");
    Console.WriteLine("                [--iters N] [--rounds N] [--warmup N] [--no-check]");
    Console.WriteLine("                                      # 同进程交替 A/B：唯一能测出 <5% 收益的口径");
    Console.WriteLine("                                      # K: attn_fuse | attn_block | attn_budget | " +
                      "attn_lazy_soft | attn_pack_reuse | gemm_threshold");
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
    Console.WriteLine("  OnnxSharp.Cli plate <image> --det det.onnx --rec rec.onnx --dict keys.txt [--beam 8]");
    Console.WriteLine("                     [--greedy] [--no-structure] [--loose] [--score 0.5]");
    Console.WriteLine("                     [--direct | --crop x,y,w,h] [--json]");
    Console.WriteLine("                                      # 车牌专用：受限字符集 + 逐位约束 + 号牌形态校验");
    Console.WriteLine("                                      # --direct：输入已是一张车牌裁剪图，跳过检测");
    Console.WriteLine("  OnnxSharp.Cli face <yunet.onnx> <image> [--conf 0.6] [--iou 0.3] [--topk 5000]");
    Console.WriteLine("                     [--max-faces N] [--json out.json] [--out marked.jpg]");
    Console.WriteLine("                                      # YuNet 人脸检测：框 + 5 点关键点，坐标是原图像素");
    Console.WriteLine("                                      # PaddleOCR：DB 检测 → 方向分类 → CTC 识别");
    Console.WriteLine("  （<image> 由 ImageFacade 解码：JPEG / PNG / BMP / PPM 都可以）");
    Console.WriteLine("  OnnxSharp.Cli sam3 <image> --models <fp32 dir> --box x1,y1,x2,y2 [--box ...]");
    Console.WriteLine("                     [--out marked.jpg] [--conf 0.3] [--format cxcywh|xyxy|px] [--json]");
    Console.WriteLine("                                      # SAM3：图像 + 框提示 → 掩码（模型需先 FP16→FP32）");
    Console.WriteLine("  OnnxSharp.Cli gemmbench [--iters N] [--shape m,k,n] [--threads T] [--threshold V]");
    Console.WriteLine("                                      # GEMM 微内核吞吐（GFLOPS），调优用；--threshold 扫并行阈值");
    Console.WriteLine("  OnnxSharp.Cli eltwisebench [--iters N] [--shape d1,d2,...]");
    Console.WriteLine("                                      # 逐元素算子 ms/GB/s，带 copy 作带宽参考");
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
        Console.Error.WriteLine("usage: dump-graph <model.onnx> [--shape 1,3,960,960] [--type <OpType>] [--limit N] [--inputs]");
        Console.Error.WriteLine("       --shape  用于全动态模型（如 PaddleOCR）：先给输入定具体尺寸，导入期才能折叠形状链");
        Console.Error.WriteLine("       --inputs 同时打印输入形状（看广播形态、算内存流量）");
        return 2;
    }

    string path = args[1];
    CliOptions options = ParseOptions(args, 2);
    string? filter = options.GetString("--type");
    int limit = options.GetInt("--limit", int.MaxValue);
    bool showInputs = options.Has("--inputs");

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

        // --inputs 打印输入形状：分析广播形态（标量 / 尾广播 / 全长）与内存流量用。
        string ins = "";
        if (showInputs)
        {
            var list = new List<string>(op.Inputs.Count);
            foreach (var arg in op.Inputs)
                list.Add(arg.Idx == 0 ? "<none>" : Describe(net.Impl.GetArgData(arg)));
            ins = "   <- " + string.Join(" ", list);
        }

        Console.WriteLine($"  [{index,4}] {op.Type,-12} {op.Name,-44} -> {string.Join(", ", outs)}{ins}");
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

    var feeds = BindInputs(net, options.Inputs, SingleShape(options), NamedShapes(options));
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
    BindInputs(net, options.Inputs, SingleShape(options), NamedShapes(options));

    int warmup = options.GetInt("--warmup", 1);
    int iters = options.GetInt("--iters", 5);

    for (int i = 0; i < warmup; i++)
        net.ForwardAll();

    // 分配量与 GC 次数是解释「单算子计时之和 ≠ 端到端」的关键：
    // 分配多就会频繁触发 gen0/gen1 GC，而 GC 暂停会被算进当时正在跑的那个算子头上。
    OnnxSharp.Tensors.Tensor.ReallocLog.Clear();
    long allocBefore = GC.GetTotalAllocatedBytes(precise: true);
    long fitBytesBefore = OnnxSharp.Tensors.Tensor.FitReallocBytes;
    long fitCountBefore = OnnxSharp.Tensors.Tensor.FitReallocCount;
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
    Console.WriteLine($"fit realloc: {(OnnxSharp.Tensors.Tensor.FitReallocBytes - fitBytesBefore) / 1024.0 / Math.Max(1, iters):F0} KB/iter " +
                      $"({(OnnxSharp.Tensors.Tensor.FitReallocCount - fitCountBefore) / Math.Max(1, iters)} 次/iter)");

    foreach (string line in OnnxSharp.Tensors.Tensor.ReallocLog.Take(8))
        Console.WriteLine($"  realloc: {line}");
    Console.WriteLine($"gc        : gen0 {GC.CollectionCount(0) - g0}  gen1 {GC.CollectionCount(1) - g1}  gen2 {GC.CollectionCount(2) - g2}  (共 {iters} 次推理)");
    Console.WriteLine($"gc pause  : {gcMsPerIter:F1} ms/iter ({gcMsPerIter / Math.Max(1e-9, samples[iters / 2]) * 100:F0}% of median)");
    return 0;
}

/// <summary>
/// 同进程交替 A/B：<b>唯一</b>能测出 5% 以下差异的口径。
/// <para>
/// 分开跑两次进程的绝对值在本机会差到 2~3 倍（IDE 抢核、温度墙），而两个引擎放在同一个进程里
/// 逐轮交替、每轮交换先后顺序，可以把「同一时刻的机器状态」对齐，剩下的差异才归因到配置上。
/// </para>
/// <para>
/// 两套配置各自建一个引擎（融合发生在建图期，只能重建），跑同一份输入，输出逐元素比对，
/// 最后给出<b>配对比值</b>（每轮 min 之比）的中位数 —— 平均值会被单个后台尖峰带偏。
/// </para>
/// </summary>
static int Ab(string[] args)
{
    if (args.Length < 2)
    {
        Console.Error.WriteLine("usage: ab <model.onnx> --input <file.bin> --a K=V[,K=V] --b K=V[,...]");
        Console.Error.WriteLine("          [--iters N] [--rounds N] [--warmup N] [--no-check]");
        Console.Error.WriteLine("       K ∈ ONNXSHARP_ATTN_FUSE_MAXSEQ | ONNXSHARP_ATTN_BLOCK_SEQ | " +
                                "ONNXSHARP_ATTN_SCORE_BUDGET(KB) | ONNXSHARP_ATTN_LAZY_SOFTMAX | " +
                                "ONNXSHARP_ATTN_PACK_REUSE | ONNXSHARP_GEMM_THRESHOLD | ONNXSHARP_NCHWC | " +
                                "ONNXSHARP_FIRSTCONV | ONNXSHARP_POOL_FAST | ONNXSHARP_RESIZE_FAST");
        return 2;
    }

    string path = args[1];
    var options = ParseOptions(args, 2);
    if (!File.Exists(path))
    {
        Console.Error.WriteLine($"model not found: {path}");
        return 2;
    }

    if (options.Inputs.Count == 0)
    {
        Console.Error.WriteLine("no --input given");
        return 2;
    }

    var knobs = AbKnobs();
    var defaults = knobs.Select(k => k.Default()).ToArray();

    List<(string Key, string Value)> configA, configB;
    try
    {
        configA = ParseAbConfig(options.GetString("--a"), "--a", knobs);
        configB = ParseAbConfig(options.GetString("--b"), "--b", knobs);
    }
    catch (FormatException error)
    {
        Console.Error.WriteLine(error.Message);
        return 2;
    }

    int iters = Math.Max(1, options.GetInt("--iters", 5));
    int rounds = Math.Max(1, options.GetInt("--rounds", 5));
    // 预热必须够多：前 2~3 次是 tier-0 代码，第一轮能虚高 19%（实测 247.8 vs 208.5）。
    int warmup = Math.Max(0, options.GetInt("--warmup", 4));
    bool check = !options.Has("--no-check");

    ApplyAbConfig(configA, knobs, defaults);
    var netA = LoadModel(path, options);
    BindInputs(netA, options.Inputs, SingleShape(options), NamedShapes(options));

    ApplyAbConfig(configB, knobs, defaults);
    var netB = LoadModel(path, options);
    BindInputs(netB, options.Inputs, SingleShape(options), NamedShapes(options));

    Console.WriteLine($"model     : {path}");
    Console.WriteLine($"config A  : {DescribeAbConfig(configA, knobs, defaults)}");
    Console.WriteLine($"config B  : {DescribeAbConfig(configB, knobs, defaults)}");
    PrintAbGraphDiff(netA, netB);
    Console.WriteLine($"iters     : {iters}  rounds {rounds}  (warmup {warmup})");
    Console.WriteLine();

    if (check)
    {
        ApplyAbConfig(configA, knobs, defaults);
        var outA = netA.ForwardAll();
        ApplyAbConfig(configB, knobs, defaults);
        var outB = netB.ForwardAll();

        PrintAbOutputDiff(outA, outB);
        Console.WriteLine();
    }

    for (int i = 0; i < warmup; i++)
    {
        ApplyAbConfig(configA, knobs, defaults);
        netA.ForwardAll();
        ApplyAbConfig(configB, knobs, defaults);
        netB.ForwardAll();
    }

    // 每轮：两个引擎交替各跑 iters 次，依次序每轮对调一次（A→B / B→A），
    // 抵消「先跑的那个吃到更凉/更热的机器」这种系统性偏差。
    var roundA = new double[rounds];
    var roundB = new double[rounds];
    var samplesA = new List<double>(rounds * iters);
    var samplesB = new List<double>(rounds * iters);

    var watch = new System.Diagnostics.Stopwatch();

    for (int r = 0; r < rounds; r++)
    {
        bool aFirst = (r & 1) == 0;
        double bestA = double.MaxValue, bestB = double.MaxValue;

        for (int i = 0; i < iters; i++)
        {
            if (aFirst)
            {
                bestA = Math.Min(bestA, AbTime(netA, configA, knobs, defaults, watch, samplesA));
                bestB = Math.Min(bestB, AbTime(netB, configB, knobs, defaults, watch, samplesB));
            }
            else
            {
                bestB = Math.Min(bestB, AbTime(netB, configB, knobs, defaults, watch, samplesB));
                bestA = Math.Min(bestA, AbTime(netA, configA, knobs, defaults, watch, samplesA));
            }
        }

        roundA[r] = bestA;
        roundB[r] = bestB;
    }

    var ratios = new double[rounds];
    for (int r = 0; r < rounds; r++)
        ratios[r] = roundA[r] / roundB[r];

    double[] sortedRatios = (double[])ratios.Clone();
    Array.Sort(sortedRatios);

    double[] sortedA = samplesA.ToArray();
    double[] sortedB = samplesB.ToArray();
    Array.Sort(sortedA);
    Array.Sort(sortedB);

    Console.WriteLine($"{"round",6}{"A min",10}{"B min",10}{"A/B",8}   顺序");
    for (int r = 0; r < rounds; r++)
        Console.WriteLine($"{r + 1,6}{roundA[r],10:F1}{roundB[r],10:F1}{ratios[r],8:F3}   {((r & 1) == 0 ? "A→B" : "B→A")}");

    Console.WriteLine();
    Console.WriteLine($"{"",6}{"min",10}{"median",10}{"mean",10}");
    Console.WriteLine($"{"A",6}{sortedA[0],10:F1}{sortedA[sortedA.Length / 2],10:F1}{sortedA.Average(),10:F1}");
    Console.WriteLine($"{"B",6}{sortedB[0],10:F1}{sortedB[sortedB.Length / 2],10:F1}{sortedB.Average(),10:F1}");

    double paired = sortedRatios[rounds / 2];
    double spread = sortedRatios[^1] - sortedRatios[0];

    // 噪声地板用 MAD（中位绝对偏差）估：单个后台尖峰只会污染一轮，不该把结论带成「不可判定」。
    var deviations = ratios.Select(r => Math.Abs(r - paired)).OrderBy(d => d).ToArray();
    double mad = deviations[deviations.Length / 2];
    double floor = Math.Max(0.01, 3.0 * mad);   // 至少 1%：再小就没有工程意义

    Console.WriteLine();
    Console.WriteLine($"paired    : A/B 中位 {paired:F3}（范围 {sortedRatios[0]:F3} ~ {sortedRatios[^1]:F3}，" +
                      $"跨度 {spread * 100:F1}%）  噪声地板 {floor * 100:F1}%（3×MAD）");

    double gain = paired - 1.0;
    string verdict = Math.Abs(gain) <= floor
        ? $"差 {Math.Abs(gain) * 100:F1}% ≤ 噪声地板：**噪声内，不足以判定**"
        : gain > 0
            ? $"**B 快 {gain * 100:F1}%**（配对中位，超过噪声地板 {floor * 100:F1}%）"
            : $"**A 快 {-gain * 100:F1}%**（配对中位，超过噪声地板 {floor * 100:F1}%）";

    Console.WriteLine($"verdict   : {verdict}");

    if (sortedRatios[0] < 1.0 && sortedRatios[^1] > 1.0)
        Console.WriteLine("           注意：每轮方向不一致（有 >1 也有 <1），别单看某一轮。");

    return 0;
}

/// <summary>
/// 逐输出比对：max / mean / p99 / 超过 1e-3 的个数，外加最差的几个下标。
/// <para>
/// 只给一个全局 max 没有诊断价值 —— 融合前后 <c>dets</c> 的 max 差 0.5（top-k 选中了不同的
/// query），但 1200 个值里只有个位数变了，靠「有几个超阈值 + 最差下标」才能一眼分辨
/// 「数值精度」和「选了不同的检测框」。
/// </para>
/// </summary>
static void PrintAbOutputDiff(List<Tensor> outA, List<Tensor> outB)
{
    if (outA.Count != outB.Count)
    {
        Console.WriteLine($"check     : 输出个数不一致（{outA.Count} vs {outB.Count}）");
        return;
    }

    var scratch = new List<double>();

    for (int i = 0; i < outA.Count; i++)
    {
        var a = outA[i].AsSpan<float>();
        var b = outB[i].AsSpan<float>();

        if (a.Length != b.Length)
        {
            Console.WriteLine($"check     : out{i} 形状不一致（{outA[i].Shape} vs {outB[i].Shape}）");
            continue;
        }

        scratch.Clear();
        var offenders = new List<(int Index, double Diff)>();
        double max = 0, sum = 0;

        for (int j = 0; j < a.Length; j++)
        {
            double diff = Math.Abs((double)a[j] - b[j]);
            if (double.IsNaN(diff))
            {
                max = double.NaN;
                break;
            }

            scratch.Add(diff);
            sum += diff;
            if (diff > max)
                max = diff;
            if (diff > 1e-3)
                offenders.Add((j, diff));
        }

        if (double.IsNaN(max))
        {
            Console.WriteLine($"check     : out{i} 含 NaN");
            continue;
        }

        scratch.Sort();
        double p99 = scratch.Count == 0 ? 0 : scratch[Math.Min(scratch.Count - 1, (int)(scratch.Count * 0.99))];

        Console.WriteLine($"check     : out{i} {outA[i].Shape,-14} max|d|={max:E2} mean={sum / Math.Max(1, a.Length):E2} " +
                          $"p99={p99:E2} >1e-3: {offenders.Count}/{a.Length}");

        foreach (var (index, diff) in offenders.OrderByDescending(o => o.Diff).Take(5))
            Console.WriteLine($"              [{index}] A={a[index]:F6}  B={b[index]:F6}  d={diff:E2}");
    }
}

/// <summary>可被 <c>ab</c> 切换的旋钮（默认值来自环境变量，见各自的属性声明）。</summary>
static (string Name, string Alias, Func<string> Default, Action<string> Apply)[] AbKnobs() =>
[
    ("ONNXSHARP_ATTN_FUSE_MAXSEQ", "attn_fuse",
        () => GraphFusion.AttentionFusionMaxSeq.ToString(),
        v => GraphFusion.AttentionFusionMaxSeq = AbParseInt("ONNXSHARP_ATTN_FUSE_MAXSEQ", v)),
    ("ONNXSHARP_ATTN_BLOCK_SEQ", "attn_block",
        () => AttentionOps.BlockKernelMinSeq.ToString(),
        v => AttentionOps.BlockKernelMinSeq = AbParseInt("ONNXSHARP_ATTN_BLOCK_SEQ", v)),
    ("ONNXSHARP_ATTN_SCORE_BUDGET", "attn_budget",
        () => (AttentionOps.ScoreBudget / 1024).ToString(),
        v =>
        {
            int kb = AbParseInt("ONNXSHARP_ATTN_SCORE_BUDGET", v);
            AttentionOps.ScoreBudget = kb > 0
                ? kb * 1024
                : throw new FormatException(
                    $"--a/--b: ONNXSHARP_ATTN_SCORE_BUDGET 期望正整数（单位 KB），收到 '{v}'");
        }),
    ("ONNXSHARP_ATTN_LAZY_SOFTMAX", "attn_lazy_soft",
        () => AttentionOps.LazySoftmax ? "1" : "0",
        v => AttentionOps.LazySoftmax = AbParseInt("ONNXSHARP_ATTN_LAZY_SOFTMAX", v) != 0),
    ("ONNXSHARP_ATTN_PACK_REUSE", "attn_pack_reuse",
        () => AttentionOps.PackReuse ? "1" : "0",
        v => AttentionOps.PackReuse = AbParseInt("ONNXSHARP_ATTN_PACK_REUSE", v) != 0),
    ("ONNXSHARP_NCHWC", "nchwc",
        () => OnnxSharp.Kernels.NchwcConv.Enabled ? "1" : "0",
        v => OnnxSharp.Kernels.NchwcConv.Enabled = AbParseInt("ONNXSHARP_NCHWC", v) != 0),
    ("ONNXSHARP_FIRSTCONV", "firstconv",
        () => OnnxSharp.Kernels.FirstConv.Enabled ? "1" : "0",
        v => OnnxSharp.Kernels.FirstConv.Enabled = AbParseInt("ONNXSHARP_FIRSTCONV", v) != 0),
    ("ONNXSHARP_POOL_FAST", "pool_fast",
        () => DnnOps.FastPool ? "1" : "0",
        v => DnnOps.FastPool = AbParseInt("ONNXSHARP_POOL_FAST", v) != 0),
    ("ONNXSHARP_RESIZE_FAST", "resize_fast",
        () => DnnOps.FastResize2x ? "1" : "0",
        v => DnnOps.FastResize2x = AbParseInt("ONNXSHARP_RESIZE_FAST", v) != 0),
    ("ONNXSHARP_GEMM_THRESHOLD", "gemm_threshold",
        () => GemmKernel.ParallelThreshold.ToString(),
        v => GemmKernel.ParallelThreshold = GemmKernel.TryParseThreshold(v, out long threshold)
            ? threshold
            : throw new FormatException($"--a/--b: ONNXSHARP_GEMM_THRESHOLD 期望 65536 / 1<<16 / 64K，收到 '{v}'")),
];

static int AbParseInt(string name, string text)
    => int.TryParse(text, out int value) && value >= 0
        ? value
        : throw new FormatException($"--a/--b: {name} 期望非负整数，收到 '{text}'");

static double AbTime(Net net, List<(string Key, string Value)> config,
    (string Name, string Alias, Func<string> Default, Action<string> Apply)[] knobs,
    string[] defaults, System.Diagnostics.Stopwatch watch, List<double> samples)
{
    // 每次计时前重放配置：建图期旋钮（融合）此时无副作用，运行期旋钮（并行阈值）必须重放。
    ApplyAbConfig(config, knobs, defaults);

    watch.Restart();
    net.ForwardAll();
    watch.Stop();

    double ms = watch.Elapsed.TotalMilliseconds;
    samples.Add(ms);
    return ms;
}

/// <summary>解析 <c>--a K=V,K=V</c>。键可以是环境变量名，也可以是短别名。</summary>
static List<(string Key, string Value)> ParseAbConfig(
    string? text, string flag, (string Name, string Alias, Func<string> Default, Action<string> Apply)[] knobs)
{
    if (string.IsNullOrWhiteSpace(text))
        throw new FormatException($"ab: 必须给 {flag}（例如 {flag} ONNXSHARP_ATTN_FUSE_MAXSEQ=0）");

    var pairs = new List<(string, string)>();
    foreach (string item in text.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
    {
        int separator = item.IndexOf('=');
        if (separator <= 0)
            throw new FormatException($"ab: {flag} 的 '{item}' 不是 K=V 形式");

        string key = item[..separator].Trim();
        string value = item[(separator + 1)..].Trim();

        bool known = knobs.Any(k =>
            string.Equals(k.Name, key, StringComparison.OrdinalIgnoreCase) ||
            string.Equals(k.Alias, key, StringComparison.OrdinalIgnoreCase));

        if (!known)
        {
            throw new FormatException($"ab: {flag} 不认识键 '{key}'，可用：" +
                string.Join(" | ", knobs.Select(k => $"{k.Alias}({k.Name})")));
        }

        pairs.Add((key, value));
    }

    if (pairs.Count == 0)
        throw new FormatException($"ab: {flag} 为空");

    return pairs;
}

/// <summary>先把所有旋钮复位到进程启动时的默认值，再套用给定配置（未列出的键 = 环境默认）。</summary>
static void ApplyAbConfig(List<(string Key, string Value)> config,
    (string Name, string Alias, Func<string> Default, Action<string> Apply)[] knobs, string[] defaults)
{
    for (int i = 0; i < knobs.Length; i++)
        knobs[i].Apply(defaults[i]);

    foreach (var (key, value) in config)
    {
        foreach (var knob in knobs)
        {
            if (string.Equals(knob.Name, key, StringComparison.OrdinalIgnoreCase) ||
                string.Equals(knob.Alias, key, StringComparison.OrdinalIgnoreCase))
            {
                knob.Apply(value);
            }
        }
    }
}

/// <summary>按配置文本渲染生效值（不读当前静态状态 —— 那已经是后一套配置了）。</summary>
static string DescribeAbConfig(List<(string Key, string Value)> config,
    (string Name, string Alias, Func<string> Default, Action<string> Apply)[] knobs, string[] defaults)
{
    var parts = new List<string>();
    for (int i = 0; i < knobs.Length; i++)
    {
        string value = defaults[i];
        foreach (var (key, overridden) in config)
        {
            if (string.Equals(knobs[i].Name, key, StringComparison.OrdinalIgnoreCase) ||
                string.Equals(knobs[i].Alias, key, StringComparison.OrdinalIgnoreCase))
            {
                value = overridden;
            }
        }

        parts.Add($"{knobs[i].Alias}={value}");
    }

    return string.Join("  ", parts);
}

/// <summary>两个引擎的算子直方图对比：只打印有差异的类型 —— 用来确认开关真的改变了图。</summary>
static void PrintAbGraphDiff(Net a, Net b)
{
    static Dictionary<string, int> Histogram(Net net)
    {
        var counts = new Dictionary<string, int>(StringComparer.Ordinal);
        if (net.MainGraph is not { } graph)
            return counts;

        foreach (var op in graph.Prog)
            counts[op.Type] = counts.GetValueOrDefault(op.Type) + 1;

        return counts;
    }

    var histogramA = Histogram(a);
    var histogramB = Histogram(b);

    int opsA = histogramA.Values.Sum();
    int opsB = histogramB.Values.Sum();
    Console.WriteLine($"graph     : ops A {opsA} / B {opsB}");

    var types = histogramA.Keys.Union(histogramB.Keys, StringComparer.Ordinal)
        .Select(type => (Type: type, A: histogramA.GetValueOrDefault(type), B: histogramB.GetValueOrDefault(type)))
        .Where(entry => entry.A != entry.B)
        .OrderByDescending(entry => Math.Abs(entry.A - entry.B))
        .ToList();

    if (types.Count == 0)
    {
        Console.WriteLine("            两个配置的图完全相同（只有运行期旋钮会不同）");
        return;
    }

    foreach (var (type, countA, countB) in types)
        Console.WriteLine($"            {type,-20} A {countA,5}  B {countB,5}  ({(countB - countA).ToString("+0;-0;0")})");
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

    // --threshold：扫描「多大工作量才值得开线程」。MLAS 的口径是 M·N·K/65536，
    // 我们默认 1<<22（等效要求 64 个任务），rfdetr 的 attention 小 k 很可能因此全程单线程。
    string? thresholdText = options.GetString("--threshold");
    if (thresholdText is not null)
    {
        if (!OnnxSharp.Tensors.GemmKernel.TryParseThreshold(thresholdText, out long parsed))
            throw new ArgumentException($"--threshold 无法解析：{thresholdText}");

        OnnxSharp.Tensors.GemmKernel.ParallelThreshold = parsed;
    }

    var shapes = new List<(int M, int K, int N)>();
    string? single = options.GetString("--shape");

    // --shape 被 ParseOptions 收进 Shapes 列表（不是通用字典），这里两种写法都支持，
    // 并且允许重复传多个 --shape。
    var custom = single is not null ? new List<string> { single } : options.Shapes;

    if (custom.Count > 0)
    {
        foreach (string text in custom)
        {
            var parts = text.Split(',', StringSplitOptions.RemoveEmptyEntries);
            shapes.Add((int.Parse(parts[0]), int.Parse(parts[1]), int.Parse(parts[2])));
        }
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
    Console.WriteLine($"thresh  : {OnnxSharp.Tensors.GemmKernel.ParallelThreshold}（m·n·k 低于此值走单线程）");
    Console.WriteLine();
    Console.WriteLine("     m       k       n    batch    per-batch ms    GFLOPS   packed ms   speedup  bview ms");

    foreach (var (m, k, n) in shapes)
    {
        // --misalign N：把三块缓冲整体后移 N 个 float，用来量「张量在共享槽位里
        // 不是 32 字节对齐」对向量 load 的影响（图里的 A/B/C 都是 byte[] 切片，
        // 起点只保证 4 字节对齐，跨 cache line 的 load 在 Zen 上要额外一拍）。
        int off = Math.Max(0, options.GetInt("--misalign", 0));

        var a = new float[off + batches * m * k];
        var b = new float[off + batches * k * n];
        var c = new float[off + batches * m * n];
        var rng = new Random(12345);
        for (int i = 0; i < a.Length; i++) a[i] = (float)rng.NextDouble();
        for (int i = 0; i < b.Length; i++) b[i] = (float)rng.NextDouble();

        // --btrans 1：把 B 物化成 [n,k] 布局，等价于 attention 里折进 GEMM 的 K^T 视图
        // （bColStride = k、bStride = 1）。用来量「视图 B」相对紧密 B 掉多少 ——
        // rfdetr 的 QK^T 走的就是视图路径（MMv），图里实测只有 gemmbench 的一半。
        float[]? bt = null;
        if (options.GetInt("--btrans", 0) == 1)
        {
            bt = new float[off + batches * n * k];
            for (int bi = 0; bi < batches; bi++)
            {
                int src = off + bi * k * n;
                int dst0 = off + bi * n * k;
                for (int i = 0; i < k; i++)
                {
                    for (int j = 0; j < n; j++)
                        bt[dst0 + j * k + i] = b[src + i * n + j];
                }
            }
        }

        double Best(Action body)
        {
            // 按时间预热：分层编译下「调用次数」比「迭代次数」更重要，固定 3 次预热会把
            // tier-0 的慢样本算进来（实测同形状能差 4 倍以上）。
            var warm = System.Diagnostics.Stopwatch.StartNew();
            do
            {
                body();
            }
            while (warm.Elapsed.TotalMilliseconds < 400);

            double best = double.MaxValue;
            for (int i = 0; i < iters; i++)
            {
                var watch = System.Diagnostics.Stopwatch.StartNew();
                body();
                watch.Stop();
                if (watch.Elapsed.TotalMilliseconds < best)
                    best = watch.Elapsed.TotalMilliseconds;
            }

            return best;
        }

        double best = Best(() => RunBatched(a, b, c, batches, m, k, n, off));
        double perBatch = best / batches;
        double gflops = 2.0 * m * k * n / (perBatch * 1e6);

        // 预打包对照：常量权重（MatMul / 量化图）每帧都不必重新打包 B。
        // batch > 1 时每个 batch 是不同的矩阵，面板不能共享，所以只比单批。
        string packedText = "        -";
        string speedupText = "        -";
        if (batches == 1)
        {
            var packed = GemmKernel.Pack(b.AsSpan(off, k * n), k, n);
            double packedBest = Best(() => GemmKernel.MultiplyPacked(a.AsSpan(off, m * k), packed, c.AsSpan(off, m * n), m));
            packedText = $"{packedBest,13:F3}";
            speedupText = $"{best / packedBest,10:F2}x";
        }

        // --merged：走 MultiplyBatched（一个并行域铺开 batch×列块），对比逐 batch 派发
        if (batches > 1)
        {
            double mergedBest = Best(() => GemmKernel.MultiplyBatched(
                a.AsSpan(off, batches * m * k),
                b.AsSpan(off, batches * k * n),
                c.AsSpan(off, batches * m * n),
                batches, m, k, n));
            packedText = $"{mergedBest,13:F3}";
            speedupText = $"{best / mergedBest,10:F2}x";
        }

        // 视图 B（K^T 布局）同口径对照
        string viewText = "        -";
        if (bt is not null)
        {
            double viewBest = Best(() => GemmKernel.MultiplyBatched(
                a.AsSpan(off, batches * m * k),
                bt.AsSpan(off, batches * n * k),
                c.AsSpan(off, batches * m * n),
                batches, m, k, n, bColStride: k, bStride: 1, bBatchStride: k * n));
            viewText = $"{viewBest,10:F3}";
        }

        Console.WriteLine($"{m,6}  {k,6}  {n,6}  {batches,6}   {perBatch,12:F3}  {gflops,8:F1} {packedText} {speedupText} {viewText}");
    }

    // --attn 1：复刻 rfdetr 的 attention 形状与视图（4 样本 × 6 head，T=257，D=64），
    // 参数取自 ONNXSHARP_MM_TRACE 的实际输出（aRow=384 aBs=64 bRow/bCol=1/384 …），
    // outer=4 × inner=6 的派发序列也和 MMv 一致。用来把「端到端 ±10% 噪声」隔离掉。
    if (options.GetInt("--attn", 0) == 1)
    {
        const int outer = 4, inner = 6, T = 257, D = 64;
        int qPlane = T * inner * D;          // [T][6][64]
        int pPlane = T * inner * T;          // [T][6][T]
        int qcbs = T * T, pcbs = T * D;

        var probe = new Random(11);
        float[] q = new float[outer * qPlane], kt = new float[outer * qPlane];
        float[] p = new float[outer * pPlane], v = new float[outer * qPlane];
        foreach (var span in new[] { q, kt, p, v })
        {
            for (int i = 0; i < span.Length; i++) span[i] = (float)probe.NextDouble();
        }

        float[] qc = new float[outer * inner * qcbs];
        float[] pc = new float[outer * inner * pcbs];

        double qkBest = Probe(() =>
        {
            for (int o = 0; o < outer; o++)
            {
                GemmKernel.MultiplyBatched(
                    q.AsSpan(o * qPlane, qPlane), kt.AsSpan(o * qPlane, qPlane),
                    qc.AsSpan(o * inner * qcbs, inner * qcbs),
                    inner, T, D, T,
                    aStride: inner * D, aBatchStride: D,
                    bColStride: inner * D, bStride: 1, bBatchStride: D);
            }
        });

        double pvBest = Probe(() =>
        {
            for (int o = 0; o < outer; o++)
            {
                GemmKernel.MultiplyBatched(
                    p.AsSpan(o * pPlane, pPlane), v.AsSpan(o * qPlane, qPlane),
                    pc.AsSpan(o * inner * pcbs, inner * pcbs),
                    inner, T, T, D,
                    aStride: T, aBatchStride: T * T,
                    bColStride: 1, bStride: inner * D, bBatchStride: D);
            }
        });

        double flops = 2.0 * outer * inner * T * D * T;
        Console.WriteLine($"attn QK : {qkBest,7:F3} ms  {flops / qkBest / 1e6,7:F1} GFLOPS");
        Console.WriteLine($"attn PV : {pvBest,7:F3} ms  {flops / pvBest / 1e6,7:F1} GFLOPS");

        double Probe(Action action)
        {
            action();
            action();
            double best = double.MaxValue;
            for (int r = 0; r < 30; r++)
            {
                long t0 = System.Diagnostics.Stopwatch.GetTimestamp();
                action();
                double ms = (System.Diagnostics.Stopwatch.GetTimestamp() - t0) * 1000.0
                    / System.Diagnostics.Stopwatch.Frequency;
                if (ms < best)
                    best = ms;
            }

            return best;
        }
    }

    return 0;

    static void RunBatched(float[] a, float[] b, float[] c, int batches, int m, int k, int n, int off)
    {
        var aSpan = a.AsSpan();
        var bSpan = b.AsSpan();
        var cSpan = c.AsSpan();
        for (int bi = 0; bi < batches; bi++)
            GemmKernel.Multiply(aSpan.Slice(off + bi * m * k, m * k), bSpan.Slice(off + bi * k * n, k * n),
                                cSpan.Slice(off + bi * m * n, m * n), m, k, n);
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

    // 和 gemmbench 一样支持 --threads：卷积内核的分块决策读 ParallelRange.Threads，
    // 不给就只能是默认物理核数，跟别的基准表对不齐。
    int threads = options.GetInt("--threads", 0);
    if (threads > 0)
        OnnxSharp.Threading.ParallelRange.MaxDegreeOfParallelism = threads;

    var configs = new List<(int Ic, int Oc, int H, int W, int K, int S, int P)>();

    // --shape 被 ParseOptions 收进 Shapes 列表（不是 _values 字典），GetString 取不到，
    // 必须读 options.Shapes。支持重复给多次，一次跑多个形状。
    foreach (var spec in options.Shapes)
    {
        var p = spec.Split(',', StringSplitOptions.RemoveEmptyEntries);
        if (p.Length != 7)
        {
            Console.Error.WriteLine($"--shape 需要 7 个数字（ic,oc,h,w,k,s,p），收到：{spec}");
            return 2;
        }
        configs.Add((int.Parse(p[0]), int.Parse(p[1]), int.Parse(p[2]), int.Parse(p[3]),
            int.Parse(p[4]), int.Parse(p[5]), int.Parse(p[6])));
    }

    if (configs.Count == 0)
    {
        configs.Add((3, 16, 640, 640, 3, 2, 1));
        configs.Add((16, 32, 320, 320, 3, 2, 1));
        configs.Add((32, 64, 160, 160, 3, 2, 1));
        configs.Add((64, 64, 160, 160, 3, 2, 1));
        configs.Add((64, 64, 160, 160, 3, 1, 1));
        configs.Add((64, 128, 80, 80, 3, 2, 1));
        configs.Add((128, 128, 80, 80, 3, 2, 1));
        configs.Add((128, 128, 80, 80, 3, 1, 1));
        configs.Add((128, 256, 40, 40, 3, 2, 1));
        configs.Add((256, 256, 40, 40, 3, 2, 1));
        configs.Add((256, 256, 40, 40, 3, 1, 1));
        configs.Add((128, 128, 20, 20, 3, 2, 1));
        configs.Add((256, 256, 20, 20, 3, 1, 1));
        configs.Add((64, 64, 160, 160, 1, 1, 0));
        configs.Add((128, 128, 80, 80, 1, 1, 0));
        configs.Add((256, 256, 40, 40, 1, 1, 0));
        configs.Add((256, 256, 20, 20, 1, 1, 0));
    }

    // 进程里第一个配置的读数永远不可信：冷启动窗口内各内核都慢 2~6 倍
    // （实测同一形状排第一 14 GMAC/s、排第二 91 GMAC/s，且与 DOTNET_* 无关）。
    // 先塞一个一次性小形状把它吃掉，读数丢弃。
    configs.Insert(0, (128, 128, 20, 20, 3, 2, 1));
    bool discard = true;

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

    // 多内核交替计时（轮流跑一轮），避免「先测的那个跑在凉 CPU 上」
    double[] BestRound(List<Action> bodies, int count)
    {
        for (int i = 0; i < 3; i++)
            foreach (var body in bodies)
                body();

        var best = new double[bodies.Count];
        Array.Fill(best, double.MaxValue);
        var watch = System.Diagnostics.Stopwatch.StartNew();

        for (int i = 0; i < count; i++)
        {
            for (int k = 0; k < bodies.Count; k++)
            {
                watch.Restart();
                bodies[k]();
                watch.Stop();
                best[k] = Math.Min(best[k], watch.Elapsed.TotalMilliseconds);
            }
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

        // 参考结果（不计时）：先跑一遍 im2col 留作对拍基准
        OnnxSharp.Kernels.ConvKernels.Im2Col(input, weight, bias, output, 1, s, s, 1, 1, p, p);
        var im2colResult = output.AsSpan<float>().ToArray();

        // 3 通道首层的直接卷积（FirstConv）：先与 im2col 结果对拍，再和 im2col 交替计时
        bool firstUsed = OnnxSharp.Kernels.FirstConv.TryConv(input, weight, bias, output, 1, s, s, 1, 1, p, p);
        double firstDiff = 0;

        if (firstUsed)
        {
            var got = output.AsSpan<float>();
            for (int i = 0; i < got.Length; i++)
                firstDiff = Math.Max(firstDiff, Math.Abs(got[i] - im2colResult[i]));
        }

        // 交替计时：笔记本上同一个 kernel 连跑会积热降频，先测谁谁占便宜（实测差 30%）
        var bodies = new List<Action>
        {
            () => OnnxSharp.Kernels.ConvKernels.Im2Col(input, weight, bias, output, 1, s, s, 1, 1, p, p),
        };

        if (firstUsed)
            bodies.Add(() => { OnnxSharp.Kernels.FirstConv.TryConv(input, weight, bias, output, 1, s, s, 1, 1, p, p); });

        var times = BestRound(bodies, iters);
        double im2col = times[0];
        double first = firstUsed ? times[1] : double.NaN;

        bool packedUsed = false;
        double packed = double.NaN;

        if (OnnxSharp.Kernels.PackedConv.TryConv(input, weight, bias, output, 1, s, s, 1, 1, p, p))
        {
            packedUsed = true;
            packed = Best(() => { OnnxSharp.Kernels.PackedConv.TryConv(input, weight, bias, output, 1, s, s, 1, 1, p, p); });
        }

        string line = $"ic={ic,-4} oc={oc,-4} {h}x{w} k{k}s{s}p{p} -> {oh}x{ow}  "
            + $"im2col {im2col,7:F2} ms ({macs / im2col / 1e6,6:F1} GMAC/s)";

        if (firstUsed)
            line += $"  firstc {first,7:F2} ms ({macs / first / 1e6,6:F1})  max|d|={firstDiff:E1}";
        else
            line += "  firstc     --";

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

        // NCHWc16 布局的 3×3 内核（planv2 §P7.7），stride 1 与 2 都报。
        // 两列分开报：kernel 是稳态代价，reorder 是进出该布局的搬运（是否摊薄取决于图级传播）。
        if (OnnxSharp.Kernels.NchwcConv.Dispatch(ic, oc, h, w, oh, ow, k, k, 1, s, s, 1, 1, p, p))
        {
            // span 进不了 lambda（Best 是静态本地函数），拷成数组
            float[] xsArray = xs.ToArray();
            float[] bsArray = bs.ToArray();
            var packedW = OnnxSharp.Kernels.NchwcConv.PackWeights(ws, ic, oc);
            var blockedIn = new float[OnnxSharp.Kernels.NchwcConv.BlockedInputCount(ic, h, w, s)];
            var blockedOut = new float[OnnxSharp.Kernels.NchwcConv.BlockedOutputCount(oc, oh, ow)];
            var unpacked = new float[blockedOut.Length];
            OnnxSharp.Kernels.NchwcConv.PackInput(xsArray, blockedIn, ic, h, w, s);

            OnnxSharp.Kernels.ConvKernels.Im2Col(input, weight, bias, output, 1, s, s, 1, 1, p, p);
            OnnxSharp.Kernels.NchwcConv.Conv(blockedIn, packedW, bsArray, blockedOut, ic, oc, h, w, oh, ow, s);
            OnnxSharp.Kernels.NchwcConv.UnpackOutput(blockedOut, unpacked, oc, oh, ow);

            var expected = output.AsSpan<float>();
            double diff = 0;
            for (int i = 0; i < unpacked.Length; i++)
                diff = Math.Max(diff, Math.Abs(unpacked[i] - expected[i]));

            double kernel = Best(() => OnnxSharp.Kernels.NchwcConv.Conv(
                blockedIn, packedW, bsArray, blockedOut, ic, oc, h, w, oh, ow, s));

            double reorder = Best(() =>
            {
                OnnxSharp.Kernels.NchwcConv.PackInput(xsArray, blockedIn, ic, h, w, s);
                OnnxSharp.Kernels.NchwcConv.UnpackOutput(blockedOut, unpacked, oc, oh, ow);
            });

            line += $"  nchwc {kernel,7:F2} ms ({macs / kernel / 1e6,6:F1}) +reorder {reorder,6:F2}"
                + $"  max|d|={diff:E1}";
        }
        else
        {
            line += "  nchwc     --";
        }

        if (discard)
            discard = false;
        else
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
        Console.Error.WriteLine("usage: profile <model.onnx> --input <file.bin> [--iters N] [--top N] [--alloc]");
        Console.Error.WriteLine("       --alloc 额外统计每个算子的分配字节数（归因「每帧分配多少是谁花的」）");
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
    BindInputs(net, options.Inputs, SingleShape(options), NamedShapes(options));

    int iters = Math.Max(1, options.GetInt("--iters", 2));
    int top = options.GetInt("--top", 20);

    // 预热不计入统计：前几次迭代是 tier-0 代码，会把计算密集算子的份额抬高一大截。
    int warmup = Math.Max(0, options.GetInt("--warmup", 5));

    // --alloc 打开逐算子分配归因（GC.GetTotalAllocatedBytes），用来回答「每帧 53 MB 是谁分配的」。
    // 它会给每个算子加两次 precise 计数调用，耗时口径会比不加时略高，所以默认关闭。
    bool traceAlloc = options.Has("--alloc");

    var watch = System.Diagnostics.Stopwatch.StartNew();
    var timings = net.ProfileAll(iters, warmup, traceAlloc, () =>
    {
        if (OnnxSharp.Kernels.ConvProf.On)
            OnnxSharp.Kernels.ConvProf.Reset();
        if (OnnxSharp.Kernels.NchwcConv.Profile)
            OnnxSharp.Kernels.NchwcConv.ResetProfile();
    });
    watch.Stop();

    double total = timings.Sum(t => t.Milliseconds);
    double totalMin = timings.Sum(t => t.MinMilliseconds);
    double totalAllocKB = timings.Sum(t => t.AllocatedBytes) / 1024.0;

    Console.WriteLine($"model     : {path}");
    Console.WriteLine($"ops       : {net.MainGraph?.Prog.Count}    iters: {iters}    warmup: {warmup}    {watch.Elapsed.TotalMilliseconds:F0} ms");
    Console.WriteLine($"total     : {total / iters:F1} ms/iter (mean)   {totalMin:F1} ms/iter (sum of min)");
    if (traceAlloc)
        Console.WriteLine($"alloc     : {totalAllocKB:F0} KB/iter (未在算子间的部分见下方「旁路」差额)");
    Console.WriteLine($"note      : share 按 sum-of-min 算；mean 会被后台负载的重尾放大，和 bench 的 median 对不上");
    Console.WriteLine();

    if (traceAlloc)
    {
        Console.WriteLine($"{"#",4}  {"type",-18}{"min ms",10}{"avg ms",10}{"share",8}{"KB",10}  name");
    }
    else
    {
        Console.WriteLine($"{"#",4}  {"type",-18}{"min ms",10}{"avg ms",10}{"share",8}  name");
    }

    int index = 0;
    foreach (var timing in timings.Take(top < 1 ? timings.Count : top))
    {
        string head = $"{++index,4}  {timing.Type,-18}{timing.MinMilliseconds,10:F2}{timing.Average,10:F2}" +
                      $"{timing.MinMilliseconds / totalMin * 100,7:F1}%";
        Console.WriteLine(traceAlloc
            ? $"{head}{timing.AllocatedBytes / 1024.0,10:F0}  {timing.Name}"
            : $"{head}  {timing.Name}");
    }

    // 按算子类型汇总：热点分散时（几百个 op 各占 1%）看这个更有用
    Console.WriteLine();
    Console.WriteLine(traceAlloc
        ? $"{"type",-20}{"ops",6}{"min ms",10}{"avg ms",10}{"share",8}{"KB",10}"
        : $"{"type",-20}{"ops",6}{"min ms",10}{"avg ms",10}{"share",8}");

    var byType = timings
        .GroupBy(t => t.Type)
        .Select(g => new
        {
            Type = g.Key,
            Ops = g.Count(),
            Min = g.Sum(t => t.MinMilliseconds),
            Milliseconds = g.Sum(t => t.Milliseconds) / iters,
            AllocKB = g.Sum(t => t.AllocatedBytes) / 1024.0,
        })
        .OrderByDescending(g => traceAlloc ? g.AllocKB : g.Min);

    foreach (var group in byType)
    {
        string head = $"{group.Type,-20}{group.Ops,6}{group.Min,10:F1}{group.Milliseconds,10:F1}{group.Min / totalMin * 100,7:F1}%";
        Console.WriteLine(traceAlloc ? $"{head}{group.AllocKB,10:F0}" : head);
    }

    // NCHWc 的分阶段耗时（只有 ONNXSHARP_NCHWC_PROF=1 时才有数）：
    // 回答「每层都付的 reorder 到底吃掉多少」，微基准里的 reorder 是热缓冲下的乐观值。
    if (OnnxSharp.Kernels.NchwcConv.Profile)
    {
        var (pack, kernel, unpack, calls) = OnnxSharp.Kernels.NchwcConv.TakeProfile();
        Console.WriteLine();
        if (calls > 0)
        {
            Console.WriteLine($"nchwc     : 墙钟 {(pack + kernel + unpack) / iters:F2} ms/帧（打包输入 {pack / iters:F2} · " +
                              $"内核 {kernel / iters:F2} · 解包输出 {unpack / iters:F2}）  " +
                              $"{calls / (double)iters:F0} 次/帧   reorder 占比 {((pack + unpack) / (pack + kernel + unpack) * 100):F1}%");
        }
        else
        {
            Console.WriteLine("nchwc     : 0 次调用（本模型没有命中 NCHWc 路径）");
        }
    }

    // 1×1 走 pointwise 时的分阶段耗时（ONNXSHARP_CONV_PROF=1）：
    // 「整段」与「GEMM」之差 = Span 切片 + 并行派发等非 GEMM 开销。
    if (OnnxSharp.Kernels.ConvProf.On)
    {
        var (pointwise, gemm, act) = OnnxSharp.Kernels.ConvProf.Take(iters);
        Console.WriteLine();
        Console.WriteLine($"pointwise : 1×1 每帧整段 {pointwise:F2} ms（其中 GEMM {gemm:F2} ms，" +
                          $"非 GEMM 开销 {pointwise - gemm:F2} ms）  融合激活 {act:F2} ms  [{iters} 帧实测]");

        var calls = OnnxSharp.Kernels.ConvProf.TakeCalls(iters, out int dropped);
        Console.WriteLine("            按「每帧贡献」排（m×k×n · 每帧次数 · 单次 ms · GMAC/s · 每帧合计 ms）：");

        foreach (var (ms, count, m, k, n) in calls.Take(10))
        {
            double gmac = (double)m * k * n / ms / 1e6;
            Console.WriteLine($"            {m,4}×{k,4}×{n,6} · {count,3} 次 · {ms,6:F2} ms · {gmac,6:F1} · {ms * count,6:F2} ms");
        }

        if (dropped > 0)
            Console.WriteLine($"            ⚠ 有 {dropped} 次调用被容量截断，次数偏低（调大 ConvProf.Capacity 或减小 --iters）");

        // im2col 路径（3×3 等）：整段 = 打包 + GEMM + 其余（回写/派发）
        var (colWall, colTotal, colPack, colGemm, colTasks) = OnnxSharp.Kernels.ConvProf.TakeIm2Col(iters);
        double packShare = colTotal > 0 ? colPack / colTotal * 100 : 0;

        Console.WriteLine();
        Console.WriteLine($"im2col    : 墙钟 {colWall:F2} ms/帧 · 任务累计 {colTotal:F2} ms（打包 {colPack:F2} = {packShare:F0}% · " +
                          $"GEMM {colGemm:F2} = {(colTotal > 0 ? colGemm / colTotal * 100 : 0):F0}%）" +
                          $"  任务 {colTasks:F0} 次/帧  [{iters} 帧实测]");

        var colCalls = OnnxSharp.Kernels.ConvProf.TakeIm2ColCalls(iters, out int colDropped);
        Console.WriteLine("            按「每帧贡献」排（m=outC × k=inC·kh·kw × n=块像素 · 每帧次数 · 单次 ms · GMAC/s · 每帧合计 ms）：");

        foreach (var (ms, count, m, k, n) in colCalls.Take(10))
        {
            double gmac = (double)m * k * n / ms / 1e6;
            Console.WriteLine($"            {m,4}×{k,4}×{n,6} · {count,3} 次 · {ms,6:F2} ms · {gmac,6:F1} · {ms * count,6:F2} ms");
        }

        if (colDropped > 0)
            Console.WriteLine($"            ⚠ 有 {colDropped} 次 im2col 任务被容量截断");

        // 面板式直接卷积（3×3 的主路径）：整段 = 打包面板 + 微内核
        var (packedWall, packedTotal, packedPack, packedMul, packedTasks) = OnnxSharp.Kernels.ConvProf.TakePacked(iters);
        string packPct = packedTotal > 0 ? $"{packedPack / packedTotal * 100:F0}%" : "n/a";
        string mulPct = packedTotal > 0 ? $"{packedMul / packedTotal * 100:F0}%" : "n/a";

        Console.WriteLine();
        Console.WriteLine($"packed    : 墙钟 {packedWall:F2} ms/帧 · 任务累计 {packedTotal:F2} ms（打包 {packedPack:F2} = {packPct} · " +
                          $"微内核 {packedMul:F2} = {mulPct}）  面板 {packedTasks:F0} 次/帧  [{iters} 帧实测]");

        var (winogradWall, winogradCalls) = OnnxSharp.Kernels.ConvProf.TakeWinograd(iters);
        Console.WriteLine($"winograd  : 墙钟 {winogradWall:F2} ms/帧  调用 {winogradCalls:F0} 次/帧");
    }

    // GEMM 指针对齐诊断（ONNXSHARP_GEMM_ALIGN=1）：微基准的缓冲是 new float[]，
    // 图里的 A/B/C 是 byte[] 槽位切片，对齐不同 → 同形状能差 10~14%（见 bench.md §30）。
    if (OnnxSharp.Tensors.GemmKernel.DbgAlign)
    {
        var (calls, alignA, alignB, alignC) = OnnxSharp.Tensors.GemmKernel.TakeAlignment();
        double Pct(long x) => calls == 0 ? 0 : x * 100.0 / calls;

        Console.WriteLine();
        Console.WriteLine($"gemm 对齐  : {calls} 次调用 · 64B 对齐率 A {Pct(alignA):F0}% / " +
                          $"B {Pct(alignB):F0}% / C {Pct(alignC):F0}%");
    }

    return 0;
}

// ---------------- 检测：图进、框出（预处理 / 解码 / NMS 都在 OnnxSharp.Vision） ----------------
// 两种检测头：YOLO（letterbox + 逐锚点 argmax + NMS）与 RF-DETR（拉伸 + 逐类 sigmoid + top-k）。
// 默认按输出个数判族（RF-DETR 是 dets + labels 双输出），也可以用 --family 强制指定。
/// <summary>
/// <c>asr &lt;wav&gt; --models &lt;dir&gt;</c>（或分别给 <c>--encoder/--decoder/--tokens</c>）：
/// wav → fbank/LFR/CMVN → encoder → CIF → decoder → argmax → 文本，全在 C# 里跑。
/// </summary>
static int Asr(string[] args)
{
    if (args.Length < 2)
    {
        Console.Error.WriteLine("usage: asr <wav> --models <dir> [--json] [--out text.txt]");
        Console.Error.WriteLine("       asr <wav> --encoder encoder.onnx --decoder decoder.onnx --tokens tokens.txt");
        return 2;
    }

    string wavPath = args[1];
    var options = ParseOptions(args, 2);

    string? directory = options.GetString("--models");
    string encoder = options.GetString("--encoder")
        ?? (directory is null ? string.Empty : Path.Combine(directory, "encoder.onnx"));
    string decoder = options.GetString("--decoder")
        ?? (directory is null ? string.Empty : Path.Combine(directory, "decoder.onnx"));
    string tokens = options.GetString("--tokens")
        ?? (directory is null ? string.Empty : Path.Combine(directory, "tokens.txt"));

    foreach (string path in new[] { wavPath, encoder, decoder, tokens })
    {
        if (string.IsNullOrEmpty(path) || !File.Exists(path))
        {
            Console.Error.WriteLine($"not found: {path}");
            return 2;
        }
    }

    var recognizer = new ParaformerRecognizer(new ParaformerOptions
    {
        EncoderPath = encoder,
        DecoderPath = decoder,
        TokensPath = tokens,
    });

    int repeat = Math.Max(1, (int)options.GetInt("--repeat", 1));
    RecognitionResult result = null!;

    for (int i = 0; i < repeat; i++)
    {
        var watch = Stopwatch.StartNew();
        result = recognizer.TranscribeWav(wavPath);
        watch.Stop();

        if (repeat > 1)
            Console.WriteLine($"iter {i + 1,-2} : {watch.Elapsed.TotalMilliseconds,7:F0} ms" +
                $"  (feat {result.FeatureMs:F0} / enc {result.EncoderMs:F0}" +
                $" / dec {result.DecoderMs:F0}; load {result.EncoderLoadMs:F0}+{result.DecoderLoadMs:F0})");
    }

    if (options.Has("--json"))
    {
        var json = new
        {
            wav = wavPath,
            text = result.Text,
            tokenIds = result.TokenIds,
            tokenCount = result.TokenCount,
            featureFrames = result.FeatureFrames,
            featureMs = Math.Round(result.FeatureMs, 1),
            encoderMs = Math.Round(result.EncoderMs, 1),
            decoderMs = Math.Round(result.DecoderMs, 1),
            totalMs = Math.Round(result.TotalMs, 1),
        };

        Console.WriteLine(JsonSerializer.Serialize(json, new JsonSerializerOptions { WriteIndented = true }));
    }
    else
    {
        Console.WriteLine($"wav    : {wavPath}");
        Console.WriteLine($"feats  : {result.FeatureFrames} frames");
        Console.WriteLine($"tokens : {result.TokenCount}  ids {string.Join(",", result.TokenIds)}");
        Console.WriteLine($"time   : {result.TotalMs:F0} ms (feat {result.FeatureMs:F0} / enc {result.EncoderMs:F0} / dec {result.DecoderMs:F0})");
        Console.WriteLine($"text   : {result.Text}");
    }

    if (options.GetString("--out") is { } outPath)
    {
        File.WriteAllText(outPath, result.Text, new UTF8Encoding(false));
        Console.WriteLine($"wrote  : {outPath}");
    }

    return 0;
}


/// <summary>
/// <c>features &lt;model.onnx&gt; &lt;wav&gt; [--out feats.f32]</c>：
/// wav → fbank(80) → LFR(7,6) → CMVN → <c>[T, 560]</c>，参数从模型 metadata 读。
/// 产出就是 Paraformer 的 <c>speech</c> 输入（可以直接喂给 <c>run</c> 或 ORT）。
/// </summary>
static int Features(string[] args)
{
    if (args.Length < 3)
    {
        Console.Error.WriteLine("usage: features <model.onnx> <wav> [--out feats.f32]");
        return 2;
    }

    string modelPath = args[1];
    string wavPath = args[2];
    var options = ParseOptions(args, 3);

    if (!File.Exists(modelPath) || !File.Exists(wavPath))
    {
        Console.Error.WriteLine($"not found: {(File.Exists(modelPath) ? wavPath : modelPath)}");
        return 2;
    }

    var extractor = SpeechFeatureExtractor.FromModel(modelPath);
    var watch = Stopwatch.StartNew();
    var features = extractor.ComputeFromWav(wavPath);
    watch.Stop();

    var metadata = OnnxFile.Load(modelPath).Metadata;
    Console.WriteLine($"wav    : {wavPath}");
    Console.WriteLine($"model  : {modelPath}");
    Console.WriteLine($"meta   : {metadata.Count} entries" +
        (metadata.TryGetValue("lfr_window_size", out string? w) ? $" (lfr {w},{metadata.GetValueOrDefault("lfr_window_shift")})" : string.Empty));
    Console.WriteLine($"feats  : [{features.Frames}, {features.Dim}]  ({watch.Elapsed.TotalMilliseconds:F1} ms)");

    if (options.GetString("--out") is { } outPath)
    {
        var bytes = new byte[features.Values.Length * sizeof(float)];
        Buffer.BlockCopy(features.Values, 0, bytes, 0, bytes.Length);
        File.WriteAllBytes(outPath, bytes);
        Console.WriteLine($"wrote  : {outPath}");
    }

    return 0;
}



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

    // 检测头每锚点的字段数（4+类别数），只有 yolo 家族才有意义
    int yoloFields = 0;

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

            if (yoloFields == 0 && detectorOptions.Layout != OutputLayout.EndToEnd && output.Shape.Rank == 3)
            {
                // 84×6300 与 6300×84 都取小的那一维（端到端 [1,300,6] 上面已经排除了）
                int first = output.Shape[1];
                int second = output.Shape[2];
                yoloFields = first < second ? first : second;
            }

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

    // 没显式给 --labels 时按检测头维度猜：84 = 4 + 80，就是 COCO 那套公开权重
    // （yolov8 / yolo11 / yolo26 都一样）。自定义类别数的模型不会误判。
    var labels = LoadLabels(options.GetString("--labels"))
        ?? (yoloFields == CocoLabels.DetectionFields ? CocoLabels.Names : null);

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

// ---------------- 人脸检测：YuNet ----------------

static int Face(string[] args)
{
    if (args.Length < 3)
    {
        Console.Error.WriteLine("usage: face <yunet.onnx> <image> [--conf 0.6] [--iou 0.3] [--topk 5000]");
        Console.Error.WriteLine("                   [--max-faces N] [--json out.json] [--out marked.jpg]");
        return 2;
    }

    string modelPath = args[1];
    string imagePath = args[2];
    var options = ParseOptions(args, 3);

    var detectorOptions = new FaceDetectorOptions
    {
        ScoreThreshold = options.GetFloat("--conf", 0.6f),
        IouThreshold = options.GetFloat("--iou", 0.3f),
        TopK = options.GetInt("--topk", 5000),
        MaxFaces = options.GetInt("--max-faces", int.MaxValue),
    };

    using CliImage image = CliImage.Load(imagePath);
    var detector = new YuNetDetector(Net.ReadModel(modelPath), detectorOptions);

    var watch = Stopwatch.StartNew();
    // CliImage.Pixels 是紧凑 RGB24（行跨度 = width*3）→ bgr: false
    IReadOnlyList<Face> faces = detector.Detect(image.Pixels, image.Width, image.Height, bgr: false);
    watch.Stop();

    Console.WriteLine($"image     : {imagePath}  {image.Width}x{image.Height} -> net {detector.InputWidth}x{detector.InputHeight}");
    Console.WriteLine($"faces     : {faces.Count}  ({watch.Elapsed.TotalMilliseconds:F1} ms 含预处理+推理+解码, conf>={detectorOptions.ScoreThreshold})");

    foreach (Face face in faces)
    {
        FaceLandmarks m = face.Landmarks;
        Console.WriteLine(
            $"  score {face.Score:F4}  box [{face.X:F1}, {face.Y:F1}, {face.Width:F1} x {face.Height:F1}]");
        Console.WriteLine(
            $"    re({m.RightEye.X:F1},{m.RightEye.Y:F1}) le({m.LeftEye.X:F1},{m.LeftEye.Y:F1}) " +
            $"nt({m.Nose.X:F1},{m.Nose.Y:F1}) rm({m.RightMouth.X:F1},{m.RightMouth.Y:F1}) " +
            $"lm({m.LeftMouth.X:F1},{m.LeftMouth.Y:F1})");
    }

    string? jsonPath = options.GetString("--json");
    if (jsonPath is not null)
    {
        var payload = new
        {
            image = new { width = image.Width, height = image.Height },
            input = new { width = detector.InputWidth, height = detector.InputHeight },
            count = faces.Count,
            faces = faces.Select(face => new
            {
                score = Math.Round(face.Score, 6),
                box = new[] { face.X, face.Y, face.Width, face.Height }.Select(v => Math.Round(v, 3)),
                landmarks = new[] { face.Landmarks.RightEye, face.Landmarks.LeftEye, face.Landmarks.Nose, face.Landmarks.RightMouth, face.Landmarks.LeftMouth }
                    .Select(p => new[] { Math.Round(p.X, 3), Math.Round(p.Y, 3) }),
            }),
        };

        File.WriteAllText(
            jsonPath,
            JsonSerializer.Serialize(payload, new JsonSerializerOptions { WriteIndented = true }));
    }

    string? outPath = options.GetString("--out");
    if (outPath is not null)
    {
        var drawn = new FacadeDetection[faces.Count];
        for (int i = 0; i < drawn.Length; i++)
        {
            Face face = faces[i];
            drawn[i] = new FacadeDetection(
                new RectF(face.X, face.Y, face.Width, face.Height),
                $"face {face.Score:F2}",
                face.Score);
        }

        ImageFacade.Pipeline.DetectionDrawing.DrawDetections(image.Image, drawn, LabelStyle(image.Width, image.Height));
        image.Image.Save(outPath, new SaveOptions { Quality = 92 });
    }

    return 0;
}

// ---------------- 画框 / 保存（ImageFacade） ----------------
/// <summary>把检测框画到源图上并另存（ImageFacade 自绘 + 内置中文字体，纯托管）。</summary>
static void SaveDetections(CliImage image, IReadOnlyList<Detection> detections, IReadOnlyList<string>? labels, string path)
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

// ---------------- 车牌 OCR：DB 找行 → 受限 CTC 解码 ----------------

static int Plate(string[] args)
{
    if (args.Length < 2)
    {
        Console.Error.WriteLine("usage: plate <image> --det det.onnx --rec rec.onnx --dict ppocr_keys.txt");
        Console.Error.WriteLine("                     [--beam 8] [--greedy] [--no-structure] [--loose] [--score 0.5] [--max 4]");
        Console.Error.WriteLine("                     [--no-aspect] [--limit 960] [--det-conf 0.3] [--box 0.6] [--unclip 1.6]");
        Console.Error.WriteLine("                     [--direct] [--crop x,y,w,h] [--iters N] [--json out.json]");
        Console.Error.WriteLine("       --direct 整张图就是一张车牌（上游检测器已抠出的裁剪图）——不跑 DB 检测；");
        Console.Error.WriteLine("       --crop x,y,w,h 在整图里指定一块车牌区域——同样跳过 DB 检测；");
        Console.Error.WriteLine("       --iters 只与 --direct / --crop 一起用，量的是单张 rec 的稳态耗时（首次含图导入）");
        return 2;
    }

    string imagePath = args[1];
    CliOptions options = ParseOptions(args, 2);

    string? detPath = options.GetString("--det");
    string? recPath = options.GetString("--rec");
    string? dictPath = options.GetString("--dict");

    if (detPath is null || recPath is null || dictPath is null)
    {
        Console.Error.WriteLine("plate 需要 --det / --rec / --dict（PaddleOCR 的 det.onnx / rec.onnx / 字典）");
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
        UseClassification = false,
    };

    var readerOptions = new OnnxSharp.Ocr.Plate.PlateReaderOptions
    {
        Decoder = new OnnxSharp.Ocr.Plate.PlateDecoderOptions
        {
            BeamWidth = options.Has("--greedy") ? 1 : options.GetInt("--beam", 8),
            UseStructure = !options.Has("--no-structure"),
            Validate = !options.Has("--loose"),
        },
        MinScore = options.GetFloat("--score", 0.5f),
        MaxPlates = options.GetInt("--max", 4),
        FilterByAspect = !options.Has("--no-aspect"),
    };

    var loadWatch = System.Diagnostics.Stopwatch.StartNew();
    PaddleOcrModel model = PaddleOcrModel.FromFiles(detPath, recPath, dictPath);
    var pipeline = new PaddleOcrPipeline(model, ocrOptions);
    var reader = new OnnxSharp.Ocr.Plate.PlateReader(pipeline, readerOptions);
    loadWatch.Stop();

    using CliImage image = CliImage.Load(imagePath);

    // 两条「跳过 DB 检测」的通路：
    //   --direct          整张图就是一张车牌（上游检测器已抠出 / 手工裁好的裁剪图）
    //   --crop x,y,w,h    在整图里指定一块车牌区域（车牌检测器已经在跑的场景）
    // 注意不能叫 --box：那是 DB 的框分数阈值（和 ocr 子命令保持一致）。
    byte[] pixels = image.Pixels;
    int imageWidth = image.Width;
    int imageHeight = image.Height;

    string? cropText = options.GetString("--crop");
    bool direct = options.Has("--direct");
    int[]? crop = null;

    if (cropText is not null)
    {
        crop = cropText.Split(',').Select(int.Parse).ToArray();
        if (crop.Length != 4)
        {
            Console.Error.WriteLine("--crop 需要 x,y,w,h");
            return 2;
        }
    }

    if (direct && crop is not null)
    {
        Console.Error.WriteLine("--direct（整图即车牌）与 --crop（指定一块）只能选一个");
        return 2;
    }

    string mode = direct ? "direct" : crop is not null ? "crop" : "detect";
    bool skipDetect = direct || crop is not null;

    // --direct / --crop / --iters 共用这一条，保证「显示的结果」与「计时测的事」是同一件事
    OnnxSharp.Ocr.Plate.PlateResult? ReadSingle() => direct
        ? reader.ReadWhole(pixels, imageWidth, imageHeight, OcrPixelFormat.Rgb24)
        : reader.Read(pixels, imageWidth, imageHeight, OcrPixelFormat.Rgb24, 0, crop![0], crop![1], crop[2], crop[3]);

    var watch = System.Diagnostics.Stopwatch.StartNew();
    IReadOnlyList<OnnxSharp.Ocr.Plate.PlateResult> plates;

    if (skipDetect)
    {
        OnnxSharp.Ocr.Plate.PlateResult? single = ReadSingle();
        plates = single is null ? [] : [single];
    }
    else
    {
        plates = reader.DetectAndRead(pixels, imageWidth, imageHeight, OcrPixelFormat.Rgb24);
    }

    watch.Stop();

    // --iters N：只给 --direct / --crop 做稳态计时（rec 图像第一次调用要付约 300 ms 的图导入）
    int iters = options.GetInt("--iters", 1);
    double bestMs = 0;
    double avgMs = 0;

    if (iters > 1 && skipDetect && plates.Count > 0)
    {
        bestMs = double.MaxValue;
        double total = 0;

        for (int i = 0; i < iters; i++)
        {
            var timer = System.Diagnostics.Stopwatch.StartNew();
            ReadSingle();
            timer.Stop();

            bestMs = Math.Min(bestMs, timer.Elapsed.TotalMilliseconds);
            total += timer.Elapsed.TotalMilliseconds;
        }

        avgMs = total / iters;
    }

    string? jsonPath = options.GetString("--json");
    if (!string.IsNullOrEmpty(jsonPath))
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
        json.WriteString("model", model.ToString());
        json.WriteString("mode", mode);
        json.WriteNumber("beam", readerOptions.Decoder.BeamWidth);
        json.WriteBoolean("structure", readerOptions.Decoder.UseStructure);
        json.WriteNumber("elapsedMs", Math.Round(watch.Elapsed.TotalMilliseconds, 2));
        json.WriteStartArray("plates");

        foreach (OnnxSharp.Ocr.Plate.PlateResult plate in plates)
        {
            json.WriteStartObject();
            json.WriteString("text", plate.Text);
            json.WriteNumber("score", Math.Round(plate.Score, 4));
            json.WriteNumber("probability", Math.Round(plate.Decoding.Probability, 6));
            json.WriteString("kind", plate.Plate?.Kind.ToString() ?? "Unknown");
            json.WriteString("province", plate.Plate?.Province.ToString() ?? string.Empty);
            json.WriteString("letter", plate.Plate?.Letter.ToString() ?? string.Empty);
            json.WriteString("serial", plate.Plate?.Serial ?? string.Empty);
            json.WriteStartArray("box");
            json.WriteNumberValue(plate.X);
            json.WriteNumberValue(plate.Y);
            json.WriteNumberValue(plate.Width);
            json.WriteNumberValue(plate.Height);
            json.WriteEndArray();
            json.WriteStartArray("chars");
            foreach (float score in plate.Decoding.CharScores)
                json.WriteNumberValue(Math.Round(score, 4));
            json.WriteEndArray();
            json.WriteEndObject();
        }

        json.WriteEndArray();
        json.WriteEndObject();
        json.Flush();

        Console.WriteLine($"{Path.GetFileName(imagePath)} {image.Width}x{image.Height}  ->  {Path.GetFullPath(jsonPath)}");
        Console.WriteLine($"  result : {plates.Count} 张，{watch.Elapsed.TotalMilliseconds:F1} ms");
        return 0;
    }

    Console.WriteLine($"{Path.GetFileName(imagePath)} {image.Width}x{image.Height}");
    Console.WriteLine($"  model  : {model}  (beam {readerOptions.Decoder.BeamWidth}, structure {(readerOptions.Decoder.UseStructure ? "on" : "off")})");
    Console.WriteLine($"  load   : {loadWatch.ElapsedMilliseconds} ms");
    Console.WriteLine(skipDetect
        ? $"  result : {plates.Count} 张，{watch.Elapsed.TotalMilliseconds:F1} ms（{(direct ? "direct：整图即车牌，无 DB" : "crop：指定块，无 DB")}）"
        : $"  result : {plates.Count} 张，{watch.Elapsed.TotalMilliseconds:F1} ms（limit {ocrOptions.DetLimitSide}）");

    if (iters > 1 && bestMs > 0)
        Console.WriteLine($"  rec    : best {bestMs:F2} ms / avg {avgMs:F2} ms  (n={iters}, beam {readerOptions.Decoder.BeamWidth})");

    foreach (OnnxSharp.Ocr.Plate.PlateResult plate in plates)
    {
        string kind = plate.Plate?.Kind.ToString() ?? "-";
        Console.WriteLine($"    {plate.Text,-10} {plate.Score,6:P1}  {kind,-9}  [{plate.X},{plate.Y} {plate.Width}x{plate.Height}]");

        if (plate.Decoding.CharScores.Length > 0)
        {
            Console.Write("        ");
            for (int i = 0; i < plate.Decoding.CharScores.Length && i < plate.Text.Length; i++)
                Console.Write($"{plate.Text[i]}{plate.Decoding.CharScores[i],5:P0} ");
            Console.WriteLine();
        }
    }

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

/// <summary>
/// 命名形状 <c>--shape name=1,96,560</c> → 字典，供多输入模型按名字覆盖
/// （如 ASR encoder 的 <c>speech</c> + <c>speech_lengths</c>：两者形状不同，共用一个 <c>--shape</c> 会失败）。
/// 没有命名形状时返回 null。
/// </summary>
static Dictionary<string, int[]>? NamedShapes(CliOptions options)
{
    Dictionary<string, int[]>? named = null;

    foreach (string text in options.Shapes)
    {
        int separator = text.IndexOf('=');
        if (separator <= 0)
            continue;

        var dims = ParseShape(text[(separator + 1)..]);
        if (dims is null)
            continue;

        named ??= new Dictionary<string, int[]>(StringComparer.Ordinal);
        named[text[..separator]] = dims;
    }

    return named;
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

static IReadOnlyList<string>? LoadLabels(string? path)
    => path is null
        ? null
        : path.Equals("coco", StringComparison.OrdinalIgnoreCase)
            ? CocoLabels.Names
            : File.ReadAllLines(path).Select(line => line.Trim()).ToArray();

static string LabelOf(IReadOnlyList<string>? labels, int classId)
    => labels is not null && classId >= 0 && classId < labels.Count ? labels[classId] : $"class{classId}";

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
    IReadOnlyList<string>? labels)
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
        if (labels is not null && detection.ClassId >= 0 && detection.ClassId < labels.Count)
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

static List<Tensor> BindInputs(
    Net net,
    List<string> inputs,
    int[]? shapeOverride = null,
    Dictionary<string, int[]>? namedShapes = null)
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

        // 命名形状按输入名优先（多输入模型每个输入形状不同）；否则退回单一 --shape
        int[]? shapeForInput = namedShapes is not null && namedShapes.TryGetValue(name, out var named)
            ? named
            : shapeOverride;

        var tensor = LoadTensor(net, name, file, shapeForInput);
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
    bytes.CopyTo(tensor.Bytes());
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
