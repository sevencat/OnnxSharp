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

    var feeds = BindInputs(net, options.Inputs);
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
    BindInputs(net, options.Inputs);

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
    BindInputs(net, options.Inputs);

    int iters = Math.Max(1, options.GetInt("--iters", 2));
    int top = options.GetInt("--top", 20);

    var watch = System.Diagnostics.Stopwatch.StartNew();
    var timings = net.ProfileAll(iters);
    watch.Stop();

    double total = timings.Sum(t => t.Milliseconds);

    Console.WriteLine($"model     : {path}");
    Console.WriteLine($"ops       : {net.MainGraph?.Prog.Count}    iters: {iters}    {watch.Elapsed.TotalMilliseconds:F0} ms");
    Console.WriteLine($"total     : {total:F1} ms  ({total / iters:F1} ms/iter)");
    Console.WriteLine();
    Console.WriteLine($"{"#",4}  {"type",-18}{"avg ms",10}{"share",8}  name");

    int index = 0;
    foreach (var timing in timings.Take(top < 1 ? timings.Count : top))
    {
        Console.WriteLine($"{++index,4}  {timing.Type,-18}{timing.Average,10:F2}{timing.Milliseconds / total * 100,7:F1}%  {timing.Name}");
    }

    // 按算子类型汇总：热点分散时（几百个 op 各占 1%）看这个更有用
    Console.WriteLine();
    Console.WriteLine($"{"type",-20}{"ops",6}{"avg ms",10}{"share",8}");

    var byType = timings
        .GroupBy(t => t.Type)
        .Select(g => new
        {
            Type = g.Key,
            Ops = g.Count(),
            Milliseconds = g.Sum(t => t.Milliseconds) / iters,
            Share = g.Sum(t => t.Milliseconds) / total * 100,
        })
        .OrderByDescending(g => g.Milliseconds);

    foreach (var group in byType)
        Console.WriteLine($"{group.Type,-20}{group.Ops,6}{group.Milliseconds,10:F1}{group.Share,7:F1}%");

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
    BindInputs(net, options.Inputs);

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
static List<Tensor> BindInputs(Net net, List<string> inputs)
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

        var tensor = LoadTensor(net, name, file);
        net.SetInput(name, tensor);
        tensors.Add(tensor);
    }

    return tensors;
}

static Tensor LoadTensor(Net net, string name, string file)
{
    var data = net.Impl.GetArgData(net.Impl.Require(name));

    if (data.DType != DType.F32)
        throw new OnnxSharpException($"input '{name}': only F32 is supported for --input (got {data.DType})");

    long count = data.Shape.Total();
    if (count <= 0)
        throw new OnnxSharpException($"input '{name}': shape {data.Shape} is not static; cannot bind a raw tensor");

    var bytes = File.ReadAllBytes(file);
    long expected = count * sizeof(float);

    if (bytes.Length != expected)
        throw new OnnxSharpException(
            $"input '{name}': {file} is {bytes.Length} B, expected {expected} B for {data.Shape}");

    var tensor = Tensor.Allocate(data.Shape, DType.F32);
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
