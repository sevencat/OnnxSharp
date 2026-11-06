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
