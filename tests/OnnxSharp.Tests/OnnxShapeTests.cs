using System.Globalization;
using OnnxSharp.Core;
using OnnxSharp.Tensors;

namespace OnnxSharp.Tests;

/// <summary>
/// M1 验收：把导入后的图里每个 Arg 的 shape/dtype 与 <c>onnx.shape_inference</c> 的结果逐一对齐。
/// oracle 由 <c>tools/scripts/dump_shapes.py</c> 生成（plan.md §8.1）。
/// </summary>
internal static class OnnxShapeTests
{
    private const string Yolo11n = @"F:\ai\models\yolo\yolo11n.onnx";

    public static IEnumerable<TestCase> Cases()
    {
        yield return new("shapes: yolo11n matches onnx.shape_inference", Yolo11nShapes);
    }

    private static void Yolo11nShapes()
    {
        if (!File.Exists(Yolo11n))
            throw new TestSkipException($"model not found: {Yolo11n}");

        var oraclePath = Path.Combine(TestPaths.RepoRoot, "tests", "data", "yolo11n.shapes.tsv");
        if (!File.Exists(oraclePath))
            throw new TestSkipException($"oracle not found: {oraclePath} (run tools/scripts/dump_shapes.py)");

        var oracle = ParseOracle(oraclePath);

        var net = Net.ReadModel(Yolo11n);
        var graph = net.MainGraph ?? throw new TestFailureException("no main graph");
        Interpreter.Run(net.Impl, graph, executeLayers: false);

        Assert.Equal(1, net.InputNames.Count, "model input count");
        Assert.Equal("images", net.InputNames[0], "model input name");

        var mismatches = new List<string>();
        int compared = 0;

        foreach (var op in graph.Prog)
        {
            foreach (var arg in op.Outputs)
            {
                if (arg.Idx == 0)
                    continue;

                var data = net.Impl.GetArgData(arg);
                if (!oracle.TryGetValue(data.Name, out var want))
                {
                    mismatches.Add($"{op.Type}: '{data.Name}' missing from oracle");
                    continue;
                }

                compared++;
                var gotDims = data.Shape.ToArray();
                if (ToOnnxCode(data.DType) != want.DType || !SameDims(want.Dims, gotDims))
                {
                    mismatches.Add(
                        $"{op.Type}/{data.Name}: want dtype={want.DType} [{string.Join(",", want.Dims)}], " +
                        $"got {data.DType} [{string.Join(",", gotDims)}]");
                }
            }
        }

        foreach (var output in graph.Outputs)
        {
            var data = net.Impl.GetArgData(output);
            if (!oracle.TryGetValue(data.Name, out var want))
                continue;
            compared++;
            if (ToOnnxCode(data.DType) != want.DType || !SameDims(want.Dims, data.Shape.ToArray()))
                mismatches.Add($"graph output '{data.Name}' mismatch");
        }

        Console.WriteLine($"        compared {compared} tensors against oracle ({oracle.Count} entries)");

        if (mismatches.Count > 0)
        {
            throw new TestFailureException(
                $"{mismatches.Count} shape mismatches, first {Math.Min(10, mismatches.Count)}:\n  "
                + string.Join("\n  ", mismatches.Take(10)));
        }

        Assert.True(compared >= 300, $"too few tensors compared ({compared})");
    }

    // ---------------- 工具 ----------------

    private static bool SameDims(int[] expected, int[] actual)
    {
        if (expected.Length != actual.Length)
            return false;
        for (int i = 0; i < expected.Length; i++)
        {
            if (expected[i] != actual[i])
                return false;
        }
        return true;
    }

    private static int ToOnnxCode(DType dtype) => dtype switch
    {
        DType.F32 => 1,
        DType.U8 => 2,
        DType.I8 => 3,
        DType.I32 => 6,
        DType.I64 => 7,
        DType.Bool => 9,
        DType.F16 => 10,
        _ => -1,
    };

    private static Dictionary<string, (int DType, int[] Dims)> ParseOracle(string path)
    {
        var result = new Dictionary<string, (int, int[])>(StringComparer.Ordinal);

        foreach (var line in File.ReadLines(path))
        {
            if (line.Length == 0)
                continue;

            var parts = line.Split('\t');
            if (parts.Length < 6)
                continue;

            var kind = parts[1];
            if (kind is not ("TENSOR" or "GRAPH_INPUT" or "GRAPH_OUTPUT"))
                continue;

            var name = parts[3];
            int dtype = int.Parse(parts[4], CultureInfo.InvariantCulture);
            int[] dims = parts[5].Length == 0
                ? Array.Empty<int>()
                : Array.ConvertAll(parts[5].Split(','),
                    s => int.Parse(s, CultureInfo.InvariantCulture));

            result[name] = (dtype, dims);
        }

        return result;
    }
}

internal static class TestPaths
{
    public static string RepoRoot { get; } = FindRepoRoot();

    private static string FindRepoRoot()
    {
        var directory = new DirectoryInfo(AppContext.BaseDirectory);
        while (directory is not null)
        {
            if (File.Exists(Path.Combine(directory.FullName, "OnnxSharp.slnx")))
                return directory.FullName;
            directory = directory.Parent;
        }
        return Directory.GetCurrentDirectory();
    }
}
