using OnnxSharp.Core;
using OnnxSharp.Tensors;

namespace OnnxSharp.Tests;

/// <summary>
/// M10-03/06：量化算子（<c>DynamicQuantizeLinear</c> / <c>MatMulInteger</c>）的单元与端到端用例。
/// 端到端用 <c>tools/scripts/make_quant_model.py</c> 生成的模型 + ORT 金标准（缺失则 SKIP）。
/// </summary>
internal static class QuantOpsTests
{
    public static IEnumerable<TestCase> Cases()
    {
        yield return new("quant: DynamicQuantizeLinear follows the ONNX spec", DynamicQuantizeLinearSpec);
        yield return new("quant: MatMulInteger matches integer arithmetic", MatMulIntegerMatchesIntegers);
        yield return new("quant: quantized graph matches onnxruntime", MatchesOnnxRuntimeGolden);
    }

    private static void DynamicQuantizeLinearSpec()
    {
        var impl = new NetImpl();
        var x = impl.Declare("x", ArgKind.Input);
        var graph = impl.NewGraph("g", new[] { x }, isMain: true);

        var layer = LayerRegistry.Create("DynamicQuantizeLinear",
            new LayerParams { Type = "DynamicQuantizeLinear", Name = "dql" });
        layer.Inputs.Add(x);

        var y = impl.Declare("y", ArgKind.Temp);
        var scale = impl.Declare("scale", ArgKind.Temp);
        var zero = impl.Declare("zp", ArgKind.Temp);
        layer.Outputs.Add(y);
        layer.Outputs.Add(scale);
        layer.Outputs.Add(zero);
        graph.Append(layer);
        graph.SetOutputs(new[] { y, scale, zero });

        impl.SetInput("x", Tensor.FromArray([-1f, 0f, 1f, 2f], new TensorShape(4)));
        var outputs = impl.ForwardAll();

        // x_min = -1, x_max = 2 → scale = 3/255, zp = round(1/scale) = 85
        float expectedScale = 3f / 255f;
        Assert.Near(expectedScale, outputs[1].AsSpan<float>()[0], 1e-7f, "scale");
        Assert.Equal((byte)85, outputs[2].AsSpan<byte>()[0], "zero point");

        var values = outputs[0].AsSpan<byte>();
        Assert.Equal((byte)0, values[0], "-1 → 0");
        Assert.Equal((byte)85, values[1], "0 → 85");
        Assert.Equal((byte)170, values[2], "1 → 170");
        Assert.Equal((byte)255, values[3], "2 → 255");
    }

    private static void MatMulIntegerMatchesIntegers()
    {
        var impl = new NetImpl();
        var a = impl.Declare("a", ArgKind.Input);
        var graph = impl.NewGraph("g", new[] { a }, isMain: true);

        var b = impl.NewConstArg("b", ToU8([1, 2, 3, 4, 5, 6], 3, 2));

        var layer = LayerRegistry.Create("MatMulInteger", new LayerParams { Type = "MatMulInteger", Name = "mmi" });
        layer.Inputs.Add(a);
        layer.Inputs.Add(b);
        var c = impl.Declare("c", ArgKind.Temp);
        layer.Outputs.Add(c);
        graph.Append(layer);
        graph.SetOutputs(new[] { c });

        impl.SetInput("a", ToU8([1, 2, 3, 4, 5, 6], 2, 3));
        var result = impl.Forward().AsSpan<int>().ToArray();

        // [[1,2,3],[4,5,6]] × [[1,2],[3,4],[5,6]]
        Assert.Equal(22, result[0], "[0,0]");
        Assert.Equal(28, result[1], "[0,1]");
        Assert.Equal(49, result[2], "[1,0]");
        Assert.Equal(64, result[3], "[1,1]");
    }

    private static void MatchesOnnxRuntimeGolden()
    {
        string? directory = FindQuantFixture();
        if (directory is null)
            throw new TestSkipException("quant fixture not found (run tools/scripts/make_quant_model.py .tmp/quant)");

        var net = Net.ReadModel(Path.Combine(directory, "quant.onnx"));
        var input = Tensor.Allocate(new TensorShape(2, 4), DType.F32);

        byte[] inputBytes = File.ReadAllBytes(Path.Combine(directory, "quant_x.f32"));
        inputBytes.CopyTo(input.Buffer, 0);

        net.SetInput("x", input);
        var output = net.Forward().AsSpan<float>();

        byte[] goldenBytes = File.ReadAllBytes(Path.Combine(directory, "quant_golden.f32"));
        var golden = new float[goldenBytes.Length / sizeof(float)];
        Buffer.BlockCopy(goldenBytes, 0, golden, 0, goldenBytes.Length);

        Assert.Equal(golden.Length, output.Length, "output length");
        for (int i = 0; i < golden.Length; i++)
            Assert.Near(golden[i], output[i], 1e-5f, $"value[{i}]");

        Console.WriteLine($"        quant graph vs ORT: {golden.Length} values, max|diff| <= 1e-5");
    }

    private static string? FindQuantFixture()
    {
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); dir is not null; dir = dir.Parent)
        {
            if (!File.Exists(Path.Combine(dir.FullName, "OnnxSharp.slnx")))
                continue;

            string candidate = Path.Combine(dir.FullName, ".tmp", "quant");
            return File.Exists(Path.Combine(candidate, "quant.onnx")) ? candidate : null;
        }

        return null;
    }

    private static Tensor ToU8(int[] values, int rows, int columns)
    {
        var tensor = Tensor.Allocate(new TensorShape(rows, columns), DType.U8);
        var span = tensor.AsSpan<byte>();
        for (int i = 0; i < values.Length; i++)
            span[i] = (byte)values[i];
        return tensor;
    }
}
