using OnnxSharp.Core;
using OnnxSharp.Tensors;

namespace OnnxSharp.Tests;

/// <summary>
/// <c>FuseAttention</c>：把整条 multi-head attention（scores → softmax → P·V → 输出重排）
/// 折成一层 <c>FusedAttention</c>。
/// <para>
/// 夹具是静态形状的教科书式 MHA，由 <c>tools/scripts/make_attention_model.py</c> 生成
/// （同时产出 ORT 参考输出），缺失则 SKIP。静态形状会走「3 维输入 + 连 Reshape/Transpose
/// 一起折」那条路；Paraformer 那种动态 head-split 走 head 空间那条路，由
/// <c>AsrEncoderTests</c> / <c>AsrEndToEndTests</c> 覆盖。
/// </para>
/// </summary>
internal static class AttentionTests
{
    public static IEnumerable<TestCase> Cases()
    {
        yield return new("attention: fused MHA matches onnxruntime", FusedMatchesOrt);
    }

    private static void FusedMatchesOrt()
    {
        string? directory = FindFixture();
        if (directory is null)
            throw new TestSkipException(
                "attention fixture not found (run tools/scripts/make_attention_model.py .tmp/attention)");

        var net = Net.ReadModel(Path.Combine(directory, "attention.onnx"));
        var prog = net.MainGraph!.Prog;

        Assert.Equal(1, prog.Count(op => op.Type == "FusedAttention"), "expected exactly one FusedAttention");
        Assert.Equal(0, prog.Count(op => op.Type == "Softmax"), "Softmax must be absorbed");
        Assert.Equal(0, prog.Count(op => op.Type == "Transpose"), "head-split transposes must be absorbed");
        Assert.Equal(0, prog.Count(op => op.Type == "Reshape"), "output reshape must be absorbed");

        var input = Tensor.Allocate(new TensorShape(1, 8, 8), DType.F32);
        File.ReadAllBytes(Path.Combine(directory, "attention_x.f32")).CopyTo(input.Buffer, 0);
        net.SetInput("x", input);

        var output = net.ForwardAll()[0].AsSpan<float>();
        var golden = ReadGolden(Path.Combine(directory, "attention_y.f32"));

        Assert.Equal(golden.Length, output.Length, "output length");

        double sum = 0;
        float max = 0;
        for (int i = 0; i < golden.Length; i++)
        {
            float diff = MathF.Abs(golden[i] - output[i]);
            sum += diff;
            if (diff > max)
                max = diff;
        }

        Console.WriteLine($"        attention vs ORT: mean|diff|={sum / golden.Length:E2}  max|diff|={max:E2}");
        Assert.True(max < 1e-5, $"max|diff| {max:E2} >= 1e-5");
    }

    private static float[] ReadGolden(string path)
    {
        byte[] bytes = File.ReadAllBytes(path);
        var values = new float[bytes.Length / sizeof(float)];
        Buffer.BlockCopy(bytes, 0, values, 0, bytes.Length);
        return values;
    }

    private static string? FindFixture()
    {
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); dir is not null; dir = dir.Parent)
        {
            if (!File.Exists(Path.Combine(dir.FullName, "OnnxSharp.slnx")))
                continue;

            string candidate = Path.Combine(dir.FullName, ".tmp", "attention");
            return File.Exists(Path.Combine(candidate, "attention.onnx")) ? candidate : null;
        }

        return null;
    }
}
