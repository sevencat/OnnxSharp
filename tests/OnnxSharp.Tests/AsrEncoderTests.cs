using OnnxSharp.Core;
using OnnxSharp.Tensors;

namespace OnnxSharp.Tests;

/// <summary>
/// ASR 回归：sherpa Paraformer 的 encoder 子图（切到第一个 <c>Loop</c> 之前）。
/// <para>
/// 这一条同时覆盖 <c>DynamicQuantizeLinear</c> / <c>MatMulInteger</c> / <c>Pad</c> / 1-D <c>Conv</c>
/// 与「运行期形状」链路。模型与金标准由脚本产生（见文件头注释），缺失则 SKIP：
/// <code>
/// python tools/scripts/split_before_control_flow.py model.int8.onnx .tmp/sherpa/encoder.onnx ^
///        /predictor/Concat_5_output_0,/predictor/Add_output_0
/// python tools/scripts/sherpa_encoder_probe.py .tmp/sherpa/encoder.onnx .tmp/sherpa 96
/// </code>
/// </para>
/// </summary>
internal static class AsrEncoderTests
{
    /// <summary>主输出（encoder 隐藏状态）的相对容差：CIF 会放大 1e-5 级的差异。</summary>
    private const float MeanTolerance = 3e-3f;

    public static IEnumerable<TestCase> Cases()
    {
        yield return new("asr: sherpa encoder sub-graph runs and tracks ORT", EncoderMatchesOrt);
    }

    private static void EncoderMatchesOrt()
    {
        string? directory = FindFixture();
        if (directory is null)
            throw new TestSkipException("encoder fixture not found (see AsrEncoderTests header)");

        var net = Net.ReadModel(
            Path.Combine(directory, "encoder.onnx"),
            new Dictionary<string, int[]>
            {
                ["speech"] = [1, 96, 560],
                ["speech_lengths"] = [1],
            });

        var speech = Tensor.Allocate(new TensorShape(1, 96, 560), DType.F32);
        File.ReadAllBytes(Path.Combine(directory, "speech.f32")).CopyTo(speech.Buffer, 0);

        var lengths = Tensor.Allocate(new TensorShape(1), DType.I32);
        File.ReadAllBytes(Path.Combine(directory, "speech_lengths.i32")).CopyTo(lengths.Buffer, 0);

        net.SetInput("speech", speech);
        net.SetInput("speech_lengths", lengths);

        var outputs = net.ForwardAll();

        // 按名字找 encoder 特征（encoder.onnx 的输出个数会随切图参数变化：
        // 现在还会吐出 decoder cross-attention 需要的量化 K/V 三个张量）
        int index = net.OutputNames.ToList().IndexOf("/predictor/Concat_5_output_0");
        Assert.True(index >= 0, "encoder output not found");

        var frame = outputs[index].AsSpan<float>();
        var golden = ReadGolden(Path.Combine(directory, "golden__predictor_Concat_5_output_0.f32"));
        Assert.Equal(golden.Length, frame.Length, "encoder output length");

        double sum = 0;
        float max = 0;
        for (int i = 0; i < golden.Length; i++)
        {
            float diff = MathF.Abs(golden[i] - frame[i]);
            sum += diff;
            if (diff > max) max = diff;
        }

        double mean = sum / golden.Length;
        Console.WriteLine($"        encoder vs ORT: mean|diff|={mean:E2}  max|diff|={max:E2}");
        Assert.True(mean < MeanTolerance, $"mean|diff| {mean:E2} >= {MeanTolerance:E2}");
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

            string candidate = Path.Combine(dir.FullName, ".tmp", "sherpa");
            return File.Exists(Path.Combine(candidate, "encoder.onnx")) &&
                   File.Exists(Path.Combine(candidate, "golden__predictor_Concat_5_output_0.f32"))
                ? candidate
                : null;
        }

        return null;
    }
}
