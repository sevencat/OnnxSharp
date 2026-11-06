using System.Diagnostics;
using System.Runtime.InteropServices;
using OnnxSharp.Core;
using OnnxSharp.Tensors;

namespace OnnxSharp.Tests;

/// <summary>
/// M2 验收：yolo11n 端到端跑通，并与 onnxruntime 的 golden 逐元素对比。
/// golden 由 <c>tools/scripts/gen_golden.py</c> 生成。
/// </summary>
internal static class YoloGoldenTests
{
    private const string Model = @"F:\ai\models\yolo\yolo11n.onnx";
    private const string GoldenPrefix = @"F:\ai\models\yolo\golden\yolo11n";

    /// <summary>卷积累加顺序不同会带来微小误差；yolo 输出量级约 0~640。</summary>
    private const float MaxAbsoluteError = 2e-2f;

    public static IEnumerable<TestCase> Cases()
    {
        yield return new("yolo11n: end-to-end matches onnxruntime golden", MatchesGolden);
    }

    private static void MatchesGolden()
    {
        if (!File.Exists(Model))
            throw new TestSkipException($"model not found: {Model}");

        var inputPath = GoldenPrefix + ".input0.bin";
        var outputPath = GoldenPrefix + ".output0.bin";
        if (!File.Exists(inputPath) || !File.Exists(outputPath))
            throw new TestSkipException($"golden not found: {GoldenPrefix}.* (run tools/scripts/gen_golden.py)");

        var net = Net.ReadModel(Model);
        var input = ReadFloats(inputPath, new TensorShape(1, 3, 640, 640));
        net.SetInput("images", input);

        var watch = Stopwatch.StartNew();
        var output = net.Forward();
        watch.Stop();

        var expected = ReadFloats(outputPath, output.Shape).AsSpan<float>();
        var actual = output.AsSpan<float>();

        Assert.Equal(expected.Length, actual.Length, "output element count");

        double maxAbs = 0;
        double sumAbs = 0;
        int worstIndex = 0;

        for (int i = 0; i < expected.Length; i++)
        {
            double diff = Math.Abs((double)expected[i] - actual[i]);
            sumAbs += diff;
            if (diff > maxAbs)
            {
                maxAbs = diff;
                worstIndex = i;
            }
        }

        double meanAbs = sumAbs / expected.Length;
        Console.WriteLine($"        forward {watch.ElapsedMilliseconds} ms   " +
                          $"max|diff|={maxAbs:E3}   mean|diff|={meanAbs:E3}   " +
                          $"worst@{worstIndex}: want={expected[worstIndex]:F6} got={actual[worstIndex]:F6}");

        Assert.True(maxAbs <= MaxAbsoluteError,
            $"max absolute error {maxAbs:F4} exceeds {MaxAbsoluteError} at index {worstIndex}");
    }

    private static Tensor ReadFloats(string path, TensorShape shape)
    {
        var tensor = Tensor.Allocate(shape, DType.F32);
        var bytes = File.ReadAllBytes(path);
        var target = MemoryMarshal.AsBytes(tensor.AsSpan<float>());

        Assert.Equal(target.Length, bytes.Length, $"byte length of {Path.GetFileName(path)}");
        bytes.CopyTo(target);
        return tensor;
    }
}
