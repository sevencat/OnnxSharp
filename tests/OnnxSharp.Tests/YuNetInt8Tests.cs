using OnnxSharp.Core;
using OnnxSharp.Tensors;

namespace OnnxSharp.Tests;

/// <summary>
/// YuNet 的 int8 版（`onnx.quantize` 导出的 **QOperator** 图：`QLinearConv`×53 / `QLinearSigmoid`×6 /
/// `QLinearAdd`×2 / 量化 `MaxPool`×4 / `Quantize/DequantizeLinear`）。
/// <para>
/// 夹具由两个脚本产生，缺一个就 SKIP：
/// <code>
/// python tools/scripts/yunet_int8_probe.py          # 录 ORT 的 12 个输出到 .tmp/yunet/int8_&lt;name&gt;.f32
/// </code>
/// 输入是 `.tmp/yunet/yunet_input.f32`（1×3×640×640，已按 YuNet 的预处理做好）。
/// </para>
/// <para>
/// 容差按实测定：我们的 QLinearConv 走「反量化→F32 卷积→重量化」，与 ORT 的整数路径在
/// **每次重量化**都可能差 1 个量化步（53 层累积），所以按 mean/max 双阈值卡，
/// 而不是逐位相等。端到端的人脸检测结果另有人工验证（框差 ~10 px）。
/// </para>
/// </summary>
internal static class YuNetInt8Tests
{
    private static readonly string[] Outputs =
    [
        "cls_8", "cls_16", "cls_32", "obj_8", "obj_16", "obj_32",
        "bbox_8", "bbox_16", "bbox_32", "kps_8", "kps_16", "kps_32",
    ];

    public static IEnumerable<TestCase> Cases()
    {
        yield return new("yunet: int8 QOperator graph tracks onnxruntime", Int8MatchesOrt);
    }

    private static void Int8MatchesOrt()
    {
        string? fixture = FindFixtureDirectory();
        if (fixture is null)
            throw new TestSkipException("yunet int8 fixture not found (see YuNetInt8Tests header)");

        string model = Path.Combine(fixture, "yunet_int8.onnx");
        if (!File.Exists(model))
            throw new TestSkipException($"model not found: {model}");

        var net = Net.ReadModel(model);

        var input = Tensor.Allocate(new TensorShape(1, 3, 640, 640), DType.F32);
        File.ReadAllBytes(Path.Combine(fixture, "yunet_input.f32")).CopyTo(input.Buffer, 0);
        net.SetInput(net.InputNames[0], input);

        var outputs = net.ForwardAll();
        Assert.True(outputs.Count >= Outputs.Length, $"output count {outputs.Count}");

        for (int i = 0; i < Outputs.Length; i++)
        {
            string golden = Path.Combine(fixture, $"int8_{Outputs[i]}.f32");
            if (!File.Exists(golden))
                throw new TestSkipException($"golden not found: {golden}");

            var actual = Unwrap(outputs[i]);
            var bytes = File.ReadAllBytes(golden);
            var expected = new float[bytes.Length / sizeof(float)];
            Buffer.BlockCopy(bytes, 0, expected, 0, bytes.Length);

            Assert.Equal(expected.Length, actual.Length, $"{Outputs[i]} length");

            double mean = 0;
            double worst = 0;
            for (int k = 0; k < expected.Length; k++)
            {
                double diff = Math.Abs(actual[k] - expected[k]);
                mean += diff;
                worst = Math.Max(worst, diff);
            }

            mean /= expected.Length;
            Console.WriteLine($"        yunet-int8 {Outputs[i],-8} mean|d|={mean:E2}  max|d|={worst:E2}");

            Assert.True(mean <= 2e-2, $"{Outputs[i]}: mean |diff| {mean:E3} > 2e-2");
            Assert.True(worst <= 0.6, $"{Outputs[i]}: max |diff| {worst:E3} > 0.6");
        }
    }

    /// <summary>输出可能带 1 个长度 1 的前导维（<c>[1,1,1,1]</c> 之类），压平后比较。</summary>
    private static Span<float> Unwrap(Tensor tensor) => tensor.AsSpan<float>();

    private static string? FindFixtureDirectory()
    {
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); dir is not null; dir = dir.Parent)
        {
            if (!File.Exists(Path.Combine(dir.FullName, "OnnxSharp.slnx")))
                continue;

            string candidate = Path.Combine(dir.FullName, ".tmp", "yunet");
            return File.Exists(Path.Combine(candidate, "yunet_input.f32"))
                   && File.Exists(Path.Combine(candidate, "int8_cls_8.f32"))
                ? candidate
                : null;
        }

        return null;
    }
}
