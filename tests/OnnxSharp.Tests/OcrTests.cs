using OnnxSharp.Core;
using OnnxSharp.Ocr.PaddleOCR;
using OnnxSharp.Tensors;

namespace OnnxSharp.Tests;

/// <summary>
/// OCR 相关用例：三个新算子（HardSigmoid / BatchNormalization / Identity）、
/// DB 后处理、CTC 解码、像素归一化，以及（模型在本地时）PP-OCRv6 全动态图的特化与端到端流水线。
/// <para>
/// 模型查找顺序：环境变量 <c>ONNXSHARP_OCR_MODELS</c> → 仓库下 <c>.tmp/ocr-models</c>
/// （由 <c>tools/scripts/extract_ocr_models.ps1</c> 从 nuget 模型包解出）。找不到就 SKIP。
/// </para>
/// </summary>
internal static class OcrTests
{
    public static IEnumerable<TestCase> Cases()
    {
        yield return new("ocr: HardSigmoid matches max(0, min(1, alpha*x+beta))", HardSigmoidMatchesDefinition);
        yield return new("ocr: BatchNormalization applies per-channel affine", BatchNormAppliesAffine);
        yield return new("ocr: Identity copies the input", IdentityCopies);
        yield return new("ocr: DB postprocess turns a probability blob into a box", DbPostprocessFindsBox);
        yield return new("ocr: CTC greedy decode drops blanks and repeats", CtcDecodeDropsBlanksAndRepeats);
        yield return new("ocr: pixel resize normalizes to NCHW", ResizeNormalizes);
        yield return new("ocr: dynamic det graph is specialized to a concrete size", DetGraphIsSpecialized);
        yield return new("ocr: paddleocr pipeline runs end to end", PipelineRuns);
    }

    // ---------------- 算子 ----------------

    private static void HardSigmoidMatchesDefinition()
    {
        var impl = new NetImpl();
        var x = impl.Declare("x", ArgKind.Input);
        var graph = impl.NewGraph("g", new[] { x }, isMain: true);

        var layer = LayerRegistry.Create("HardSigmoid", new LayerParams { Type = "HardSigmoid", Name = "hs" });
        layer.Inputs.Add(x);
        var y = impl.Declare("y", ArgKind.Temp);
        layer.Outputs.Add(y);
        graph.Append(layer);
        graph.SetOutputs(new[] { y });

        float[] input = { -10f, -1f, 0f, 1f, 2f, 10f };
        impl.SetInput("x", Tensor.FromArray(input, new TensorShape(6)));

        var result = impl.Forward().AsSpan<float>();
        for (int i = 0; i < input.Length; i++)
        {
            float expected = MathF.Min(1f, MathF.Max(0f, 0.2f * input[i] + 0.5f));
            Assert.Near(expected, result[i], 1e-6f, $"value[{i}]");
        }
    }

    private static void BatchNormAppliesAffine()
    {
        var impl = new NetImpl();
        var x = impl.Declare("x", ArgKind.Input);
        var graph = impl.NewGraph("g", new[] { x }, isMain: true);

        var layer = LayerRegistry.Create("BatchNormalization", new LayerParams { Type = "BatchNormalization", Name = "bn" });
        layer.Inputs.Add(x);
        layer.Inputs.Add(impl.NewConstArg("scale", Tensor.FromArray(new[] { 2f, 0.5f }, new TensorShape(2))));
        layer.Inputs.Add(impl.NewConstArg("bias", Tensor.FromArray(new[] { 1f, -1f }, new TensorShape(2))));
        layer.Inputs.Add(impl.NewConstArg("mean", Tensor.FromArray(new[] { 1f, 2f }, new TensorShape(2))));
        layer.Inputs.Add(impl.NewConstArg("var", Tensor.FromArray(new[] { 4f, 0.25f }, new TensorShape(2))));

        var y = impl.Declare("y", ArgKind.Temp);
        layer.Outputs.Add(y);
        graph.Append(layer);
        graph.SetOutputs(new[] { y });

        // [1, 2, 1, 2]：通道 0 两个像素、通道 1 两个像素
        impl.SetInput("x", Tensor.FromArray(new[] { 3f, 5f, 1f, 3f }, new TensorShape(1, 2, 1, 2)));

        var result = impl.Forward().AsSpan<float>().ToArray();
        const float eps = 1e-5f;

        float a0 = 2f / MathF.Sqrt(4f + eps);
        float b0 = 1f - 1f * a0;
        float a1 = 0.5f / MathF.Sqrt(0.25f + eps);
        float b1 = -1f - 2f * a1;

        Assert.Near(3f * a0 + b0, result[0], 1e-5f, "channel 0, x=3");
        Assert.Near(5f * a0 + b0, result[1], 1e-5f, "channel 0, x=5");
        Assert.Near(1f * a1 + b1, result[2], 1e-5f, "channel 1, x=1");
        Assert.Near(3f * a1 + b1, result[3], 1e-5f, "channel 1, x=3");
    }

    private static void IdentityCopies()
    {
        var impl = new NetImpl();
        var x = impl.Declare("x", ArgKind.Input);
        var graph = impl.NewGraph("g", new[] { x }, isMain: true);

        var layer = LayerRegistry.Create("Identity", new LayerParams { Type = "Identity", Name = "id" });
        layer.Inputs.Add(x);
        var y = impl.Declare("y", ArgKind.Temp);
        layer.Outputs.Add(y);
        graph.Append(layer);
        graph.SetOutputs(new[] { y });

        impl.SetInput("x", Tensor.FromArray(new[] { 1f, -2f, 3f }, new TensorShape(1, 3, 1, 1)));
        var result = impl.Forward();

        Assert.Equal(new TensorShape(1, 3, 1, 1), result.Shape, "shape");
        var values = result.AsSpan<float>();
        Assert.Near(1f, values[0], 1e-6f, "value[0]");
        Assert.Near(-2f, values[1], 1e-6f, "value[1]");
        Assert.Near(3f, values[2], 1e-6f, "value[2]");
    }

    // ---------------- DB 后处理 / CTC ----------------

    private static void DbPostprocessFindsBox()
    {
        const int width = 40, height = 30;
        var probability = new float[width * height];

        // 一块 11x7 的高概率区域（模拟一行文字）
        for (int y = 10; y < 17; y++)
            for (int x = 12; x < 23; x++)
                probability[y * width + x] = 0.9f;

        var boxes = DbPostprocess.Detect(probability, width, height, 0.3f, 0.6f, 1.6f, minSize: 3);

        Assert.Equal(1, boxes.Count, "box count");
        var box = boxes[0];
        Assert.Near(0.9f, box.Score, 1e-5f, "score");

        // unclip 之后必须比原始区域大，且中心保持不变
        Assert.True(box.Width > 11 && box.Height > 7, $"unclip should grow the box, got {box.Width}x{box.Height}");
        Assert.Near(17.5f, box.X + box.Width / 2, 0.6f, "center x");
        Assert.Near(13.5f, box.Y + box.Height / 2, 0.6f, "center y");
    }

    private static void CtcDecodeDropsBlanksAndRepeats()
    {
        const int classes = 4;              // blank, a, b, space
        string[] dictionary = ["", "a", "b", " "];

        // T=5：blank → a → a → blank → b
        float[,] rows =
        {
            { 0.9f, 0.05f, 0.03f, 0.02f },
            { 0.05f, 0.8f, 0.1f, 0.05f },
            { 0.05f, 0.7f, 0.2f, 0.05f },
            { 0.9f, 0.02f, 0.03f, 0.05f },
            { 0.05f, 0.05f, 0.6f, 0.3f },
        };

        var probabilities = new float[rows.Length * classes];
        for (int t = 0; t < rows.GetLength(0); t++)
            for (int c = 0; c < classes; c++)
                probabilities[t * classes + c] = rows[t, c];

        (string text, float score) = CtcDecoder.Decode(probabilities, classes, dictionary);

        Assert.Equal("ab", text, "decoded text");
        Assert.Near((0.8f + 0.6f) / 2, score, 1e-5f, "mean score of kept steps");
    }

    private static void ResizeNormalizes()
    {
        // 2x2 RGB：红 / 绿 / 蓝 / 白
        byte[] pixels =
        [
            255, 0, 0,    0, 255, 0,
            0, 0, 255,    255, 255, 255,
        ];

        var destination = new float[3 * 2 * 2];
        OcrImage.ToChw(pixels, 2, 2, OcrPixelFormat.Rgb24, 0,
            0, 0, 2, 2, 2, 2,
            OcrImage.Half, OcrImage.Half, destination, 2);

        // 归一化后 (v/255 - 0.5) / 0.5：255 → 1，0 → -1；平面按通道优先排列
        Assert.Near(1f, destination[0], 1e-6f, "R of red pixel");
        Assert.Near(-1f, destination[1], 1e-6f, "R of green pixel");
        Assert.Near(1f, destination[5], 1e-6f, "G of green pixel");
        Assert.Near(1f, destination[10], 1e-6f, "B of blue pixel");
    }

    // ---------------- 模型相关（无模型时 SKIP） ----------------

    private static string? ModelDirectory()
    {
        string? environment = Environment.GetEnvironmentVariable("ONNXSHARP_OCR_MODELS");
        if (!string.IsNullOrEmpty(environment) && File.Exists(Path.Combine(environment, "det.onnx")))
            return environment;

        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); dir is not null; dir = dir.Parent)
        {
            if (!File.Exists(Path.Combine(dir.FullName, "OnnxSharp.slnx")))
                continue;

            string candidate = Path.Combine(dir.FullName, ".tmp", "ocr-models");
            return File.Exists(Path.Combine(candidate, "det.onnx")) ? candidate : null;
        }

        return null;
    }

    private static void DetGraphIsSpecialized()
    {
        string? directory = ModelDirectory();
        if (directory is null)
            throw new TestSkipException("OCR models not found (run tools/scripts/extract_ocr_models.ps1)");

        // det 是全动态导出（x[N,3,H,W]）：给了尺寸才能折叠形状链并存成静态图
        var net = Net.ReadModel(Path.Combine(directory, "det.onnx"), new[] { 1, 3, 64, 64 });
        Assert.Equal(new TensorShape(1, 3, 64, 64), net.Impl.GetArgData(net.Impl.Require("x")).Shape, "input shape");

        net.SetInput("x", Tensor.Allocate(new TensorShape(1, 3, 64, 64), DType.F32));
        var output = net.Forward();

        Assert.Equal(new TensorShape(1, 1, 64, 64), output.Shape, "output shape");
        foreach (float value in output.AsSpan<float>())
        {
            if (value is < 0f or > 1f)
                throw new TestFailureException($"DB output is not a probability: {value}");
        }
    }

    private static void PipelineRuns()
    {
        string? directory = ModelDirectory();
        if (directory is null)
            throw new TestSkipException("OCR models not found (run tools/scripts/extract_ocr_models.ps1)");

        var model = PaddleOcrModel.FromFiles(
            Path.Combine(directory, "det.onnx"),
            Path.Combine(directory, "rec.onnx"),
            Path.Combine(directory, "dict.txt"));

        // PP-OCRv6 tiny 的 rec 输出 6906 类：blank + 字典 + 空格
        Assert.Equal(6906, model.Classes.Count, "class count");

        var pipeline = new PaddleOcrPipeline(model, new PaddleOcrOptions { DetLimitSide = 128 });

        // 白底 + 一条黑杠（没有字库也能覆盖「检测 → 裁剪 → 识别」整条链路）
        const int width = 192, height = 96;
        var pixels = new byte[width * height * 3];
        Array.Fill(pixels, (byte)255);
        for (int y = 40; y < 58; y++)
        {
            for (int x = 20; x < 170; x++)
            {
                int index = (y * width + x) * 3;
                pixels[index] = pixels[index + 1] = pixels[index + 2] = 0;
            }
        }

        IReadOnlyList<OcrLine> lines = pipeline.Run(pixels, width, height, OcrPixelFormat.Rgb24);

        foreach (OcrLine line in lines)
        {
            Assert.True(line.X >= 0 && line.Y >= 0, $"box origin inside image: {line}");
            Assert.True(line.X + line.Width <= width && line.Y + line.Height <= height, $"box inside image: {line}");
            Assert.True(line.Score is >= 0 and <= 1, $"score in [0,1]: {line}");
        }

        // 直接喂一块区域给识别器：即使检测把黑杠丢掉了，rec 模型（CTC + 字典）也必须跑通
        (string text, float score) = pipeline.Recognize(
            pixels, width, height, OcrPixelFormat.Rgb24, 0, 20, 40, 150, 18, rotate: false);
        Assert.True(score is >= 0 and <= 1, $"recognizer score in [0,1]: {score}");
        Assert.True(text.Length <= 40, $"recognizer produced {text.Length} characters on a blank bar");

        Console.WriteLine(
            $"        pipeline: {lines.Count} line(s) on a synthetic bar image, " +
            $"recognizer -> '{text}' ({score:F3})");
    }
}
