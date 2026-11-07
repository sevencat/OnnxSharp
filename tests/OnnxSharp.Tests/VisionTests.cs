using OnnxSharp.Tensors;
using OnnxSharp.Vision;

namespace OnnxSharp.Tests;

/// <summary>
/// letterbox / NMS / decode 三条路径的纯数值校验：
/// 不加载模型，全部用手工构造的像素和检测头张量，失败点能直接定位到公式。
/// </summary>
internal static class VisionTests
{
    public static IEnumerable<TestCase> Cases()
    {
        yield return new("vision: letterbox 16:9 -> square pads top/bottom", LetterboxPadsTopBottom);
        yield return new("vision: letterbox fills padding with 114", LetterboxPadValue);
        yield return new("vision: letterbox downsample averages the block", LetterboxAveragesBlock);
        yield return new("vision: letterbox honours bgr order", LetterboxBgr);
        yield return new("vision: iou of identical and disjoint boxes", IouBasics);
        yield return new("vision: nms keeps the best of overlapping same class", NmsSuppressesSameClass);
        yield return new("vision: nms keeps overlapping different classes", NmsKeepsDifferentClasses);
        yield return new("vision: nms class-agnostic suppresses across classes", NmsClassAgnostic);
        yield return new("vision: decode channels-first maps back to source", DecodeChannelsFirst);
        yield return new("vision: decode channels-last matches channels-first", DecodeChannelsLast);
        yield return new("vision: decode end-to-end treats coords as corners", DecodeEndToEnd);
        yield return new("vision: decode clips boxes to image bounds", DecodeClips);
        yield return new("vision: decode rejects ambiguous layout", DecodeRejectsAmbiguous);
    }

    // ---------------- letterbox ----------------

    private static void LetterboxPadsTopBottom()
    {
        var pixels = new byte[1280 * 720 * 3];
        var (input, transform) = ImagePreprocessor.Letterbox(pixels, 1280, 720, 640, 640);

        Require(input.Shape.Rank == 4 && input.Shape[1] == 3, $"unexpected input shape {input.Shape}");
        Require(input.Shape[2] == 640 && input.Shape[3] == 640, $"unexpected input shape {input.Shape}");

        // 1280x720 缩到 640x360，上下各填 140。
        Near(transform.ScaleX, 0.5f, "scaleX");
        Near(transform.ScaleY, 0.5f, "scaleY");
        Require(transform.PadX == 0, $"padX {transform.PadX}");
        Require(transform.PadY == 140, $"padY {transform.PadY}");
    }

    private static void LetterboxPadValue()
    {
        var pixels = White(4, 4);
        var (input, transform) = ImagePreprocessor.Letterbox(pixels, 4, 4, 8, 4);

        // 4x4 不缩放，左右各填 2 列。
        Require(transform.PadX == 2 && transform.PadY == 0, $"unexpected pad {transform.PadX},{transform.PadY}");

        var data = input.AsSpan<float>();

        Near(data[0], ImagePreprocessor.PadValue / 255f, "left pad pixel");
        Near(data[2], 1f, "first image pixel");
        Near(data[5], 1f, "last image pixel");
    }

    private static void LetterboxAveragesBlock()
    {
        // 4x4 图，每 2x2 一块同色，缩到 2x2 后每个输出点正好是块颜色。
        var pixels = new byte[4 * 4 * 3];
        SetBlock(pixels, 4, 0, 0, 255, 0, 0);    // 左上 红
        SetBlock(pixels, 4, 2, 0, 0, 255, 0);    // 右上 绿
        SetBlock(pixels, 4, 0, 2, 0, 0, 255);    // 左下 蓝
        SetBlock(pixels, 4, 2, 2, 255, 255, 0);  // 右下 黄

        var (input, _) = ImagePreprocessor.Letterbox(pixels, 4, 4, 2, 2);
        var data = input.AsSpan<float>();
        int plane = 2 * 2;

        Near(data[0], 1f, "red.r");
        Near(data[plane], 0f, "red.g");
        Near(data[1], 0f, "green.r");
        Near(data[plane + 1], 1f, "green.g");
        Near(data[2], 0f, "blue.r");
        Near(data[2 * plane + 2], 1f, "blue.b");
    }

    private static void LetterboxBgr()
    {
        var pixels = new byte[2 * 2 * 3];
        for (int i = 0; i < 4; i++)
        {
            pixels[i * 3] = 10;     // R
            pixels[i * 3 + 1] = 20; // G
            pixels[i * 3 + 2] = 30; // B
        }

        var (rgb, _) = ImagePreprocessor.Letterbox(pixels, 2, 2, 2, 2);
        var (bgr, _) = ImagePreprocessor.Letterbox(pixels, 2, 2, 2, 2, bgr: true);

        var source = rgb.AsSpan<float>();
        var swapped = bgr.AsSpan<float>();
        int plane = 2 * 2;

        Near(source[0], 10 / 255f, "rgb.r");
        Near(source[2 * plane], 30 / 255f, "rgb.b");
        Near(swapped[0], 30 / 255f, "bgr.r");
        Near(swapped[2 * plane], 10 / 255f, "bgr.b");
    }

    // ---------------- iou / nms ----------------

    private static void IouBasics()
    {
        var a = new Detection(0, 1f, 0, 0, 10, 10);

        Near(Detection.IoU(a, a), 1f, "self iou");
        Near(Detection.IoU(a, new Detection(0, 1f, 20, 20, 10, 10)), 0f, "disjoint iou");
        // 交 5x5=25，并 100+100-25=175
        Near(Detection.IoU(a, new Detection(0, 1f, 5, 5, 10, 10)), 25f / 175f, "partial iou");
    }

    private static void NmsSuppressesSameClass()
    {
        var kept = DetectionNms.Suppress(new List<Detection>
        {
            new(0, 0.6f, 0, 0, 10, 10),
            new(0, 0.9f, 1, 1, 10, 10),   // 与第一个重叠，分数更高 → 保留
            new(0, 0.5f, 100, 100, 10, 10),
        }, iouThreshold: 0.5f);

        Require(kept.Count == 2, $"expected 2 kept, got {kept.Count}");
        Near(kept[0].Score, 0.9f, "first kept should be the highest score");
        Near(kept[1].Score, 0.5f, "second kept");
    }

    private static void NmsKeepsDifferentClasses()
    {
        var kept = DetectionNms.Suppress(new List<Detection>
        {
            new(0, 0.9f, 0, 0, 10, 10),
            new(1, 0.8f, 1, 1, 10, 10),
        }, iouThreshold: 0.5f);

        Require(kept.Count == 2, $"classes must not suppress each other, got {kept.Count}");
    }

    private static void NmsClassAgnostic()
    {
        var kept = DetectionNms.Suppress(new List<Detection>
        {
            new(0, 0.9f, 0, 0, 10, 10),
            new(1, 0.8f, 1, 1, 10, 10),
        }, iouThreshold: 0.5f, classAgnostic: true);

        Require(kept.Count == 1, $"class-agnostic should keep one box, got {kept.Count}");
        Require(kept[0].ClassId == 0, "should keep the higher score");
    }

    // ---------------- decode ----------------

    private static void DecodeChannelsFirst()
    {
        // 原图 200x100 经 letterbox 到 100x100：scale 0.5、上填 25 行。
        var transform = new Letterbox(200, 100, 100, 100, 0.5f, 0.5f, 0, 25);

        var output = Tensor.Allocate(new TensorShape(1, 6, 3), DType.F32);
        var data = output.AsSpan<float>();

        // anchor0：class 0，0.9；anchor1/2：class 1，重叠 → 后者被 NMS 掉。
        WriteAnchorColumn(data, 3, 0, 20, 30, 10, 8, 0.9f, 0.1f);
        WriteAnchorColumn(data, 3, 1, 50, 50, 10, 10, 0.1f, 0.8f);
        WriteAnchorColumn(data, 3, 2, 50, 50, 10, 10, 0.1f, 0.7f);

        var detections = YoloDecoder.Decode(output, transform, new DetectorOptions { Confidence = 0.25f, ClassCount = 2 });

        Require(detections.Count == 2, $"expected 2 detections, got {detections.Count}");

        var first = detections[0];
        Require(first.ClassId == 0, $"first class {first.ClassId}");
        Near(first.Score, 0.9f, "first score");
        Near(first.X, 30f, "first x");
        Near(first.Y, 2f, "first y");
        Near(first.Width, 20f, "first width");
        Near(first.Height, 16f, "first height");

        var second = detections[1];
        Require(second.ClassId == 1, $"second class {second.ClassId}");
        Near(second.X, 90f, "second x");
        Near(second.Y, 40f, "second y");
    }

    private static void DecodeChannelsLast()
    {
        var transform = new Letterbox(200, 100, 100, 100, 0.5f, 0.5f, 0, 25);

        // 同样的数据，按 [1, N, 4+C] 放。
        var output = Tensor.Allocate(new TensorShape(1, 3, 6), DType.F32);
        var data = output.AsSpan<float>();
        WriteRow(data, 0, 20, 30, 10, 8, 0.9f, 0.1f);
        WriteRow(data, 1, 50, 50, 10, 10, 0.1f, 0.8f);
        WriteRow(data, 2, 50, 50, 10, 10, 0.1f, 0.7f);

        var detections = YoloDecoder.Decode(output, transform, new DetectorOptions { Confidence = 0.25f, ClassCount = 2 });

        Require(detections.Count == 2, $"expected 2 detections, got {detections.Count}");
        Near(detections[0].X, 30f, "first x");
        Near(detections[0].Y, 2f, "first y");
    }

    private static void DecodeEndToEnd()
    {
        var transform = new Letterbox(200, 100, 100, 100, 0.5f, 0.5f, 0, 25);

        var output = Tensor.Allocate(new TensorShape(1, 2, 6), DType.F32);
        var data = output.AsSpan<float>();

        // x1, y1, x2, y2, score, class —— 端到端输出已经是角点。
        WriteRow(data, 0, 10, 20, 30, 40, 0.75f, 3f);
        WriteRow(data, 1, 0, 0, 5, 5, 0.10f, 3f);   // 低于阈值

        var detections = YoloDecoder.Decode(output, transform, new DetectorOptions { Confidence = 0.25f });

        Require(detections.Count == 1, $"expected 1 detection, got {detections.Count}");
        Require(detections[0].ClassId == 3, $"class {detections[0].ClassId}");
        Near(detections[0].X, 20f, "x");         // (10 - 0) / 0.5
        Near(detections[0].Y, 0f, "y");          // (20 - 25) / 0.5 = -10 → 裁到 0
        Near(detections[0].Width, 40f, "width"); // (30 - 10) / 0.5
        Near(detections[0].Height, 30f, "height"); // (40 - 25) / 0.5 - 0
    }

    private static void DecodeClips()
    {
        var transform = new Letterbox(100, 100, 100, 100, 1f, 1f, 0, 0);

        var output = Tensor.Allocate(new TensorShape(1, 6, 1), DType.F32);
        var data = output.AsSpan<float>();
        WriteAnchorColumn(data, 1, 0, 50, 50, 200, 200, 0.9f, 0.1f);

        var detections = YoloDecoder.Decode(output, transform, new DetectorOptions { Confidence = 0.25f, ClassCount = 2 });

        Require(detections.Count == 1, $"expected 1 detection, got {detections.Count}");
        var box = detections[0];
        Near(box.X, 0f, "clipped x");
        Near(box.Y, 0f, "clipped y");
        Near(box.Width, 100f, "clipped width");
        Near(box.Height, 100f, "clipped height");
    }

    private static void DecodeRejectsAmbiguous()
    {
        var transform = new Letterbox(10, 10, 10, 10, 1f, 1f, 0, 0);
        var output = Tensor.Allocate(new TensorShape(1, 6, 6), DType.F32);

        bool threw = false;
        try
        {
            YoloDecoder.Decode(output, transform, new DetectorOptions());
        }
        catch (ArgumentException)
        {
            threw = true;
        }

        Require(threw, "6x6 output must be rejected as ambiguous");
    }

    // ---------------- helpers ----------------

    private static byte[] White(int width, int height)
    {
        var pixels = new byte[width * height * 3];
        Array.Fill(pixels, (byte)255);
        return pixels;
    }

    private static void SetBlock(byte[] pixels, int width, int x, int y, byte r, byte g, byte b)
    {
        for (int dy = 0; dy < 2; dy++)
        {
            for (int dx = 0; dx < 2; dx++)
            {
                int index = ((y + dy) * width + x + dx) * 3;
                pixels[index] = r;
                pixels[index + 1] = g;
                pixels[index + 2] = b;
            }
        }
    }

    /// <summary>写一个 ChannelsFirst（<c>[field, anchor]</c>）布局的锚点，两个类别分。</summary>
    private static void WriteAnchorColumn(
        Span<float> data, int anchors, int anchor,
        float cx, float cy, float w, float h, float score0, float score1)
    {
        data[0 * anchors + anchor] = cx;
        data[1 * anchors + anchor] = cy;
        data[2 * anchors + anchor] = w;
        data[3 * anchors + anchor] = h;
        data[4 * anchors + anchor] = score0;
        data[5 * anchors + anchor] = score1;
    }

    /// <summary>写一行（<c>[anchor, field]</c> 布局，6 个字段）。</summary>
    private static void WriteRow(
        Span<float> data, int row, float f0, float f1, float f2, float f3, float f4, float f5)
    {
        int start = row * 6;
        data[start] = f0;
        data[start + 1] = f1;
        data[start + 2] = f2;
        data[start + 3] = f3;
        data[start + 4] = f4;
        data[start + 5] = f5;
    }

    private static void Require(bool condition, string message)
    {
        if (!condition)
            throw new InvalidOperationException(message);
    }

    private static void Near(float actual, float expected, string what)
    {
        if (MathF.Abs(actual - expected) > 1e-4f)
            throw new InvalidOperationException($"{what}: expected {expected}, got {actual}");
    }
}
