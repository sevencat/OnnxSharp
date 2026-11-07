using OnnxSharp.Tensors;

namespace OnnxSharp.Vision;

/// <summary>YOLO 检测头的字段布局。</summary>
public enum OutputLayout
{
    /// <summary>按张量维度推断（<c>[1, 4+C, N]</c> 与 <c>[1, N, 4+C]</c> 靠「哪个维度更小」区分）。</summary>
    Auto,

    /// <summary><c>[1, 4+C, N]</c>：ultralytics 导出的默认布局。</summary>
    ChannelsFirst,

    /// <summary><c>[1, N, 4+C]</c>：部分第三方导出的转置布局。</summary>
    ChannelsLast,

    /// <summary><c>[1, N, 6]</c>：每行 <c>x1, y1, x2, y2, score, class</c>，已内置 NMS（YOLOv10/26 之类）。</summary>
    EndToEnd,
}

/// <summary>检测后处理参数。</summary>
public sealed class DetectorOptions
{
    /// <summary>置信度阈值，低于它的锚点直接丢弃。</summary>
    public float Confidence { get; init; } = 0.25f;

    /// <summary>NMS 的 IoU 阈值（严格大于才抑制）。</summary>
    public float IouThreshold { get; init; } = 0.45f;

    /// <summary>最多返回多少个框。</summary>
    public int MaxDetections { get; init; } = 300;

    /// <summary>参与 NMS 的候选上限（先按分数取前 N 个，防止极端场景下 NMS 退化成平方级）。</summary>
    public int MaxCandidates { get; init; } = 30000;

    /// <summary>true 表示不同类别之间也互相抑制。</summary>
    public bool ClassAgnosticNms { get; init; }

    /// <summary>类别数；0 表示从输出维度推断（<c>4 + ClassCount == 字段数</c>）。</summary>
    public int ClassCount { get; init; }

    /// <summary>检测头布局；<see cref="OutputLayout.Auto"/> 会自动推断。</summary>
    public OutputLayout Layout { get; init; } = OutputLayout.Auto;

    /// <summary>模型输入为动态尺寸时使用的宽（0 表示 640）。</summary>
    public int TargetWidth { get; init; }

    /// <summary>模型输入为动态尺寸时使用的高（0 表示 640）。</summary>
    public int TargetHeight { get; init; }
}

/// <summary>
/// 把 YOLO 检测头解成原图坐标的检测框：逐锚点取最大类别分 → 阈值筛选 → NMS → letterbox 反变换 → 裁剪。
/// <para>
/// 顺序很重要：<b>NMS 必须在模型坐标系里做</b>。先还原到原图再 NMS 会让被裁剪过的框互相「变形」，
/// 重叠关系与原图上的真实重叠不一致。
/// </para>
/// </summary>
public static class YoloDecoder
{
    /// <summary>解码一批锚点并返回裁剪到原图内的检测框（按分数降序）。</summary>
    /// <param name="output">检测头张量，形状为 <c>[1, 4+C, N]</c>、<c>[1, N, 4+C]</c> 或 <c>[1, N, 6]</c>。</param>
    /// <param name="transform">预处理时的 letterbox 参数，用于把坐标还原到原图。</param>
    /// <param name="options">阈值与布局；null 使用默认值。</param>
    /// <exception cref="NotSupportedException">输出不是 F32、不是三维、或批次不为 1。</exception>
    /// <exception cref="ArgumentException">布局无法推断，或类别数与字段数不匹配。</exception>
    public static IReadOnlyList<Detection> Decode(Tensor output, in Letterbox transform, DetectorOptions? options = null)
    {
        options ??= new DetectorOptions();

        if (output.DType != DType.F32)
            throw new NotSupportedException($"detect: output dtype {output.DType} is not supported (F32 only)");

        var shape = output.Shape;
        if (shape.Rank != 3 || shape[0] != 1)
            throw new NotSupportedException($"detect: expected a single-image rank-3 output, got {shape}");

        var layout = ResolveLayout(shape[1], shape[2], options);
        bool channelsFirst = layout == OutputLayout.ChannelsFirst;
        bool endToEnd = layout == OutputLayout.EndToEnd;

        int fields = channelsFirst ? shape[1] : shape[2];
        int anchors = channelsFirst ? shape[2] : shape[1];
        int classFields = options.ClassCount > 0 ? 4 + options.ClassCount : fields;

        if (endToEnd ? fields != 6 : classFields < 5 || classFields > fields)
            throw new ArgumentException(
                $"detect: output {shape} does not match layout {layout} with ClassCount={options.ClassCount}");

        var data = output.AsReadOnlySpan<float>();

        // 两种布局只差一个步长：ChannelsFirst 是 [field, anchor]，ChannelsLast 是 [anchor, field]。
        // 用步长而不是分支，内层循环里就没有 if 了。
        int anchorStride = channelsFirst ? 1 : fields;
        int fieldStride = channelsFirst ? anchors : 1;

        var candidates = new List<Detection>();
        for (int a = 0; a < anchors; a++)
        {
            int rowStart = a * anchorStride;

            int classId;
            float score;

            if (endToEnd)
            {
                score = data[rowStart + 4 * fieldStride];
                float classValue = data[rowStart + 5 * fieldStride];
                if (!float.IsFinite(classValue) || classValue < 0 || classValue != MathF.Truncate(classValue))
                    continue;
                classId = (int)classValue;
            }
            else
            {
                score = float.NegativeInfinity;
                classId = 0;
                for (int c = 4; c < classFields; c++)
                {
                    float value = data[rowStart + c * fieldStride];
                    if (value > score)
                    {
                        score = value;
                        classId = c - 4;
                    }
                }
            }

            if (!float.IsFinite(score) || score < options.Confidence)
                continue;

            float x = data[rowStart];
            float y = data[rowStart + fieldStride];
            float w = data[rowStart + 2 * fieldStride];
            float h = data[rowStart + 3 * fieldStride];
            if (!float.IsFinite(x) || !float.IsFinite(y) || !float.IsFinite(w) || !float.IsFinite(h))
                continue;

            if (endToEnd)
            {
                // 端到端输出给的是左上角 + 右下角。
                w -= x;
                h -= y;
            }
            else
            {
                // 原始检测头给的是中心点 + 宽高。
                x -= w * 0.5f;
                y -= h * 0.5f;
            }

            if (w <= 0 || h <= 0)
                continue;

            candidates.Add(new Detection(classId, score, x, y, w, h));
        }

        if (candidates.Count == 0)
            return Array.Empty<Detection>();

        candidates.Sort((left, right) => right.Score.CompareTo(left.Score));
        if (candidates.Count > options.MaxCandidates)
            candidates.RemoveRange(options.MaxCandidates, candidates.Count - options.MaxCandidates);

        var kept = DetectionNms.SuppressSorted(
            candidates, options.IouThreshold, options.MaxDetections, options.ClassAgnosticNms);

        // 先把框还原到原图并裁到图内，再丢掉被裁成零面积的（完全落在填边区里的框）。
        var result = new List<Detection>(kept.Count);
        foreach (var detection in kept)
        {
            float x1 = Math.Clamp(transform.ToSourceX(detection.X), 0, transform.SourceWidth);
            float y1 = Math.Clamp(transform.ToSourceY(detection.Y), 0, transform.SourceHeight);
            float x2 = Math.Clamp(transform.ToSourceX(detection.X + detection.Width), 0, transform.SourceWidth);
            float y2 = Math.Clamp(transform.ToSourceY(detection.Y + detection.Height), 0, transform.SourceHeight);

            if (x2 <= x1 || y2 <= y1)
                continue;

            result.Add(detection with { X = x1, Y = y1, Width = x2 - x1, Height = y2 - y1 });
        }

        return result;
    }

    private static OutputLayout ResolveLayout(int dimension1, int dimension2, DetectorOptions options)
    {
        if (options.Layout != OutputLayout.Auto)
            return options.Layout;

        int expectedFields = options.ClassCount > 0 ? 4 + options.ClassCount : 0;

        if (expectedFields > 0 && dimension1 == expectedFields && dimension2 != expectedFields)
            return OutputLayout.ChannelsFirst;
        if (expectedFields > 0 && dimension2 == expectedFields && dimension1 != expectedFields)
            return OutputLayout.ChannelsLast;

        // [1, 6, N] 只可能是「2 类 + 通道在前」的检测头（N 远大于 6）。
        if (dimension1 == 6 && dimension2 > 6)
            return OutputLayout.ChannelsFirst;

        // [1, N, 6] 是端到端（YOLOv10/26）。注意它与「2 类 + 通道在后」在形状上无法区分，
        // 后者必须显式指定 Layout = ChannelsLast。
        if (dimension2 == 6 && dimension1 != 6)
            return OutputLayout.EndToEnd;

        if (dimension1 >= 5 && dimension1 < dimension2)
            return OutputLayout.ChannelsFirst;
        if (dimension2 >= 5 && dimension2 < dimension1)
            return OutputLayout.ChannelsLast;

        throw new ArgumentException(
            $"detect: ambiguous output layout {dimension1}x{dimension2}; set DetectorOptions.Layout explicitly");
    }
}
