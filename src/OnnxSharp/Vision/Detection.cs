namespace OnnxSharp.Vision;

/// <summary>
/// 一个检测框。坐标原点在<b>原图</b>左上角，单位是原图像素
/// —— <see cref="YoloDecoder"/> 已按 <see cref="Letterbox"/> 的缩放/填边参数还原并裁剪到图内。
/// </summary>
/// <param name="ClassId">类别编号，从 0 开始。</param>
/// <param name="Score">类别置信度。YOLOv8/v11 的输出没有 objectness，这里就是 sigmoid 之后的类别分。</param>
/// <param name="X">左上角横坐标。</param>
/// <param name="Y">左上角纵坐标。</param>
/// <param name="Width">框宽，恒大于 0。</param>
/// <param name="Height">框高，恒大于 0。</param>
public readonly record struct Detection(int ClassId, float Score, float X, float Y, float Width, float Height)
{
    public float Right => X + Width;

    public float Bottom => Y + Height;

    public float Area => Width * Height;

    /// <summary>交并比。两者不相交时返回 0（不会返回负值）。</summary>
    public static float IoU(in Detection a, in Detection b)
    {
        float x1 = MathF.Max(a.X, b.X);
        float y1 = MathF.Max(a.Y, b.Y);
        float x2 = MathF.Min(a.Right, b.Right);
        float y2 = MathF.Min(a.Bottom, b.Bottom);

        float intersectionWidth = x2 - x1;
        float intersectionHeight = y2 - y1;
        if (intersectionWidth <= 0 || intersectionHeight <= 0)
            return 0;

        float intersection = intersectionWidth * intersectionHeight;
        float union = a.Area + b.Area - intersection;
        return union <= 0 ? 0 : intersection / union;
    }

    public override string ToString()
        => $"class {ClassId} score {Score:F4} box [{X:F1}, {Y:F1}, {Width:F1} x {Height:F1}]";
}

/// <summary>
/// 贪心非极大值抑制。语义与 ultralytics 的默认后处理一致：
/// 先按分数降序，再逐个与已保留的框比较，<b>默认只在同类之间抑制</b>。
/// </summary>
public static class DetectionNms
{
    /// <param name="candidates">候选框（顺序无所谓，内部会按分数降序排）。</param>
    /// <param name="iouThreshold">IoU 严格大于该值的同类框会被抑制。</param>
    /// <param name="maxDetections">最多保留多少个框。</param>
    /// <param name="classAgnostic">true 表示不同类别之间也互相抑制（如人群检测只保留一个框）。</param>
    public static List<Detection> Suppress(
        IReadOnlyList<Detection> candidates,
        float iouThreshold,
        int maxDetections = int.MaxValue,
        bool classAgnostic = false)
    {
        if (candidates.Count == 0 || maxDetections <= 0)
            return new List<Detection>();

        var sorted = new List<Detection>(candidates);
        sorted.Sort((a, b) => b.Score.CompareTo(a.Score));

        return SuppressSorted(sorted, iouThreshold, maxDetections, classAgnostic);
    }

    /// <summary>
    /// 已按分数降序排好的候选直接做抑制，省掉一次排序。
    /// 解码路径里候选数可达 3 万，多排一次不值。
    /// </summary>
    internal static List<Detection> SuppressSorted(
        List<Detection> sorted,
        float iouThreshold,
        int maxDetections,
        bool classAgnostic)
    {
        var kept = new List<Detection>();
        if (sorted.Count == 0 || maxDetections <= 0)
            return kept;

        foreach (var candidate in sorted)
        {
            bool suppressed = false;
            foreach (var other in kept)
            {
                if (!classAgnostic && other.ClassId != candidate.ClassId)
                    continue;

                if (Detection.IoU(other, candidate) > iouThreshold)
                {
                    suppressed = true;
                    break;
                }
            }

            if (suppressed)
                continue;

            kept.Add(candidate);
            if (kept.Count >= maxDetections)
                break;
        }

        return kept;
    }
}
