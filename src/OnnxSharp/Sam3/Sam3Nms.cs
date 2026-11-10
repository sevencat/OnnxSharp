namespace OnnxSharp.Sam3;

/// <summary>
/// SAM3 结果去重：按分数降序的**框级贪心 NMS**（单类别，所以是 class-agnostic）。
/// <para>
/// 为什么必须做：框提示下解码器会输出 200 个 query，同一个人被 10+ 个高度重叠的框反复命中；
/// 官方 demo 也会做类似的去重，否则可视化就是一团叠框。
/// </para>
/// </summary>
internal static class Sam3Nms
{
    public static List<Sam3Item> Suppress(List<Sam3Item> sortedByScore, float iouThreshold, int maxItems)
    {
        if (sortedByScore.Count == 0)
            return sortedByScore;

        if (iouThreshold <= 0)
            return Trim(sortedByScore, maxItems);

        var kept = new List<Sam3Item>(sortedByScore.Count);

        foreach (Sam3Item candidate in sortedByScore)
        {
            bool suppressed = false;
            foreach (Sam3Item other in kept)
            {
                if (IoU(candidate.Box, other.Box) > iouThreshold)
                {
                    suppressed = true;
                    break;
                }
            }

            if (suppressed)
                continue;

            kept.Add(candidate);
            if (maxItems > 0 && kept.Count >= maxItems)
                break;
        }

        return kept;
    }

    private static List<Sam3Item> Trim(List<Sam3Item> items, int maxItems)
        => maxItems > 0 && items.Count > maxItems ? items.GetRange(0, maxItems) : items;

    internal static float IoU(Sam3Box a, Sam3Box b)
    {
        float left = MathF.Max(a.X1, b.X1);
        float top = MathF.Max(a.Y1, b.Y1);
        float right = MathF.Min(a.X2, b.X2);
        float bottom = MathF.Min(a.Y2, b.Y2);

        float width = right - left;
        float height = bottom - top;
        if (width <= 0 || height <= 0)
            return 0f;

        float intersection = width * height;
        float union = a.Width * a.Height + b.Width * b.Height - intersection;
        return union <= 0 ? 0f : intersection / union;
    }
}
