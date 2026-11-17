namespace OnnxSharp.Ocr.PaddleOCR;

/// <summary>
/// DB（Differentiable Binarization）检测后处理：概率图 → 文本框。
/// <para>
/// 简化版 PaddleOCR <c>DBPostProcess</c>：邻域连通域 → 轴对齐外接框 → 按
/// <c>面积 × unclip_ratio / 周长</c> 外扩（PaddleOCR 用 pyclipper 做多边形外扩，
/// 这里只在轴对齐框上外扩：竖排 / 倾斜文本仍能框住，只是框是正的）。
/// </para>
/// </summary>
internal static class DbPostprocess
{
    internal readonly record struct Box(float X, float Y, float Width, float Height, float Score);

    public static List<Box> Detect(
        ReadOnlySpan<float> probability,
        int width,
        int height,
        float threshold = 0.3f,
        float boxThreshold = 0.6f,
        float unclipRatio = 1.6f,
        int minSize = 3)
    {
        if (probability.Length < (long)width * height)
            throw new ArgumentException(
                $"probability map has {probability.Length} values, {width}x{height} needed", nameof(probability));

        var visited = new bool[(long)width * height];
        var queue = new Queue<int>();
        var boxes = new List<Box>();

        for (int start = 0; start < visited.Length; start++)
        {
            if (visited[start] || probability[start] <= threshold)
                continue;

            int minX = int.MaxValue, minY = int.MaxValue, maxX = -1, maxY = -1;
            double sum = 0;
            int count = 0;

            visited[start] = true;
            queue.Enqueue(start);

            while (queue.Count > 0)
            {
                int index = queue.Dequeue();
                int x = index % width;
                int y = index / width;

                sum += probability[index];
                count++;
                if (x < minX) minX = x;
                if (x > maxX) maxX = x;
                if (y < minY) minY = y;
                if (y > maxY) maxY = y;

                for (int dy = -1; dy <= 1; dy++)
                {
                    int ny = y + dy;
                    if ((uint)ny >= (uint)height)
                        continue;

                    for (int dx = -1; dx <= 1; dx++)
                    {
                        int nx = x + dx;
                        if ((uint)nx >= (uint)width)
                            continue;

                        int next = ny * width + nx;
                        if (visited[next] || probability[next] <= threshold)
                            continue;

                        visited[next] = true;
                        queue.Enqueue(next);
                    }
                }
            }

            float boxWidth = maxX - minX + 1;
            float boxHeight = maxY - minY + 1;
            float score = (float)(sum / count);
            if (boxWidth < minSize || boxHeight < minSize || score < boxThreshold)
                continue;

            double area = boxWidth * boxHeight;
            double perimeter = 2 * (boxWidth + boxHeight);
            float grow = (float)(area * unclipRatio / perimeter);
            boxes.Add(new Box(minX - grow, minY - grow, boxWidth + 2 * grow, boxHeight + 2 * grow, score));
        }

        // 阅读顺序：先上后下、先左后右（同一行内按 x 排）
        boxes.Sort((a, b) =>
        {
            float dy = a.Y - b.Y;
            if (MathF.Abs(dy) > MathF.Min(a.Height, b.Height) * 0.5f)
                return dy < 0 ? -1 : 1;
            return a.X.CompareTo(b.X);
        });

        return boxes;
    }

    /// <summary>把概率图坐标系的框缩放到原图，并裁剪到图内。</summary>
    public static Box Scale(Box box, float scaleX, float scaleY, int imageWidth, int imageHeight)
    {
        float x = MathF.Max(0, box.X * scaleX);
        float y = MathF.Max(0, box.Y * scaleY);
        float right = MathF.Min(imageWidth, (box.X + box.Width) * scaleX);
        float bottom = MathF.Min(imageHeight, (box.Y + box.Height) * scaleY);
        return new Box(x, y, MathF.Max(1, right - x), MathF.Max(1, bottom - y), box.Score);
    }
}
