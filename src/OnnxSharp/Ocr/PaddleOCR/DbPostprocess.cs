namespace OnnxSharp.Ocr.PaddleOCR;

/// <summary>
/// DB锛圖ifferentiable Binarization锛夋娴嬪悗澶勭悊锛氭鐜囧浘 鈫?鏂囨湰妗嗐€?
/// <para>
/// 绠€鍖栫増 PaddleOCR <c>DBPostProcess</c>锛? 閭诲煙杩為€氬煙 鈫?杞村榻愬鎺ユ 鈫?鎸?
/// <c>闈㈢Н 脳 unclip_ratio / 鍛ㄩ暱</c> 澶栨墿锛圥addleOCR 鐢?pyclipper 鍋氬杈瑰舰澶栨墿锛?
/// 杩欓噷鍙湪杞村榻愭涓婂鎵╋細绔栨帓 / 鍊炬枩鏂囨湰浠嶈兘妗嗕綇锛屽彧鏄鏄鐨勶級銆?
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

        // 闃呰椤哄簭锛氬厛涓婂悗涓嬨€佸厛宸﹀悗鍙筹紙鍚屼竴琛屽唴鎸?x 鎺掞級
        boxes.Sort((a, b) =>
        {
            float dy = a.Y - b.Y;
            if (MathF.Abs(dy) > MathF.Min(a.Height, b.Height) * 0.5f)
                return dy < 0 ? -1 : 1;
            return a.X.CompareTo(b.X);
        });

        return boxes;
    }

    /// <summary>鎶婃鐜囧浘鍧愭爣绯荤殑妗嗙缉鏀惧埌鍘熷浘锛屽苟瑁佸壀鍒板浘鍐呫€?/summary>
    public static Box Scale(Box box, float scaleX, float scaleY, int imageWidth, int imageHeight)
    {
        float x = MathF.Max(0, box.X * scaleX);
        float y = MathF.Max(0, box.Y * scaleY);
        float right = MathF.Min(imageWidth, (box.X + box.Width) * scaleX);
        float bottom = MathF.Min(imageHeight, (box.Y + box.Height) * scaleY);
        return new Box(x, y, MathF.Max(1, right - x), MathF.Max(1, bottom - y), box.Score);
    }
}
