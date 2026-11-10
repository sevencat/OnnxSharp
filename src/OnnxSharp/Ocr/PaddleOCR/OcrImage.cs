namespace OnnxSharp.Ocr.PaddleOCR;

/// <summary>
/// 像素 → NCHW float 的转换（裁剪 + 双线性缩放 + 归一化一次做完）。
/// <para>
/// 采样公式与 OpenCV <c>INTER_LINEAR</c> 一致：<c>(dst + 0.5) * srcSize / dstSize - 0.5</c>，
/// 越界按边缘复制。PaddleOCR 的 det / rec / cls 预处理都是这条约定，差半像素会让框整体偏 1~2 px。
/// </para>
/// </summary>
internal static class OcrImage
{
    /// <summary>检测的归一化（ImageNet 统计量）。</summary>
    internal static readonly float[] DetMean = [0.485f, 0.456f, 0.406f];

    internal static readonly float[] DetStd = [0.229f, 0.224f, 0.225f];

    /// <summary>识别 / 方向分类的归一化（<c>(v/255 - 0.5) / 0.5</c>）。</summary>
    internal static readonly float[] Half = [0.5f, 0.5f, 0.5f];

    internal static int BytesPerPixel(OcrPixelFormat format)
        => format is OcrPixelFormat.Rgba32 or OcrPixelFormat.Bgra32 ? 4 : 3;

    private static void ChannelOffsets(OcrPixelFormat format, out int r, out int g, out int b)
    {
        switch (format)
        {
            case OcrPixelFormat.Rgb24:
            case OcrPixelFormat.Rgba32:
                r = 0; g = 1; b = 2;
                break;
            default:
                r = 2; g = 1; b = 0;
                break;
        }
    }

    /// <summary>
    /// 把源图的一块区域缩放到 <paramref name="outWidth"/>×<paramref name="outHeight"/>，
    /// 输出 <c>[3, outHeight, destinationWidth]</c> 的 NCHW。右侧多出来的列保持原值
    /// （调用方先清零，即 PaddleOCR 识别阶段的右侧 0 padding）。
    /// </summary>
    /// <param name="rotate180">true 表示整块旋转 180°（方向分类判为倒置时用）。</param>
    internal static void ToChw(
        ReadOnlySpan<byte> pixels,
        int imageWidth,
        int imageHeight,
        OcrPixelFormat format,
        int rowStride,
        int cropX,
        int cropY,
        int cropWidth,
        int cropHeight,
        int outWidth,
        int outHeight,
        ReadOnlySpan<float> mean,
        ReadOnlySpan<float> std,
        float[] destination,
        int destinationWidth,
        bool rotate180 = false)
    {
        int bytesPerPixel = BytesPerPixel(format);
        if (rowStride <= 0)
            rowStride = imageWidth * bytesPerPixel;
        if (pixels.Length < (long)rowStride * (imageHeight - 1) + (long)imageWidth * bytesPerPixel)
            throw new ArgumentException("pixel buffer is smaller than width/height/stride require", nameof(pixels));

        cropX = Math.Clamp(cropX, 0, imageWidth - 1);
        cropY = Math.Clamp(cropY, 0, imageHeight - 1);
        cropWidth = Math.Clamp(cropWidth, 1, imageWidth - cropX);
        cropHeight = Math.Clamp(cropHeight, 1, imageHeight - cropY);

        int needed = checked(3 * outHeight * destinationWidth);
        if (destination.Length < needed)
            throw new ArgumentException(
                $"destination has {destination.Length} floats, {needed} needed for [3,{outHeight},{destinationWidth}]",
                nameof(destination));

        ChannelOffsets(format, out int rOffset, out int gOffset, out int bOffset);
        Span<int> offsets = stackalloc int[3] { rOffset, gOffset, bOffset };

        // 采样坐标：先算好两张表，内层循环只剩两次乘加。
        var xs = new int[outWidth * 2];
        var xw = new float[outWidth];
        for (int dx = 0; dx < outWidth; dx++)
        {
            int sourceX = rotate180 ? outWidth - 1 - dx : dx;
            float fx = (sourceX + 0.5f) * cropWidth / outWidth - 0.5f;
            int x0 = (int)MathF.Floor(fx);
            float wx = fx - x0;
            if (x0 < 0) { x0 = 0; wx = 0; }
            if (x0 >= cropWidth - 1) { x0 = cropWidth - 1; wx = 0; }
            xs[dx * 2] = cropX + x0;
            xs[dx * 2 + 1] = cropX + Math.Min(x0 + 1, cropWidth - 1);
            xw[dx] = wx;
        }

        var ys = new int[outHeight * 2];
        var yw = new float[outHeight];
        for (int dy = 0; dy < outHeight; dy++)
        {
            int sourceY = rotate180 ? outHeight - 1 - dy : dy;
            float fy = (sourceY + 0.5f) * cropHeight / outHeight - 0.5f;
            int y0 = (int)MathF.Floor(fy);
            float wy = fy - y0;
            if (y0 < 0) { y0 = 0; wy = 0; }
            if (y0 >= cropHeight - 1) { y0 = cropHeight - 1; wy = 0; }
            ys[dy * 2] = cropY + y0;
            ys[dy * 2 + 1] = cropY + Math.Min(y0 + 1, cropHeight - 1);
            yw[dy] = wy;
        }

        for (int c = 0; c < 3; c++)
        {
            int channelOffset = offsets[c];
            float scale = 1f / (255f * std[c]);
            float shift = -mean[c] / std[c];
            int plane = c * outHeight * destinationWidth;

            for (int dy = 0; dy < outHeight; dy++)
            {
                int rowBase = plane + dy * destinationWidth;
                int y0 = ys[dy * 2], y1 = ys[dy * 2 + 1];
                float wy = yw[dy];
                int row0 = y0 * rowStride + channelOffset;
                int row1 = y1 * rowStride + channelOffset;

                for (int dx = 0; dx < outWidth; dx++)
                {
                    int x0 = xs[dx * 2] * bytesPerPixel;
                    int x1 = xs[dx * 2 + 1] * bytesPerPixel;
                    float wx = xw[dx];

                    float top = pixels[row0 + x0] + (pixels[row0 + x1] - pixels[row0 + x0]) * wx;
                    float bottom = pixels[row1 + x0] + (pixels[row1 + x1] - pixels[row1 + x0]) * wx;
                    float value = top + (bottom - top) * wy;

                    destination[rowBase + dx] = value * scale + shift;
                }
            }
        }
    }
}
