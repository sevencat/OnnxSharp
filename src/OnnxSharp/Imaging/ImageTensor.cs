using OnnxSharp.Vision;

namespace OnnxSharp.Imaging;

/// <summary>
/// 像素 → NCHW 输入张量的唯一实现：<b>裁剪 + 缩放 + 归一化 + 填边</b>一次做完。
/// <para>三处调用方都收口到这里，差异只用参数表达：</para>
/// <list type="bullet">
/// <item><b>YOLO</b>：<see cref="ImageGeometry.Contain"/> 居中填 114、<c>/255</c>；</item>
/// <item><b>RF-DETR</b>：<see cref="ImageGeometry.Stretch"/> 不填边、ImageNet 归一化；</item>
/// <item><b>PaddleOCR</b>：裁剪文本行、高 48、右侧补 0、<c>(v/255-0.5)/0.5</c>，可 180° 旋转；</item>
/// <item><b>SAM3</b>：左上对齐、白边（归一化后填 1.0）、<c>/255</c>。</item>
/// </list>
/// </summary>
public static class ImageTensor
{
    /// <summary>目标像素数超过该阈值时按行并行。</summary>
    private const int ParallelRowsThreshold = 32768;

    /// <summary>
    /// 把 <paramref name="source"/> 的一块区域缩放写入 <c>[3, outHeight, destinationStride]</c> 的
    /// <paramref name="destination"/>。整个缓冲**先填 <paramref name="padValue"/>**，再写缩放区域 ——
    /// letterbox 灰边、OCR 右侧 0 padding、SAM3 白边都走同一个机制。
    /// </summary>
    /// <param name="cropX">裁剪起点 X（源图像素）。</param>
    /// <param name="cropY">裁剪起点 Y。</param>
    /// <param name="cropWidth">裁剪宽；0 表示「到源图右边界」。</param>
    /// <param name="cropHeight">裁剪高；0 表示「到源图下边界」。</param>
    /// <param name="outWidth">写进目标缓冲的区域宽（不是缓冲宽）。</param>
    /// <param name="outHeight">写进目标缓冲的区域高。</param>
    /// <param name="offsetX">区域在目标缓冲里的 X 偏移（居中填边时 = padX）。</param>
    /// <param name="offsetY">区域在目标缓冲里的 Y 偏移。</param>
    /// <param name="destinationStride">目标缓冲每行宽度（右侧留白 = stride - outWidth）。</param>
    /// <param name="padValue">填边值（<b>归一化之后</b>：YOLO 114/255、OCR 0、SAM3 1.0）。</param>
    /// <param name="destination">目标缓冲，长度至少 <c>3 × outHeight × destinationStride</c>。</param>
    /// <param name="normalization">逐通道归一化；null = 只做 <c>/255</c>。</param>
    /// <param name="interpolation">双线性（默认，半像素对齐）或最近邻。</param>
    /// <param name="rotate180">true 表示把这块区域旋转 180° 再写（OCR 方向纠偏）。</param>
    public static unsafe void ToChw(
        ImagePixels source,
        int cropX,
        int cropY,
        int cropWidth,
        int cropHeight,
        int outWidth,
        int outHeight,
        int offsetX,
        int offsetY,
        int destinationHeight,
        int destinationStride,
        float padValue,
        Span<float> destination,
        ChannelNormalization? normalization = null,
        ResizeInterpolation interpolation = ResizeInterpolation.Bilinear,
        bool rotate180 = false)
    {
        // Span 不需要判空；长度在下面对照 needed 校验。

        if (outWidth <= 0 || outHeight <= 0)
            throw new ArgumentException($"invalid output size {outWidth}x{outHeight}");
        if (destinationStride < outWidth)
            throw new ArgumentException($"destinationStride {destinationStride} < outWidth {outWidth}");
        if (offsetX < 0 || offsetY < 0 || offsetX + outWidth > destinationStride || offsetY + outHeight > destinationHeight)
            throw new ArgumentException("output region does not fit the destination stride");

        cropX = Math.Clamp(cropX, 0, source.Width - 1);
        cropY = Math.Clamp(cropY, 0, source.Height - 1);
        cropWidth = cropWidth <= 0 ? source.Width - cropX : Math.Clamp(cropWidth, 1, source.Width - cropX);
        cropHeight = cropHeight <= 0 ? source.Height - cropY : Math.Clamp(cropHeight, 1, source.Height - cropY);

        int needed = checked(3 * destinationHeight * destinationStride);
        if (destination.Length < needed)
            throw new ArgumentException(
                $"destination has {destination.Length} floats, {needed} needed for [3,{destinationHeight},{destinationStride}]",
                nameof(destination));

        ChannelNormalization norm = normalization ?? ChannelNormalization.Divide255;
        if (norm.Std(0) <= 0 || norm.Std(1) <= 0 || norm.Std(2) <= 0)
            throw new ArgumentException("normalization standard deviation must be positive", nameof(normalization));

        float gain0 = 1f / (255f * norm.Std(0));
        float gain1 = 1f / (255f * norm.Std(1));
        float gain2 = 1f / (255f * norm.Std(2));
        float bias0 = -norm.Mean(0) / norm.Std(0);
        float bias1 = -norm.Mean(1) / norm.Std(1);
        float bias2 = -norm.Mean(2) / norm.Std(2);

        destination[..needed].Fill(padValue);

        bool nearest = interpolation == ResizeInterpolation.Nearest;
        AxisMap xMap = nearest ? AxisMap.Nearest(cropWidth, outWidth) : AxisMap.Bilinear(cropWidth, outWidth);
        AxisMap yMap = nearest ? AxisMap.Nearest(cropHeight, outHeight) : AxisMap.Bilinear(cropHeight, outHeight);

        int[] xLower = xMap.Lower, xUpper = xMap.Upper;
        float[] xFraction = xMap.Fraction;
        int[] yLower = yMap.Lower, yUpper = yMap.Upper;
        float[] yFraction = yMap.Fraction;

        int bytesPerPixel = source.BytesPerPixel;
        int rowStride = source.RowStride;
        int rOffset = source.ChannelOffset(0);
        int gOffset = source.ChannelOffset(1);
        int bOffset = source.ChannelOffset(2);
        int plane = checked(destinationHeight * destinationStride);

        fixed (byte* sourcePointer = source.Pixels)
        fixed (float* destinationPointer = destination)
        {
            nint sourceAddress = (nint)sourcePointer;
            nint destinationAddress = (nint)destinationPointer;

            void Row(int dy)
            {
                byte* src = (byte*)sourceAddress;
                float* dst = (float*)destinationAddress;

                int sampleRow = rotate180 ? outHeight - 1 - dy : dy;
                int y0 = cropY + yLower[sampleRow];
                int y1 = cropY + yUpper[sampleRow];
                float fy = yFraction[sampleRow];
                float oneMinusFy = 1 - fy;

                int row = (offsetY + dy) * destinationStride + offsetX;
                int top = y0 * rowStride;
                int bottom = y1 * rowStride;

                for (int dx = 0; dx < outWidth; dx++)
                {
                    int sampleColumn = rotate180 ? outWidth - 1 - dx : dx;
                    int x0 = (cropX + xLower[sampleColumn]) * bytesPerPixel;
                    int x1 = (cropX + xUpper[sampleColumn]) * bytesPerPixel;
                    float fx = xFraction[sampleColumn];
                    float oneMinusFx = 1 - fx;

                    WriteChannel(dst, 0, plane, row + dx, src, top, bottom, x0, x1, rOffset,
                        fx, oneMinusFx, oneMinusFy, fy, gain0, bias0);
                    WriteChannel(dst, 1, plane, row + dx, src, top, bottom, x0, x1, gOffset,
                        fx, oneMinusFx, oneMinusFy, fy, gain1, bias1);
                    WriteChannel(dst, 2, plane, row + dx, src, top, bottom, x0, x1, bOffset,
                        fx, oneMinusFx, oneMinusFy, fy, gain2, bias2);
                }
            }

            if ((long)outWidth * outHeight >= ParallelRowsThreshold)
                System.Threading.Tasks.Parallel.For(0, outHeight, Row);
            else
                for (int dy = 0; dy < outHeight; dy++)
                    Row(dy);
        }
    }

    private static unsafe void WriteChannel(
        float* destination,
        int channel,
        int plane,
        int index,
        byte* source,
        int topRow,
        int bottomRow,
        int x0,
        int x1,
        int channelOffset,
        float fx,
        float oneMinusFx,
        float oneMinusFy,
        float fy,
        float gain,
        float bias)
    {
        float top = source[topRow + x0 + channelOffset] * oneMinusFx
                    + source[topRow + x1 + channelOffset] * fx;
        float bottom = source[bottomRow + x0 + channelOffset] * oneMinusFx
                       + source[bottomRow + x1 + channelOffset] * fx;

        destination[channel * plane + index] = (top * oneMinusFy + bottom * fy) * gain + bias;
    }
}
