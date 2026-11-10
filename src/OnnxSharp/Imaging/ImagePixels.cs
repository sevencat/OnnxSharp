namespace OnnxSharp.Imaging;

/// <summary>交错像素的排列（四通道的 alpha 会被忽略，只取 RGB）。</summary>
public enum PixelFormat
{
    Rgb24,
    Bgr24,
    Rgba32,
    Bgra32,
}

/// <summary>重采样方式。两者都是「目标像素中心 → 源像素」的半像素对齐。</summary>
public enum ResizeInterpolation
{
    /// <summary><c>src = (dst + 0.5) * srcSize / dstSize - 0.5</c>，与 OpenCV <c>INTER_LINEAR</c> 一致。</summary>
    Bilinear,

    /// <summary>取最近的源像素（掩码上采样之类的场景）。</summary>
    Nearest,
}

/// <summary>
/// 已解码图像的像素视图（<b>不拷贝、不拥有内存</b>）。把三处调用方的输入约定收成一套：
/// <c>Vision</c>（RGB/BGR + rowStride）、<c>Ocr</c>（4 种排列）、<c>Sam3</c>（letterbox 白边）。
/// </summary>
public readonly ref struct ImagePixels
{
    public ImagePixels(
        ReadOnlySpan<byte> pixels,
        int width,
        int height,
        PixelFormat format = PixelFormat.Rgb24,
        int rowStride = 0)
    {
        if (width <= 0 || height <= 0)
            throw new ArgumentException($"invalid image size {width}x{height}", nameof(width));

        int bytesPerPixel = BytesPerPixelOf(format);
        rowStride = rowStride == 0 ? checked(width * bytesPerPixel) : rowStride;

        if (rowStride < checked(width * bytesPerPixel))
            throw new ArgumentException(
                $"rowStride {rowStride} is smaller than width * bytesPerPixel ({width * bytesPerPixel})",
                nameof(rowStride));

        if (pixels.Length < (long)(height - 1) * rowStride + width * bytesPerPixel)
            throw new ArgumentException("pixel buffer is too short for the declared size", nameof(pixels));

        Pixels = pixels;
        Width = width;
        Height = height;
        Format = format;
        RowStride = rowStride;
    }

    public ReadOnlySpan<byte> Pixels { get; }

    public int Width { get; }

    public int Height { get; }

    public PixelFormat Format { get; }

    public int RowStride { get; }

    public int BytesPerPixel => BytesPerPixelOf(Format);

    /// <summary>true 表示内存里是 BGR(A) 顺序（OpenCV / 多数相机 SDK）。</summary>
    public bool IsBgr => Format is PixelFormat.Bgr24 or PixelFormat.Bgra32;

    /// <summary>R/G/B 在像素内的字节偏移（通道 0=R、1=G、2=B）。</summary>
    public int ChannelOffset(int channel) => IsBgr ? 2 - channel : channel;

    public static int BytesPerPixelOf(PixelFormat format)
        => format is PixelFormat.Rgba32 or PixelFormat.Bgra32 ? 4 : 3;

    /// <summary>便捷重载：<c>bgr = true</c> 等价于 <see cref="PixelFormat.Bgr24"/>。</summary>
    public static ImagePixels FromBgr(ReadOnlySpan<byte> pixels, int width, int height, bool bgr, int rowStride = 0)
        => new(pixels, width, height, bgr ? PixelFormat.Bgr24 : PixelFormat.Rgb24, rowStride);
}
