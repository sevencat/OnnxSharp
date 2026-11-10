using OnnxSharp.Imaging;
using OnnxSharp.Vision;

namespace OnnxSharp.Ocr.PaddleOCR;

/// <summary>
/// PaddleOCR 的像素 → NCHW 约定（裁剪 + 双线性 + 归一化）。实现委托给共享的
/// <see cref="ImageTensor.ToChw"/>，本类型只保留 PaddleOCR 特有的参数默认值。
/// </summary>
internal static class OcrImage
{
    /// <summary>检测的归一化（ImageNet 统计量）。</summary>
    internal static readonly float[] DetMean = [0.485f, 0.456f, 0.406f];

    internal static readonly float[] DetStd = [0.229f, 0.224f, 0.225f];

    /// <summary>识别 / 方向分类的归一化（<c>(v/255 - 0.5) / 0.5</c>）。</summary>
    internal static readonly float[] Half = [0.5f, 0.5f, 0.5f];

    internal static int BytesPerPixel(OcrPixelFormat format) => format switch
    {
        OcrPixelFormat.Rgba32 or OcrPixelFormat.Bgra32 => 4,
        _ => 3,
    };

    internal static PixelFormat ToPixelFormat(OcrPixelFormat format) => format switch
    {
        OcrPixelFormat.Rgb24 => PixelFormat.Rgb24,
        OcrPixelFormat.Rgba32 => PixelFormat.Rgba32,
        OcrPixelFormat.Bgra32 => PixelFormat.Bgra32,
        _ => PixelFormat.Bgr24,
    };

    /// <summary>
    /// 把源图的一块区域缩放到 <paramref name="outWidth"/>×<paramref name="outHeight"/>，
    /// 输出 <c>[3, outHeight, destinationWidth]</c> 的 NCHW。
    /// 右侧多出来的列保持原值（调用方先清零，即 PaddleOCR 识别阶段的右侧 0 padding）。
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
        var normalization = new ChannelNormalization(
            mean[0], mean[1], mean[2],
            std[0], std[1], std[2]);

        var source = new ImagePixels(pixels, imageWidth, imageHeight, ToPixelFormat(format), rowStride);

        ImageTensor.ToChw(
            source,
            cropX, cropY, cropWidth, cropHeight,
            outWidth, outHeight,
            offsetX: 0, offsetY: 0,
            destinationHeight: outHeight,
            destinationStride: destinationWidth,
            padValue: 0f,                     // PaddleOCR 的右侧 padding 是「归一化后的 0」
            destination,
            normalization,
            ResizeInterpolation.Bilinear,
            rotate180);
    }
}
