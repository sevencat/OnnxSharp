using OnnxSharp.Imaging;
using OnnxSharp.Tensors;

namespace OnnxSharp.Vision;

/// <summary>
/// letterbox 的等比缩放 + 居中填边参数。解码时用它把模型坐标还原到原图。
/// <para>
/// 之所以要保留它而不是自己重算：缩放后宽高是 <b>四舍五入取整</b> 的，
/// 实际比例 <see cref="ScaleX"/> 与理论比例 <c>TargetWidth / SourceWidth</c> 并不相等，
/// 用理论比例还原会带来 1~2 像素的系统偏差。
/// </para>
/// </summary>
/// <param name="SourceWidth">原图宽。</param>
/// <param name="SourceHeight">原图高。</param>
/// <param name="TargetWidth">模型输入宽。</param>
/// <param name="TargetHeight">模型输入高。</param>
/// <param name="ScaleX">实际横向比例（缩放后宽 / 原图宽）。</param>
/// <param name="ScaleY">实际纵向比例（缩放后高 / 原图高）。</param>
/// <param name="PadX">左侧填边像素数。</param>
/// <param name="PadY">顶部填边像素数。</param>
public readonly record struct Letterbox(
    int SourceWidth,
    int SourceHeight,
    int TargetWidth,
    int TargetHeight,
    float ScaleX,
    float ScaleY,
    int PadX,
    int PadY)
{
    /// <summary>把模型坐标系里的横坐标还原到原图。</summary>
    public float ToSourceX(float x) => (x - PadX) / ScaleX;

    /// <summary>把模型坐标系里的纵坐标还原到原图。</summary>
    public float ToSourceY(float y) => (y - PadY) / ScaleY;

    public override string ToString()
        => $"{SourceWidth}x{SourceHeight} -> {TargetWidth}x{TargetHeight} (scale {ScaleX:F4}x{ScaleY:F4}, pad {PadX},{PadY})";
}

/// <summary>
/// 逐通道归一化参数：<c>out = (v / 255 - Mean[c]) / Std[c]</c>。
/// </summary>
public readonly record struct ChannelNormalization(
    float MeanR,
    float MeanG,
    float MeanB,
    float StdR,
    float StdG,
    float StdB)
{
    /// <summary>只做 <c>/255</c>（YOLO 系列，等价于 ultralytics 的默认预处理）。</summary>
    public static ChannelNormalization Divide255 => new(0, 0, 0, 1, 1, 1);

    /// <summary>ImageNet 均值方差（RF-DETR 的官方预处理）。</summary>
    public static ChannelNormalization ImageNet => new(0.485f, 0.456f, 0.406f, 0.229f, 0.224f, 0.225f);

    public float Mean(int channel) => channel switch
    {
        0 => MeanR,
        1 => MeanG,
        _ => MeanB,
    };

    public float Std(int channel) => channel switch
    {
        0 => StdR,
        1 => StdG,
        _ => StdB,
    };
}
/// <summary>
/// 把已解码的三通道图像转换成归一化 NCHW 输入。两种几何变换：
/// <list type="bullet">
/// <item><see cref="Letterbox"/>：等比缩放 + 居中填 114 灰边（ultralytics / SimdYolo 的 YOLO 约定）；</item>
/// <item><see cref="Stretch"/>：直接拉伸到目标尺寸、不填边（RF-DETR / torchvision <c>Resize</c> 的约定）。</item>
/// </list>
/// <para>
/// 实现委托给共享的 <see cref="Imaging.ImageTensor"/> / <see cref="Imaging.ImageGeometry"/> ——
/// PaddleOCR 的文本行裁剪与 SAM3 的白边 letterbox 用的是同一套代码，只是参数不同。
/// 采样公式仍是 <c>(dst + 0.5) * src / dst - 0.5</c>（半像素中心对齐，与 OpenCV <c>INTER_LINEAR</c> 一致）。
/// </para>
/// </summary>
public static class ImagePreprocessor
{
    /// <summary>填边灰度值（与 ultralytics / SimdYolo 一致）。</summary>
    public const byte PadValue = 114;

    /// <summary>等比缩放 + 居中填边，输出 <c>[1, 3, targetHeight, targetWidth]</c> 的 F32 张量。</summary>
    /// <param name="pixels">RGB24 或 BGR24 的像素（每行 <paramref name="rowStride"/> 字节）。</param>
    /// <param name="width">原图宽，像素。</param>
    /// <param name="height">原图高，像素。</param>
    /// <param name="targetWidth">模型输入宽。</param>
    /// <param name="targetHeight">模型输入高。</param>
    /// <param name="bgr">true 表示输入是 BGR 顺序（OpenCV 默认），false 为 RGB。</param>
    /// <param name="rowStride">源图每行字节数；0 表示 <c>width * 3</c>（无行尾填充）。</param>
    public static (Tensor Input, Letterbox Transform) Letterbox(
        ReadOnlySpan<byte> pixels,
        int width,
        int height,
        int targetWidth,
        int targetHeight,
        bool bgr = false,
        int rowStride = 0)
    {
        var source = ImagePixels.FromBgr(pixels, width, height, bgr, rowStride);
        ImagePlacement placement = ImageGeometry.Contain(width, height, targetWidth, targetHeight, centered: true);

        var input = Tensor.Allocate(new TensorShape(1, 3, targetHeight, targetWidth), DType.F32);
        ImageTensor.ToChw(
            source,
            cropX: 0,
            cropY: 0,
            cropWidth: width,
            cropHeight: height,
            outWidth: placement.ResizedWidth,
            outHeight: placement.ResizedHeight,
            offsetX: placement.OffsetX,
            offsetY: placement.OffsetY,
            destinationHeight: targetHeight,
            destinationStride: targetWidth,
            padValue: PadValue / 255f,
            destination: input.AsSpan<float>(),
            normalization: ChannelNormalization.Divide255);

        return (input, new Letterbox(
            width,
            height,
            targetWidth,
            targetHeight,
            placement.ScaleX,
            placement.ScaleY,
            placement.OffsetX,
            placement.OffsetY));
    }

    /// <summary>直接把整幅图拉伸到目标尺寸（<b>不保持长宽比、不填边</b>），输出 <c>[1, 3, targetHeight, targetWidth]</c>。</summary>
    /// <param name="pixels">RGB24 或 BGR24 的像素（每行 <paramref name="rowStride"/> 字节）。</param>
    /// <param name="width">原图宽，像素。</param>
    /// <param name="height">原图高，像素。</param>
    /// <param name="targetWidth">模型输入宽。</param>
    /// <param name="targetHeight">模型输入高。</param>
    /// <param name="bgr">true 表示输入是 BGR 顺序（OpenCV 默认），false 为 RGB。</param>
    /// <param name="rowStride">源图每行字节数；0 表示 <c>width * 3</c>（无行尾填充）。</param>
    /// <param name="normalization">逐通道归一化；null 表示只做 <c>/255</c>。</param>
    public static Tensor Stretch(
        ReadOnlySpan<byte> pixels,
        int width,
        int height,
        int targetWidth,
        int targetHeight,
        bool bgr = false,
        int rowStride = 0,
        ChannelNormalization? normalization = null)
    {
        var source = ImagePixels.FromBgr(pixels, width, height, bgr, rowStride);
        ImagePlacement placement = ImageGeometry.Stretch(width, height, targetWidth, targetHeight);

        var input = Tensor.Allocate(new TensorShape(1, 3, targetHeight, targetWidth), DType.F32);
        ImageTensor.ToChw(
            source,
            cropX: 0,
            cropY: 0,
            cropWidth: width,
            cropHeight: height,
            outWidth: placement.ResizedWidth,
            outHeight: placement.ResizedHeight,
            offsetX: 0,
            offsetY: 0,
            destinationHeight: targetHeight,
            destinationStride: targetWidth,
            padValue: 0f,                       // 拉伸不填边，缩放区域覆盖整幅
            destination: input.AsSpan<float>(),
            normalization: normalization ?? ChannelNormalization.Divide255);

        return input;
    }
}

