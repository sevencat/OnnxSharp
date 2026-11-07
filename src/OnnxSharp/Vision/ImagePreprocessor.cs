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
/// 两者都用半像素中心对齐的双线性插值，与 OpenCV <c>INTER_LINEAR</c> 一致。
/// </summary>
public static class ImagePreprocessor
{
    /// <summary>填边灰度值（与 ultralytics / SimdYolo 一致）。</summary>
    public const byte PadValue = 114;

    /// <summary>像素总数超过该阈值时按行并行；640² 量级一定走并行。</summary>
    private const int ParallelRowsThreshold = 32768;

    /// <summary>
    /// 等比缩放 + 居中填边，输出 <c>[1, 3, targetHeight, targetWidth]</c> 的 F32 张量。
    /// <para>
    /// 采样点公式是 <c>(dst + 0.5) * src / dst - 0.5</c>（半像素中心对齐），与 OpenCV 的
    /// <c>INTER_LINEAR</c>、ultralytics、SimdYolo 完全一致 —— 这样同一张图交给不同引擎，
    /// 送进模型的输入张量可以逐元素对上。
    /// </para>
    /// </summary>
    /// <param name="pixels">RGB24 或 BGR24 的紧凑像素（每行 <paramref name="rowStride"/> 字节）。</param>
    /// <param name="width">原图宽，像素。</param>
    /// <param name="height">原图高，像素。</param>
    /// <param name="targetWidth">模型输入宽。</param>
    /// <param name="targetHeight">模型输入高。</param>
    /// <param name="bgr">true 表示输入是 BGR 顺序（OpenCV 默认），false 为 RGB。</param>
    /// <param name="rowStride">源图每行字节数；0 表示 <c>width * 3</c>（无行尾填充）。</param>
    /// <returns>归一化后的输入张量与还原参数。</returns>
    /// <exception cref="ArgumentException">尺寸非正、行跨度不足或缓冲区长度不够。</exception>
    /// <exception cref="OverflowException">尺寸相乘超出 Int32。</exception>
    public static unsafe (Tensor Input, Letterbox Transform) Letterbox(
        ReadOnlySpan<byte> pixels,
        int width,
        int height,
        int targetWidth,
        int targetHeight,
        bool bgr = false,
        int rowStride = 0)
    {
        if (width <= 0 || height <= 0 || targetWidth <= 0 || targetHeight <= 0)
            throw new ArgumentException("Image and target dimensions must be positive.");

        rowStride = rowStride == 0 ? checked(width * 3) : rowStride;
        if (rowStride < checked(width * 3) || pixels.Length < (long)(height - 1) * rowStride + width * 3)
            throw new ArgumentException("Image buffer is too short for the declared size.");

        float gain = MathF.Min((float)targetWidth / width, (float)targetHeight / height);
        int resizedWidth = Math.Clamp((int)MathF.Round(width * gain), 1, targetWidth);
        int resizedHeight = Math.Clamp((int)MathF.Round(height * gain), 1, targetHeight);
        int padX = (targetWidth - resizedWidth) / 2;
        int padY = (targetHeight - resizedHeight) / 2;

        var input = Tensor.Allocate(new TensorShape(1, 3, targetHeight, targetWidth), DType.F32);
        var data = input.AsSpan<float>();
        data.Fill(PadValue / 255f);

        // 采样下标只与列有关，按列预算一次（避免每像素重算乘除与 clamp）。
        BuildAxisMap(width, resizedWidth, out var x0Map, out var x1Map, out var fractions);

        int plane = checked(targetWidth * targetHeight);

        fixed (byte* source = pixels)
        fixed (float* destination = data)
        {
            nint sourceAddress = (nint)source;
            nint destinationAddress = (nint)destination;

            // 每个线程只写自己那一行，三通道互不重叠，可安全并行。
            void Row(int y)
            {
                byte* src = (byte*)sourceAddress;
                float* dst = (float*)destinationAddress;

                float sy = Math.Clamp((y + 0.5f) * height / resizedHeight - 0.5f, 0, height - 1);
                int y0 = (int)sy;
                int y1 = Math.Min(y0 + 1, height - 1);
                float fy = sy - y0;
                int rowOffset = (y + padY) * targetWidth + padX;

                for (int x = 0; x < resizedWidth; x++)
                {
                    int x0 = x0Map[x];
                    int x1 = x1Map[x];
                    float fx = fractions[x];

                    for (int c = 0; c < 3; c++)
                    {
                        int channel = bgr ? 2 - c : c;
                        float top = src[y0 * rowStride + x0 * 3 + channel] * (1 - fx)
                                    + src[y0 * rowStride + x1 * 3 + channel] * fx;
                        float bottom = src[y1 * rowStride + x0 * 3 + channel] * (1 - fx)
                                       + src[y1 * rowStride + x1 * 3 + channel] * fx;

                        dst[c * plane + rowOffset + x] = (top * (1 - fy) + bottom * fy) / 255f;
                    }
                }
            }

            if ((long)resizedWidth * resizedHeight >= ParallelRowsThreshold)
                Parallel.For(0, resizedHeight, Row);
            else
                for (int y = 0; y < resizedHeight; y++)
                    Row(y);
        }

        return (input, new Letterbox(
            width,
            height,
            targetWidth,
            targetHeight,
            (float)resizedWidth / width,
            (float)resizedHeight / height,
            padX,
            padY));
    }

    /// <summary>
    /// 直接把整幅图拉伸到目标尺寸（<b>不保持长宽比、不填边</b>），输出 <c>[1, 3, targetHeight, targetWidth]</c>。
    /// <para>
    /// 这是 RF-DETR 的约定（<c>torchvision.transforms.functional.resize</c>，
    /// <c>antialias=False</c>，即半像素中心双线性），框因此可以直接按 <c>原图宽/目标宽</c> 线性还原。
    /// 采样公式与 <see cref="Letterbox"/> 相同：<c>(dst + 0.5) * src / dst - 0.5</c>。
    /// </para>
    /// </summary>
    /// <param name="pixels">RGB24 或 BGR24 的紧凑像素（每行 <paramref name="rowStride"/> 字节）。</param>
    /// <param name="width">原图宽，像素。</param>
    /// <param name="height">原图高，像素。</param>
    /// <param name="targetWidth">模型输入宽。</param>
    /// <param name="targetHeight">模型输入高。</param>
    /// <param name="bgr">true 表示输入是 BGR 顺序（OpenCV 默认），false 为 RGB。</param>
    /// <param name="rowStride">源图每行字节数；0 表示 <c>width * 3</c>（无行尾填充）。</param>
    /// <param name="normalization">逐通道归一化；null 表示只做 <c>/255</c>（<see cref="ChannelNormalization.Divide255"/>）。</param>
    /// <returns>归一化后的输入张量。</returns>
    /// <exception cref="ArgumentException">尺寸非正、行跨度不足、缓冲区长度不够或标准差为 0。</exception>
    /// <exception cref="OverflowException">尺寸相乘超出 Int32。</exception>
    public static unsafe Tensor Stretch(
        ReadOnlySpan<byte> pixels,
        int width,
        int height,
        int targetWidth,
        int targetHeight,
        bool bgr = false,
        int rowStride = 0,
        ChannelNormalization? normalization = null)
    {
        if (width <= 0 || height <= 0 || targetWidth <= 0 || targetHeight <= 0)
            throw new ArgumentException("Image and target dimensions must be positive.");

        rowStride = rowStride == 0 ? checked(width * 3) : rowStride;
        if (rowStride < checked(width * 3) || pixels.Length < (long)(height - 1) * rowStride + width * 3)
            throw new ArgumentException("Image buffer is too short for the declared size.");

        ChannelNormalization norm = normalization ?? ChannelNormalization.Divide255;
        if (norm.Std(0) <= 0 || norm.Std(1) <= 0 || norm.Std(2) <= 0)
            throw new ArgumentException("Normalization standard deviation must be positive.", nameof(normalization));

        // 归一化写成一次乘法 + 一次加法，省掉每个像素两次除法。
        float gain0 = 1f / (255f * norm.Std(0));
        float gain1 = 1f / (255f * norm.Std(1));
        float gain2 = 1f / (255f * norm.Std(2));
        float bias0 = -norm.Mean(0) / norm.Std(0);
        float bias1 = -norm.Mean(1) / norm.Std(1);
        float bias2 = -norm.Mean(2) / norm.Std(2);

        BuildAxisMap(width, targetWidth, out var x0Map, out var x1Map, out var fxMap);
        BuildAxisMap(height, targetHeight, out var y0Map, out var y1Map, out var fyMap);

        var input = Tensor.Allocate(new TensorShape(1, 3, targetHeight, targetWidth), DType.F32);
        var data = input.AsSpan<float>();
        int plane = checked(targetWidth * targetHeight);

        fixed (byte* source = pixels)
        fixed (float* destination = data)
        {
            nint sourceAddress = (nint)source;
            nint destinationAddress = (nint)destination;

            // 每个线程只写自己那一行，三通道互不重叠，可安全并行。
            void Row(int y)
            {
                byte* src = (byte*)sourceAddress;
                float* dst = (float*)destinationAddress;

                int y0 = y0Map[y];
                int y1 = y1Map[y];
                float fy = fyMap[y];
                float oneMinusFy = 1 - fy;
                int rowOffset = y * targetWidth;

                for (int x = 0; x < targetWidth; x++)
                {
                    int x0 = x0Map[x];
                    int x1 = x1Map[x];
                    float fx = fxMap[x];
                    float oneMinusFx = 1 - fx;

                    int topLeft = y0 * rowStride + x0 * 3;
                    int topRight = y0 * rowStride + x1 * 3;
                    int bottomLeft = y1 * rowStride + x0 * 3;
                    int bottomRight = y1 * rowStride + x1 * 3;

                    for (int c = 0; c < 3; c++)
                    {
                        int channel = bgr ? 2 - c : c;
                        float top = src[topLeft + channel] * oneMinusFx + src[topRight + channel] * fx;
                        float bottom = src[bottomLeft + channel] * oneMinusFx + src[bottomRight + channel] * fx;
                        float value = top * oneMinusFy + bottom * fy;

                        float gain = c == 0 ? gain0 : c == 1 ? gain1 : gain2;
                        float bias = c == 0 ? bias0 : c == 1 ? bias1 : bias2;
                        dst[c * plane + rowOffset + x] = value * gain + bias;
                    }
                }
            }

            if ((long)targetWidth * targetHeight >= ParallelRowsThreshold)
                Parallel.For(0, targetHeight, Row);
            else
                for (int y = 0; y < targetHeight; y++)
                    Row(y);
        }

        return input;
    }

    /// <summary>
    /// 预算一维的重采样坐标：半像素中心对齐的双线性，
    /// <c>src = (dst + 0.5) * srcSize / dstSize - 0.5</c>（与 OpenCV <c>INTER_LINEAR</c> 一致）。
    /// </summary>
    private static void BuildAxisMap(
        int sourceSize,
        int destinationSize,
        out int[] lower,
        out int[] upper,
        out float[] fraction)
    {
        lower = new int[destinationSize];
        upper = new int[destinationSize];
        fraction = new float[destinationSize];

        for (int i = 0; i < destinationSize; i++)
        {
            float sample = Math.Clamp((i + 0.5f) * sourceSize / destinationSize - 0.5f, 0, sourceSize - 1);
            lower[i] = (int)sample;
            upper[i] = Math.Min(lower[i] + 1, sourceSize - 1);
            fraction[i] = sample - lower[i];
        }
    }
}
