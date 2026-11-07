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
/// 把已解码的三通道图像转换成归一化 NCHW 输入。
/// 等价于 ultralytics 的 <c>letterbox(img, new_shape, color=(114,114,114))</c>：
/// 等比缩放、居中填 114 灰边、双线性插值、BGR→RGB、除以 255。
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
        var x0Map = new int[resizedWidth];
        var x1Map = new int[resizedWidth];
        var fractions = new float[resizedWidth];
        for (int x = 0; x < resizedWidth; x++)
        {
            float sx = Math.Clamp((x + 0.5f) * width / resizedWidth - 0.5f, 0, width - 1);
            x0Map[x] = (int)sx;
            x1Map[x] = Math.Min(x0Map[x] + 1, width - 1);
            fractions[x] = sx - x0Map[x];
        }

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
}
