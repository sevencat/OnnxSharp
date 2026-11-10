namespace OnnxSharp.Imaging;

/// <summary>
/// 一维重采样表：预算好「下界 / 上界 / 权重」，内层循环不再做乘除与 clamp。
/// Vision 的 letterbox、OCR 的裁剪缩放、SAM3 的输入预处理共用同一份坐标公式
/// （<c>src = (dst + 0.5) * srcSize / dstSize - 0.5</c>）。
/// </summary>
public sealed class AxisMap
{
    private AxisMap(int[] lower, int[] upper, float[] fraction)
    {
        Lower = lower;
        Upper = upper;
        Fraction = fraction;
    }

    public int[] Lower { get; }

    public int[] Upper { get; }

    public float[] Fraction { get; }

    public int Length => Lower.Length;

    public static AxisMap Bilinear(int sourceSize, int destinationSize)
    {
        if (sourceSize <= 0 || destinationSize <= 0)
            throw new ArgumentException($"invalid axis sizes {sourceSize} -> {destinationSize}");

        var lower = new int[destinationSize];
        var upper = new int[destinationSize];
        var fraction = new float[destinationSize];

        for (int i = 0; i < destinationSize; i++)
        {
            float sample = Math.Clamp((i + 0.5f) * sourceSize / destinationSize - 0.5f, 0, sourceSize - 1);
            lower[i] = (int)sample;
            upper[i] = Math.Min(lower[i] + 1, sourceSize - 1);
            fraction[i] = sample - lower[i];
        }

        return new AxisMap(lower, upper, fraction);
    }

    public static AxisMap Nearest(int sourceSize, int destinationSize)
    {
        if (sourceSize <= 0 || destinationSize <= 0)
            throw new ArgumentException($"invalid axis sizes {sourceSize} -> {destinationSize}");

        var lower = new int[destinationSize];
        var upper = new int[destinationSize];
        var fraction = new float[destinationSize];

        for (int i = 0; i < destinationSize; i++)
        {
            int nearest = Math.Clamp((int)((i + 0.5f) * sourceSize / destinationSize), 0, sourceSize - 1);
            lower[i] = nearest;
            upper[i] = nearest;
            fraction[i] = 0f;
        }

        return new AxisMap(lower, upper, fraction);
    }
}

/// <summary>
/// letterbox / stretch 的几何参数：缩放后的尺寸、填边偏移与实际比例。
/// <see cref="ScaleX"/> / <see cref="ScaleY"/> 用的是<b>取整后</b>的尺寸算出来的，
/// 用理论比例还原会带来 1~2 像素的系统偏差。
/// </summary>
public readonly record struct ImagePlacement(
    int SourceWidth,
    int SourceHeight,
    int TargetWidth,
    int TargetHeight,
    int ResizedWidth,
    int ResizedHeight,
    int OffsetX,
    int OffsetY)
{
    public float ScaleX => (float)ResizedWidth / SourceWidth;

    public float ScaleY => (float)ResizedHeight / SourceHeight;

    public override string ToString()
        => $"{SourceWidth}x{SourceHeight} -> {ResizedWidth}x{ResizedHeight} in {TargetWidth}x{TargetHeight} " +
           $"(offset {OffsetX},{OffsetY})";
}

/// <summary>几何规划：等比缩放到目标框（可居中）或直接拉伸。</summary>
public static class ImageGeometry
{
    /// <summary>
    /// 等比缩放到目标框内（会放大），尺寸四舍五入取整。
    /// <paramref name="centered"/> = true 时居中填边（YOLO），false 时左上对齐（SAM3）。
    /// </summary>
    public static ImagePlacement Contain(int sourceWidth, int sourceHeight, int targetWidth, int targetHeight,
        bool centered = true)
    {
        if (sourceWidth <= 0 || sourceHeight <= 0 || targetWidth <= 0 || targetHeight <= 0)
            throw new ArgumentException("image and target dimensions must be positive");

        float gain = MathF.Min((float)targetWidth / sourceWidth, (float)targetHeight / sourceHeight);
        int resizedWidth = Math.Clamp((int)MathF.Round(sourceWidth * gain), 1, targetWidth);
        int resizedHeight = Math.Clamp((int)MathF.Round(sourceHeight * gain), 1, targetHeight);

        int offsetX = centered ? (targetWidth - resizedWidth) / 2 : 0;
        int offsetY = centered ? (targetHeight - resizedHeight) / 2 : 0;

        return new ImagePlacement(sourceWidth, sourceHeight, targetWidth, targetHeight,
            resizedWidth, resizedHeight, offsetX, offsetY);
    }

    /// <summary>直接拉伸到目标尺寸（RF-DETR / torchvision Resize 的约定）。</summary>
    public static ImagePlacement Stretch(int sourceWidth, int sourceHeight, int targetWidth, int targetHeight)
        => new(sourceWidth, sourceHeight, targetWidth, targetHeight, targetWidth, targetHeight, 0, 0);
}
