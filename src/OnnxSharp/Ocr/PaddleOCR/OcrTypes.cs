namespace OnnxSharp.Ocr.PaddleOCR;

/// <summary>
/// 交错像素的排列。OCR 的核心只吃像素内存，不解码图片，所以这里只描述「内存里怎么排」。
/// </summary>
public enum OcrPixelFormat
{
    /// <summary>RGB 三通道（每像素 3 字节）。</summary>
    Rgb24,

    /// <summary>BGR 三通道（OpenCV / 多数相机 SDK 的默认顺序）。</summary>
    Bgr24,

    /// <summary>RGB + 忽略的 alpha（每像素 4 字节）。</summary>
    Rgba32,

    /// <summary>BGR + 忽略的 alpha（Windows GDI <c>Format32bppArgb</c> 的字节序）。</summary>
    Bgra32,
}

/// <summary>一行 OCR 结果（<b>源图像素坐标</b>，轴对齐框）。与具体识别引擎无关。</summary>
public readonly record struct OcrLine(string Text, float Score, float X, float Y, float Width, float Height)
{
    public override string ToString()
        => $"{Text}  ({Score:F3})  [{X:F0},{Y:F0},{Width:F0}x{Height:F0}]";
}
