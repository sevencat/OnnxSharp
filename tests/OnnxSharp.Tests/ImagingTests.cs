using OnnxSharp.Imaging;

namespace OnnxSharp.Tests;

/// <summary>
/// 共享图像预处理（<see cref="ImagePixels"/> / <see cref="AxisMap"/> / <see cref="ImageGeometry"/> /
/// <see cref="ImageTensor"/>）的几何与数值用例。Vision / Ocr / Sam3 都跑在这套实现上。
/// </summary>
internal static class ImagingTests
{
    public static IEnumerable<TestCase> Cases()
    {
        yield return new("imaging: axis map uses the half-pixel centre formula", AxisMapHalfPixel);
        yield return new("imaging: contain keeps aspect and rounds the resized size", ContainPlacement);
        yield return new("imaging: BGR24 with row stride matches packed RGB24", BgrWithStrideMatchesRgb);
        yield return new("imaging: nearest interpolation picks the source pixel", NearestPicksSourcePixel);
        yield return new("imaging: rotate180 flips both axes", Rotate180Flips);
        yield return new("imaging: padding stays at the pad value", PaddingStaysAtPadValue);
    }

    private static void AxisMapHalfPixel()
    {
        var map = AxisMap.Bilinear(4, 2);

        // (0 + 0.5) * 4 / 2 - 0.5 = 0.5 → lower 0 / upper 1 / weight 0.5
        Assert.Equal(0, map.Lower[0], "lower[0]");
        Assert.Equal(1, map.Upper[0], "upper[0]");
        Assert.Near(0.5f, map.Fraction[0], 1e-6f, "fraction[0]");

        // (1 + 0.5) * 2 - 0.5 = 2.5 → lower 2 / upper 3 / weight 0.5
        Assert.Equal(2, map.Lower[1], "lower[1]");
        Assert.Equal(3, map.Upper[1], "upper[1]");
        Assert.Near(0.5f, map.Fraction[1], 1e-6f, "fraction[1]");

        // 放大时同样遵守半像素公式：4 → 8 的第 0 个点落在 -0.25（夹到 0）
        var upscale = AxisMap.Bilinear(4, 8);
        Assert.Equal(0, upscale.Lower[0], "upscale lower[0]");
        Assert.Near(0f, upscale.Fraction[0], 1e-6f, "upscale fraction[0]");
    }

    private static void ContainPlacement()
    {
        var topLeft = ImageGeometry.Contain(1568, 1068, 1008, 1008, centered: false);
        Assert.Equal(1008, topLeft.ResizedWidth, "resized width");
        Assert.Equal(687, topLeft.ResizedHeight, "resized height");
        Assert.Equal(0, topLeft.OffsetX, "top-left offset x");
        Assert.Equal(0, topLeft.OffsetY, "top-left offset y");
        Assert.Near(1008f / 1568f, topLeft.ScaleX, 1e-6f, "scale x");

        var centered = ImageGeometry.Contain(1568, 1068, 1008, 1008);
        Assert.Equal(160, centered.OffsetY, "centered offset y");
        Assert.Equal(0, centered.OffsetX, "centered offset x");

        var stretch = ImageGeometry.Stretch(640, 480, 512, 512);
        Assert.Equal(512, stretch.ResizedWidth, "stretch width");
        Assert.Equal(512, stretch.ResizedHeight, "stretch height");
    }

    private static void BgrWithStrideMatchesRgb()
    {
        // 2x2：红、绿 / 蓝、白
        byte[] rgb =
        [
            255, 0, 0,    0, 255, 0,
            0, 0, 255,    255, 255, 255,
        ];

        // 同一张图的 BGR 版本，每行多 4 字节填充
        int stride = 2 * 3 + 4;
        var bgr = new byte[stride * 2];
        for (int y = 0; y < 2; y++)
        {
            for (int x = 0; x < 2; x++)
            {
                int source = (y * 2 + x) * 3;
                int target = y * stride + x * 3;
                bgr[target] = rgb[source + 2];
                bgr[target + 1] = rgb[source + 1];
                bgr[target + 2] = rgb[source];
            }
        }

        float[] fromRgb = ToChw(new ImagePixels(rgb, 2, 2, PixelFormat.Rgb24));
        float[] fromBgr = ToChw(new ImagePixels(bgr, 2, 2, PixelFormat.Bgr24, stride));

        Assert.Equal(fromRgb.Length, fromBgr.Length, "plane size");
        for (int i = 0; i < fromRgb.Length; i++)
            Assert.Near(fromRgb[i], fromBgr[i], 1e-6f, $"value[{i}]");
    }

    private static void NearestPicksSourcePixel()
    {
        byte[] pixels =
        [
            10, 20, 30,    40, 50, 60,
        ];

        var destination = new float[3 * 2 * 2];
        ImageTensor.ToChw(new ImagePixels(pixels, 2, 1, PixelFormat.Rgb24),
            0, 0, 0, 0, outWidth: 2, outHeight: 1,
            offsetX: 0, offsetY: 0, destinationHeight: 1,
            destinationStride: 2,
            padValue: 0f, destination,
            interpolation: ResizeInterpolation.Nearest);

        // 最近邻放大到同宽：两个像素原样取
        Assert.Near(10f / 255f, destination[0], 1e-6f, "left R");
        Assert.Near(40f / 255f, destination[1], 1e-6f, "right R");
    }

    private static void Rotate180Flips()
    {
        byte[] pixels =
        [
            1, 0, 0,    2, 0, 0,
            3, 0, 0,    4, 0, 0,
        ];

        var normal = ToChw(new ImagePixels(pixels, 2, 2, PixelFormat.Rgb24));
        var rotated = new float[3 * 2 * 2];
        ImageTensor.ToChw(new ImagePixels(pixels, 2, 2, PixelFormat.Rgb24),
            0, 0, 0, 0, outWidth: 2, outHeight: 2,
            offsetX: 0, offsetY: 0, destinationHeight: 2,
            destinationStride: 2,
            padValue: 0f, rotated, rotate180: true);

        // 4 → 1、3 → 2 的位置互换
        Assert.Near(normal[0], rotated[3], 1e-6f, "top-left becomes bottom-right");
        Assert.Near(normal[3], rotated[0], 1e-6f, "bottom-right becomes top-left");
    }

    private static void PaddingStaysAtPadValue()
    {
        byte[] pixels = [255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255];

        var destination = new float[3 * 4 * 4];
        ImageTensor.ToChw(new ImagePixels(pixels, 2, 2, PixelFormat.Rgb24),
            0, 0, 0, 0, outWidth: 2, outHeight: 2,
            offsetX: 1, offsetY: 1, destinationHeight: 4,
            destinationStride: 4,
            padValue: 0.25f, destination);

        Assert.Near(0.25f, destination[0], 1e-6f, "pad top-left");
        Assert.Near(1f, destination[4 + 1 + 0], 1e-6f, "image starts at (1,1)");
    }

    private static float[] ToChw(ImagePixels source)
    {
        var destination = new float[3 * source.Height * source.Width];
        ImageTensor.ToChw(source, 0, 0, 0, 0, source.Width, source.Height,
            offsetX: 0, offsetY: 0, destinationHeight: source.Height,
            destinationStride: source.Width,
            padValue: 0f, destination);
        return destination;
    }
}
