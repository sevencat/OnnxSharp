using OnnxSharp.Sam3;

namespace OnnxSharp.Tests;

/// <summary>SAM3 门面的纯几何/去重用例（不需要模型文件）。</summary>
internal static class Sam3Tests
{
    public static IEnumerable<TestCase> Cases()
    {
        yield return new("sam3: letterbox fit keeps aspect for wide and tall images", FitSize);
        yield return new("sam3: nms suppresses overlapping boxes", NmsSuppresses);
        yield return new("sam3: nms honours max items", NmsMaxItems);
        yield return new("sam3: mask scale maps the 288 grid back to the source", MaskScale);
    }

    private static void FitSize()
    {
        Assert.Equal((1008, 687), Sam3Segmenter.FitSize(1568, 1068, 1008), "wide image");
        Assert.Equal((504, 1008), Sam3Segmenter.FitSize(600, 1200, 1008), "tall image");
        Assert.Equal((1008, 1008), Sam3Segmenter.FitSize(1008, 1008, 1008), "square image");
    }

    private static void NmsSuppresses()
    {
        var items = new List<Sam3Item>
        {
            Item(0.90f, 10, 10, 110, 210),
            Item(0.80f, 12, 12, 108, 208),   // 与第一个高度重叠 → 抑制
            Item(0.70f, 200, 10, 300, 210),  // 不重叠 → 保留
        };

        List<Sam3Item> kept = Sam3Nms.Suppress(items, iouThreshold: 0.5f, maxItems: 0);

        Assert.Equal(2, kept.Count, "kept count");
        Assert.Near(0.90f, kept[0].Score, 1e-6f, "best score first");
        Assert.Near(0.70f, kept[1].Score, 1e-6f, "second box kept");
    }

    private static void NmsMaxItems()
    {
        var items = new List<Sam3Item>
        {
            Item(0.9f, 0, 0, 10, 10),
            Item(0.8f, 100, 0, 110, 10),
            Item(0.7f, 200, 0, 210, 10),
        };

        Assert.Equal(2, Sam3Nms.Suppress(items, 0.5f, maxItems: 2).Count, "max items");
        Assert.Equal(3, Sam3Nms.Suppress(items, 0.5f, maxItems: 0).Count, "unlimited");
        Assert.Equal(3, Sam3Nms.Suppress(items, 0f, maxItems: 0).Count, "iou <= 0 disables nms");
    }

    private static void MaskScale()
    {
        var result = new Sam3Result(Array.Empty<Sam3Item>(), 1f,
            imageWidth: 1568, imageHeight: 1068,
            embeddedWidth: 1008, embeddedHeight: 687,
            imageSize: 1008, maskSize: 288);

        // 掩码 288 覆盖的是 1008 的嵌入画布；嵌入画布按 (1008/1568, 687/1068) 缩放过，
        // 所以源图坐标 = mask × (1008/288) × (源尺寸/嵌入尺寸)。
        Assert.Near(1568f / 288f, result.MaskScaleX, 1e-3f, "scale x");
        Assert.Near(1008f / 288f * 1068f / 687f, result.MaskScaleY, 1e-3f, "scale y");
    }

    private static Sam3Item Item(float score, float x1, float y1, float x2, float y2)
        => new(score, new Sam3Box(x1, y1, x2, y2), new float[1], 1, 1);
}
