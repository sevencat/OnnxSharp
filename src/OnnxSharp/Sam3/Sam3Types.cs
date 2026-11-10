namespace OnnxSharp.Sam3;

/// <summary>提示框在模型输入里的坐标约定（SAM3 的 <c>geometry-encoder</c> 吃 <c>[1,N,4]</c>）。</summary>
public enum Sam3PromptBoxFormat
{
    /// <summary>中心点 + 宽高，除以模型边长归一化到 [0,1]（SAM2/3 的默认导出约定）。</summary>
    CxcywhNormalized,

    /// <summary>左上 + 右下，除以模型边长归一化到 [0,1]。</summary>
    XyxyNormalized,

    /// <summary>左上 + 右下，直接用模型像素坐标（0..1008）。</summary>
    XyxyModelPixels,
}

/// <summary>轴对齐框（坐标原点左上角；作为提示时是<b>源图像素</b>）。</summary>
public readonly record struct Sam3Box(float X1, float Y1, float X2, float Y2)
{
    public float Width => X2 - X1;

    public float Height => Y2 - Y1;

    public override string ToString() => $"[{X1:F1},{Y1:F1} - {X2:F1},{Y2:F1}]";
}

/// <summary>
/// SAM3 分段参数。默认值对齐官方导出：图像 1008²、掩码 288²、分数阈值 0.3、掩码阈值 0.5。
/// </summary>
public sealed class Sam3Options
{
    /// <summary>fp32 模型目录（含 <c>vision-encoder.onnx</c> / <c>geometry-encoder.onnx</c> / <c>decoder.onnx</c>）。</summary>
    public string ModelDirectory { get; set; } = string.Empty;

    /// <summary>编码器输入边长（导出固定 1008）。</summary>
    public int ImageSize { get; set; } = 1008;

    /// <summary>解码器输出的掩码边长（导出固定 288）。</summary>
    public int MaskSize { get; set; } = 288;

    /// <summary>分数阈值：<c>sigmoid(pred_logits) × sigmoid(presence_logits)</c>。</summary>
    public float ScoreThreshold { get; set; } = 0.3f;

    /// <summary>
    /// 框级 NMS 的 IoU 阈值。SAM3 一次会输出 200 个 query，同一目标通常有多个高度重叠的候选，
    /// 不做抑制会看到一串几乎一样的框（实测同一人 10+ 个）。&lt;= 0 表示关闭 NMS。
    /// </summary>
    public float IouThreshold { get; set; } = 0.5f;

    /// <summary>NMS 之后最多保留多少个结果；0 = 不限制。</summary>
    public int MaxItems { get; set; }

    /// <summary>提示框的 label（SAM3 的 <c>input_boxes_labels</c>；官方约定 1 = 框，0 = 空提示）。</summary>
    public long BoxLabel { get; set; } = 1;

    /// <summary>提示框坐标约定。</summary>
    public Sam3PromptBoxFormat BoxFormat { get; set; } = Sam3PromptBoxFormat.CxcywhNormalized;

    /// <summary>geometry-encoder / decoder 的形状缓存份数（每个框数 / 提示长度各一份图）。</summary>
    public int ShapeCacheSize { get; set; } = 4;

    /// <summary>诊断输出（每个中间张量的统计）。null = 不输出。</summary>
    public Action<string>? Diagnostic { get; set; }
}

/// <summary>一个分割结果：分数、源图坐标的框、<c>MaskSize²</c> 的原始 logits 掩码。</summary>
public sealed record Sam3Item(float Score, Sam3Box Box, float[] Mask, int MaskWidth, int MaskHeight);

/// <summary>一次 <see cref="Sam3Segmenter.Segment"/> 的结果。</summary>
public sealed class Sam3Result
{
    internal Sam3Result(IReadOnlyList<Sam3Item> items, float presenceScore, int imageWidth, int imageHeight,
        int embeddedWidth, int embeddedHeight, int imageSize, int maskSize)
    {
        Items = items;
        PresenceScore = presenceScore;
        ImageWidth = imageWidth;
        ImageHeight = imageHeight;
        EmbeddedWidth = embeddedWidth;
        EmbeddedHeight = embeddedHeight;
        ImageSize = imageSize;
        MaskSize = maskSize;
    }

    public IReadOnlyList<Sam3Item> Items { get; }

    /// <summary><c>sigmoid(presence_logits)</c>：整张图「有没有目标」的分数。</summary>
    public float PresenceScore { get; }

    public int ImageWidth { get; }

    public int ImageHeight { get; }

    /// <summary>letterbox 之后实际占用的模型像素宽（左上对齐，其余是白边）。</summary>
    public int EmbeddedWidth { get; }

    public int EmbeddedHeight { get; }

    public int ImageSize { get; }

    public int MaskSize { get; }

    /// <summary>
    /// 掩码坐标 → 源图坐标的缩放比例：掩码 <c>(mx,my)</c> 对应源图
    /// <c>(mx × MaskScaleX, my × MaskScaleY)</c>。
    /// </summary>
    public float MaskScaleX => (float)ImageSize / MaskSize * ImageWidth / EmbeddedWidth;

    public float MaskScaleY => (float)ImageSize / MaskSize * ImageHeight / EmbeddedHeight;
}
