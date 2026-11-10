namespace OnnxSharp.Ocr.PaddleOCR;

/// <summary>
/// PaddleOCR 流水线的参数。默认值对齐 PaddleOCR 官方实现：
/// 检测最长边 960、二值化阈值 0.3、框平均分 0.6、外扩 1.6、识别高 48。
/// </summary>
public sealed class PaddleOcrOptions
{
    /// <summary>检测输入的最长边上限（PaddleOCR 的 <c>limit_side_len</c>）。大图小字调大（如 1600）。</summary>
    public int DetLimitSide { get; set; } = 960;

    /// <summary>概率图二值化阈值。</summary>
    public float DetThreshold { get; set; } = 0.3f;

    /// <summary>文本框内概率平均分阈值。</summary>
    public float BoxThreshold { get; set; } = 0.6f;

    /// <summary>面积/周长外扩比例（PaddleOCR 的 <c>unclip_ratio</c>）。</summary>
    public float UnclipRatio { get; set; } = 1.6f;

    /// <summary>是否用文本行方向分类（模型没有 cls 时自动跳过）。</summary>
    public bool UseClassification { get; set; } = true;

    /// <summary>方向分类判定为 180° 的概率阈值。</summary>
    public float UpsideDownThreshold { get; set; } = 0.9f;

    /// <summary>识别输入高度（PP-OCRv6 固定 48）。</summary>
    public int RecHeight { get; set; } = 48;

    /// <summary>识别输入宽度上限；超过的行会被压到该宽度（PaddleOCR 的 <c>rec_image_shape</c> 上限）。</summary>
    public int RecMaxWidth { get; set; } = 640;

    /// <summary>
    /// 检测网络的缓存份数。每个 <c>(H, W)</c> 组合都要导入+折叠一份图（约 50 ms / 10 MB），
    /// 缓存让同一尺寸的后续调用零成本；超出后整表丢弃（列表很短，没必要做 LRU）。
    /// </summary>
    public int DetShapeCacheSize { get; set; } = 2;

    /// <summary>识别网络的缓存份数（宽度分桶，每档一份图）。</summary>
    public int RecShapeCacheSize { get; set; } = 8;
}
