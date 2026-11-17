namespace OnnxSharp.Ocr.PaddleOCR;

/// <summary>
/// 一次 <see cref="PaddleOcrPipeline.Run"/> 的分段耗时（毫秒）。
/// 只在 <see cref="PaddleOcrOptions.CollectTimings"/> 打开时产生，用于定位「慢在哪一段」。
/// </summary>
/// <param name="Preprocess">检测输入预处理（缩放 + 归一化 + 转 CHW）。</param>
/// <param name="DetForward">DB 检测网络前向。</param>
/// <param name="DetPostprocess">概率图 → 文本框（二值化 + 连通域 + 外扩）。</param>
/// <param name="Classification">所有文本行的方向分类（含裁剪与预处理）。</param>
/// <param name="Recognition">所有文本行的识别（含裁剪、预处理、CTC 解码）。</param>
public sealed record OcrTimings(
    double Preprocess,
    double DetForward,
    double DetPostprocess,
    double Classification,
    double Recognition)
{
    /// <summary>五段之和（不含框间调度开销，故略小于端到端）。</summary>
    public double Total => Preprocess + DetForward + DetPostprocess + Classification + Recognition;

    public override string ToString() =>
        $"pre {Preprocess:F2}  det {DetForward:F2}  post {DetPostprocess:F2}  " +
        $"cls {Classification:F2}  rec {Recognition:F2}  = {Total:F2} ms";
}
