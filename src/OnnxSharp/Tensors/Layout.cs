namespace OnnxSharp.Tensors;

/// <summary>
/// 张量存储布局。
/// <para>
/// 首版只有 NCHW 一种取值，<b>不做图级 blocked layout</b>（OpenCV 的 NC8HW8 + <c>TransformLayout</c>）。
/// 本枚举与 <see cref="Tensor.Layout"/>、<see cref="Core.ArgData.Layout"/>、
/// <see cref="Core.LayerInfo.GetLayouts"/> 一起构成接口预留：将来真要做 blocked layout 时
/// 是增量扩展而非返工，成本已被压到接近零。
/// </para>
/// <para>
/// 注意区分：kernel 内部的列面板打包（SimdYolo 的 16/24/48 列 panel）属于 kernel 私有实现，
/// <b>不体现在本枚举上</b>，张量对外永远是 NCHW。
/// </para>
/// </summary>
public enum Layout
{
    /// <summary>NCHW 直通（唯一已实现布局）。</summary>
    NCHW = 0,
}
