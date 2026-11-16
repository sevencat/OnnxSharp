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
    /// <summary>NCHW 直通（默认布局，其余算子全部按它实现）。</summary>
    NCHW = 0,

    /// <summary>
    /// 通道分块 16：物理顺序 <c>[C/16][H][W][16]</c>，同一个像素的 16 个通道连续
    /// （正好 1 个 <c>zmm</c>）。逻辑形状仍是 NCHW，元素数不变，只是顺序不同。
    /// <para>
    /// 只有 <see cref="Kernels.NchwcConv"/> 认识它；进出该布局的算子必须显式
    /// reorder（<see cref="Kernels.NchwcConv.PackInput"/> / <see cref="Kernels.NchwcConv.UnpackOutput"/>）。
    /// 阶段 1 仅 yolo 的 3×3/s2 层使用。
    /// </para>
    /// </summary>
    NCHWc16 = 1,
}
