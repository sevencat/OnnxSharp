using OnnxSharp.Core;
using OnnxSharp.Tensors;

namespace OnnxSharp.Tests;

/// <summary>
/// 模拟 <c>Reshape</c>/<c>Slice</c> 这类「形状由常量输入决定」的算子：
/// 在 <see cref="GetMemoryShapes"/> 里用 <see cref="LayerInfo.In"/> 读常量值。
/// 这条链路是 M0.5 的核心验证点（OpenCV 的 <c>getNetImpl(this)-&gt;argTensor(...)</c>）。
/// </summary>
internal sealed class ReadConstShapeLayer : Layer
{
    /// <summary>记录最后一次拿到的 temp 引用，用于验证全局 temp 池跨 forward 复用。</summary>
    public Tensor? LastTemp;

    public override void GetMemoryShapes(
        IReadOnlyList<TensorShape> inShapes,
        int requiredOutputs,
        List<TensorShape> outShapes,
        List<TensorShape> tempShapes)
    {
        if (!IsConstInput(1))
            throw new InvalidOperationException($"{Name}: input #1 must be a constant");

        int n = (int)In(1).AsSpan<long>()[0];
        outShapes.Add(new TensorShape(n));
        tempShapes.Add(new TensorShape(4));
    }

    public override void Forward(
        IReadOnlyList<Tensor> inputs,
        IReadOnlyList<Tensor> outputs,
        IReadOnlyList<Tensor> temps)
    {
        LastTemp = temps[0];

        var src = inputs[0].AsSpan<float>();
        var dst = outputs[0].AsSpan<float>();
        for (int i = 0; i < dst.Length; i++)
            dst[i] = src[i % src.Length];
    }
}

/// <summary>
/// 模拟 <c>Where/Expand/ConstantOfShape</c> 这类输出形状由运行期数值决定的算子：
/// <see cref="DynamicOutputShapes"/> 为 true，输出**没有被预先 Fit**，必须自己 Fit。
/// </summary>
internal sealed class DynamicFillLayer : Layer
{
    private readonly float _value;

    public DynamicFillLayer(float value) => _value = value;

    public override bool DynamicOutputShapes() => true;

    public override void Forward(
        IReadOnlyList<Tensor> inputs,
        IReadOnlyList<Tensor> outputs,
        IReadOnlyList<Tensor> temps)
    {
        int n = (int)inputs[0].AsSpan<long>()[0];

        // 引擎没有替我们 Fit，这里必须自己做（对齐 OpenCV net_impl2.cpp:1891-1898）。
        outputs[0].Fit(new TensorShape(n), DType.F32);
        outputs[0].AsSpan<float>().Fill(_value);
    }
}
