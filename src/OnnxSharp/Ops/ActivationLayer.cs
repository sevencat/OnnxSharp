using OnnxSharp.Core;
using OnnxSharp.Tensors;

namespace OnnxSharp.Ops;

/// <summary>一元逐元素算子（Relu/Sigmoid/Sqrt/Exp/Neg/Abs）。</summary>
public sealed class ActivationLayer : Layer
{
    private readonly UnaryOpKind _op;

    private ActivationLayer(LayerParams parameters, UnaryOpKind op)
    {
        Type = parameters.Type;
        Name = parameters.Name;
        _op = op;
    }

    public static void Register()
    {
        LayerRegistry.Register("Relu", p => new ActivationLayer(p, UnaryOpKind.Relu));
        LayerRegistry.Register("Sigmoid", p => new ActivationLayer(p, UnaryOpKind.Sigmoid));
        LayerRegistry.Register("Sqrt", p => new ActivationLayer(p, UnaryOpKind.Sqrt));
        LayerRegistry.Register("Exp", p => new ActivationLayer(p, UnaryOpKind.Exp));
        LayerRegistry.Register("Neg", p => new ActivationLayer(p, UnaryOpKind.Neg));
        LayerRegistry.Register("Abs", p => new ActivationLayer(p, UnaryOpKind.Abs));
        LayerRegistry.Register("Erf", p => new ActivationLayer(p, UnaryOpKind.Erf));
        LayerRegistry.Register("Sin", p => new ActivationLayer(p, UnaryOpKind.Sin));
        LayerRegistry.Register("Cos", p => new ActivationLayer(p, UnaryOpKind.Cos));

        // 不是 ONNX 算子：图级融合（Sigmoid → Mul）的产物，见 GraphFusion。
        LayerRegistry.Register("Silu", p => new ActivationLayer(p, UnaryOpKind.Silu));
    }

    public override void GetMemoryShapes(
        IReadOnlyList<TensorShape> inShapes,
        int requiredOutputs,
        List<TensorShape> outShapes,
        List<TensorShape> tempShapes)
    {
        if (inShapes.Count != 1)
            throw new OnnxSharpException($"'{Type}' expects 1 input, got {inShapes.Count}");
        outShapes.Add(inShapes[0]);
    }

    public override void Forward(
        IReadOnlyList<Tensor> inputs,
        IReadOnlyList<Tensor> outputs,
        IReadOnlyList<Tensor> temps)
        => TensorOps.ElementWiseUnary(_op, inputs[0], outputs[0]);
}
