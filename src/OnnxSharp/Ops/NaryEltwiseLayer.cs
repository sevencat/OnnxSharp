using OnnxSharp.Core;
using OnnxSharp.Tensors;

namespace OnnxSharp.Ops;

/// <summary>二元逐元素算子（Add/Sub/Mul/Div），支持 ONNX 右对齐广播。</summary>
public sealed class NaryEltwiseLayer : Layer
{
    private readonly BinaryOpKind _op;

    private NaryEltwiseLayer(LayerParams parameters, BinaryOpKind op)
    {
        Type = parameters.Type;
        Name = parameters.Name;
        _op = op;
    }

    public static void Register()
    {
        LayerRegistry.Register("Add", p => new NaryEltwiseLayer(p, BinaryOpKind.Add));
        LayerRegistry.Register("Sub", p => new NaryEltwiseLayer(p, BinaryOpKind.Sub));
        LayerRegistry.Register("Mul", p => new NaryEltwiseLayer(p, BinaryOpKind.Mul));
        LayerRegistry.Register("Div", p => new NaryEltwiseLayer(p, BinaryOpKind.Div));
        LayerRegistry.Register("Max", p => new NaryEltwiseLayer(p, BinaryOpKind.Max));
        LayerRegistry.Register("Min", p => new NaryEltwiseLayer(p, BinaryOpKind.Min));
    }

    public override void GetMemoryShapes(
        IReadOnlyList<TensorShape> inShapes,
        int requiredOutputs,
        List<TensorShape> outShapes,
        List<TensorShape> tempShapes)
    {
        if (inShapes.Count != 2)
            throw new OnnxSharpException($"'{Type}' expects 2 inputs, got {inShapes.Count}");
        outShapes.Add(TensorOps.BroadcastShape(inShapes[0], inShapes[1]));
    }

    public override void Forward(
        IReadOnlyList<Tensor> inputs,
        IReadOnlyList<Tensor> outputs,
        IReadOnlyList<Tensor> temps)
        => TensorOps.ElementWiseBinary(_op, inputs[0], inputs[1], outputs[0]);
}
