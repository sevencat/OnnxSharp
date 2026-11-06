using OnnxSharp.Tensors;

namespace OnnxSharp.Core;

/// <summary>
/// 线性解释执行器：按 <c>prog</c> 顺序逐算子推断形状、绑定缓冲、执行。
/// 对应 OpenCV <c>Net::Impl::forwardGraph</c>。
/// </summary>
internal static class Interpreter
{
    public static void Run(NetImpl net, Graph graph)
    {
        var inpMats = new List<Tensor>();
        var inpShapes = new List<TensorShape>();
        var inpTypes = new List<DType>();

        var outMats = new List<Tensor>();
        var outShapes = new List<TensorShape>();
        var outTypes = new List<DType>();

        var tempMats = new List<Tensor>();
        var tempShapes = new List<TensorShape>();
        var tempTypes = new List<DType>();

        foreach (var op in graph.Prog)
        {
            inpMats.Clear();
            inpShapes.Clear();
            inpTypes.Clear();
            foreach (var arg in op.Inputs)
            {
                var t = net.GetArgTensor(arg);
                inpMats.Add(t);
                inpShapes.Add(t.Shape);
                inpTypes.Add(t.DType);
            }

            outShapes.Clear();
            tempShapes.Clear();
            op.GetMemoryShapes(inpShapes, op.Outputs.Count, outShapes, tempShapes);

            if (outShapes.Count != op.Outputs.Count)
                throw new OnnxSharpException(
                    $"op '{op.Name}' ({op.Type}) produced {outShapes.Count} output shapes, {op.Outputs.Count} expected");

            outTypes.Clear();
            tempTypes.Clear();
            op.GetTypes(inpTypes, op.Outputs.Count, tempShapes.Count, outTypes, tempTypes);

            outMats.Clear();
            for (int i = 0; i < op.Outputs.Count; i++)
                outMats.Add(net.BindArgTensor(op.Outputs[i], outShapes[i], outTypes[i]));

            tempMats.Clear();
            for (int i = 0; i < tempShapes.Count; i++)
                tempMats.Add(Tensor.Allocate(tempShapes[i], tempTypes[i]));

            if (op is not Layer layer)
                throw new OnnxSharpException($"op '{op.Name}' ({op.Type}) has no executor");

            bool shapesChanged = !layer.FinalizedOnce || !SameShapes(layer.LastInShapes, inpShapes);
            if (shapesChanged)
            {
                layer.PrepackWeights();
                layer.Finalize(inpMats, outMats);
                layer.LastInShapes.Clear();
                layer.LastInShapes.AddRange(inpShapes);
                layer.FinalizedOnce = true;
            }

            layer.Forward(inpMats, outMats, tempMats);
        }
    }

    private static bool SameShapes(List<TensorShape> a, List<TensorShape> b)
    {
        if (a.Count != b.Count)
            return false;
        for (int i = 0; i < a.Count; i++)
        {
            if (!a[i].Equals(b[i]))
                return false;
        }
        return true;
    }
}
