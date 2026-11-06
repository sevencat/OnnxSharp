using OnnxSharp.Tensors;

namespace OnnxSharp.Core;

/// <summary>
/// 线性解释执行器：按 <c>prog</c> 顺序逐算子推断形状、绑定缓冲、执行。
/// 对应 OpenCV <c>Net::Impl::forwardGraph</c>（<c>net_impl2.cpp:1750</c>）。
/// </summary>
internal static class Interpreter
{
    private static readonly IReadOnlyList<Tensor> NoTemps = Array.Empty<Tensor>();

    public static void Run(NetImpl net, Graph graph)
    {
        // 绑定 LayerInfo.Net 并按生命周期分槽（图结构变化时才重算）。
        net.PrepareForGraph(graph);

        var inpMats = new List<Tensor>();
        var inpShapes = new List<TensorShape>();
        var inpTypes = new List<DType>();

        var outShapes = new List<TensorShape>();
        var tempShapes = new List<TensorShape>();
        var outTypes = new List<DType>();
        var tempTypes = new List<DType>();

        var outMats = new List<Tensor>();
        var dynamicOut = new List<Tensor>();

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

            IReadOnlyList<Tensor> outs;
            IReadOnlyList<Tensor> temps;

            if (!op.DynamicOutputShapes())
            {
                outShapes.Clear();
                tempShapes.Clear();
                op.GetMemoryShapes(inpShapes, op.Outputs.Count, outShapes, tempShapes);

                if (outShapes.Count != op.Outputs.Count)
                    throw new OnnxSharpException(
                        $"op '{op.Name}' ({op.Type}) produced {outShapes.Count} output shapes, {op.Outputs.Count} expected");

                outTypes.Clear();
                tempTypes.Clear();
                op.GetTypes(inpTypes, op.Outputs.Count, tempShapes.Count, outTypes, tempTypes);

                if (outTypes.Count != op.Outputs.Count || tempTypes.Count != tempShapes.Count)
                    throw new OnnxSharpException(
                        $"op '{op.Name}' ({op.Type}) produced mismatched types " +
                        $"({outTypes.Count} outputs, {tempTypes.Count} temps)");

                outMats.Clear();
                for (int i = 0; i < op.Outputs.Count; i++)
                    outMats.Add(net.BindArgTensor(op.Outputs[i], outShapes[i], outTypes[i]));

                temps = net.FitTemps(tempShapes, tempTypes);
                outs = outMats;
            }
            else
            {
                // 动态形状：输出**不预 fit**，由 layer 自己在 Forward 里 Fit（对齐 net_impl2.cpp:1891-1898）。
                dynamicOut.Clear();
                foreach (var arg in op.Outputs)
                    dynamicOut.Add(net.GetArgTensor(arg));

                temps = NoTemps;
                outs = dynamicOut;
            }

            if (op is not Layer layer)
                throw new OnnxSharpException($"op '{op.Name}' ({op.Type}) has no executor");

            bool signatureChanged = !layer.FinalizedOnce
                || !SameShapes(layer.LastInShapes, inpShapes)
                || !SameTypes(layer.LastInTypes, inpTypes);

            if (signatureChanged)
            {
                layer.PrepackWeights();
                layer.Finalize(inpMats, outs);
                layer.LastInShapes.Clear();
                layer.LastInShapes.AddRange(inpShapes);
                layer.LastInTypes.Clear();
                layer.LastInTypes.AddRange(inpTypes);
                layer.FinalizedOnce = true;
            }

            layer.Forward(inpMats, outs, temps);

            if (op.DynamicOutputShapes())
            {
                // 动态 op 自己决定形状，回填 ArgData 供下游/输出读取。
                for (int i = 0; i < op.Outputs.Count; i++)
                {
                    var data = net.GetArgData(op.Outputs[i]);
                    data.Shape = outs[i].Shape;
                    data.DType = outs[i].DType;
                    data.Layout = outs[i].Layout;
                }
            }
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

    private static bool SameTypes(List<DType> a, List<DType> b)
    {
        if (a.Count != b.Count)
            return false;
        for (int i = 0; i < a.Count; i++)
        {
            if (a[i] != b[i])
                return false;
        }
        return true;
    }
}
