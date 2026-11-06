using OnnxSharp.Tensors;

namespace OnnxSharp.Core;

/// <summary>
/// 常量折叠：把「所有输入都是常量」的算子提前算掉，结果写回成常量 Arg 并从图上删除该节点。
/// <para>
/// 对应 OpenCV 的 <c>ConstFolder</c>（<c>net_impl_fuse.cpp</c>）。RF-DETR 大量用
/// <c>Shape → Gather → Unsqueeze → Concat → Reshape</c> 这条链计算 reshape/expand 目标形状，
/// 不折叠的话 <c>Reshape</c> 的 shape 输入永远是「运行时才知道」，形状推断直接卡死。
/// </para>
/// <para>
/// 折叠是就地改写 Arg（不新增 Arg、不改引用），所以必须在<b>缓冲槽分配之前</b>、并且在
/// 任何一次真正的 forward 之前跑。
/// </para>
/// </summary>
internal static class ConstFolder
{
    /// <summary>执行折叠，返回被折叠掉的节点数。</summary>
    public static int Run(NetImpl net, Graph graph)
    {
        net.PrepareForGraph(graph);

        int total = 0;
        bool changed = true;

        while (changed)
        {
            changed = false;

            // 先做一轮形状传播：`Shape` 之类「只看形状」的算子要靠它拿到静态形状才能折叠
            if (PropagateShapes(net, graph) > 0)
                changed = true;

            foreach (var op in graph.Prog.ToList())
            {
                if (op is not Layer layer)
                    continue;

                // 动态形状的算子无法在不跑图的情况下定型；宁可不折叠
                if (op.DynamicOutputShapes() || op.Inputs.Count == 0)
                    continue;

                bool folded;

                if (AllInputsConst(net, op))
                {
                    folded = Fold(net, layer);
                }
                else if (TryFoldByShape(net, layer))
                {
                    folded = true;
                }
                else
                {
                    continue;
                }

                if (!folded)
                    continue;

                graph.RemoveAll(candidate => ReferenceEquals(candidate, op));
                total++;
                changed = true;
            }
        }

        return total;
    }

    /// <summary>
    /// 静态形状传播：按拓扑序对「输入形状全部已知」的算子跑一遍 <see cref="LayerInfo.GetMemoryShapes"/>
    /// 并写回 <see cref="ArgData.Shape"/>。返回新确定形状的 Arg 数量。
    /// <para>
    /// 很多 ONNX 导出（如 RF-DETR）不带 <c>value_info</c>，中间张量形状完全未知；
    /// 而 <c>Shape</c> 折叠、<c>Reshape</c> 目标形状都依赖它。推断失败（需要常量输入等）就跳过，
    /// 等下一轮折叠后再试。
    /// </para>
    /// </summary>
    private static int PropagateShapes(NetImpl net, Graph graph)
    {
        int resolved = 0;

        foreach (var op in graph.Prog)
        {
            if (op.DynamicOutputShapes() || op.Inputs.Count == 0 || op.Outputs.Count == 0)
                continue;

            var inShapes = new List<TensorShape>(op.Inputs.Count);
            var inTypes = new List<DType>(op.Inputs.Count);
            bool ready = true;

            foreach (var arg in op.Inputs)
            {
                if (arg.Idx == 0)
                {
                    inShapes.Add(TensorShape.Scalar);
                    inTypes.Add(DType.F32);
                    continue;
                }

                var data = net.GetArgData(arg);
                if (!data.HasShape)
                {
                    ready = false;
                    break;
                }

                inShapes.Add(data.Shape);
                inTypes.Add(data.DType);
            }

            if (!ready)
                continue;

            var outShapes = new List<TensorShape>();
            var tempShapes = new List<TensorShape>();
            var outTypes = new List<DType>();
            var tempTypes = new List<DType>();

            try
            {
                op.GetMemoryShapes(inShapes, op.Outputs.Count, outShapes, tempShapes);
                op.GetTypes(inTypes, op.Outputs.Count, tempShapes.Count, outTypes, tempTypes);
            }
            catch (OnnxSharpException)
            {
                continue;   // 需要常量输入等；等下一轮
            }

            if (outShapes.Count != op.Outputs.Count)
                continue;

            for (int i = 0; i < op.Outputs.Count; i++)
            {
                var arg = op.Outputs[i];
                if (arg.Idx == 0)
                    continue;

                var data = net.GetArgData(arg);
                if (data.Kind is ArgKind.Const or ArgKind.Input || data.HasShape)
                    continue;

                data.Shape = outShapes[i];
                data.HasShape = true;
                data.DType = outTypes[i];
                resolved++;
            }
        }

        return resolved;
    }

    private static bool AllInputsConst(NetImpl net, LayerInfo op)
    {
        foreach (var arg in op.Inputs)
        {
            // Idx == 0 是「可选输入缺失」的占位
            if (arg.Idx == 0)
                continue;

            if (!net.IsConstArg(arg))
                return false;
        }

        return true;
    }

    /// <summary>
    /// 输入是运行时张量、但<b>形状静态已知</b>时也能折叠的算子（如 <c>Shape</c>）。
    /// 只有单输出算子走这条路。
    /// </summary>
    private static bool TryFoldByShape(NetImpl net, Layer layer)
    {
        if (layer.Outputs.Count != 1 || layer.Outputs[0].Idx == 0)
            return false;

        var shapes = new List<TensorShape>(layer.Inputs.Count);
        foreach (var arg in layer.Inputs)
        {
            var data = net.GetArgData(arg);
            if (arg.Idx == 0 || !data.HasShape)
                return false;
            shapes.Add(data.Shape);
        }

        if (!layer.TryFoldShapeOnly(shapes, out var result) || result is null)
            return false;

        var output = net.GetArgData(layer.Outputs[0]);
        if (output.Kind is ArgKind.Input or ArgKind.Const)
            return false;

        SetConst(net, layer.Outputs[0], result);
        return true;
    }

    private static void SetConst(NetImpl net, Arg arg, Tensor value)
    {
        var data = net.GetArgData(arg);
        data.Kind = ArgKind.Const;
        data.Const = value;
        data.Shape = value.Shape;
        data.HasShape = true;
        data.DType = value.DType;
        data.Layout = value.Layout;
        data.BufferIndex = -1;
        data.Runtime = null;
    }

    private static bool Fold(NetImpl net, Layer layer)
    {
        var inputs = new List<Tensor>(layer.Inputs.Count);
        var inShapes = new List<TensorShape>(layer.Inputs.Count);
        var inTypes = new List<DType>(layer.Inputs.Count);

        foreach (var arg in layer.Inputs)
        {
            var tensor = net.GetArgTensor(arg);
            inputs.Add(tensor);
            inShapes.Add(tensor.Shape);
            inTypes.Add(tensor.DType);
        }

        var outShapes = new List<TensorShape>();
        var tempShapes = new List<TensorShape>();
        var outTypes = new List<DType>();
        var tempTypes = new List<DType>();

        layer.GetMemoryShapes(inShapes, layer.Outputs.Count, outShapes, tempShapes);
        layer.GetTypes(inTypes, layer.Outputs.Count, tempShapes.Count, outTypes, tempTypes);

        if (outShapes.Count != layer.Outputs.Count)
            return false;

        var outputs = new List<Tensor>(outShapes.Count);
        for (int i = 0; i < outShapes.Count; i++)
            outputs.Add(Tensor.Allocate(outShapes[i], outTypes[i]));

        var temps = new List<Tensor>(tempShapes.Count);
        for (int i = 0; i < tempShapes.Count; i++)
            temps.Add(Tensor.Allocate(tempShapes[i], tempTypes[i]));

        layer.PrepackWeights();
        layer.Finalize(inputs, outputs);
        layer.Forward(inputs, outputs, temps);

        for (int i = 0; i < layer.Outputs.Count; i++)
        {
            var arg = layer.Outputs[i];
            if (arg.Idx == 0)
                continue;

            var data = net.GetArgData(arg);
            if (data.Kind is ArgKind.Input or ArgKind.Const)
                continue;   // 已经是常量 / 图输入（in-place 算子把输入当输出时会出现）

            SetConst(net, arg, outputs[i]);
        }

        return true;
    }
}
