using System.Diagnostics;
using OnnxSharp.Tensors;

namespace OnnxSharp.Core;

/// <summary>
/// 线性解释执行器：按 <c>prog</c> 顺序逐算子推断形状、绑定缓冲、执行。
/// 对应 OpenCV <c>Net::Impl::forwardGraph</c>（<c>net_impl2.cpp:1750</c>）。
/// </summary>
internal static class Interpreter
{
    private static readonly IReadOnlyList<Tensor> NoTemps = Array.Empty<Tensor>();

    /// <summary>
    /// 关掉「按输入签名缓存形状/类型推断」（<c>ONNXSHARP_NO_SHAPECACHE=1</c>）。
    /// 只用于 A/B 度量：本机负载抖动很大，性能结论必须靠交替测量。
    /// </summary>
    private static readonly bool DisableShapeCache =
        Environment.GetEnvironmentVariable("ONNXSHARP_NO_SHAPECACHE") is { Length: > 0 };

    /// <summary>
    /// 执行整张图。<paramref name="executeLayers"/> = false 时只做形状/类型推断与缓冲绑定，
    /// 不调用 <see cref="Layer.Forward"/>（供 `dump-graph` 与形状对齐测试使用）。
    /// <para>
    /// <paramref name="timings"/> 非空时累计<b>每个算子</b>的耗时（诊断用，
    /// 对应 OpenCV 的 <c>ENABLE_PROFILING</c> / <c>getPerfProfile</c>）。
    /// 计时本身有开销，只在 profile 时传。
    /// </para>
    /// </summary>
    public static void Run(
        NetImpl net,
        Graph graph,
        bool executeLayers = true,
        Dictionary<LayerInfo, (double Milliseconds, int Calls)>? timings = null,
        bool detectNaN = false)
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
            if (op is not Layer layer)
                throw new OnnxSharpException($"op '{op.Name}' ({op.Type}) has no executor");

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

            // 签名（输入形状 + 类型）只比一次：缓存命中判定和 Finalize 判定用的是同一份数据，
            // 之前这里比了两遍（2076 个节点的 decoder 上就是几千次多余的数组遍历）。
            bool sameSignature = layer.ShapeCacheValid
                && SameShapes(layer.CachedInShapes, inpShapes)
                && SameTypes(layer.CachedInTypes, inpTypes);

            if (!op.DynamicOutputShapes())
            {
                bool cacheHit = !DisableShapeCache && sameSignature;

                if (cacheHit)
                {
                    outs = BindAll(net, op, layer.CachedOutShapes, layer.CachedOutTypes, ref outMats);
                    temps = net.FitTemps(layer.CachedTempShapes, layer.CachedTempTypes);
                }
                else
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

                    // 存成缓存副本（形状推断第二次调用会改写 outShapes 等列表）
                    layer.CachedInShapes.Clear();
                    layer.CachedInShapes.AddRange(inpShapes);
                    layer.CachedInTypes.Clear();
                    layer.CachedInTypes.AddRange(inpTypes);
                    layer.CachedOutShapes.Clear();
                    layer.CachedOutShapes.AddRange(outShapes);
                    layer.CachedTempShapes.Clear();
                    layer.CachedTempShapes.AddRange(tempShapes);
                    layer.CachedOutTypes.Clear();
                    layer.CachedOutTypes.AddRange(outTypes);
                    layer.CachedTempTypes.Clear();
                    layer.CachedTempTypes.AddRange(tempTypes);
                    layer.ShapeCacheValid = true;

                    outs = BindAll(net, op, outShapes, outTypes, ref outMats);
                    temps = net.FitTemps(tempShapes, tempTypes);
                }
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

            if (!executeLayers)
            {
                // 形状对齐模式：动态 op 的形状无法在不执行的情况下推断
                if (op.DynamicOutputShapes())
                    throw new OnnxSharpException(
                        $"op '{op.Name}' ({op.Type}) has dynamic output shapes; " +
                        "shape-only mode cannot infer them");
                continue;
            }

            bool signatureChanged = !layer.FinalizedOnce || !sameSignature;

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

            if (timings is null)
            {
                layer.Forward(inpMats, outs, temps);
            }
            else
            {
                long start = Stopwatch.GetTimestamp();
                layer.Forward(inpMats, outs, temps);

                double ms = Stopwatch.GetElapsedTime(start).TotalMilliseconds;
                var acc = timings.GetValueOrDefault(layer);
                timings[layer] = (acc.Milliseconds + ms, acc.Calls + 1);
            }

            if (detectNaN)
                CheckNaN(layer, outs);

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

    /// <summary>
    /// 调试用：检查该算子的每个 F32 输出是否含 NaN（第一个命中的输出直接抛，带算子名与形状）。
    /// 只在 <c>detectNaN</c> 打开时调用；共享缓冲槽不参与判断（只看本算子刚写完的输出）。
    /// </summary>
    private static void CheckNaN(Layer layer, IReadOnlyList<Tensor> outputs)
    {
        for (int i = 0; i < outputs.Count; i++)
        {
            var tensor = outputs[i];
            if (tensor is null || tensor.DType != DType.F32 || tensor.Count == 0)
                continue;

            var span = tensor.AsSpan<float>();
            for (int j = 0; j < span.Length; j++)
            {
                if (float.IsNaN(span[j]))
                {
                    throw new OnnxSharpException(
                        $"op '{layer.Name}' ({layer.Type}) output #{i} {tensor.Shape} contains NaN at index {j}");
                }
            }
        }
    }

    /// <summary>按缓存的形状/类型把输出 Arg 绑到缓冲区（复用 <paramref name="buffer"/>）。</summary>
    private static List<Tensor> BindAll(
        NetImpl net,
        LayerInfo op,
        List<TensorShape> shapes,
        List<DType> types,
        ref List<Tensor> buffer)
    {
        buffer.Clear();
        for (int i = 0; i < op.Outputs.Count; i++)
            buffer.Add(net.BindArgTensor(op.Outputs[i], shapes[i], types[i]));

        return buffer;
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
