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
        bool detectNaN = false,
        Dictionary<LayerInfo, long>? allocs = null)
    {
        // 绑定 LayerInfo.Net 并按生命周期分槽（图结构变化时才重算）。
        net.PrepareForGraph(graph);

        var inpMats = new List<Tensor>();
        var inpShapes = new List<TensorShape>();
        var inpTypes = new List<DType>();

        var outMats = new List<Tensor>();
        var dynamicOut = new List<Tensor>();

        foreach (var op in graph.Prog)
        {
            if (op is not Layer layer)
                throw new OnnxSharpException($"op '{op.Name}' ({op.Type}) has no executor");

            // 分配归因要覆盖整层的全部动作：形状推断、BindAll（输出 Fit）、
            // PrepackWeights/Finalize（动态形状层每帧都会重跑）。只包 Forward 会漏掉大头。
            long allocBefore = allocs is null ? 0 : GC.GetTotalAllocatedBytes(precise: true);

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

            // 输入签名（shape + type）只比一次：缓存复用与 Finalize 失效判定读同一个事实，
            // 之前这里比了两遍，且两套签名分别维护（动态形状层那套永远对不上）。
            var plan = layer.Plan;
            bool sameSignature = plan.Matches(inpShapes, inpTypes);
            bool dynamicOutShapes = op.DynamicOutputShapes();

            if (dynamicOutShapes)
            {
                // 动态形状：输出**不预 fit**，由 layer 自己在 Forward 里 Fit（对齐 net_impl2.cpp:1891-1898）。
                dynamicOut.Clear();
                foreach (var arg in op.Outputs)
                    dynamicOut.Add(net.GetArgTensor(arg));

                temps = NoTemps;
                outs = dynamicOut;
            }
            else if (!DisableShapeCache && plan.State >= PlanState.Shaped && sameSignature)
            {
                outs = BindAll(net, op, plan.OutShapes, plan.OutTypes, ref outMats);
                temps = net.FitTemps(plan.TempShapes, plan.TempTypes);
            }
            else
            {
                plan.OutShapes.Clear();
                plan.TempShapes.Clear();
                op.GetMemoryShapes(inpShapes, op.Outputs.Count, plan.OutShapes, plan.TempShapes);

                if (plan.OutShapes.Count != op.Outputs.Count)
                    throw new OnnxSharpException(
                        $"op '{op.Name}' ({op.Type}) produced {plan.OutShapes.Count} output shapes, {op.Outputs.Count} expected");

                plan.OutTypes.Clear();
                plan.TempTypes.Clear();
                op.GetTypes(inpTypes, op.Outputs.Count, plan.TempShapes.Count, plan.OutTypes, plan.TempTypes);

                if (plan.OutTypes.Count != op.Outputs.Count || plan.TempTypes.Count != plan.TempShapes.Count)
                    throw new OnnxSharpException(
                        $"op '{op.Name}' ({op.Type}) produced mismatched types " +
                        $"({plan.OutTypes.Count} outputs, {plan.TempTypes.Count} temps)");

                plan.WriteInputs(inpShapes, inpTypes);
                if (plan.State < PlanState.Shaped)
                    plan.State = PlanState.Shaped;

                outs = BindAll(net, op, plan.OutShapes, plan.OutTypes, ref outMats);
                temps = net.FitTemps(plan.TempShapes, plan.TempTypes);
            }

            if (!executeLayers)
            {
                // 形状对齐模式：动态 op 的形状无法在不执行的情况下推断
                if (dynamicOutShapes)
                    throw new OnnxSharpException(
                        $"op '{op.Name}' ({op.Type}) has dynamic output shapes; " +
                        "shape-only mode cannot infer them");

                if (allocs is not null)
                    allocs[op] = allocs.GetValueOrDefault(op) + (GC.GetTotalAllocatedBytes(precise: true) - allocBefore);
                continue;
            }

            // 第三段：Finalize。与形状缓存读**同一条签名**，动态形状层也在这里把签名写进去，
            // 所以它下一帧就能判定「没变」—— 之前两套签名不同步，动态层每帧重跑
            // PrepackWeights + Finalize，rfdetr-small 上等于每帧白分配 52 MB。
            if (plan.State != PlanState.Finalized || !sameSignature)
            {
                plan.WriteInputs(inpShapes, inpTypes);   // 动态层没走形状分支，签名在这里补写

                layer.PrepackWeights();
                layer.Finalize(inpMats, outs);
                plan.State = PlanState.Finalized;
            }

            if (timings is null && allocs is null)
            {
                layer.Forward(inpMats, outs, temps);
            }
            else
            {
                // 分配量用 precise 口径：它会把其他工作线程尚未归集的 TLS 缓冲一起刷进来，
                // 否则 Conv/GEMM 这类多线程层的分配会被漏掉。代价是每次调用要扫一遍线程，
                // 所以只在诊断（传了 timings/allocs）时开。
                long start = timings is null ? 0 : Stopwatch.GetTimestamp();

                layer.Forward(inpMats, outs, temps);

                if (timings is not null)
                {
                    double ms = Stopwatch.GetElapsedTime(start).TotalMilliseconds;
                    var acc = timings.GetValueOrDefault(layer);
                    timings[layer] = (acc.Milliseconds + ms, acc.Calls + 1);
                }

                if (allocs is not null)
                {
                    long delta = GC.GetTotalAllocatedBytes(precise: true) - allocBefore;
                    allocs[layer] = allocs.GetValueOrDefault(layer) + delta;
                }
            }

            if (detectNaN)
                CheckNaN(layer, outs);

            if (dynamicOutShapes)
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
}
