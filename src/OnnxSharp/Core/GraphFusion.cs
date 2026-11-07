using OnnxSharp.Tensors;

namespace OnnxSharp.Core;

/// <summary>
/// 图级融合：把若干个节点合并成一个等价但更省带宽的节点。
/// <para>
/// 与常量折叠的分工：折叠处理「可以提前算掉」的节点（不留下运行时工作），
/// 融合处理「必须留在运行时、但可以合并成一次读写」的节点。两者都在
/// <see cref="NetImpl.PrepareForGraph"/> 分配缓冲槽之前跑，改写 Arg 引用不会失效。
/// </para>
/// </summary>
internal static class GraphFusion
{
    /// <summary>执行所有融合，返回被消掉的节点数。</summary>
    public static int Run(NetImpl net, Graph graph)
    {
        int total = 0;

        // 顺序有讲究：先造出 SiLU，再把它吸收进上游卷积。
        total += FuseSilu(net, graph);
        total += FuseActivationIntoProducer(net, graph);

        return total;
    }

    /// <summary>
    /// 把紧跟在上游算子后面的激活（SiLU / Relu）吸收进上游算子：
    /// 激活层的输出 Arg 改由上游产出，上游在写完输出后原地激活。
    /// <para>
    /// 这样整条链上少一次算子调用、少一轮「读一遍整张输出、再写一遍整张输出」。
    /// 只在上游输出恰好只有这一个消费者时成立，否则上游的其它消费者会拿到激活后的值。
    /// </para>
    /// </summary>
    private static int FuseActivationIntoProducer(NetImpl net, Graph graph)
    {
        var prog = graph.Prog;
        if (prog.Count < 2)
            return 0;

        var consumers = new Dictionary<int, int>();
        var consumerOf = new Dictionary<int, LayerInfo>();

        foreach (var op in prog)
        {
            foreach (var arg in op.Inputs)
            {
                if (arg.IsEmpty)
                    continue;

                consumers[arg.Idx] = consumers.TryGetValue(arg.Idx, out int count) ? count + 1 : 1;
                consumerOf.TryAdd(arg.Idx, op);
            }
        }

        HashSet<LayerInfo>? drop = null;
        int fused = 0;

        foreach (var op in prog)
        {
            if (op is not IActivationFusable fusable || op.Outputs.Count != 1)
                continue;

            var produced = op.Outputs[0];
            if (produced.IsEmpty)
                continue;

            if (!consumers.TryGetValue(produced.Idx, out int uses) || uses != 1)
                continue;

            if (!consumerOf.TryGetValue(produced.Idx, out var actOp) || actOp is not Layer actLayer)
                continue;

            UnaryOpKind? kind = actOp.Type switch
            {
                "Silu" => UnaryOpKind.Silu,
                "Relu" => UnaryOpKind.Relu,
                _ => null,
            };

            if (kind is null || actOp.Inputs.Count != 1 || actOp.Outputs.Count != 1)
                continue;

            if (!ShapesMatch(net, produced, actOp.Outputs[0]))
                continue;

            // 激活层本身不能再被融合走（否则会丢一次激活）
            if (drop is not null && drop.Contains(actOp))
                continue;

            fusable.FusedActivation = kind;
            op.Outputs[0] = actOp.Outputs[0];

            (drop ??= new HashSet<LayerInfo>()).Add(actOp);
            fused++;
        }

        if (drop is null)
            return 0;

        var result = new List<LayerInfo>(prog.Count - drop.Count);
        foreach (var op in prog)
        {
            if (!drop.Contains(op))
                result.Add(op);
        }

        graph.Reset(result);
        return fused;
    }

    /// <summary>
    /// <c>x → Sigmoid → s</c>、<c>Mul(s, x) → y</c> 融成单个 <c>Silu(x) → y</c>。
    /// <para>
    /// ONNX 导出的 SiLU（yolo 系列的每一个卷积后面都有）永远是这两个节点：
    /// 分开跑要写一遍 s、再读 s 和 x 各一遍、再写一遍 y，共 4 个张量流量；
    /// 融成一个内核后只剩读 x、写 y，流量减 2.5 倍，还省一次线程派发。
    /// </para>
    /// <para>
    /// 只融合「Sigmoid 结果仅此一个消费者、且不是图输出」的情形，
    /// 否则删掉 Sigmoid 会破坏其它消费者。
    /// </para>
    /// </summary>
    private static int FuseSilu(NetImpl net, Graph graph)
    {
        var prog = graph.Prog;
        if (prog.Count < 2)
            return 0;

        var producer = new Dictionary<int, LayerInfo>();
        var consumers = new Dictionary<int, int>();

        foreach (var op in prog)
        {
            foreach (var arg in op.Outputs)
            {
                if (arg.Idx != 0)
                    producer[arg.Idx] = op;
            }

            foreach (var arg in op.Inputs)
            {
                if (arg.Idx == 0)
                    continue;
                consumers[arg.Idx] = consumers.TryGetValue(arg.Idx, out int count) ? count + 1 : 1;
            }
        }

        HashSet<int>? graphOutputs = null;

        Dictionary<LayerInfo, LayerInfo>? replace = null;
        HashSet<LayerInfo>? drop = null;

        foreach (var op in prog)
        {
            if (op.Type != "Mul" || op.Inputs.Count != 2 || op.Outputs.Count != 1)
                continue;

            for (int side = 0; side < 2; side++)
            {
                Arg sig = op.Inputs[side];
                Arg x = op.Inputs[1 - side];

                if (sig.IsEmpty || x.IsEmpty || sig.Idx == x.Idx)
                    continue;

                if (!producer.TryGetValue(sig.Idx, out var sign) || sign.Type != "Sigmoid")
                    continue;

                if (sign.Inputs.Count != 1 || sign.Inputs[0].Idx != x.Idx || sign.Outputs.Count != 1)
                    continue;

                // Sigmoid 的输出只能被这个 Mul 用，否则删了会断链
                if (!consumers.TryGetValue(sig.Idx, out int uses) || uses != 1)
                    continue;

                graphOutputs ??= CollectOutputs(graph);
                if (graphOutputs.Contains(sig.Idx))
                    continue;

                if (!ShapesMatch(net, x, op.Outputs[0]))
                    continue;

                // 换成 SiLU：输入直接吃 Sigmoid 的输入，输出沿用 Mul 的输出 Arg
                var silu = (Layer)LayerRegistry.Create(
                    "Silu", new LayerParams { Type = "Silu", Name = sign.Name });

                silu.Inputs.Add(x);
                silu.Outputs.Add(op.Outputs[0]);

                (replace ??= new Dictionary<LayerInfo, LayerInfo>()).Add(op, silu);
                (drop ??= new HashSet<LayerInfo>()).Add(sign);
                break;
            }
        }

        if (replace is null || drop is null)
            return 0;

        var result = new List<LayerInfo>(prog.Count);
        foreach (var op in prog)
        {
            if (drop.Contains(op))
                continue;

            result.Add(replace.TryGetValue(op, out var fused) ? fused : op);
        }

        graph.Reset(result);
        return replace.Count;
    }

    private static HashSet<int> CollectOutputs(Graph graph)
    {
        var set = new HashSet<int>();
        foreach (var arg in graph.Outputs)
        {
            if (arg.Idx != 0)
                set.Add(arg.Idx);
        }

        return set;
    }

    /// <summary>SiLU 不做广播：输入输出必须同形同类型（形状未知时交给运行时去报错）。</summary>
    private static bool ShapesMatch(NetImpl net, Arg input, Arg output)
    {
        var inData = net.GetArgData(input);
        var outData = net.GetArgData(output);

        if (inData.HasShape && inData.DType != DType.F32)
            return false;

        if (inData.HasShape && outData.HasShape && !inData.Shape.Equals(outData.Shape))
            return false;

        return true;
    }
}
