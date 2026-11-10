using OnnxSharp.Ops;
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

        // 逐元素链融合放最后：前面两步可能已经把 Sigmoid+Mul 变成单个 Silu，
        // 那样链里就少一个节点，融合结果的寄存器也更省。
        total += FuseElementWiseChains(net, graph);

        return total;
    }

    /// <summary>可以进融合链的逐元素算子（都是同形状、无广播的 F32 运算）。</summary>
    private static readonly Dictionary<string, EltwiseKind> FusableEltwise = new()
    {
        ["Add"] = EltwiseKind.Add,
        ["Sub"] = EltwiseKind.Sub,
        ["Mul"] = EltwiseKind.Mul,
        ["Div"] = EltwiseKind.Div,
        ["Min"] = EltwiseKind.Min,
        ["Max"] = EltwiseKind.Max,
        ["Sqrt"] = EltwiseKind.Sqrt,
        ["Erf"] = EltwiseKind.Erf,
        ["Exp"] = EltwiseKind.Exp,
        ["Neg"] = EltwiseKind.Neg,
        ["Abs"] = EltwiseKind.Abs,
        ["Relu"] = EltwiseKind.Relu,
        ["Sigmoid"] = EltwiseKind.Sigmoid,
    };

    /// <summary>一条融合链最多这么长（再长寄存器也不够用，收益也早饱和了）。</summary>
    private const int MaxChainLength = 8;

    /// <summary>达到这个长度的链无条件融合（省下的流量足够多）。</summary>
    private const int MinChainLength = 4;

    /// <summary>不足 <see cref="MinChainLength"/> 的短链，张量至少这么大才值得融合。</summary>
    private const long MinShortChainElements = 1L << 18;

    /// <summary>
    /// 把同形状的逐元素算子链融合成一个 <see cref="FusedEltwiseLayer"/>。
    /// <para>
    /// 典型目标是 GELU：<c>Erf → Add → Mul → Mul</c> 四个节点各自「读一遍整张张量、写一遍」，
    /// 融合后整条链只读外部输入一遍、写最终输出一遍，中间结果留在 YMM 寄存器里。
    /// RF-DETR 有 12 条这样的链，每推理省 613 MB 内存流量。
    /// </para>
    /// <para>
    /// 成立条件（任一不满足就停在那一步，只融已经确定安全的部分）：
    /// 中间结果<b>只被链上下一个节点消费</b>、不是图输出、形状静态已知且整条链相同；
    /// 链上的其它操作数要么同形，要么是元素数为 1 的常量（GELU 的 <c>+1</c>／<c>×0.5</c>）。
    /// </para>
    /// </summary>
    private static int FuseElementWiseChains(NetImpl net, Graph graph)
    {
        var prog = graph.Prog;
        if (prog.Count < 2)
            return 0;

        // 每个 Arg 被用了多少次、被谁用（多次消费时 consumerOf 的值不可信，用 count 判定）
        var consumers = new Dictionary<int, int>();
        var consumerOf = new Dictionary<int, LayerInfo>();

        foreach (var op in prog)
        {
            foreach (var arg in op.Inputs)
            {
                if (arg.IsEmpty)
                    continue;

                consumers[arg.Idx] = consumers.TryGetValue(arg.Idx, out int count) ? count + 1 : 1;
                consumerOf[arg.Idx] = op;
            }
        }

        var graphOutputs = CollectOutputs(graph);
        var consumed = new HashSet<LayerInfo>();

        // 最后一个节点 → 融合层（融合层插在最后一个节点的位置，保证外部输入的产出都在它之前）
        Dictionary<LayerInfo, LayerInfo>? replace = null;

        foreach (var head in prog)
        {
            if (consumed.Contains(head) || !TryGetKind(head, out _))
                continue;

            if (!TryGetOutputShape(net, head, out var shape))
                continue;

            var chain = new List<LayerInfo> { head };

            while (chain.Count < MaxChainLength)
            {
                var cur = chain[^1];
                if (cur.Outputs.Count != 1 || cur.Outputs[0].IsEmpty)
                    break;

                int outIdx = cur.Outputs[0].Idx;
                if (graphOutputs.Contains(outIdx))
                    break;
                if (!consumers.TryGetValue(outIdx, out int uses) || uses != 1)
                    break;
                if (!consumerOf.TryGetValue(outIdx, out var next) || !TryGetKind(next, out _))
                    break;
                if (next.Outputs.Count != 1 || !TryGetOutputShape(net, next, out var nextShape))
                    break;
                if (!nextShape.Equals(shape))
                    break;
                if (chain.Contains(next))
                    break;   // 防环（推理图里不该有，但别让一个 bug 变成死循环）
                if (!InputsCompatible(net, next, shape))
                    break;

                chain.Add(next);
            }

            if (chain.Count < 2)
                continue;

            // 短链（2~3 个节点）省下的流量本来就少，却要和通用内核的
            // pin/闭包/寄存器数组索引开销对抵，小张量上净收益是负的（yolo 系列实测 median 变慢 4%）。
            // 只有长链或大张量才值得。
            if (chain.Count < MinChainLength && shape.Total() < MinShortChainElements)
                continue;

            // 链头自己的输入也要能同形处理，否则整条放弃
            if (!InputsCompatible(net, chain[0], shape))
                continue;

            var fused = BuildFusedLayer(net, chain);
            if (fused is null)
                continue;

            foreach (var op in chain)
                consumed.Add(op);

            (replace ??= new Dictionary<LayerInfo, LayerInfo>()).Add(chain[^1], fused);
        }

        if (replace is null)
            return 0;

        var result = new List<LayerInfo>(prog.Count);
        foreach (var op in prog)
        {
            if (replace.TryGetValue(op, out var fused))
                result.Add(fused);
            else if (!consumed.Contains(op))
                result.Add(op);
        }

        graph.Reset(result);
        return consumed.Count - replace.Count;
    }

    private static bool TryGetKind(LayerInfo op, out EltwiseKind kind)
    {
        kind = default;
        if (op.Inputs.Count is < 1 or > 2 || op.Outputs.Count != 1)
            return false;
        return FusableEltwise.TryGetValue(op.Type, out kind);
    }

    private static bool TryGetOutputShape(NetImpl net, LayerInfo op, out TensorShape shape)
    {
        var data = net.GetArgData(op.Outputs[0]);
        shape = data.HasShape ? data.Shape : default;
        return data.HasShape && data.DType == DType.F32;
    }

    /// <summary>
    /// 输入的每个操作数：要么与链<b>严格同形</b>，要么元素数为 1（标量常量，splat 处理）。
    /// <para>
    /// 不能只比元素总数：`[1,1,256]` 与 `[256]` 的元素数相同但广播语义不同，
    /// SAM3 的 decoder 里就有这种 Add —— 融合层只做线性遍历，不理解广播，
    /// 放行会在运行期抛「shape does not match」。
    /// </para>
    /// </summary>
    private static bool InputsCompatible(NetImpl net, LayerInfo op, TensorShape shape)
    {
        foreach (var arg in op.Inputs)
        {
            var data = net.GetArgData(arg);
            if (!data.HasShape || data.DType != DType.F32)
                return false;

            if (data.Shape.Total() == 1)
                continue;

            if (!data.Shape.Equals(shape))
                return false;
        }

        return true;
    }

    private static LayerInfo? BuildFusedLayer(NetImpl net, List<LayerInfo> chain)
    {
        // 链内中间结果：这些 Arg 不出现在融合层的输入里
        var internalArgs = new HashSet<int>();
        for (int i = 0; i < chain.Count - 1; i++)
        {
            if (chain[i].Outputs.Count == 1 && !chain[i].Outputs[0].IsEmpty)
                internalArgs.Add(chain[i].Outputs[0].Idx);
        }

        // 第一遍：收集外部输入（按首次出现顺序分配寄存器 0..n-1）；
        // 链内中间结果是上一字节点的产出，跳过，第二遍再按步骤序号编号。
        var inputArgs = new List<Arg>();
        var regOf = new Dictionary<int, int>();

        foreach (var op in chain)
        {
            foreach (var arg in op.Inputs)
            {
                if (arg.IsEmpty)
                    return null;
                if (internalArgs.Contains(arg.Idx) || regOf.ContainsKey(arg.Idx))
                    continue;

                regOf[arg.Idx] = inputArgs.Count;
                inputArgs.Add(arg);
            }
        }

        // 第二遍：发指令。中间寄存器从 nIn 开始编号。
        int nIn = inputArgs.Count;
        var steps = new EltwiseStep[chain.Count];

        for (int i = 0; i < chain.Count; i++)
        {
            var op = chain[i];
            if (!FusableEltwise.TryGetValue(op.Type, out var kind))
                return null;

            // 操作数若由链内前序步骤产出，则取该步骤的目标寄存器
            int RegOf(Arg arg)
            {
                if (regOf.TryGetValue(arg.Idx, out int r))
                    return r;
                throw new InvalidOperationException($"fused chain is not topologically ordered at {op.Name}");
            }

            int a = RegOf(op.Inputs[0]);
            int b = op.Inputs.Count > 1 ? RegOf(op.Inputs[1]) : -1;

            steps[i] = new EltwiseStep(kind, a, b, nIn + i);

            if (op.Outputs.Count == 1 && !op.Outputs[0].IsEmpty)
                regOf[op.Outputs[0].Idx] = nIn + i;
        }

        var last = chain[^1];
        var fused = new FusedEltwiseLayer(
            new LayerParams { Type = "FusedEltwise", Name = last.Name },
            steps,
            nIn + steps.Length,
            nIn + steps.Length - 1);

        foreach (var arg in inputArgs)
            fused.Inputs.Add(arg);

        fused.Outputs.Add(last.Outputs[0]);
        return fused;
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
