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
        total += FuseMatMulArgMax(net, graph);
        total += FuseQuantizedMatMul(net, graph);
        total += FuseAttention(net, graph);
        total += FuseLayerNorm(net, graph);

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
    /// <summary>
    /// <c>MatMul(a, b) → ArgMax(axis = -1, keepdims = 0)</c>（中间可夹一个「常量 bias 的 <c>Add</c>」）
    /// 融成单层 <c>MatMulArgMax</c>：**不物化 logits**。
    /// <para>
    /// 收益取决于 logits 的大小：逐 token 解码 <c>m=1, n=8000</c> 只省一次 32 KB 的写读，
    /// 但 CTC / 强制对齐是 <c>m = 帧数（几百）</c>×<c>n = 词表（几千）</c>，能省几 MB 的流量，
    /// 顺带省掉一次算子调度。
    /// </para>
    /// <para>
    /// 只融 <c>MatMul</c>（<c>Gemm</c> 的 transA/transB/alpha/beta 没有暴露给融合层，先不做），
    /// 且要求 MatMul 的输出<b>只有一个消费者</b>、也不是图输出。
    /// </para>
    /// </summary>
    private static int FuseMatMulArgMax(NetImpl net, Graph graph)
    {
        var prog = graph.Prog;
        if (prog.Count < 2)
            return 0;

        var uses = new Dictionary<int, int>();
        var consumerOf = new Dictionary<int, LayerInfo>();
        var consumerCount = new Dictionary<int, int>();

        foreach (var op in prog)
        {
            foreach (var arg in op.Inputs)
            {
                if (arg.IsEmpty)
                    continue;

                uses[arg.Idx] = uses.TryGetValue(arg.Idx, out int count) ? count + 1 : 1;
                consumerCount[arg.Idx] = consumerCount.TryGetValue(arg.Idx, out int seen) ? seen + 1 : 1;
                if (consumerCount[arg.Idx] == 1)
                    consumerOf[arg.Idx] = op;
            }
        }

        var graphOutputs = CollectOutputs(graph);
        var replacements = new Dictionary<LayerInfo, LayerInfo>();
        var drop = new HashSet<LayerInfo>();
        int fused = 0;

        foreach (var op in prog)
        {
            if (op.Type != "MatMul" || op.Outputs.Count != 1 || op.Outputs[0].IsEmpty)
                continue;

            var produced = op.Outputs[0];
            if (graphOutputs.Contains(produced.Idx))
                continue;

            if (!uses.TryGetValue(produced.Idx, out int count) || count != 1)
                continue;

            if (!consumerOf.TryGetValue(produced.Idx, out var consumer) || consumer is not Layer mid)
                continue;

            LayerInfo? argMaxOp = null;
            LayerInfo? addOp = null;
            int biasSlotIndex = -1;

            if (mid.Type == "ArgMax")
            {
                argMaxOp = mid;
            }
            else if (mid.Type == "Add" && mid.Inputs.Count == 2 && mid.Outputs.Count == 1 && !mid.Outputs[0].IsEmpty)
            {
                // Add(MatMul_out, bias) —— bias 必须是**常量向量**，且这个 Add 只有 ArgMax 一个消费者
                int biasSlot = mid.Inputs[0].Idx == produced.Idx ? 1 : 0;
                if (!mid.IsConstInput(biasSlot))
                    continue;

                var addOutput = mid.Outputs[0];
                if (graphOutputs.Contains(addOutput.Idx))
                    continue;
                if (!uses.TryGetValue(addOutput.Idx, out int addCount) || addCount != 1)
                    continue;
                if (!consumerOf.TryGetValue(addOutput.Idx, out var afterAdd) || afterAdd.Type != "ArgMax")
                    continue;

                addOp = mid;
                biasSlotIndex = biasSlot;
                argMaxOp = afterAdd;
            }

            if (argMaxOp is not Ops.ArgMaxOps.ILastAxisArgMax { IsLastAxisNoKeep: true })
                continue;
            if (argMaxOp.Inputs.Count != 1 || argMaxOp.Outputs.Count != 1)
                continue;

            // MatMul 输出必须是 2-D [m, n]，融合层只处理这一种
            if (!TryGetOutputShape(net, op, out var matMulShape) || matMulShape.Rank != 2)
                continue;

            var parameters = new LayerParams { Type = "MatMulArgMax", Name = op.Name + "/argmax" };
            if (LayerRegistry.Create("MatMulArgMax", parameters) is not Layer fusedLayer)
                continue;

            fusedLayer.Inputs.Add(op.Inputs[0]);
            fusedLayer.Inputs.Add(op.Inputs[1]);
            if (biasSlotIndex >= 0 && addOp is not null)
                fusedLayer.Inputs.Add(addOp.Inputs[biasSlotIndex]);
            fusedLayer.Outputs.Add(argMaxOp.Outputs[0]);

            replacements[op] = fusedLayer;
            drop.Add(argMaxOp);
            if (addOp is not null)
                drop.Add(addOp);
            fused++;
        }

        if (replacements.Count == 0)
            return 0;

        var result = new List<LayerInfo>(prog.Count);
        foreach (var op in prog)
        {
            if (drop.Contains(op))
                continue;

            result.Add(replacements.TryGetValue(op, out var replacement) ? replacement : op);
        }

        graph.Reset(result);
        return fused;
    }

    /// <summary>
    /// <c>DQL → MatMulInteger → Cast → Mul(a_scale·w_scale) → [Add(bias)]</c> 折成一层
    /// <c>DynamicQuantizeMatMul</c>（对齐 ORT 的同名融合）。
    /// <para>
    /// 前提：<c>a_scale</c> 只喂那个「组合 scale」的 <c>Mul</c>、权重/零点/偏置都是常量、
    /// 以及链路上每个中间张量只有一个消费者。
    /// </para>
    /// </summary>
    private static int FuseQuantizedMatMul(NetImpl net, Graph graph)
    {
        var prog = graph.Prog;
        if (prog.Count < 5)
            return 0;

        var producerOf = new Dictionary<int, LayerInfo>();
        var consumerOf = new Dictionary<int, LayerInfo>();
        var uses = new Dictionary<int, int>();

        foreach (var op in prog)
        {
            foreach (var arg in op.Outputs)
            {
                if (!arg.IsEmpty)
                    producerOf[arg.Idx] = op;          // 输出 → 生产者
            }

            foreach (var arg in op.Inputs)
            {
                if (arg.IsEmpty)
                    continue;

                uses[arg.Idx] = uses.TryGetValue(arg.Idx, out int count) ? count + 1 : 1;
                consumerOf.TryAdd(arg.Idx, op);        // 输入 → 消费者（第一个）
            }
        }

        int Uses(Arg arg) => arg.IsEmpty ? 0 : uses.GetValueOrDefault(arg.Idx);

        var graphOutputs = CollectOutputs(graph);
        var replacements = new Dictionary<LayerInfo, LayerInfo>();
        var drop = new HashSet<LayerInfo>();
        int fused = 0;

        foreach (var matmul in prog)
        {
            if (matmul.Type != "MatMulInteger" || matmul.Inputs.Count < 2 || matmul.Outputs.Count != 1)
                continue;
            if (!matmul.IsConstInput(1))
                continue;

            if (!producerOf.TryGetValue(matmul.Inputs[0].Idx, out var dql) || dql.Type != "DynamicQuantizeLinear")
                continue;
            if (dql.Outputs.Count < 3 || Uses(dql.Outputs[0]) != 1 || Uses(dql.Outputs[1]) != 1 || Uses(dql.Outputs[2]) != 1)
                continue;

            if (matmul.Inputs.Count > 2 && !matmul.Inputs[2].IsEmpty && matmul.Inputs[2].Idx != dql.Outputs[2].Idx)
                continue;
            if (matmul.Inputs.Count > 3 && !matmul.Inputs[3].IsEmpty && !matmul.IsConstInput(3))
                continue;

            // Mul(a_scale, w_scale_const) → 组合 scale
            if (!consumerOf.TryGetValue(dql.Outputs[1].Idx, out var scaleMul)
                || scaleMul.Type != "Mul" || scaleMul.Inputs.Count != 2 || scaleMul.Outputs.Count != 1)
                continue;

            int scaleSlot = scaleMul.IsConstInput(0) ? 0 : scaleMul.IsConstInput(1) ? 1 : -1;
            if (scaleSlot < 0)
                continue;

            // 权重是 per-channel 量化的：w_scale 是 [n] 向量（也兼容标量）
            var weightScale = scaleMul.Inputs[scaleSlot];

            // Cast(int32 → f32)
            if (Uses(matmul.Outputs[0]) != 1 || !consumerOf.TryGetValue(matmul.Outputs[0].Idx, out var cast)
                || cast.Type != "Cast" || cast.Outputs.Count != 1)
                continue;
            if (net.GetArgData(cast.Outputs[0]).DType != DType.F32)
                continue;

            // Mul(cast_out, combined_scale)
            if (Uses(cast.Outputs[0]) != 1 || !consumerOf.TryGetValue(cast.Outputs[0].Idx, out var mul)
                || mul.Type != "Mul" || mul.Inputs.Count != 2 || mul.Outputs.Count != 1)
                continue;
            if (Uses(scaleMul.Outputs[0]) != 1)
                continue;
            if (mul.Inputs[0].Idx != scaleMul.Outputs[0].Idx && mul.Inputs[1].Idx != scaleMul.Outputs[0].Idx)
                continue;

            // 可选：Add(bias_const)
            LayerInfo tail = mul;
            int biasSlotIndex = -1;

            if (Uses(mul.Outputs[0]) == 1 && consumerOf.TryGetValue(mul.Outputs[0].Idx, out var add)
                && add.Type == "Add" && add.Inputs.Count == 2 && add.Outputs.Count == 1)
            {
                int slot = add.IsConstInput(0) ? 0 : add.IsConstInput(1) ? 1 : -1;
                if (slot >= 0)
                {
                    tail = add;
                    biasSlotIndex = slot;
                }
            }

            // 只在「既不是图输出、也没人用」时才跳过（那是死节点，不值得我们建融合层）
            if (!graphOutputs.Contains(tail.Outputs[0].Idx) && Uses(tail.Outputs[0]) == 0)
                continue;

            var parameters = new LayerParams { Type = "DynamicQuantizeMatMul", Name = matmul.Name + "/fused" };

            if (LayerRegistry.Create("DynamicQuantizeMatMul", parameters) is not Layer fusedLayer)
                continue;

            fusedLayer.Inputs.Add(dql.Inputs[0]);                                   // A(f32)
            fusedLayer.Inputs.Add(matmul.Inputs[1]);                                // W(int8/u8, const)
            fusedLayer.Inputs.Add(matmul.Inputs.Count > 3 ? matmul.Inputs[3] : default);  // w_zero_point
            fusedLayer.Inputs.Add(weightScale);                                     // w_scale（per-column 或标量）
            fusedLayer.Inputs.Add(biasSlotIndex >= 0 ? tail.Inputs[biasSlotIndex] : default);
            fusedLayer.Outputs.Add(tail.Outputs[0]);

            replacements[matmul] = fusedLayer;
            drop.Add(dql);
            drop.Add(scaleMul);
            drop.Add(cast);
            drop.Add(mul);
            if (tail != mul)
                drop.Add(tail);

            fused++;
        }

        if (replacements.Count == 0)
            return 0;

        var result = new List<LayerInfo>(prog.Count);
        foreach (var op in prog)
        {
            if (drop.Contains(op))
                continue;

            result.Add(replacements.TryGetValue(op, out var replacement) ? replacement : op);
        }

        graph.Reset(result);
        return fused;
    }

    /// <summary>
    /// 整条 multi-head attention 折成一层 <c>FusedAttention</c>：
    /// <code>
    /// Qh = Transpose(Reshape(q), [0,2,1,3])      [B,H,T,D]
    /// Kt = Transpose(Reshape(k), [0,2,3,1])      [B,H,D,T]
    /// Vh = Transpose(Reshape(v), [0,2,1,3])      [B,H,T,D]
    /// S  = MatMul([Mul(Qh, scale),] Kt) → [Add(·, mask)] → Softmax(-1)
    /// O  = Reshape(Transpose(MatMul(S, Vh), [0,2,1,3]))
    /// </code>
    /// <para>
    /// 两种形态：能读出 head 数（<c>Reshape</c> 目标静态）时融合层的输入是 3 维投影张量，
    /// 连 3 个 <c>Reshape + Transpose</c> 一起折掉；读不出时（Paraformer 的 head-split 目标是
    /// <c>Shape→Gather→Concat</c> 动态链）输入留在 head 空间，H/D 由形状给出，
    /// 仍然折掉 scores 物化、Softmax、两次 MatMul、输出 Transpose 与 Reshape。
    /// </para>
    /// </summary>
    private static int FuseAttention(NetImpl net, Graph graph)
    {
        var prog = graph.Prog;
        if (prog.Count < 6)
            return 0;

        var producerOf = new Dictionary<int, LayerInfo>();
        var consumerOf = new Dictionary<int, LayerInfo>();
        var consumersOf = new Dictionary<int, List<LayerInfo>>();

        var uses = new Dictionary<int, int>();

        foreach (var op in prog)
        {
            foreach (var arg in op.Outputs)
            {
                if (!arg.IsEmpty)
                    producerOf[arg.Idx] = op;          // 输出 → 生产者
            }

            foreach (var arg in op.Inputs)
            {
                if (arg.IsEmpty)
                    continue;

                uses[arg.Idx] = uses.TryGetValue(arg.Idx, out int count) ? count + 1 : 1;
                consumerOf.TryAdd(arg.Idx, op);        // 输入 → 消费者（第一个）
                if (!consumersOf.TryGetValue(arg.Idx, out var list))
                    consumersOf[arg.Idx] = list = new List<LayerInfo>();
                list.Add(op);

            }
        }

        int Uses(Arg arg) => arg.IsEmpty ? 0 : uses.GetValueOrDefault(arg.Idx);

        var replacements = new Dictionary<LayerInfo, LayerInfo>();
        var drop = new HashSet<LayerInfo>();
        int fused = 0;
        foreach (var softmax in prog)
        {
            // 只认「沿最后一维」的 Softmax（属性可能写成 -1，也可能写成 rank-1）
            if (softmax.Type != "Softmax" || softmax is not Ops.DnnOps.ISoftmaxAxis { Axis: -1 or 3 })
                continue;
            if (softmax.Inputs.Count != 1 || softmax.Outputs.Count != 1)
                continue;

            // O = MatMul(softmax, Vh)
            if (Uses(softmax.Outputs[0]) != 1
                || !consumerOf.TryGetValue(softmax.Outputs[0].Idx, out var pv)
                || pv.Type != "MatMul" || pv.Inputs.Count != 2 || pv.Outputs.Count != 1
                || pv.Inputs[0].Idx != softmax.Outputs[0].Idx)
                continue;

            if (!producerOf.TryGetValue(pv.Inputs[1].Idx, out var vTranspose)
                || vTranspose is not Ops.ShapeOps.ITransposePerm { Perm: [0, 2, 1, 3] })
                continue;

            // Softmax 的输入可能是 Add(scores, mask)
            var scoreArg = softmax.Inputs[0];
            var maskArg = default(Arg);
            LayerInfo? maskAdd = null;

            if (producerOf.TryGetValue(scoreArg.Idx, out var addNode)
                && addNode.Type == "Add" && addNode.Inputs.Count == 2 && addNode.Outputs.Count == 1
                && Uses(addNode.Outputs[0]) == 1)
            {
                bool firstIsMatMul = producerOf.TryGetValue(addNode.Inputs[0].Idx, out var firstProducer)
                    && firstProducer.Type == "MatMul";
                bool secondIsMatMul = producerOf.TryGetValue(addNode.Inputs[1].Idx, out var secondProducer)
                    && secondProducer.Type == "MatMul";

                if (firstIsMatMul == secondIsMatMul)
                    continue;

                scoreArg = firstIsMatMul ? addNode.Inputs[0] : addNode.Inputs[1];
                maskArg = firstIsMatMul ? addNode.Inputs[1] : addNode.Inputs[0];
                maskAdd = addNode;
            }

            // S = MatMul(Qh(·scale), Kt)
            if (!producerOf.TryGetValue(scoreArg.Idx, out var qk)
                || qk.Type != "MatMul" || qk.Inputs.Count != 2 || qk.Outputs.Count != 1)
                continue;

            var qArg = qk.Inputs[0];
            var ktArg = qk.Inputs[1];
            float scale = 1f;
            LayerInfo? qScale = null;

            if (producerOf.TryGetValue(qArg.Idx, out var scaleNode)
                && scaleNode.Type == "Mul" && scaleNode.Inputs.Count == 2 && scaleNode.Outputs.Count == 1
                && Uses(scaleNode.Outputs[0]) == 1)
            {
                int scaleSlot = scaleNode.IsConstInput(0) ? 0 : scaleNode.IsConstInput(1) ? 1 : -1;
                if (scaleSlot < 0)
                    continue;

                var scaleTensor = scaleNode.In(scaleSlot);
                if (scaleTensor.Count != 1)
                    continue;

                scale = scaleTensor.AsSpan<float>()[0];
                qArg = scaleNode.Inputs[1 - scaleSlot];
                qScale = scaleNode;
            }

            if (!producerOf.TryGetValue(qArg.Idx, out var qTranspose)
                || qTranspose is not Ops.ShapeOps.ITransposePerm { Perm: [0, 2, 1, 3] })
                continue;
            if (!producerOf.TryGetValue(ktArg.Idx, out var kTranspose)
                || kTranspose is not Ops.ShapeOps.ITransposePerm { Perm: [0, 2, 3, 1] })
                continue;

            // 输出侧：Transpose([0,2,1,3]) → Reshape
            if (Uses(pv.Outputs[0]) != 1
                || !consumerOf.TryGetValue(pv.Outputs[0].Idx, out var oTranspose)
                || oTranspose is not Ops.ShapeOps.ITransposePerm { Perm: [0, 2, 1, 3] })
                continue;

            if (!consumersOf.TryGetValue(oTranspose.Outputs[0].Idx, out var transposeConsumers))
                continue;

            LayerInfo? oReshape = null;
            foreach (var consumer in transposeConsumers)
            {
                if (consumer.Type == "Reshape" && consumer.Outputs.Count == 1)
                {
                    oReshape = consumer;
                    break;
                }
            }

            // 没有 Reshape（只有 Shape 链）说明这不是我们要的形态
            if (oReshape is null)
                continue;

            // Transpose 的输出只被这个 Reshape 用 → 连 Transpose+Reshape 一起折掉；
            // 否则（动态 Reshape 的目标由 Shape(Transpose 输出) 建出来）保留它们，
            // 融合层输出 [B,H,T,D]。
            bool foldOutput = transposeConsumers.Count == 1;

            // ---- 建融合层 ----
            // 优先「3 维输入」形态：三个 Transpose 的输入都是 Reshape，且能静态读出 head 数
            int heads = 0;
            var qReshape = default(LayerInfo);
            var kReshape = default(LayerInfo);
            var vReshape = default(LayerInfo);

            if (producerOf.TryGetValue(qTranspose.Inputs[0].Idx, out var qr)
                && qr.Type == "Reshape" && Uses(qr.Outputs[0]) == 1
                && producerOf.TryGetValue(kTranspose.Inputs[0].Idx, out var kr)
                && kr.Type == "Reshape" && Uses(kr.Outputs[0]) == 1
                && producerOf.TryGetValue(vTranspose.Inputs[0].Idx, out var vr)
                && vr.Type == "Reshape" && Uses(vr.Outputs[0]) == 1
                && TryGetOutputShape(net, qTranspose, out var qhShape) && qhShape.Rank == 4
                && TryGetOutputShape(net, vTranspose, out var vhShape) && vhShape.Rank == 4
                && qhShape.Dims[1] > 0 && qhShape.Dims[3] > 0
                && qhShape.Dims[0] == vhShape.Dims[0] && qhShape.Dims[1] == vhShape.Dims[1]
                && qhShape.Dims[2] == vhShape.Dims[2] && qhShape.Dims[3] == vhShape.Dims[3])
            {
                heads = qhShape.Dims[1];
                qReshape = qr;
                kReshape = kr;
                vReshape = vr;
            }

            var parameters = new LayerParams { Type = "FusedAttention", Name = softmax.Name + "/fused" };
            parameters.Set("scale", (double)scale);
            if (heads > 0)
                parameters.Set("heads", (long)heads);
            if (!foldOutput)
                parameters.Set("output_head_space", 1L);

            if (LayerRegistry.Create("FusedAttention", parameters) is not Layer fusedLayer)
                continue;

            if (heads > 0)
            {
                fusedLayer.Inputs.Add(qReshape!.Inputs[0]);
                fusedLayer.Inputs.Add(kReshape!.Inputs[0]);
                fusedLayer.Inputs.Add(vReshape!.Inputs[0]);
            }
            else
            {
                fusedLayer.Inputs.Add(qTranspose.Outputs[0]);
                fusedLayer.Inputs.Add(kTranspose.Outputs[0]);
                fusedLayer.Inputs.Add(vTranspose.Outputs[0]);
            }

            fusedLayer.Inputs.Add(maskArg);
            fusedLayer.Outputs.Add(foldOutput ? oReshape.Outputs[0] : pv.Outputs[0]);

            replacements[softmax] = fusedLayer;

            drop.Add(softmax);
            drop.Add(qk);
            drop.Add(pv);

            if (foldOutput)
            {
                drop.Add(oTranspose);
                drop.Add(oReshape);
            }

            if (maskAdd is not null)
                drop.Add(maskAdd);
            if (qScale is not null)
                drop.Add(qScale);

            if (heads > 0)
            {
                // 3 维形态：head-split 的 Reshape 与 Transpose 全部进融合层
                drop.Add(qTranspose);
                drop.Add(kTranspose);
                drop.Add(vTranspose);
                drop.Add(qReshape!);
                drop.Add(kReshape!);
                drop.Add(vReshape!);
            }

            fused++;
        }

        if (replacements.Count == 0)
            return 0;

        var result = new List<LayerInfo>(prog.Count);
        foreach (var op in prog)
        {
            // 先查替换表：被替换的节点往往也在 drop 里，顺序反了融合层就丢了
            if (replacements.TryGetValue(op, out var replacement))
            {
                result.Add(replacement);
                continue;
            }

            if (drop.Contains(op))
                continue;

            result.Add(op);
        }

        graph.Reset(result);
        return fused;
    }

    /// <summary>
    /// 把手写的 LayerNorm 折成单个 <c>LayerNormalization</c>（思路来自 SimdOnnx 的 <c>FuseLayerNorm</c>）：
    /// <code>
    /// x → ReduceMean → Sub(x, mean) → Pow(2) → ReduceMean → Add(eps) → Sqrt
    ///   → Div(Sub, Sqrt) → Mul(gamma) → Add(beta)
    /// </code>
    /// 一共 9 个算子、对 <c>[T, C]</c> 要走 8 遍内存；融成一层后内核只扫两遍。
    /// <para>
    /// 匹配的关键判据（漏一个就会算错）：每个中间张量**只有一个消费者**、
    /// <c>gamma/beta</c> 长度等于最后一维、<c>eps</c> 是标量常量。
    /// </para>
    /// </summary>
    private static int FuseLayerNorm(NetImpl net, Graph graph)
    {
        var prog = graph.Prog;
        if (prog.Count < 9)
            return 0;

        var producerOf = new Dictionary<int, LayerInfo>();
        var uses = new Dictionary<int, int>();
        foreach (var op in prog)
        {
            foreach (var arg in op.Outputs)
            {
                if (!arg.IsEmpty)
                    producerOf[arg.Idx] = op;          // 输出 → 生产者
            }

            foreach (var arg in op.Inputs)
            {
                if (arg.IsEmpty)
                    continue;

                uses[arg.Idx] = uses.TryGetValue(arg.Idx, out int count) ? count + 1 : 1;
            }
        }

        int Uses(Arg arg) => arg.IsEmpty ? 0 : uses.GetValueOrDefault(arg.Idx);

        var graphOutputs = CollectOutputs(graph);
        var replacements = new Dictionary<LayerInfo, LayerInfo>();
        var drop = new HashSet<LayerInfo>();
        int fused = 0;

        foreach (var tail in prog)
        {
            if (tail.Type != "Add" || tail.Inputs.Count != 2 || tail.Outputs.Count != 1)
                continue;
            if (drop.Contains(tail) || graphOutputs.Contains(tail.Outputs[0].Idx))
                continue;

            int betaSlot = tail.IsConstInput(0) ? 0 : tail.IsConstInput(1) ? 1 : -1;
            if (betaSlot < 0)
                continue;

            var beta = tail.Inputs[betaSlot];
            var mulArg = tail.Inputs[1 - betaSlot];
            if (!producerOf.TryGetValue(mulArg.Idx, out var mul) || mul.Type != "Mul" || mul.Inputs.Count != 2)
                continue;

            int gammaSlot = mul.IsConstInput(0) ? 0 : mul.IsConstInput(1) ? 1 : -1;
            if (gammaSlot < 0)
                continue;

            var gamma = mul.Inputs[gammaSlot];
            var divArg = mul.Inputs[1 - gammaSlot];
            if (!producerOf.TryGetValue(divArg.Idx, out var div) || div.Type != "Div" || div.Inputs.Count != 2)
                continue;

            var subArg = div.Inputs[0];
            var sqrtArg = div.Inputs[1];
            if (!producerOf.TryGetValue(sqrtArg.Idx, out var sqrt) || sqrt.Type != "Sqrt" || sqrt.Inputs.Count != 1)
                continue;
            if (!producerOf.TryGetValue(subArg.Idx, out var sub) || sub.Type != "Sub" || sub.Inputs.Count != 2)
                continue;

            var epsArg = sqrt.Inputs[0];
            if (!producerOf.TryGetValue(epsArg.Idx, out var epsAdd) || epsAdd.Type != "Add" || epsAdd.Inputs.Count != 2)
                continue;

            int epsSlot = epsAdd.IsConstInput(0) ? 0 : epsAdd.IsConstInput(1) ? 1 : -1;
            if (epsSlot < 0)
                continue;

            var variance = epsAdd.Inputs[1 - epsSlot];
            if (!producerOf.TryGetValue(variance.Idx, out var varianceMean) || varianceMean.Type != "ReduceMean")
                continue;

            var powArg = varianceMean.Inputs[0];
            if (!producerOf.TryGetValue(powArg.Idx, out var pow))
                continue;

            bool squared = (pow.Type == "Pow" && pow.Inputs.Count == 2 && pow.Inputs[0].Idx == subArg.Idx)
                        || (pow.Type == "Mul" && pow.Inputs.Count == 2
                            && pow.Inputs[0].Idx == subArg.Idx && pow.Inputs[1].Idx == subArg.Idx);
            if (!squared)
                continue;

            // Sub(x, ReduceMean(x))：确认减的是同一路输入的均值
            var x = sub.Inputs[0];
            if (!producerOf.TryGetValue(sub.Inputs[1].Idx, out var xMean) || xMean.Type != "ReduceMean")
                continue;
            if (xMean.Inputs[0].Idx != x.Idx)
                continue;

            // 每个中间张量只能有一个消费者，否则删掉会破坏别的分支
            if (Uses(mulArg) != 1 || Uses(divArg) != 1 || Uses(sqrtArg) != 1 || Uses(epsArg) != 1 ||
                Uses(variance) != 1 || Uses(powArg) != 1 || Uses(xMean.Outputs[0]) != 1)
                continue;

            // gamma/beta 必须是「最后一维」的向量，长度对不上说明不是 LN。
            // 裁剪过的子图常常没有 value_info，这时退化成「gamma 与 beta 等长」的自洽检查
            // （LayerNormalization 的输出形状本来就等于输入形状，不依赖这里推出来的形状）。
            int betaCount = (int)tail.In(betaSlot).Count;
            int gammaCount = (int)mul.In(gammaSlot).Count;
            if (betaCount != gammaCount || betaCount < 1)
                continue;

            if (TryGetOutputShape(net, tail, out var outputShape) && outputShape.Rank >= 1
                && outputShape.Dims[outputShape.Rank - 1] != betaCount)
                continue;

            var epsTensor = epsAdd.In(epsSlot);
            if (epsTensor.Count != 1)
                continue;

            var parameters = new LayerParams { Type = "LayerNormalization", Name = tail.Name + "/fused" };
            parameters.Set("axis", -1L);
            parameters.Set("epsilon", (double)epsTensor.AsSpan<float>()[0]);

            if (LayerRegistry.Create("LayerNormalization", parameters) is not Layer fusedLayer)
                continue;

            fusedLayer.Inputs.Add(x);
            fusedLayer.Inputs.Add(gamma);
            fusedLayer.Inputs.Add(beta);
            fusedLayer.Outputs.Add(tail.Outputs[0]);

            replacements[tail] = fusedLayer;
            foreach (var op in new[] { mul, div, sqrt, epsAdd, varianceMean, pow, xMean, sub })
            {
                if (op != tail)
                    drop.Add(op);
            }

            fused++;
        }

        if (replacements.Count == 0)
            return 0;

        var result = new List<LayerInfo>(prog.Count);
        foreach (var op in prog)
        {
            if (drop.Contains(op))
                continue;

            result.Add(replacements.TryGetValue(op, out var replacement) ? replacement : op);
        }

        graph.Reset(result);
        return fused;
    }

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
