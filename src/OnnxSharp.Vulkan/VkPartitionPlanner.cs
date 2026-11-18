// 分区规划（P1-8）：找出「一段连续、都能在 GPU 上跑」的算子，作为一次段执行的候选。
//
// 这一层刻意只碰 `Graph`/`LayerInfo`/`Arg`，不碰设备、不碰形状推断，
// 于是「边界划得对不对」可以在没有显卡的情况下验证——这是最容易出错、
// 也最难在数值上暴露出来的部分（少划一个 arg 就静默多拷或少拷一整块数据）。
//
// 三条规则：
//   1. 只有**连续**的可发射算子能拼成一段。中间夹一个 CPU 算子就断成两段，
//      因为段内不允许回到 host（回了就退化成逐算子往返）。
//   2. 段外生产、段内消费的 arg 是 live-in（进 stage in）；
//      段内生产、段外消费（或本身是图输出）的 arg 是 live-out（进 stage out）；
//      **常量输入不算 live-in**：它由发射器绑成权重切片，常驻显存，不该每帧重拷。
//   3. 少于 MinOpsPerSegment 的连续段不发射：一次提交 + 两趟搬运的固定成本
//      （实测约 100 us 提交 + 每 400 KB 往返 200 us）比单算子 CPU 计算还贵。
using OnnxSharp.Core;

namespace OnnxSharp.Vulkan;

/// <summary>一个候选段：连续的一段算子，以及进出段需要搬运的 arg。</summary>
internal sealed class VkSegmentPlan
{
    public VkSegmentPlan(int start, List<LayerInfo> ops, List<Arg> liveIn, List<Arg> liveOut)
    {
        Start = start;
        Ops = ops;
        LiveIn = liveIn;
        LiveOut = liveOut;
    }

    /// <summary>段首算子在 <c>graph.Prog</c> 里的下标（供 <see cref="Graph.Reset"/> 拼接）。</summary>
    public int Start { get; }

    public int Count => Ops.Count;

    /// <summary>段内的算子，保持原拓扑顺序。</summary>
    public IReadOnlyList<LayerInfo> Ops { get; }

    /// <summary>段外生产、段内消费的 arg（按首次使用顺序）。stage in 就按这个顺序。</summary>
    public IReadOnlyList<Arg> LiveIn { get; }

    /// <summary>段内生产、段外消费的 arg（按生产顺序，含图输出）。stage out 按这个顺序。</summary>
    public IReadOnlyList<Arg> LiveOut { get; }

    public override string ToString() =>
        $"SegmentPlan(ops={Count} @{Start}, in={LiveIn.Count}, out={LiveOut.Count})";
}

internal static class VkPartitionPlanner
{
    /// <summary>
    /// 默认最短段长。取 3 而不是 2 的理由：探针里单算子 400 KB 的往返是 234 us，
    /// 而 10 万元素的逐元素算子在 CPU 上只要几十 us——1~2 个算子成段是净亏。
    /// </summary>
    public const int DefaultMinOpsPerSegment = 3;

    /// <summary>可调下限（对应计划里的 <c>MinOpsPerSegment</c>）。设 1 表示「只要连续就发射」。</summary>
    public static int MinOpsPerSegment = DefaultMinOpsPerSegment;

    /// <summary>
    /// 规划。返回的段按 <see cref="VkSegmentPlan.Start"/> 升序，互不重叠，
    /// 且只包含 <paramref name="canEmit"/> 认可、长度 ≥ <paramref name="minOpsPerSegment"/> 的连续段。
    /// <para>
    /// <paramref name="isConst"/> 给了就把常量输入排除在 live-in 之外（交给发射器绑权重）；
    /// 不给则一律当 live-in，是「不知道谁是常量」时的保守口径。
    /// </para>
    /// </summary>
    public static List<VkSegmentPlan> Plan(Graph graph, Func<LayerInfo, bool> canEmit,
        int minOpsPerSegment = -1, Func<Arg, bool>? isConst = null)
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(canEmit);

        int minOps = minOpsPerSegment < 0 ? MinOpsPerSegment : minOpsPerSegment;
        if (minOps < 1) minOps = 1;

        IReadOnlyList<LayerInfo> prog = graph.Prog;
        var plans = new List<VkSegmentPlan>();
        if (prog.Count == 0) return plans;

        // arg → 最后一个消费它的算子下标。判定 live-out 用：只有「段外还有人读」才需要拷出去。
        var lastUse = new Dictionary<int, int>();
        for (int k = 0; k < prog.Count; k++)
            foreach (Arg a in prog[k].Inputs)
                if (!a.IsEmpty) lastUse[a.Idx] = k;

        var graphOutputs = new HashSet<int>();
        foreach (Arg a in graph.Outputs)
            if (!a.IsEmpty) graphOutputs.Add(a.Idx);

        int i = 0;
        while (i < prog.Count)
        {
            if (!canEmit(prog[i]))
            {
                i++;
                continue;
            }

            int start = i;
            while (i < prog.Count && canEmit(prog[i])) i++;
            int count = i - start;

            // 太短的段不发射：固定成本（提交 + 搬运）吃掉了收益
            if (count < minOps) continue;

            plans.Add(Build(prog, start, count, lastUse, graphOutputs, isConst));
        }

        return plans;
    }

    private static VkSegmentPlan Build(IReadOnlyList<LayerInfo> prog, int start, int count,
        Dictionary<int, int> lastUse, HashSet<int> graphOutputs, Func<Arg, bool>? isConst)
    {
        int end = start + count;

        var producedInside = new HashSet<int>();
        for (int k = start; k < end; k++)
            foreach (Arg a in prog[k].Outputs)
                if (!a.IsEmpty) producedInside.Add(a.Idx);

        // live-in：段内要读、但段内不生产。常量输入不算 live-in——它由发射器绑成权重切片，
        // 常驻显存，每帧再 stage 一遍纯属浪费（BN 的 gamma/beta 就是这么被白拷的）。
        var liveIn = new List<Arg>();
        var seenIn = new HashSet<int>();
        for (int k = start; k < end; k++)
        {
            foreach (Arg a in prog[k].Inputs)
            {
                if (a.IsEmpty || producedInside.Contains(a.Idx)) continue;
                if (isConst is not null && isConst(a)) continue;
                if (seenIn.Add(a.Idx)) liveIn.Add(a);
            }
        }

        // live-out：段内生产，且「是图输出」或「段外还有人读」
        var liveOut = new List<Arg>();
        var seenOut = new HashSet<int>();
        for (int k = start; k < end; k++)
        {
            foreach (Arg a in prog[k].Outputs)
            {
                if (a.IsEmpty) continue;
                bool external = graphOutputs.Contains(a.Idx) ||
                                (lastUse.TryGetValue(a.Idx, out int last) && last >= end);
                if (external && seenOut.Add(a.Idx)) liveOut.Add(a);
            }
        }

        var ops = new List<LayerInfo>(count);
        for (int k = start; k < end; k++) ops.Add(prog[k]);

        return new VkSegmentPlan(start, ops, liveIn, liveOut);
    }

    /// <summary>
    /// 自检（探针与将来的一致性测试用）：段内每个输入要么是 live-in、要么由段内生产；
    /// 段内每个输出要么在段内被消费、要么是 live-out。返回问题描述，空表示自洽。
    /// </summary>
    public static string? SelfCheck(VkSegmentPlan plan, Func<Arg, bool>? isConst = null)
    {
        var liveIn = new HashSet<int>();
        foreach (Arg a in plan.LiveIn) liveIn.Add(a.Idx);
        var liveOut = new HashSet<int>();
        foreach (Arg a in plan.LiveOut) liveOut.Add(a.Idx);

        var produced = new HashSet<int>();
        foreach (LayerInfo op in plan.Ops)
            foreach (Arg a in op.Outputs)
                if (!a.IsEmpty) produced.Add(a.Idx);

        foreach (LayerInfo op in plan.Ops)
        {
            foreach (Arg a in op.Inputs)
            {
                if (a.IsEmpty) continue;
                if (produced.Contains(a.Idx) || liveIn.Contains(a.Idx)) continue;
                // 常量输入不是 live-in（发射器绑权重），但必须承认它是合法来源，
                // 否则「常量被错算成 live-in」这个 bug 会被自检掩盖成「本来就该如此」
                if (isConst is not null && isConst(a)) continue;
                return $"op '{op.Name}' 读了 arg#{a.Idx}，但既不是段内生产也不是 live-in";
            }
        }

        // 段内生产但没人读、也不是 live-out 的 arg：说明多分配了一块内存
        var readInside = new HashSet<int>();
        foreach (LayerInfo op in plan.Ops)
            foreach (Arg a in op.Inputs)
                if (!a.IsEmpty) readInside.Add(a.Idx);

        foreach (LayerInfo op in plan.Ops)
            foreach (Arg a in op.Outputs)
                if (!a.IsEmpty && !readInside.Contains(a.Idx) && !liveOut.Contains(a.Idx))
                    return $"arg#{a.Idx} 由 op '{op.Name}' 生产，段内没人读，也不是 live-out（死槽位）";

        foreach (Arg a in plan.LiveOut)
            if (!produced.Contains(a.Idx))
                return $"live-out arg#{a.Idx} 不是本段生产的";

        return null;
    }
}
