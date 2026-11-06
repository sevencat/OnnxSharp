using OnnxSharp.Core;

namespace OnnxSharp.Engine;

/// <summary>
/// 按生命周期区间给运行期 Arg 分配共享缓冲槽。
/// 对应 OpenCV 的 <c>Net::Impl::assignBuffers()</c>（<c>graph_buffer_allocator.cpp</c>）。
/// <para>
/// 算法：
/// <list type="number">
/// <item>每个 Arg 的存活区间 = [首个写它的 op 下标, 最后一次读它的 op 下标]；
/// 图输出额外活到图末尾（下标 = <c>prog.Count</c>）。</item>
/// <item>按 firstWrite 升序贪心着色：在已释放的槽里挑「最晚释放」的复用，否则开新槽。</item>
/// <item>区间不重叠的 Arg 共用一个槽，也就共用一个 <see cref="Tensors.Tensor"/> 对象；
/// 槽只增不减（沿用 <c>Tensor.Fit</c> 的复用语义）。</item>
/// <item>常量与图输入不参与（它们持有自己的存储）。</item>
/// </list>
/// </para>
/// <para>
/// <b>为什么不按尺寸着色</b>：形状要到运行时才推断，而槽可以在这之前定下来
/// （槽第一次被 Fit 时按需扩容）。与 OpenCV 一致。
/// </para>
/// <para>
/// <b>保守点</b>：相邻两个 Arg 若「前者的最后读 == 后者的首次写」（同一个 op 上读完再写），
/// 本实现<b>不</b>复用（严格 <c>&lt;</c> 判定），避免任何“先写后读”的算子踩坏输入。
/// 代价是峰值多出一代张量。
/// </para>
/// </summary>
public static class BufferAllocator
{
    /// <summary>表示「不分配槽」。</summary>
    public const int NoSlot = -1;

    /// <summary>表示「没有任何算子写它」（图输入、常量、死 Arg）。</summary>
    public const int NeverWritten = int.MaxValue;

    /// <summary>
    /// 计算每个 Arg 的共享缓冲槽编号。
    /// 返回数组长度 = <paramref name="argCount"/>，元素为槽号或 <see cref="NoSlot"/>。
    /// </summary>
    public static int[] Assign(
        IReadOnlyList<LayerInfo> prog,
        IReadOnlyList<Arg> graphOutputs,
        int argCount,
        out int slotCount)
    {
        ArgumentNullException.ThrowIfNull(prog);
        ArgumentNullException.ThrowIfNull(graphOutputs);

        if (argCount <= 0)
        {
            slotCount = 0;
            return Array.Empty<int>();
        }

        var slotOf = new int[argCount];
        Array.Fill(slotOf, NoSlot);

        var (firstWrite, lastRead) = Analyze(prog, graphOutputs, argCount);

        // 收集需要槽的 Arg，按首次写入顺序排序（同序号按 idx，保证确定性）。
        var order = new List<int>();
        for (int i = 0; i < argCount; i++)
        {
            if (firstWrite[i] != NeverWritten)
                order.Add(i);
        }

        order.Sort((a, b) => firstWrite[a] != firstWrite[b]
            ? firstWrite[a].CompareTo(firstWrite[b])
            : a.CompareTo(b));

        var slotFreeAt = new List<int>();
        foreach (int idx in order)
        {
            int start = firstWrite[idx];
            int end = Math.Max(lastRead[idx], start);

            int chosen = NoSlot;
            int bestFree = int.MinValue;
            for (int s = 0; s < slotFreeAt.Count; s++)
            {
                // 严格小于：同一算子下标上「读完即写」不复用。
                if (slotFreeAt[s] < start && slotFreeAt[s] > bestFree)
                {
                    bestFree = slotFreeAt[s];
                    chosen = s;
                }
            }

            if (chosen == NoSlot)
            {
                chosen = slotFreeAt.Count;
                slotFreeAt.Add(end);
            }
            else
            {
                slotFreeAt[chosen] = end;
            }

            slotOf[idx] = chosen;
        }

        slotCount = slotFreeAt.Count;
        return slotOf;
    }

    /// <summary>
    /// 计算每个 Arg 的首次写入下标与最后读取下标。
    /// <see cref="NeverWritten"/> 表示从未被写入；<c>-1</c> 表示从未被读取。
    /// </summary>
    public static (int[] FirstWrite, int[] LastRead) Analyze(
        IReadOnlyList<LayerInfo> prog,
        IReadOnlyList<Arg> graphOutputs,
        int argCount)
    {
        ArgumentNullException.ThrowIfNull(prog);
        ArgumentNullException.ThrowIfNull(graphOutputs);

        var firstWrite = new int[Math.Max(argCount, 0)];
        var lastRead = new int[Math.Max(argCount, 0)];
        Array.Fill(firstWrite, NeverWritten);
        Array.Fill(lastRead, -1);

        int nops = prog.Count;
        for (int i = 0; i < nops; i++)
        {
            var op = prog[i];
            foreach (var arg in op.Outputs)
            {
                if ((uint)arg.Idx < (uint)argCount && firstWrite[arg.Idx] == NeverWritten)
                    firstWrite[arg.Idx] = i;
            }

            foreach (var arg in op.Inputs)
            {
                if ((uint)arg.Idx < (uint)argCount && lastRead[arg.Idx] < i)
                    lastRead[arg.Idx] = i;
            }
        }

        // 图输出必须活到图末尾，否则会被后续算子覆盖。
        foreach (var arg in graphOutputs)
        {
            if ((uint)arg.Idx < (uint)argCount && lastRead[arg.Idx] < nops)
                lastRead[arg.Idx] = nops;
        }

        return (firstWrite, lastRead);
    }
}
