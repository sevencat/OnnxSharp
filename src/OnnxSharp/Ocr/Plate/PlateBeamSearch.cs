using System.Runtime.InteropServices;

namespace OnnxSharp.Ocr.Plate;

/// <summary>
/// CTC 前缀 beam search（log 域），带「第几位能出什么字」的位置约束。
/// <para>
/// 状态是「已折叠的前缀」：<c>Pb</c> = 该前缀且当前帧停在 blank 的对数概率，
/// <c>Pnb</c> = 该前缀且当前帧停在前缀末字上的对数概率。三条转移与教科书实现一致：
/// 吐 blank（两个状态都落到 <c>Pb</c>）、重复末字（只有 <c>Pnb</c> 能折叠回自己）、
/// 追加新字（新字与末字相同时必须隔一个 blank，所以只从 <c>Pb</c> 转移）。
/// </para>
/// <para>
/// 前缀打包进一个 <see cref="long"/>：低 56 位每 7 位放一个 rank，高 8 位放长度。
/// 车牌最多 8 位、字符集 ≤ 127，正好装下 —— 省掉每步的字符串分配。
/// </para>
/// </summary>
internal static class PlateBeamSearch
{
    private const int SlotBits = 7;
    private const int LengthShift = 56;
    private const double NegInf = double.NegativeInfinity;

    private struct State
    {
        public double Pb;
        public double Pnb;
    }

    public static int[] Run(
        float[] logp,
        int count,
        int minLength,
        int maxLength,
        int[][] candidates,
        int beamWidth,
        out double totalLog)
    {
        int width = count + 1;
        int steps = logp.Length / width;

        var beam = new Dictionary<long, State>(beamWidth * 4)
        {
            [0L] = new State { Pb = 0, Pnb = NegInf },
        };

        for (int t = 0; t < steps; t++)
        {
            int at = t * width;
            float blank = logp[at];
            var next = new Dictionary<long, State>(beam.Count * 4);

            foreach (KeyValuePair<long, State> entry in beam)
            {
                long key = entry.Key;
                State state = entry.Value;
                int length = (int)(key >> LengthShift);
                double sum = LogAdd(state.Pb, state.Pnb);

                Accumulate(next, key, sum + blank, NegInf);

                if (length > 0)
                {
                    int last = LastRank(key, length);
                    Accumulate(next, key, NegInf, state.Pnb + logp[at + last]);
                }

                if (length >= maxLength)
                    continue;

                int previous = length > 0 ? LastRank(key, length) : 0;

                foreach (int rank in candidates[length])
                {
                    float emit = logp[at + rank];
                    if (float.IsNegativeInfinity(emit))
                        continue;

                    long appended = (key | ((long)rank << (SlotBits * length))) + (1L << LengthShift);

                    if (rank == previous)
                        Accumulate(next, appended, NegInf, state.Pb + emit);
                    else
                        Accumulate(next, appended, NegInf, sum + emit);
                }
            }

            beam = next.Count <= beamWidth ? next : Prune(next, beamWidth);
        }

        long bestKey = 0;
        totalLog = NegInf;

        foreach (KeyValuePair<long, State> entry in beam)
        {
            int length = (int)(entry.Key >> LengthShift);
            if (length < minLength || length > maxLength)
                continue;

            double total = LogAdd(entry.Value.Pb, entry.Value.Pnb);
            if (total > totalLog)
            {
                totalLog = total;
                bestKey = entry.Key;
            }
        }

        // 没有合法长度的前缀（严重模糊/空白图）：退回整体最优，结果随后会被 Validate 拦掉
        if (double.IsNegativeInfinity(totalLog))
        {
            foreach (KeyValuePair<long, State> entry in beam)
            {
                int length = (int)(entry.Key >> LengthShift);
                if (length == 0)
                    continue;

                double total = LogAdd(entry.Value.Pb, entry.Value.Pnb);
                if (total > totalLog)
                {
                    totalLog = total;
                    bestKey = entry.Key;
                }
            }
        }

        if (double.IsNegativeInfinity(totalLog))
            return [];

        int lengthOfBest = (int)(bestKey >> LengthShift);
        var sequence = new int[lengthOfBest];

        for (int i = 0; i < lengthOfBest; i++)
            sequence[i] = (int)((bestKey >> (SlotBits * i)) & 0x7F);

        return sequence;
    }

    private static void Accumulate(Dictionary<long, State> map, long key, double pb, double pnb)
    {
        ref State slot = ref CollectionsMarshal.GetValueRefOrAddDefault(map, key, out bool exists);

        if (!exists)
            slot = new State { Pb = NegInf, Pnb = NegInf };

        slot.Pb = LogAdd(slot.Pb, pb);
        slot.Pnb = LogAdd(slot.Pnb, pnb);
    }

    private static Dictionary<long, State> Prune(Dictionary<long, State> states, int beamWidth)
    {
        var scored = new (long Key, double Total)[states.Count];
        int i = 0;

        foreach (KeyValuePair<long, State> entry in states)
            scored[i++] = (entry.Key, LogAdd(entry.Value.Pb, entry.Value.Pnb));

        Array.Sort(scored, static (a, b) => b.Total.CompareTo(a.Total));

        var kept = new Dictionary<long, State>(beamWidth);

        for (int k = 0; k < beamWidth && k < scored.Length; k++)
            kept[scored[k].Key] = states[scored[k].Key];

        return kept;
    }

    private static int LastRank(long key, int length)
        => (int)((key >> (SlotBits * (length - 1))) & 0x7F);

    private static double LogAdd(double a, double b)
    {
        if (a < b)
            (a, b) = (b, a);

        if (double.IsNegativeInfinity(a) || double.IsNegativeInfinity(b))
            return a;

        return a + Math.Log(1 + Math.Exp(b - a));
    }
}
