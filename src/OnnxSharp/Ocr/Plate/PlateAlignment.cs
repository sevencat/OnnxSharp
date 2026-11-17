namespace OnnxSharp.Ocr.Plate;

/// <summary>
/// 对已定下来的字序列做 CTC Viterbi 对齐（2L+1 个状态：blank / 字交替），
/// 顺带取出每个字在对齐区间内的最大后验 —— beam 只给整串概率，逐字置信度要靠它。
/// </summary>
internal static class PlateAlignment
{
    public static (int[] Steps, float[] Scores) Run(float[] logp, int count, int[] sequence)
    {
        int width = count + 1;
        int steps = logp.Length / width;
        int labels = sequence.Length;

        if (labels == 0 || steps == 0)
            return ([], []);

        int stateCount = 2 * labels + 1;
        var dp = new double[stateCount];
        var back = new byte[steps * stateCount];

        Array.Fill(dp, double.NegativeInfinity);
        dp[0] = logp[0];

        for (int t = 1; t < steps; t++)
        {
            int at = t * width;
            var next = new double[stateCount];

            for (int s = 0; s < stateCount; s++)
            {
                double best = dp[s];
                byte from = 0;

                if (s > 0 && dp[s - 1] > best)
                {
                    best = dp[s - 1];
                    from = 1;
                }

                // 只有在字状态上才能「跳一个字」；两个相同字之间必须过 blank
                if (s > 1 && (s & 1) == 1 && sequence[(s - 1) / 2] != sequence[(s - 3) / 2] && dp[s - 2] > best)
                {
                    best = dp[s - 2];
                    from = 2;
                }

                if (double.IsNegativeInfinity(best))
                {
                    next[s] = double.NegativeInfinity;
                    back[t * stateCount + s] = 0;
                    continue;
                }

                int emit = (s & 1) == 0 ? 0 : sequence[(s - 1) / 2];
                next[s] = best + logp[at + emit];
                back[t * stateCount + s] = from;
            }

            dp = next;
        }

        // 末帧可以停在末尾 blank，也可以直接停在最后一个字上
        int state = stateCount - 1;
        if (labels > 0 && dp[stateCount - 2] > dp[state])
            state = stateCount - 2;

        var charSteps = new int[labels];
        var charScores = new float[labels];
        Array.Fill(charSteps, -1);

        for (int t = steps - 1; t >= 0; t--)
        {
            if ((state & 1) == 1)
            {
                int index = (state - 1) / 2;
                float value = MathF.Exp(logp[t * width + sequence[index]]);

                charSteps[index] = t;

                if (value > charScores[index])
                    charScores[index] = value;
            }

            if (t == 0)
                break;

            state -= back[t * stateCount + state];
        }

        for (int i = 0; i < labels; i++)
        {
            if (charSteps[i] < 0)
                charSteps[i] = 0;

            if (charScores[i] <= 0)
                charScores[i] = 1e-6f;
        }

        return (charSteps, charScores);
    }
}
