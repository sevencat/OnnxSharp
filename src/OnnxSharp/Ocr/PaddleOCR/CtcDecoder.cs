namespace OnnxSharp.Ocr.PaddleOCR;

/// <summary>
/// CTC 贪心解码（PaddleOCR 的识别头）：逐时间步取 argmax → 去掉重复 → 去掉 blank(0) → 查字典。
/// <para>
/// PP-OCRv6 的 rec 图末尾自带 <c>Softmax</c>，输出已经是概率；换成分输出 logits 的模型时
/// 这里会自己补一次 softmax 再算分数（用 <c>max &gt; 1</c> 判据，和 SimdOnnxExt 的做法一致）。
/// </para>
/// </summary>
internal static class CtcDecoder
{
    public static (string Text, float Score) Decode(
        ReadOnlySpan<float> probabilities,
        int classCount,
        IReadOnlyList<string> dictionary)
    {
        if (classCount <= 0)
            return (string.Empty, 0f);

        int steps = probabilities.Length / classCount;
        if (steps <= 0)
            return (string.Empty, 0f);

        // 先判一次「输出是不是概率」：概率图里所有值都 ≤ 1。
        bool isProbability = true;
        for (int i = 0; i < probabilities.Length; i++)
        {
            if (probabilities[i] > 1.0001f)
            {
                isProbability = false;
                break;
            }
        }

        var builder = new System.Text.StringBuilder(steps / 2);
        double scoreSum = 0;
        int kept = 0;
        int previous = -1;

        for (int t = 0; t < steps; t++)
        {
            ReadOnlySpan<float> row = probabilities.Slice(t * classCount, classCount);

            int best = 0;
            float bestValue = row[0];
            for (int c = 1; c < classCount; c++)
            {
                if (row[c] > bestValue)
                {
                    bestValue = row[c];
                    best = c;
                }
            }

            if (best == 0 || best == previous)
            {
                previous = best;
                continue;
            }

            previous = best;
            if ((uint)best < (uint)dictionary.Count)
                builder.Append(dictionary[best]);

            scoreSum += isProbability ? bestValue : Softmax(row, best, bestValue);
            kept++;
        }

        return (builder.ToString(), kept == 0 ? 0f : (float)(scoreSum / kept));
    }

    private static float Softmax(ReadOnlySpan<float> row, int index, float max)
    {
        double total = 0;
        for (int i = 0; i < row.Length; i++)
            total += Math.Exp(row[i] - max);
        return total > 0 ? (float)(1.0 / total) : 0f;
    }
}
