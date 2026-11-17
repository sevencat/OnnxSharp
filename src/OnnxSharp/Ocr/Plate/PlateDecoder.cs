namespace OnnxSharp.Ocr.Plate;

/// <summary>
/// 车牌专用 CTC 解码：通用 rec 模型的输出只保留车牌字符集中的类，再按号牌逐位规则做
/// <b>前缀 beam search</b>（不是逐帧 argmax）。
/// <para>
/// <b>为什么要 beam</b>：贪心在「某个字只占一两帧」或「两个相同的字相邻」时会漏字/并字；
/// 车牌只有 7~8 个字、搜索空间极小，beam 全开也只多花零点几毫秒，还能顺便拿到逐字后验。
/// </para>
/// <para>
/// <b>为什么快</b>：字典 6 千多类，但这里<b>只对车牌字符集的 70 来个类取 log</b>（blank 单算），
/// 其余类别按概率 0 处理 —— 相当于把 98% 的输出维度白扔掉。
/// </para>
/// </summary>
public static class PlateDecoder
{
    /// <summary>按车牌字符集 + 逐位规则解码。</summary>
    /// <param name="probabilities">rec 网络输出，行主序 <c>steps × classCount</c>（softmax 后的概率或原始 logits 均可）。</param>
    /// <param name="classCount">类别数（等于 rec 模型最后一维）。</param>
    /// <param name="dictionary">模型字典，<b>索引与类别一一对应</b>（0 = blank）。</param>
    public static PlateDecoding Decode(
        ReadOnlySpan<float> probabilities,
        int classCount,
        IReadOnlyList<string> dictionary,
        PlateCharset? charset = null,
        PlateDecoderOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(dictionary);

        charset ??= PlateCharset.Default;
        options ??= new PlateDecoderOptions();

        if (classCount <= 1 || dictionary.Count < classCount)
            throw new ArgumentException(
                $"dictionary has {dictionary.Count} entries but the model has {classCount} classes", nameof(dictionary));

        int steps = probabilities.Length / classCount;
        if (steps <= 0)
            return Empty();

        var table = new RankTable(charset, dictionary, classCount);
        int maxLength = Math.Clamp(options.MaxLength, 1, PlateCharset.MaxLength);
        int minLength = Math.Clamp(options.MinLength, 1, maxLength);

        float[] logp = BuildLogProbabilities(probabilities, classCount, steps, table);

        var candidates = new int[maxLength][];
        for (int position = 0; position < maxLength; position++)
            candidates[position] = table.RanksAt(position, options.UseStructure);

        int[] sequence;
        double totalLog;

        if (options.BeamWidth <= 1)
            sequence = GreedySearch(logp, table.Count, minLength, maxLength, candidates, out totalLog);
        else
            sequence = PlateBeamSearch.Run(logp, table.Count, minLength, maxLength,
                candidates, options.BeamWidth, out totalLog);

        if (sequence.Length == 0)
            return Empty();

        var builder = new System.Text.StringBuilder(sequence.Length);
        var classIndices = new int[sequence.Length];

        for (int i = 0; i < sequence.Length; i++)
        {
            builder.Append(charset.RankToChar[sequence[i]]);
            classIndices[i] = table.ClassOfRank[sequence[i]];
        }

        (int[] stepsOfChars, float[] charScores) = PlateAlignment.Run(logp, table.Count, sequence);

        double sum = 0;
        foreach (float score in charScores)
            sum += Math.Log(Math.Max(score, 1e-30f));

        string text = builder.ToString();
        PlateNumber.TryParse(text, out PlateNumber? plate, charset);

        if (options.Validate && plate is null)
            return Empty();

        return new PlateDecoding(
            text,
            (float)Math.Exp(totalLog),
            (float)Math.Exp(sum / Math.Max(1, charScores.Length)),
            classIndices,
            stepsOfChars,
            charScores,
            plate);
    }

    private static PlateDecoding Empty() => new(string.Empty, 0f, 0f, [], [], [], null);

    /// <summary>
    /// 整理成 <c>[blank, rank 1..K]</c> 的 log 概率表（行主序，每行 <c>count+1</c> 个）。
    /// </summary>
    private static float[] BuildLogProbabilities(
        ReadOnlySpan<float> probabilities,
        int classCount,
        int steps,
        RankTable table)
    {
        int width = table.Count + 1;
        var logp = new float[steps * width];

        // 先判一次「是不是已经 softmax 过」：概率图里所有值都 ≤ 1
        bool isProbability = true;
        for (int i = 0; i < probabilities.Length; i++)
        {
            if (probabilities[i] > 1.0001f)
            {
                isProbability = false;
                break;
            }
        }

        for (int t = 0; t < steps; t++)
        {
            ReadOnlySpan<float> row = probabilities.Slice(t * classCount, classCount);
            int at = t * width;

            if (isProbability)
            {
                logp[at] = MathF.Log(MathF.Max(row[0], 1e-30f));

                for (int rank = 1; rank <= table.Count; rank++)
                {
                    int c = table.ClassOfRank[rank];
                    logp[at + rank] = c < 0 ? float.NegativeInfinity : MathF.Log(MathF.Max(row[c], 1e-30f));
                }

                continue;
            }

            // 原始 logits：先减 max 再按整行归一化（这里必须扫全 6 千类，只发生在非 softmax 模型上）
            float max = float.NegativeInfinity;
            for (int c = 0; c < classCount; c++)
                max = MathF.Max(max, row[c]);

            double total = 0;
            for (int c = 0; c < classCount; c++)
                total += Math.Exp(row[c] - max);

            if (total <= 0)
                total = 1;

            float norm = (float)Math.Log(total);
            logp[at] = row[0] - max - norm;

            for (int rank = 1; rank <= table.Count; rank++)
            {
                int c = table.ClassOfRank[rank];
                logp[at + rank] = c < 0 ? float.NegativeInfinity : row[c] - max - norm;
            }
        }

        return logp;
    }

    /// <summary>逐位受限贪心：每帧在「blank ∪ 当前位允许的字符」里取 argmax。</summary>
    private static int[] GreedySearch(
        float[] logp,
        int count,
        int minLength,
        int maxLength,
        int[][] candidates,
        out double totalLog)
    {
        int width = count + 1;
        int steps = logp.Length / width;

        var output = new List<int>(maxLength);
        int previous = 0;
        totalLog = 0;

        for (int t = 0; t < steps; t++)
        {
            int at = t * width;
            int best = 0;
            float bestLog = logp[at];

            if (output.Count < maxLength)
            {
                foreach (int rank in candidates[output.Count])
                {
                    float value = logp[at + rank];
                    if (value > bestLog)
                    {
                        bestLog = value;
                        best = rank;
                    }
                }
            }

            if (best == 0 || best == previous)
            {
                previous = best;
                continue;
            }

            previous = best;
            output.Add(best);
            totalLog += bestLog;
        }

        return output.Count < minLength || output.Count > maxLength ? [] : [.. output];
    }
}
