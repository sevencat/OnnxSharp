namespace OnnxSharp.Ocr.Plate;

/// <summary>车牌解码的可调项。</summary>
public sealed class PlateDecoderOptions
{
    /// <summary>
    /// 前缀 beam 宽度。1 = 退化成「逐位受限贪心」（最快，清晰图够用）；
    /// 默认 8（80 步 × 70 类的表，单张约 0.3 ms）。
    /// </summary>
    public int BeamWidth { get; set; } = 8;

    /// <summary>最短位数（普通号牌 7 位）。</summary>
    public int MinLength { get; set; } = 7;

    /// <summary>最长位数（新能源 8 位）。</summary>
    public int MaxLength { get; set; } = PlateCharset.MaxLength;

    /// <summary>
    /// 是否按「第几位能出什么字」约束（省 + 字母 + 数字的逐位规则）。
    /// 关掉后只在全局车牌字符集上解码 —— 用来做 A/B，看结构先验到底值多少。
    /// </summary>
    public bool UseStructure { get; set; } = true;

    /// <summary>结果必须通过 <see cref="PlateNumber.TryParse"/>；关掉则只要字符集内就返回。</summary>
    public bool Validate { get; set; } = true;
}

/// <summary>一次车牌解码的结果。</summary>
public sealed class PlateDecoding
{
    internal PlateDecoding(
        string text,
        float probability,
        float score,
        int[] classIndices,
        int[] steps,
        float[] charScores,
        PlateNumber? plate)
    {
        Text = text;
        Probability = probability;
        Score = score;
        ClassIndices = classIndices;
        Steps = steps;
        CharScores = charScores;
        Plate = plate;
    }

    /// <summary>解码出的文本（已按车牌字符集约束）。</summary>
    public string Text { get; }

    /// <summary>CTC 序列概率 <c>P(labels | x)</c>（0~1）。</summary>
    public float Probability { get; }

    /// <summary>逐字概率的几何平均（0~1）：比 <see cref="Probability"/> 更贴近「每个字有多确定」。</summary>
    public float Score { get; }

    /// <summary>每个字对应的 CTC 类别下标（可回头查模型字典）。</summary>
    public int[] ClassIndices { get; }

    /// <summary>每个字落在第几个时间步（对齐后）。</summary>
    public int[] Steps { get; }

    /// <summary>每个字的对齐概率。</summary>
    public float[] CharScores { get; }

    /// <summary>通过形态校验时的结构化号牌；未通过为 <c>null</c>。</summary>
    public PlateNumber? Plate { get; }

    /// <summary>是否是一个形态合法的车牌。</summary>
    public bool IsPlate => Plate is not null;

    public override string ToString() => $"{Text} ({Score:P1})";
}
