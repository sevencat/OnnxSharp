namespace OnnxSharp.Ocr.Plate;

/// <summary>
/// 字符集 rank ↔ rec 模型类别下标的映射。
/// <para>
/// PP-OCRv6 的字典是「索引 0 = blank，末尾补一个空格」的 6 千多类表；这里只留下车牌字符集里
/// 出现过的单字符类（≤ 70 个），其余类别在解码时按概率 0 处理。
/// </para>
/// </summary>
internal sealed class RankTable
{
    private readonly int[][] _byPosition;

    public RankTable(PlateCharset charset, IReadOnlyList<string> dictionary, int classCount)
    {
        Count = charset.Count;
        ClassOfRank = new int[Count + 1];
        Array.Fill(ClassOfRank, -1);

        for (int c = 1; c < classCount; c++)
        {
            string text = dictionary[c];
            if (text.Length != 1)
                continue;

            int rank = charset.RankOf(text[0]);

            // 同一个字在字典里可能出现两次（大小写/全角变体），取第一个
            if (rank > 0 && ClassOfRank[rank] < 0)
                ClassOfRank[rank] = c;
        }

        // 每个位置可用的 rank：结构模式按号牌逐位规则，非结构模式退化成整个字符集
        _byPosition = new int[PlateCharset.MaxLength][];

        for (int position = 0; position < PlateCharset.MaxLength; position++)
        {
            ReadOnlySpan<char> allowed = charset.AllowedAt(position, PlateCharset.MaxLength);
            var ranks = new List<int>(allowed.Length);

            foreach (char c in allowed)
            {
                int rank = charset.RankOf(c);
                if (rank > 0 && ClassOfRank[rank] >= 0)
                    ranks.Add(rank);
            }

            _byPosition[position] = [.. ranks];
        }

        var all = new List<int>(Count);

        for (int rank = 1; rank <= Count; rank++)
        {
            if (ClassOfRank[rank] >= 0)
                all.Add(rank);
        }

        AllRanks = [.. all];
    }

    /// <summary>字符集大小（rank 上界）。</summary>
    public int Count { get; }

    /// <summary>rank → 类别下标；该字不在字典里时为 -1。</summary>
    public int[] ClassOfRank { get; }

    /// <summary>整个字符集里「字典里确实存在」的 rank（非结构模式用）。</summary>
    public int[] AllRanks { get; }

    /// <summary>
    /// 第 <paramref name="position"/> 位可用的 rank。
    /// <para>
    /// 注意按 <see cref="PlateCharset.MaxLength"/> 位取「并集」：解码时还不知道最终是 7 位还是 8 位，
    /// 第 6 位既可能是 7 位号的末位（可出 学/警/挂…）也可能是 8 位号的中间位，所以取宽的。
    /// 最终合法性由 <see cref="PlateNumber.TryParse"/> 兜底。
    /// </para>
    /// </summary>
    public int[] RanksAt(int position, bool useStructure)
        => useStructure ? _byPosition[Math.Min(position, _byPosition.Length - 1)] : AllRanks;
}
