namespace OnnxSharp.Ocr.Plate;

/// <summary>
/// 车牌字符集与<b>逐位</b>允许字符表（GA 36-2018 民用号牌）。
/// <para>
/// 车牌是极少数「字符集固定且位置有语义」的 OCR 场景，把它当先验直接喂给 CTC 解码，
/// 比在几千类通用字典上做无约束贪心稳定得多：
/// <list type="bullet">
/// <item>第 0 位只能是省份简称（或 使 / 领）；</item>
/// <item>第 1 位只能是字母（<b>不含 I、O</b>，避免与 1、0 混淆）；</item>
/// <item>中间位字母 + 数字；末位额外允许 学 / 警 / 挂 / 港 / 澳 / 领。</item>
/// </list>
/// 识别模型仍是 PP-OCRv6 那份 6 千多类的通用 rec 图 —— 这里不改模型，只改解码时的约束
/// （等价于把「其他 6800 类」的概率按 0 处理）。
/// </para>
/// </summary>
public sealed class PlateCharset
{
    /// <summary>省份简称（31 个）。</summary>
    public const string Provinces = "京津冀晋蒙辽吉黑沪苏浙皖闽赣鲁豫鄂湘粤桂琼渝川贵云藏陕甘青宁新";

    /// <summary>车牌用到的字母：A-Z 去掉 I 和 O（24 个）。</summary>
    public const string Letters = "ABCDEFGHJKLMNPQRSTUVWXYZ";

    /// <summary>数字。</summary>
    public const string Digits = "0123456789";

    /// <summary>只能出现在<b>末位</b>的汉字（教练车 / 警车 / 挂车 / 港澳 / 使馆）。</summary>
    public const string TailChars = "学警挂港澳领";

    /// <summary>号牌最大位数（新能源 8 位）。</summary>
    public const int MaxLength = 8;

    private static readonly char[] LeadingSet = (Provinces + "使领").ToCharArray();
    private static readonly char[] LetterSet = Letters.ToCharArray();
    private static readonly char[] AlnumSet = (Letters + Digits).ToCharArray();
    private static readonly char[] TailSet = (Letters + Digits + TailChars).ToCharArray();

    private static readonly PlateCharset DefaultInstance = new();

    private readonly byte[] _rank = new byte[ushort.MaxValue + 1];

    private PlateCharset()
    {
        // 去重后按序发放 rank（1..Count）：AllSet 里 领 同时属于「首位」与「末位」，必须只算一次
        var chars = new List<char>(96);

        void Add(string text)
        {
            foreach (char c in text)
            {
                if (_rank[c] == 0 && !chars.Contains(c))
                {
                    chars.Add(c);
                    _rank[c] = (byte)(chars.Count);   // rank 从 1 开始，0 = 不在字符集里
                }
            }
        }

        Add(Provinces);
        Add("使领");
        Add(Letters);
        Add(Digits);
        Add(TailChars);

        var rankToChar = new char[chars.Count + 1];
        for (int i = 0; i < chars.Count; i++)
            rankToChar[i + 1] = chars[i];

        RankToChar = rankToChar;
    }

    /// <summary>默认字符集实例（无状态，可共享）。</summary>
    public static PlateCharset Default => DefaultInstance;

    /// <summary>rank（1..<see cref="Count"/>）→ 字符；下标 0 不用。</summary>
    public char[] RankToChar { get; }

    /// <summary>字符集大小（含末位专用字）。</summary>
    public int Count => RankToChar.Length - 1;

    /// <summary>是否为车牌合法字符（大写）。</summary>
    public bool Contains(char c) => _rank[c] != 0;

    /// <summary>字符的 rank；不在字符集里返回 0。</summary>
    public int RankOf(char c) => _rank[c];

    /// <summary>
    /// 第 <paramref name="position"/> 位允许的字符（号牌共 <paramref name="length"/> 位）。
    /// 返回内部数组的视图，<b>不要修改</b>。
    /// </summary>
    public ReadOnlySpan<char> AllowedAt(int position, int length)
    {
        if (position == 0)
            return LeadingSet;

        if (position == 1)
            return LetterSet;

        return position >= length - 1 ? TailSet : AlnumSet;
    }
}
