namespace OnnxSharp.Ocr.Plate;

/// <summary>号牌类型（只按字符串形态判，不查车管所规则）。</summary>
public enum PlateKind
{
    /// <summary>没通过校验。</summary>
    Unknown = 0,

    /// <summary>普通燃油车，7 位。</summary>
    Fuel,

    /// <summary>新能源，8 位。</summary>
    NewEnergy,

    /// <summary>警用（末位 警）。</summary>
    Police,

    /// <summary>教练车（末位 学）。</summary>
    Coach,

    /// <summary>挂车（末位 挂）。</summary>
    Trailer,

    /// <summary>香港入出境（末位 港）。</summary>
    HongKong,

    /// <summary>澳门入出境（末位 澳）。</summary>
    Macao,

    /// <summary>使馆 / 领馆（首位 使 / 领）。</summary>
    Embassy,
}

/// <summary>
/// 解析后的车牌号：把一段文本按「省 + 发牌机关字母 + 序号」切开，并判类型。
/// <para>
/// <see cref="TryParse"/> 只做<b>形态</b>校验（长度、逐位字符集、I/O 排除），
/// 不保证真实存在 —— 它的用途是给解码结果打分/过滤，不是做车辆登记核验。
/// </para>
/// </summary>
public sealed class PlateNumber
{
    private PlateNumber(string text, char province, char letter, string serial, PlateKind kind)
    {
        Text = text;
        Province = province;
        Letter = letter;
        Serial = serial;
        Kind = kind;
    }

    /// <summary>归一化后的号牌文本（大写、去掉分隔符）。</summary>
    public string Text { get; }

    /// <summary>首位：省份简称，或 使 / 领。</summary>
    public char Province { get; }

    /// <summary>第 2 位：发牌机关代号（字母）。</summary>
    public char Letter { get; }

    /// <summary>序号部分（第 3 位起）。</summary>
    public string Serial { get; }

    /// <summary>号牌类型。</summary>
    public PlateKind Kind { get; }

    /// <summary>号牌位数（7 或 8）。</summary>
    public int Length => Text.Length;

    /// <summary>
    /// 归一化 + 形态校验。允许输入里带空格 / 中点 / 全角，其它多余字符会导致解析失败。
    /// </summary>
    public static bool TryParse(string? text, out PlateNumber? plate, PlateCharset? charset = null)
    {
        plate = null;
        if (string.IsNullOrEmpty(text))
            return false;

        charset ??= PlateCharset.Default;

        Span<char> buffer = stackalloc char[PlateCharset.MaxLength];

        int length = 0;
        foreach (char raw in text)
        {
            char c = char.ToUpperInvariant(raw);

            // 分隔符直接丢掉（「京A·12345」「京A 12345」都常见）
            if (c is ' ' or '·' or '・' or '-' or '　')
                continue;

            if (!charset.Contains(c) || length == PlateCharset.MaxLength)
                return false;

            buffer[length++] = c;
        }

        if (length is not (7 or 8))
            return false;

        ReadOnlySpan<char> span = buffer[..length];

        // 逐位对照「该位允许的字符」
        for (int position = 0; position < length; position++)
        {
            if (charset.AllowedAt(position, length).IndexOf(span[position]) < 0)
                return false;
        }

        string normalized = new(span);
        char tail = span[^1];

        PlateKind kind = tail switch
        {
            '学' => PlateKind.Coach,
            '警' => PlateKind.Police,
            '挂' => PlateKind.Trailer,
            '港' => PlateKind.HongKong,
            '澳' => PlateKind.Macao,
            _ => span[0] is '使' or '领'
                ? PlateKind.Embassy
                : length == 8 ? PlateKind.NewEnergy : PlateKind.Fuel,
        };

        plate = new PlateNumber(normalized, span[0], span[1], normalized[2..], kind);
        return true;
    }

    public override string ToString() => Text;
}
