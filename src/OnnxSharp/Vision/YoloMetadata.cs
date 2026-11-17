namespace OnnxSharp.Vision;

/// <summary>
/// ultralytics 导出模型 <c>metadata_props</c> 里的类别表（<c>names</c>）。
/// <para>
/// 该字段是 Python <c>dict</c> 的 <c>str()</c> 形式，不是 JSON：
/// <c>{0: 'person', 1: 'bicycle', …, 79: 'toothbrush'}</c>。解析后按<b>类别编号</b>索引，
/// 编号不连续（裁剪过的模型）时中间留空串，因此下标本身就是模型输出的类别 id。
/// </para>
/// <para>
/// 没有 <c>names</c>（自训练后手工转的 ONNX、老版本导出）时返回 <see cref="Empty"/>，
/// 调用方应回退到自己的名字表（如 <see cref="CocoLabels"/>）。
/// </para>
/// </summary>
public sealed class YoloMetadata
{
    /// <summary>类别编号上限：超过就认为 metadata 不可信，直接忽略（防止恶意模型撑爆数组）。</summary>
    private const int MaxClassId = 100_000;

    private static readonly string[] NoNames = Array.Empty<string>();

    /// <summary>没有 metadata 时的空实例（<see cref="Names"/> 为空表）。</summary>
    public static YoloMetadata Empty { get; } = new(NoNames, null, null);

    private YoloMetadata(string[] names, int? inputHeight, int? inputWidth)
    {
        Names = names;
        InputHeight = inputHeight;
        InputWidth = inputWidth;
    }

    /// <summary>按类别编号索引的名字表；缺号的槽位是空串。没有 metadata 时为空表。</summary>
    public IReadOnlyList<string> Names { get; }

    /// <summary>类别数（<see cref="Names"/> 的长度，含缺号）。</summary>
    public int Count => Names.Count;

    /// <summary>模型 metadata 声明的训练输入高（<c>imgsz</c>）；没有则为 <c>null</c>。</summary>
    public int? InputHeight { get; }

    /// <summary>模型 metadata 声明的训练输入宽（<c>imgsz</c>）；没有则为 <c>null</c>。</summary>
    public int? InputWidth { get; }

    /// <summary>取类别名；没有该编号（越界或缺号）返回 <c>null</c>。</summary>
    public string? NameOf(int classId)
        => (uint)classId < (uint)Names.Count && Names[classId].Length != 0 ? Names[classId] : null;

    /// <summary>取类别名，越界时退回 <c>class{id}</c>（日志 / 可视化用，永不返回 null）。</summary>
    public string LabelOf(int classId) => NameOf(classId) ?? $"class{classId}";

    /// <summary>从模型的 <c>metadata_props</c> 解析；没有 <c>names</c> 时返回 <see cref="Empty"/>。</summary>
    /// <param name="metadata"><see cref="Core.Net.Metadata"/> 之类原样字典；null 视为空。</param>
    public static YoloMetadata Parse(IReadOnlyDictionary<string, string>? metadata)
    {
        if (metadata is null || metadata.Count == 0)
            return Empty;

        metadata.TryGetValue("names", out string? namesText);
        metadata.TryGetValue("imgsz", out string? imageSizeText);

        var names = ParseNames(namesText);
        var (height, width) = ParseImageSize(imageSizeText);

        return names.Length == 0 && height is null
            ? Empty
            : new YoloMetadata(names, height, width);
    }

    /// <summary>
    /// 解析 <c>{0: 'person', 1: "bicycle"}</c>。容忍单/双引号、带引号的键、多余空白和换行；
    /// 名字里的 <c>\'</c> / <c>\\</c> 转义会还原。解析不出任何条目时返回空表。
    /// </summary>
    internal static string[] ParseNames(string? text)
    {
        if (string.IsNullOrWhiteSpace(text))
            return NoNames;

        var entries = new SortedDictionary<int, string>();
        int i = 0;
        int length = text.Length;

        while (i < length)
        {
            // 键：裸数字（0:）或引号串（'0': / "0":）。
            while (i < length && !char.IsAsciiDigit(text[i]) && text[i] is not ('\'' or '"'))
                i++;
            if (i >= length)
                break;

            int key;
            if (char.IsAsciiDigit(text[i]))
            {
                int start = i;
                while (i < length && char.IsAsciiDigit(text[i]))
                    i++;

                if (!int.TryParse(text.AsSpan(start, i - start), out key))
                    break;
            }
            else if (!int.TryParse(ReadQuoted(text, ref i), out key))
            {
                continue;
            }

            while (i < length && char.IsWhiteSpace(text[i]))
                i++;
            if (i >= length || text[i] != ':')
                continue;
            i++;

            while (i < length && char.IsWhiteSpace(text[i]))
                i++;

            string? name = ReadQuoted(text, ref i);
            if (name is null)
                continue;

            if ((uint)key <= MaxClassId)
                entries[key] = name;
            else
                return NoNames;  // 编号离谱：整份 metadata 不可信
        }

        if (entries.Count == 0)
            return NoNames;

        int count = entries.Keys.Max() + 1;
        var names = new string[count];
        Array.Fill(names, "");
        foreach (var (id, name) in entries)
            names[id] = name;

        return names;
    }

    /// <summary>解析 <c>[480, 640]</c>（高, 宽）；只有一维时宽高相同。</summary>
    private static (int? Height, int? Width) ParseImageSize(string? text)
    {
        if (string.IsNullOrWhiteSpace(text))
            return (null, null);

        Span<int> values = stackalloc int[2];
        int found = 0;
        int i = 0;

        while (i < text.Length && found < 2)
        {
            if (!char.IsAsciiDigit(text[i]))
            {
                i++;
                continue;
            }

            int start = i;
            while (i < text.Length && char.IsAsciiDigit(text[i]))
                i++;

            if (int.TryParse(text.AsSpan(start, i - start), out int value) && value > 0)
                values[found++] = value;
        }

        return found switch
        {
            2 => (values[0], values[1]),
            1 => (values[0], values[0]),
            _ => (null, null),
        };
    }

    /// <summary>读一个引号串（'…' 或 "…"），指针停在收尾引号之后；不是引号则返回 null。</summary>
    private static string? ReadQuoted(string text, ref int index)
    {
        if (index >= text.Length || text[index] is not ('\'' or '"'))
            return null;

        char quote = text[index++];
        var builder = new System.Text.StringBuilder();

        while (index < text.Length)
        {
            char ch = text[index++];
            if (ch == quote)
                return builder.ToString();

            if (ch == '\\' && index < text.Length)
            {
                // 只还原「引号 / 反斜杠」两种转义：其它（\n、\t）在类名里没有实际意义，
                // 按 Python repr 的原样保留会被误读，所以直接取下一个字符。
                builder.Append(text[index++]);
                continue;
            }

            builder.Append(ch);
        }

        return null;  // 引号没闭合：当这条坏掉
    }
}
