using System.Globalization;
using OnnxSharp.Core;

namespace OnnxSharp.Speech;

/// <summary>
/// <c>tokens.txt</c>（每行 <c>"&lt;token&gt; &lt;id&gt;"</c>）与 Paraformer 的解码规则。
/// <para>
/// 规则（与 sherpa / funasr 一致）：逐帧取 argmax → 跳过 <c>&lt;blank&gt;</c>(0) →
/// <b>折叠连续重复</b> → 遇到 <c>&lt;/s&gt;</c>(2) 停止。
/// </para>
/// </summary>
public sealed class TokenTable
{
    public const int BlankId = 0;
    public const int SosId = 1;
    public const int EosId = 2;

    private readonly string[] _tokens;

    private TokenTable(string[] tokens) => _tokens = tokens;

    public int Count => _tokens.Length;

    public string this[int id] => (uint)id < (uint)_tokens.Length ? _tokens[id] : string.Empty;

    /// <summary>读取 <c>tokens.txt</c>。兼容两种行格式：<c>"token id"</c> 与 <c>"id token"</c>。</summary>
    public static TokenTable Load(string path)
    {
        string[] lines = File.ReadAllLines(path);
        var tokens = new List<string>(lines.Length);

        foreach (string raw in lines)
        {
            string line = raw.Trim();
            if (line.Length == 0)
                continue;

            string token;
            int id;

            int split = line.LastIndexOf(' ');
            if (split > 0 && int.TryParse(line.AsSpan(split + 1), NumberStyles.Integer,
                    CultureInfo.InvariantCulture, out int tail))
            {
                // "<token> <id>"
                token = line[..split];
                id = tail;
            }
            else
            {
                // "<id> <token>"
                int head = line.IndexOf(' ');
                if (head <= 0 || !int.TryParse(line.AsSpan(0, head), NumberStyles.Integer,
                        CultureInfo.InvariantCulture, out id))
                {
                    throw new OnnxSharpException($"{path}: cannot parse tokens line '{raw}'");
                }

                token = line[(head + 1)..];
            }

            while (tokens.Count <= id)
                tokens.Add(string.Empty);

            tokens[id] = token;
        }

        return new TokenTable(tokens.ToArray());
    }

    /// <summary>argmax 序列 → token id（去 blank、折叠重复、遇 EOS 停）。</summary>
    public List<int> DecodeIds(ReadOnlySpan<long> argMax, int limit = int.MaxValue)
    {
        var result = new List<int>(argMax.Length);
        int previous = -1;

        int count = Math.Min(argMax.Length, limit);
        for (int i = 0; i < count; i++)
        {
            int token = (int)argMax[i];
            if (token == BlankId)
                continue;
            if (token == previous)
                continue;
            if (token == EosId)
                break;

            previous = token;
            result.Add(token);
        }

        return result;
    }

    /// <summary>token id → 文本（直接拼接，不做分词/标点处理）。</summary>
    public string ToText(IReadOnlyList<int> ids)
    {
        var builder = new System.Text.StringBuilder(ids.Count * 2);
        foreach (int id in ids)
            builder.Append(this[id]);

        return builder.ToString();
    }
}
