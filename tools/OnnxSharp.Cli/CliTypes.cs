// CLI 用的小类型。放在独立文件里：Program.cs 是顶层语句文件，
// 类型声明必须位于所有语句之后，混在中间会编译不过。
//
// 注意：YOLO 的 Detection / NMS / letterbox 现在在库里的 OnnxSharp.Vision，
// 不再是 CLI 的私有示例 —— CLI 只是它的一个调用方。

/// <summary>极简命令行参数容器。</summary>
internal sealed class CliOptions
{
    private readonly Dictionary<string, string> _values = new(StringComparer.Ordinal);
    private readonly List<(string Key, string Value)> _entries = new();

    public List<string> Inputs { get; } = new();

    /// <summary>
    /// 所有 <c>--shape</c> 取值。可以是 <c>1,3,640,640</c>（单输入模型），
    /// 也可以是 <c>name=1,256,288,288</c>（多输入模型，如 SAM3 的 decoder）。
    /// 同一参数可重复出现，按输入名各自生效。
    /// </summary>
    public List<string> Shapes { get; } = new();

    public string? Compare { get; set; }

    public void Add(string key, string value)
    {
        _values[key] = value;
        _entries.Add((key, value));
    }

    /// <summary>同一个开关出现多次时按顺序返回（如 SAM3 的 <c>--box</c>）。</summary>
    public IEnumerable<string> Values(string key)
    {
        foreach (var entry in _entries)
        {
            if (string.Equals(entry.Key, key, StringComparison.Ordinal))
                yield return entry.Value;
        }
    }

    /// <summary>是否给了某个开关（<c>--flag</c> 这种不带值的写法也算）。</summary>
    public bool Has(string key) => _values.ContainsKey(key);

    public int GetInt(string key, int fallback)
        => _values.TryGetValue(key, out var text) && text.Length > 0 ? int.Parse(text) : fallback;

    public float GetFloat(string key, float fallback)
        => _values.TryGetValue(key, out var text) && text.Length > 0 ? float.Parse(text) : fallback;

    public string? GetString(string key)
        => _values.TryGetValue(key, out var text) ? text : null;
}
