// CLI 用的小类型。放在独立文件里：Program.cs 是顶层语句文件，
// 类型声明必须位于所有语句之后，混在中间会编译不过。

/// <summary>YOLO 检测框（decode + NMS 用，仅 CLI 示例）。</summary>
internal sealed class Detection
{
    public int ClassId;
    public float Score;
    public float X, Y, W, H;

    public float Area => W * H;

    public static float IoU(Detection a, Detection b)
    {
        float x1 = MathF.Max(a.X, b.X);
        float y1 = MathF.Max(a.Y, b.Y);
        float x2 = MathF.Min(a.X + a.W, b.X + b.W);
        float y2 = MathF.Min(a.Y + a.H, b.Y + b.H);

        float interW = x2 - x1;
        float interH = y2 - y1;
        if (interW <= 0 || interH <= 0)
            return 0;

        float intersection = interW * interH;
        float union = a.Area + b.Area - intersection;
        return union <= 0 ? 0 : intersection / union;
    }
}

/// <summary>极简命令行参数容器。</summary>
internal sealed class CliOptions
{
    private readonly Dictionary<string, string> _values = new(StringComparer.Ordinal);

    public List<string> Inputs { get; } = new();

    public string? Compare { get; set; }

    public void Add(string key, string value) => _values[key] = value;

    public int GetInt(string key, int fallback)
        => _values.TryGetValue(key, out var text) ? int.Parse(text) : fallback;

    public float GetFloat(string key, float fallback)
        => _values.TryGetValue(key, out var text) ? float.Parse(text) : fallback;

    public string? GetString(string key)
        => _values.TryGetValue(key, out var text) ? text : null;
}
