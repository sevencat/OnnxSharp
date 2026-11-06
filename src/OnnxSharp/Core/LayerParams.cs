using OnnxSharp.Tensors;

namespace OnnxSharp.Core;

/// <summary>层/算子参数集合（对应 OpenCV <c>LayerParams</c>）。</summary>
public sealed class LayerParams
{
    private readonly Dictionary<string, DictValue> _dict = new(StringComparer.Ordinal);

    public string Name { get; set; } = string.Empty;

    public string Type { get; set; } = string.Empty;

    public IReadOnlyDictionary<string, DictValue> Values => _dict;

    public bool Has(string key) => _dict.ContainsKey(key);

    public void Set(string key, DictValue value) => _dict[key] = value;

    public void Set(string key, long value) => _dict[key] = DictValue.FromInt(value);

    public void Set(string key, double value) => _dict[key] = DictValue.FromFloat(value);

    public void Set(string key, string value) => _dict[key] = DictValue.FromString(value);

    public void Set(string key, long[] value) => _dict[key] = DictValue.FromInts(value);

    public void Set(string key, int[] value) => _dict[key] = DictValue.FromInts(Array.ConvertAll(value, x => (long)x));

    public void Set(string key, float[] value) => _dict[key] = DictValue.FromFloats(Array.ConvertAll(value, x => (double)x));

    public void Set(string key, Tensor value) => _dict[key] = DictValue.FromTensor(value);

    public DictValue Get(string key) =>
        _dict.TryGetValue(key, out var v)
            ? v
            : throw new KeyNotFoundException($"attribute '{key}' is missing in op '{Type}'");

    public DictValue? TryGet(string key) => _dict.TryGetValue(key, out var v) ? v : null;

    public long GetInt(string key, long defaultValue)
        => _dict.TryGetValue(key, out var v) ? v.AsInt() : defaultValue;

    public double GetFloat(string key, double defaultValue)
        => _dict.TryGetValue(key, out var v) ? v.AsFloat() : defaultValue;

    public string GetString(string key, string defaultValue)
        => _dict.TryGetValue(key, out var v) ? v.AsString() : defaultValue;

    public long[]? GetInts(string key)
        => _dict.TryGetValue(key, out var v) ? v.AsInts() : null;

    public double[]? GetFloats(string key)
        => _dict.TryGetValue(key, out var v) ? v.AsFloats() : null;

    public override string ToString() => $"{Type}({Name})[{_dict.Count} attrs]";
}
