using OnnxSharp.Tensors;

namespace OnnxSharp.Core;

public enum DictValueKind
{
    None = 0,
    Int,
    Float,
    String,
    Ints,
    Floats,
    Strings,
    Tensor,
}

/// <summary>ONNX 属性值 / 层参数值（对应 OpenCV <c>DictValue</c>）。</summary>
public sealed class DictValue
{
    private readonly long _i;
    private readonly double _f;
    private readonly string? _s;
    private readonly long[]? _ints;
    private readonly double[]? _floats;
    private readonly string[]? _strings;

    private DictValue(DictValueKind kind, long i, double f, string? s,
                      long[]? ints, double[]? floats, string[]? strings, Tensor? tensor)
    {
        Kind = kind;
        _i = i;
        _f = f;
        _s = s;
        _ints = ints;
        _floats = floats;
        _strings = strings;
        Tensor = tensor;
    }

    public DictValueKind Kind { get; }

    public Tensor? Tensor { get; }

    public static DictValue FromInt(long v) => new(DictValueKind.Int, v, 0, null, null, null, null, null);

    public static DictValue FromFloat(double v) => new(DictValueKind.Float, 0, v, null, null, null, null, null);

    public static DictValue FromString(string v) => new(DictValueKind.String, 0, 0, v, null, null, null, null);

    public static DictValue FromInts(long[] v) => new(DictValueKind.Ints, 0, 0, null, v, null, null, null);

    public static DictValue FromFloats(double[] v) => new(DictValueKind.Floats, 0, 0, null, null, v, null, null);

    public static DictValue FromStrings(string[] v) => new(DictValueKind.Strings, 0, 0, null, null, null, v, null);

    public static DictValue FromTensor(Tensor v) => new(DictValueKind.Tensor, 0, 0, null, null, null, null, v);

    public int Count => Kind switch
    {
        DictValueKind.Int or DictValueKind.Float or DictValueKind.String => 1,
        DictValueKind.Ints => _ints!.Length,
        DictValueKind.Floats => _floats!.Length,
        DictValueKind.Strings => _strings!.Length,
        DictValueKind.Tensor => (int)(Tensor?.Count ?? 0),
        _ => 0,
    };

    public long AsInt(int index = 0) => Kind switch
    {
        DictValueKind.Int => _i,
        DictValueKind.Ints => _ints![index],
        DictValueKind.Float => (long)_f,
        DictValueKind.Floats => (long)_floats![index],
        _ => throw new InvalidCastException($"DictValue({Kind}) is not an integer"),
    };

    public double AsFloat(int index = 0) => Kind switch
    {
        DictValueKind.Float => _f,
        DictValueKind.Floats => _floats![index],
        DictValueKind.Int => _i,
        DictValueKind.Ints => _ints![index],
        _ => throw new InvalidCastException($"DictValue({Kind}) is not a number"),
    };

    public string AsString(int index = 0) => Kind switch
    {
        DictValueKind.String => _s!,
        DictValueKind.Strings => _strings![index],
        _ => throw new InvalidCastException($"DictValue({Kind}) is not a string"),
    };

    public long[] AsInts() => Kind switch
    {
        DictValueKind.Ints => _ints!,
        DictValueKind.Int => new[] { _i },
        _ => throw new InvalidCastException($"DictValue({Kind}) is not an integer array"),
    };

    public double[] AsFloats() => Kind switch
    {
        DictValueKind.Floats => _floats!,
        DictValueKind.Float => new[] { _f },
        _ => throw new InvalidCastException($"DictValue({Kind}) is not a float array"),
    };

    public override string ToString() => Kind switch
    {
        DictValueKind.Int => _i.ToString(System.Globalization.CultureInfo.InvariantCulture),
        DictValueKind.Float => _f.ToString("G9", System.Globalization.CultureInfo.InvariantCulture),
        DictValueKind.String => _s!,
        DictValueKind.Ints => "[" + string.Join(",", _ints!) + "]",
        DictValueKind.Floats => "[" + string.Join(",", _floats!) + "]",
        DictValueKind.Strings => "[" + string.Join(",", _strings!) + "]",
        DictValueKind.Tensor => Tensor!.ToString(),
        _ => "<none>",
    };
}
