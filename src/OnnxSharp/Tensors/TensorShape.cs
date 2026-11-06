namespace OnnxSharp.Tensors;

/// <summary>
/// 多维形状。等价于 OpenCV 的 <c>MatShape</c>。
/// 默认值（未初始化）视为标量 rank=0、元素数 1。
/// </summary>
public readonly struct TensorShape : IEquatable<TensorShape>
{
    private readonly int[]? _dims;

    public TensorShape(params int[] dims) => _dims = dims;

    public static TensorShape Scalar => new(Array.Empty<int>());

    public int Rank => _dims?.Length ?? 0;

    public ReadOnlySpan<int> Dims => _dims ?? Array.Empty<int>();

    public int this[int axis]
    {
        get
        {
            var d = _dims ?? Array.Empty<int>();
            if ((uint)axis >= (uint)d.Length)
                throw new ArgumentOutOfRangeException(nameof(axis));
            return d[axis];
        }
    }

    public bool IsScalar => Rank == 0;

    /// <summary>元素总数。</summary>
    public long Total()
    {
        var d = _dims;
        if (d is null || d.Length == 0)
            return 1;
        long total = 1;
        for (int i = 0; i < d.Length; i++)
        {
            if (d[i] < 0)
                throw new InvalidOperationException($"negative dimension in {this}");
            total *= d[i];
        }
        return total;
    }

    /// <summary>返回替换指定轴后的新形状。</summary>
    public TensorShape With(int axis, int value)
    {
        var d = (_dims ?? Array.Empty<int>()).ToArray();
        d[axis] = value;
        return new TensorShape(d);
    }

    public int[] ToArray() => (_dims ?? Array.Empty<int>()).ToArray();

    public bool Equals(TensorShape other)
    {
        var a = Dims;
        var b = other.Dims;
        if (a.Length != b.Length)
            return false;
        return a.SequenceEqual(b);
    }

    public override bool Equals(object? obj) => obj is TensorShape s && Equals(s);

    public override int GetHashCode()
    {
        var h = new HashCode();
        foreach (var d in Dims)
            h.Add(d);
        return h.ToHashCode();
    }

    public static bool operator ==(TensorShape a, TensorShape b) => a.Equals(b);

    public static bool operator !=(TensorShape a, TensorShape b) => !a.Equals(b);

    public override string ToString() => "[" + string.Join(",", Dims.ToArray()) + "]";
}
