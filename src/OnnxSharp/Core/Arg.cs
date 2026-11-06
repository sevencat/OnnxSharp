using OnnxSharp.Tensors;

namespace OnnxSharp.Core;

public enum ArgKind
{
    Empty = 0,
    Const = 1,
    Input = 2,
    Output = 3,
    Temp = 4,
}

/// <summary>SSA 值编号。0 表示“空参数”。</summary>
public readonly struct Arg : IEquatable<Arg>
{
    public Arg(int idx) => Idx = idx;

    public int Idx { get; }

    public bool IsEmpty => Idx <= 0;

    public bool Equals(Arg other) => Idx == other.Idx;

    public override bool Equals(object? obj) => obj is Arg a && Equals(a);

    public override int GetHashCode() => Idx;

    public static bool operator ==(Arg a, Arg b) => a.Idx == b.Idx;

    public static bool operator !=(Arg a, Arg b) => a.Idx != b.Idx;

    public override string ToString() => $"arg#{Idx}";
}

/// <summary>Arg 的元信息（对应 OpenCV <c>ArgData</c>）。</summary>
public sealed class ArgData
{
    public string Name = string.Empty;

    public ArgKind Kind = ArgKind.Empty;

    public TensorShape Shape = TensorShape.Scalar;

    public DType DType = DType.F32;

    /// <summary>仅 <see cref="ArgKind.Const"/> 有效：常量张量。</summary>
    public Tensor? Const;

    /// <summary>运行期张量缓冲（跨 forward 复用）。</summary>
    public Tensor? Runtime;

    public override string ToString() => $"{Name}({Kind}, {DType}, {Shape})";
}
