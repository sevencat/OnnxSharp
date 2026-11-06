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

    /// <summary>存储布局（首版恒为 NCHW，仅预留）。</summary>
    public Layout Layout = Layout.NCHW;

    /// <summary>仅 <see cref="ArgKind.Const"/> 有效：常量张量。</summary>
    public Tensor? Const;

    /// <summary>运行期张量缓冲（跨 forward 复用）。</summary>
    public Tensor? Runtime;

    /// <summary>
    /// 共享缓冲槽编号（-1 表示无槽，例如常量、图输入或未被写入的 Arg）。
    /// 由 <see cref="Engine.BufferAllocator"/> 按生命周期区间着色分配；
    /// 生命周期不重叠的多个 Arg 会共用同一个槽（也就共用一个 <see cref="Tensor"/> 对象）。
    /// </summary>
    public int BufferIndex = -1;

    public override string ToString() => $"{Name}({Kind}, {DType}, {Shape})";
}
