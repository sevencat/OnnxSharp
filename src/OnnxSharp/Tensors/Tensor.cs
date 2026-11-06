using System.Runtime.InteropServices;

namespace OnnxSharp.Tensors;

/// <summary>
/// 稠密多维张量（行主序）。等价于 OpenCV 的 <c>cv::Mat</c>，但只维护
/// “紧凑布局 + 字节偏移”两种形态，支持内存复用（<see cref="Fit"/>）。
/// </summary>
public sealed class Tensor
{
    public TensorShape Shape { get; private set; } = TensorShape.Scalar;

    public DType DType { get; private set; } = DType.F32;

    /// <summary>底层缓冲（可能大于实际使用长度，供复用）。</summary>
    public byte[] Buffer { get; private set; } = Array.Empty<byte>();

    public int ByteOffset { get; private set; }

    public int ByteLength { get; private set; }

    /// <summary>null 表示紧凑布局。</summary>
    public int[]? Strides { get; private set; }

    /// <summary>元素个数。</summary>
    public long Count => Shape.Total();

    public bool IsContiguous => Strides is null;

    public static Tensor Allocate(TensorShape shape, DType dtype)
    {
        var t = new Tensor();
        t.Fit(shape, dtype, reuse: false);
        return t;
    }

    public static Tensor FromArray(ReadOnlySpan<float> data, TensorShape? shape = null)
    {
        var t = new Tensor();
        t.Fit(shape ?? new TensorShape(data.Length), DType.F32, reuse: false);
        data.CopyTo(t.AsSpan<float>());
        return t;
    }

    public static Tensor FromInt64(ReadOnlySpan<long> data, TensorShape? shape = null)
    {
        var t = new Tensor();
        t.Fit(shape ?? new TensorShape(data.Length), DType.I64, reuse: false);
        data.CopyTo(t.AsSpan<long>());
        return t;
    }

    /// <summary>
    /// 重新绑定形状/类型。缓冲足够时复用，否则重新分配（对齐 OpenCV <c>Mat::fit</c> 语义）。
    /// </summary>
    public void Fit(TensorShape shape, DType dtype, bool reuse = true)
    {
        long need = checked(shape.Total() * dtype.SizeOf());
        if (need > int.MaxValue)
            throw new NotSupportedException($"tensor too large: {shape}");

        if (!reuse || DType != dtype || Buffer.Length < need)
        {
            Buffer = need == 0 ? Array.Empty<byte>() : new byte[need];
            DType = dtype;
        }

        Shape = shape;
        ByteOffset = 0;
        ByteLength = (int)need;
        Strides = null;
    }

    /// <summary>以指定元素类型访问数据（要求紧凑布局）。</summary>
    public Span<T> AsSpan<T>() where T : struct
    {
        if (!IsContiguous)
            throw new NotSupportedException("non-contiguous tensor");
        return MemoryMarshal.Cast<byte, T>(Buffer.AsSpan(ByteOffset, ByteLength));
    }

    public ReadOnlySpan<T> AsReadOnlySpan<T>() where T : struct
    {
        if (!IsContiguous)
            throw new NotSupportedException("non-contiguous tensor");
        return MemoryMarshal.Cast<byte, T>(Buffer.AsSpan(ByteOffset, ByteLength));
    }

    public Tensor Clone()
    {
        var t = Allocate(Shape, DType);
        Buffer.AsSpan(ByteOffset, ByteLength).CopyTo(t.Buffer);
        return t;
    }

    public override string ToString() => $"Tensor({DType} {Shape})";
}
