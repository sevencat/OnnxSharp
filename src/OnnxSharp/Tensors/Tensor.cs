using System.Collections.Concurrent;
using System.Runtime.InteropServices;
using System.Threading;

namespace OnnxSharp.Tensors;

/// <summary>
/// 稠密多维张量（行主序）。等价于 OpenCV 的 <c>cv::Mat</c>，但只维护
/// “紧凑布局 + 字节偏移”两种形态，支持内存复用（<see cref="Fit"/>）。
/// </summary>
public sealed class Tensor
{
    public TensorShape Shape { get; private set; } = TensorShape.Scalar;

    public DType DType { get; private set; } = DType.F32;

    /// <summary>
    /// 存储布局。首版恒为 <see cref="Tensors.Layout.NCHW"/>；
    /// 保留该属性是为了将来图级 blocked layout 不必改 API。
    /// </summary>
    public Layout Layout { get; private set; } = Layout.NCHW;

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

    /// <summary>诊断：<see cref="Fit"/> 触发缓冲重分配的次数与总字节数。</summary>
    internal static long FitReallocCount;

    /// <summary>诊断：<see cref="Fit"/> 触发缓冲重分配的总字节数。</summary>
    internal static long FitReallocBytes;

    /// <summary>诊断：重分配现场（形状/需求/现有容量），用于定位「为什么又要分配」。</summary>
    internal static readonly ConcurrentQueue<string> ReallocLog = new();

    /// <summary>
    /// 重新绑定形状/类型。缓冲足够时复用，否则重新分配（对齐 OpenCV <c>Mat::fit</c> 语义）。
    /// </summary>
    public void Fit(TensorShape shape, DType dtype, bool reuse = true)
    {
        long need = checked(shape.Total() * dtype.SizeOf());
        if (need > int.MaxValue)
            throw new NotSupportedException(
                $"tensor too large: {shape} {dtype} needs {need} bytes, max {int.MaxValue}");

        // 只在「显式要求重分配」或「容量不够」时重建缓冲。
        // dtype 变化**不算**重建理由：Buffer 是 byte[]，dtype 只是解释视图，长度够就能直接复用。
        // 这条曾经是个真 bug —— 共享槽位会被 F32 大张量与 I64 小张量轮流使用，
        // 每次 dtype 切换都 new byte[need]（精确大小），小张量把大缓冲缩成几百字节，
        // 下一帧大张量只好重新分配。rfdetr-small 因此每帧白分配 53 MB、触发 19 次重分配。
        if (!reuse || Buffer.Length < need)
        {
            // 诊断计数器：区分「缓冲被复用」和「每帧重新分配」。跨线程累加用 Interlocked，
            // 只有诊断时读（Net/Tensor 不暴露公开 API，CLI 直接读 internal 字段）。
            Interlocked.Increment(ref FitReallocCount);
            Interlocked.Add(ref FitReallocBytes, need);
            if (ReallocLog.Count < 24)
                ReallocLog.Enqueue(
                    $"#{System.Runtime.CompilerServices.RuntimeHelpers.GetHashCode(this):x8} " +
                    $"old=[{string.Join(",", Shape.Dims.ToArray())}] need={need} have={Buffer.Length} " +
                    $"reuse={reuse} dtype={DType}->{dtype}");
            Buffer = need == 0 ? Array.Empty<byte>() : new byte[need];
        }

        DType = dtype;

        Layout = Layout.NCHW;
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
