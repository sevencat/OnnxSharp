using OnnxSharp.Core;
using OnnxSharp.Tensors;

namespace OnnxSharp.Ops;

/// <summary>算子实现共用的形状/常量读取工具。</summary>
internal static class OpUtil
{
    /// <summary>把常量张量读成 int[]（支持 I64 / I32 / F32，OpenCV 三种都接受）。</summary>
    public static int[] ReadInts(Tensor tensor)
    {
        int count = checked((int)tensor.Count);
        var result = new int[count];

        if (tensor.DType == DType.I64)
        {
            var span = tensor.AsSpan<long>();
            for (int i = 0; i < count; i++)
                result[i] = checked((int)span[i]);
        }
        else if (tensor.DType == DType.I32)
        {
            tensor.AsSpan<int>().CopyTo(result);
        }
        else if (tensor.DType == DType.F32)
        {
            var span = tensor.AsSpan<float>();
            for (int i = 0; i < count; i++)
                result[i] = (int)span[i];
        }
        else
        {
            throw new OnnxSharpException($"cannot read integers from {tensor.DType} tensor");
        }

        return result;
    }

    public static float[] ReadFloats(Tensor tensor)
    {
        if (tensor.DType != DType.F32)
            throw new OnnxSharpException($"cannot read floats from {tensor.DType} tensor");
        return tensor.AsSpan<float>().ToArray();
    }

    /// <summary>按 rank 归一化负轴，并校验范围。</summary>
    public static int NormalizeAxis(long axis, int rank, string context)
    {
        int a = (int)axis;
        if (a < 0)
            a += rank;
        if ((uint)a >= (uint)rank)
            throw new OnnxSharpException($"{context}: axis {axis} out of range for rank {rank}");
        return a;
    }

    public static int Product(ReadOnlySpan<int> dims)
    {
        int product = 1;
        foreach (var d in dims)
            product = checked(product * d);
        return product;
    }

    /// <summary>行主序步长（单位：元素）。</summary>
    public static int[] Strides(ReadOnlySpan<int> dims)
    {
        var strides = new int[dims.Length];
        int running = 1;
        for (int i = dims.Length - 1; i >= 0; i--)
        {
            strides[i] = running;
            running = checked(running * dims[i]);
        }
        return strides;
    }

    /// <summary>把左右两端尺寸补齐到同一 rank（左补 1），用于广播对齐。</summary>
    public static int[] AlignLeft(ReadOnlySpan<int> dims, int rank)
    {
        var result = new int[rank];
        int offset = rank - dims.Length;
        for (int i = 0; i < dims.Length; i++)
            result[offset + i] = dims[i];
        for (int i = 0; i < offset; i++)
            result[i] = 1;
        return result;
    }

    /// <summary>取第 <paramref name="index"/> 个输入常量；不是常量则抛异常并指出算子。</summary>
    public static Tensor RequireConstInput(LayerInfo op, int index)
    {
        if (!op.IsConstInput(index))
            throw new OnnxSharpException(
                $"op '{op.Name}' ({op.Type}): input #{index} must be a constant, but it is computed at runtime " +
                "(needs constant folding)");
        return op.In(index);
    }

    /// <summary>按元素字节宽度整块拷贝。</summary>
    public static void CopyBytes(Tensor src, long srcElementOffset, Tensor dst, long dstElementOffset, long elements)
    {
        int elementSize = src.DType.SizeOf();
        var srcBytes = src.Buffer.AsSpan(src.ByteOffset + checked((int)(srcElementOffset * elementSize)),
            checked((int)(elements * elementSize)));
        var dstBytes = dst.Buffer.AsSpan(dst.ByteOffset + checked((int)(dstElementOffset * elementSize)),
            checked((int)(elements * elementSize)));
        srcBytes.CopyTo(dstBytes);
    }
}
