using OnnxSharp.Core;
using OnnxSharp.Tensors;

namespace OnnxSharp.Ops;

/// <summary>算子实现共用的形状/常量读取工具。</summary>
internal static class OpUtil
{
    /// <summary>
    /// 把常量张量读成 int[]（支持 I64 / I32 / F32，OpenCV 三种都接受）。
    /// <para>
    /// 越界值按 int 范围饱和：ONNX 的 <c>Slice</c> 用 <c>INT64_MAX</c> / <c>INT64_MIN</c>
    /// 作为「到边界」的哨兵值，读取时会溢出，饱和后由各算子自行按维度裁剪。
    /// </para>
    /// </summary>
    public static int[] ReadInts(Tensor tensor)
    {
        int count = checked((int)tensor.Count);
        var result = new int[count];

        if (tensor.DType == DType.I64)
        {
            var span = tensor.AsSpan<long>();
            for (int i = 0; i < count; i++)
                result[i] = span[i] > int.MaxValue ? int.MaxValue
                          : span[i] < int.MinValue ? int.MinValue
                          : (int)span[i];
        }
        else if (tensor.DType == DType.I32)
        {
            tensor.AsSpan<int>().CopyTo(result);
        }
        else if (tensor.DType == DType.U8 || tensor.DType == DType.Bool)
        {
            // 量化算子的 zero point 与量化数据类型一致（常见是 uint8）
            var span = tensor.AsSpan<byte>();
            for (int i = 0; i < count; i++)
                result[i] = span[i];
        }
        else if (tensor.DType == DType.I8)
        {
            var span = tensor.AsSpan<sbyte>();
            for (int i = 0; i < count; i++)
                result[i] = span[i];
        }
        else if (tensor.DType == DType.F32)
        {
            var span = tensor.AsSpan<float>();
            for (int i = 0; i < count; i++)
                result[i] = span[i] >= int.MaxValue ? int.MaxValue
                          : span[i] <= int.MinValue ? int.MinValue
                          : (int)span[i];
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

    /// <summary>
    /// 读取 axes：opset 13+ 优先第 1 个「常量」输入，老版本用 attribute。
    /// 返回<b>未归一化</b>的原始值（负数轴由调用方按各自的 rank 归一化：
    /// <c>Squeeze</c> 按输入 rank、<c>Unsqueeze</c> 按输出 rank）。
    /// </summary>
    public static int[] Axes(LayerInfo op, int[]? attrAxes)
    {
        if (op.Inputs.Count > 1 && op.Inputs[1].Idx != 0)
            return ReadInts(RequireConstInput(op, 1));
        return attrAxes ?? Array.Empty<int>();
    }

    /// <summary>归一化（负轴 +rank）、排序、去重。</summary>
    public static int[] NormalizeAxes(ReadOnlySpan<int> axes, int rank, string context)
    {
        if (axes.Length == 0)
            return Array.Empty<int>();

        var result = new int[axes.Length];
        for (int i = 0; i < axes.Length; i++)
            result[i] = NormalizeAxis(axes[i], rank, context);

        Array.Sort(result);

        int unique = 1;
        for (int i = 1; i < result.Length; i++)
        {
            if (result[i] != result[unique - 1])
                result[unique++] = result[i];
        }

        if (unique != result.Length)
            Array.Resize(ref result, unique);

        return result;
    }

    /// <summary>布尔掩码 / 索引张量读成 bool[]（Bool/U8/I64/I32/F32 都接受）。</summary>
    public static bool[] ReadBools(Tensor tensor)
    {
        int count = checked((int)tensor.Count);
        var result = new bool[count];

        if (tensor.DType is DType.Bool or DType.U8)
        {
            var span = tensor.AsSpan<byte>();
            for (int i = 0; i < count; i++)
                result[i] = span[i] != 0;
        }
        else if (tensor.DType == DType.I64)
        {
            var span = tensor.AsSpan<long>();
            for (int i = 0; i < count; i++)
                result[i] = span[i] != 0;
        }
        else if (tensor.DType == DType.I32)
        {
            var span = tensor.AsSpan<int>();
            for (int i = 0; i < count; i++)
                result[i] = span[i] != 0;
        }
        else if (tensor.DType == DType.F32)
        {
            var span = tensor.AsSpan<float>();
            for (int i = 0; i < count; i++)
                result[i] = span[i] != 0f;
        }
        else
        {
            throw new OnnxSharpException($"cannot read booleans from {tensor.DType} tensor");
        }

        return result;
    }

    /// <summary>多输入广播后的输出形状（numpy 右对齐规则）。</summary>
    public static int[] BroadcastDims(IReadOnlyList<TensorShape> shapes)
    {
        int rank = 0;
        foreach (var shape in shapes)
            rank = Math.Max(rank, shape.Rank);

        var dims = new int[rank];
        Array.Fill(dims, 1);

        foreach (var shape in shapes)
        {
            int offset = rank - shape.Rank;
            for (int i = 0; i < shape.Rank; i++)
            {
                int target = dims[offset + i];
                int source = shape[i];

                if (source == target || source == 1)
                    continue;

                if (target != 1)
                    throw new OnnxSharpException(
                        $"cannot broadcast shapes (axis {offset + i}: {target} vs {source})");

                dims[offset + i] = source;
            }
        }

        return dims;
    }

    /// <summary>
    /// 把张量形状对齐到 <paramref name="outDims"/> 后的元素步长；
    /// 广播维（源为 1、目标 &gt; 1）步长为 0，因此可以直接做线性遍历。
    /// </summary>
    public static int[] BroadcastStrides(TensorShape shape, int[] outDims)
    {
        var dims = shape.ToArray();
        var strides = Strides(dims);

        int offset = outDims.Length - dims.Length;
        if (offset < 0)
            throw new OnnxSharpException($"cannot broadcast {shape} to [{string.Join(",", outDims)}]");

        var result = new int[outDims.Length];
        for (int i = 0; i < dims.Length; i++)
        {
            int target = outDims[offset + i];
            if (dims[i] != 1 && dims[i] != target)
                throw new OnnxSharpException($"cannot broadcast {shape} to [{string.Join(",", outDims)}]");
            result[offset + i] = dims[i] == 1 && target != 1 ? 0 : strides[i];
        }

        return result;
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
