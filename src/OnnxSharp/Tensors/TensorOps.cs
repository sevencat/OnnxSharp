using OnnxSharp.Core;

namespace OnnxSharp.Tensors;

public enum BinaryOpKind
{
    Add,
    Sub,
    Mul,
    Div,
    Max,
    Min,
}

public enum UnaryOpKind
{
    Relu,
    Sigmoid,
    Sqrt,
    Exp,
    Neg,
    Abs,
}

/// <summary>
/// 基础逐元素运算（首版仅 F32 + ONNX 右对齐广播）。
/// 后续在 M4 增加无广播时的单循环快速路径与 SIMD 版本。
/// </summary>
public static class TensorOps
{
    public static TensorShape BroadcastShape(TensorShape a, TensorShape b)
    {
        int rank = Math.Max(a.Rank, b.Rank);
        var dims = new int[rank];
        for (int i = 0; i < rank; i++)
        {
            int da = DimAt(a, i, rank);
            int db = DimAt(b, i, rank);
            if (da != db && da != 1 && db != 1)
                throw new OnnxSharpException($"cannot broadcast {a} with {b}");
            dims[i] = da == 1 ? db : da;
        }
        return new TensorShape(dims);
    }

    private static int DimAt(TensorShape shape, int i, int outRank)
    {
        int offset = outRank - shape.Rank;
        return i < offset ? 1 : shape[i - offset];
    }

    /// <summary>c = op(a, b)，要求 c 的形状是 a/b 的广播结果。</summary>
    public static void ElementWiseBinary(BinaryOpKind op, Tensor a, Tensor b, Tensor c)
    {
        if (a.DType != DType.F32 || b.DType != DType.F32 || c.DType != DType.F32)
            throw new NotSupportedException($"element-wise kernel supports F32 only, got {a.DType}/{b.DType}/{c.DType}");

        var sa = a.AsSpan<float>();
        var sb = b.AsSpan<float>();
        var sc = c.AsSpan<float>();

        int rank = c.Shape.Rank;
        var dims = c.Shape.ToArray();
        var strideA = ComputeStrides(a.Shape, rank);
        var strideB = ComputeStrides(b.Shape, rank);
        var index = new int[rank];

        if (a.Shape.Equals(c.Shape) && b.Shape.Equals(c.Shape))
        {
            // 无广播快速路径
            for (int i = 0; i < sc.Length; i++)
                sc[i] = Apply(op, sa[i], sb[i]);
            return;
        }

        int ia = 0, ib = 0;
        for (int n = 0; n < sc.Length; n++)
        {
            sc[n] = Apply(op, sa[ia], sb[ib]);

            for (int axis = rank - 1; axis >= 0; axis--)
            {
                index[axis]++;
                ia += strideA[axis];
                ib += strideB[axis];
                if (index[axis] < dims[axis])
                    break;
                ia -= strideA[axis] * dims[axis];
                ib -= strideB[axis] * dims[axis];
                index[axis] = 0;
            }
        }
    }

    public static void ElementWiseUnary(UnaryOpKind op, Tensor a, Tensor c)
    {
        if (a.DType != DType.F32 || c.DType != DType.F32)
            throw new NotSupportedException("unary kernel supports F32 only");

        var sa = a.AsSpan<float>();
        var sc = c.AsSpan<float>();
        if (sa.Length != sc.Length)
            throw new OnnxSharpException($"unary op size mismatch: {sa.Length} vs {sc.Length}");

        for (int i = 0; i < sc.Length; i++)
            sc[i] = Apply(op, sa[i]);
    }

    private static int[] ComputeStrides(TensorShape shape, int outRank)
    {
        var strides = new int[outRank];
        int offset = outRank - shape.Rank;
        int acc = 1;
        for (int i = shape.Rank - 1; i >= 0; i--)
        {
            int dim = shape[i];
            strides[i + offset] = dim == 1 ? 0 : acc;
            acc *= dim;
        }
        return strides;
    }

    private static float Apply(BinaryOpKind op, float a, float b) => op switch
    {
        BinaryOpKind.Add => a + b,
        BinaryOpKind.Sub => a - b,
        BinaryOpKind.Mul => a * b,
        BinaryOpKind.Div => a / b,
        BinaryOpKind.Max => MathF.Max(a, b),
        BinaryOpKind.Min => MathF.Min(a, b),
        _ => throw new ArgumentOutOfRangeException(nameof(op), op, null),
    };

    private static float Apply(UnaryOpKind op, float a) => op switch
    {
        UnaryOpKind.Relu => a > 0f ? a : 0f,
        UnaryOpKind.Sigmoid => 1f / (1f + MathF.Exp(-a)),
        UnaryOpKind.Sqrt => MathF.Sqrt(a),
        UnaryOpKind.Exp => MathF.Exp(a),
        UnaryOpKind.Neg => -a,
        UnaryOpKind.Abs => MathF.Abs(a),
        _ => throw new ArgumentOutOfRangeException(nameof(op), op, null),
    };
}
