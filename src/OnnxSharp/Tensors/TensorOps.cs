using System.Numerics;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using OnnxSharp.Core;
using OnnxSharp.Threading;

namespace OnnxSharp.Tensors;

public enum BinaryOpKind
{
    Add,
    Sub,
    Mul,
    Div,
    Max,
    Min,

    /// <summary>a 的 b 次幂（RT-DETR 的位置编码里用）。</summary>
    Pow,

    /// <summary>取模，按 ONNX 默认的 fmod=0（floor 语义）。</summary>
    Mod,
}

public enum UnaryOpKind
{
    Relu,
    Sigmoid,
    Sqrt,
    Exp,
    Neg,
    Abs,
    Erf,
    Sin,
    Cos,
    Floor,
    Ceil,
    Log,
    Tanh,
    Reciprocal,
    Sign,
    Round,

    /// <summary>
    /// <c>x · sigmoid(x)</c>。ONNX 里 SiLU（Swish）总是导出成 <c>Sigmoid → Mul</c> 两个节点，
    /// 图级融合后走这一个内核，省掉一整遍读+写。
    /// </summary>
    Silu,
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

    /// <summary>c = op(a, b)，要求 c 的形状是 a/b 的广播结果。支持 F32 与 I64（后者供形状运算/常量折叠）。</summary>
    public static void ElementWiseBinary(BinaryOpKind op, Tensor a, Tensor b, Tensor c)
    {
        if (a.DType == DType.I64 && b.DType == DType.I64 && c.DType == DType.I64)
        {
            ElementWiseBinaryInt64(op, a, b, c);
            return;
        }

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

        // Pow 的指数是标量、且为常见值时改写成更便宜的算子：
        // LayerNorm 的 Pow(x, 2) 在 ASR 图里有 81 个（每个 97×320），逐 lane 调 MathF.Pow
        // 要 ~14 ns/元素，改写成 Mul 后走向量化路径（~0.5 ns）。
        if (op == BinaryOpKind.Pow && b.Count == 1 && a.Shape.Equals(c.Shape))
        {
            float exponent = b.AsSpan<float>()[0];
            if (exponent == 2f)
            {
                ElementWiseBinary(BinaryOpKind.Mul, a, a, c);
                return;
            }

            if (exponent == 1f)
            {
                a.Buffer.AsSpan(a.ByteOffset, a.ByteLength).CopyTo(c.Buffer.AsSpan(c.ByteOffset, c.ByteLength));
                return;
            }

            if (exponent == 0.5f)
            {
                ElementWiseUnary(UnaryOpKind.Sqrt, a, c);
                return;
            }
        }

        if (a.Shape.Equals(c.Shape) && b.Shape.Equals(c.Shape))
        {
            // 无广播快速路径
            ApplyContiguous(op, sa, sb, sc);
            return;
        }

        // 标量广播（GELU 的 x*0.5、x/√2 都是）：单独一条路径，否则会退化成
        // 「每 1 个元素调用一次 ApplyContiguous」，比不优化还慢。
        if (a.Shape.Equals(c.Shape) && b.Count == 1)
        {
            ApplyScalar(op, sa, sb[0], sc, scalarIsLeft: false);
            return;
        }

        if (b.Shape.Equals(c.Shape) && a.Count == 1)
        {
            ApplyScalar(op, sb, sa[0], sc, scalarIsLeft: true);
            return;
        }

        // 「尾部广播」快速路径：transformer 里的 bias-add / scale-mul 都是这种形状
        // （如 [1,1024,384] + [384]），此时广播方的索引就是 i % inner，可以整段向量化。
        if (a.Shape.Equals(c.Shape) && IsTailBroadcast(b.Shape, c.Shape, sc.Length, out int innerB))
        {
            ApplyBroadcast(op, sa, sb, sc, innerB, leftIsFull: true);
            return;
        }

        if (b.Shape.Equals(c.Shape) && IsTailBroadcast(a.Shape, c.Shape, sc.Length, out int innerA))
        {
            // 参数顺序固定为 (全长, 重复)，再靠 leftIsFull 决定谁在 operator 左边
            ApplyBroadcast(op, sb, sa, sc, innerA, leftIsFull: false);
            return;
        }

        // 「每段一个标量」广播：LayerNorm 的 x - mean[1,97,1]、掩码相乘 [1,97,1] × [1,97,320] 都是这种。
        // 之前只能落到通用循环（逐元素算 6 个索引，~13 ns/元素）；这里按段广播后在段内走向量化路径。
        if (a.Shape.Equals(c.Shape) && IsInnerConstantBroadcast(b.Shape, c.Shape, out int runB))
        {
            ApplyInnerScalar(op, a.Shape, b.Shape, sa, sb, sc, runB, operandIsLeft: false);
            return;
        }

        if (b.Shape.Equals(c.Shape) && IsInnerConstantBroadcast(a.Shape, c.Shape, out int runA))
        {
            ApplyInnerScalar(op, b.Shape, a.Shape, sb, sa, sc, runA, operandIsLeft: true);
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

    private static void ElementWiseBinaryInt64(BinaryOpKind op, Tensor a, Tensor b, Tensor c)
    {
        var sa = a.AsSpan<long>();
        var sb = b.AsSpan<long>();
        var sc = c.AsSpan<long>();

        int rank = c.Shape.Rank;
        var dims = c.Shape.ToArray();
        var strideA = ComputeStrides(a.Shape, rank);
        var strideB = ComputeStrides(b.Shape, rank);
        var index = new int[rank];

        int ia = 0, ib = 0;
        for (int n = 0; n < sc.Length; n++)
        {
            sc[n] = ApplyInt64(op, sa[ia], sb[ib]);

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

    private static long ApplyInt64(BinaryOpKind op, long a, long b) => op switch
    {
        BinaryOpKind.Add => a + b,
        BinaryOpKind.Sub => a - b,
        BinaryOpKind.Mul => a * b,
        BinaryOpKind.Div => b == 0 ? 0 : a / b,
        BinaryOpKind.Max => Math.Max(a, b),
        BinaryOpKind.Min => Math.Min(a, b),
        BinaryOpKind.Mod => b == 0 ? 0 : a - b * (long)Math.Floor((double)a / b),
        _ => throw new OnnxSharpException($"integer element-wise op '{op}' is not supported"),
    };

    /// <summary>
    /// erf(x) = 2/√π ∫₀ˣ e^(−t²) dt。Abramowitz &amp; Stegun 7.1.26 有理逼近，
    /// 绝对误差 &lt; 1.5e-7 —— 对 F32 已经足够（F32 自身 eps 约 1.2e-7）。
    /// .NET 没有内建 erf（Math/MathF 只有 Gamma 系列），只能自己写。
    /// </summary>
    public static float Erf(float x)
    {
        const float p = 0.3275911f;
        const float a1 = 0.254829592f;
        const float a2 = -0.284496736f;
        const float a3 = 1.421413741f;
        const float a4 = -1.453152027f;
        const float a5 = 1.061405429f;

        float sign = x < 0 ? -1f : 1f;
        float ax = MathF.Abs(x);
        float t = 1f / (1f + p * ax);
        float poly = ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t;

        return sign * (1f - poly * MathF.Exp(-ax * ax));
    }

    /// <summary>低于该元素数时逐元素算子不开线程（调度开销盖过收益）。</summary>
    private const int ParallelElementThreshold = 1 << 16;

    public static unsafe void ElementWiseUnary(UnaryOpKind op, Tensor a, Tensor c)
    {
        if (a.DType != DType.F32 || c.DType != DType.F32)
            throw new NotSupportedException("unary kernel supports F32 only");

        var sa = a.AsSpan<float>();
        var sc = c.AsSpan<float>();
        if (sa.Length != sc.Length)
            throw new OnnxSharpException($"unary op size mismatch: {sa.Length} vs {sc.Length}");

        int n = sc.Length;
        if (n <= 0)
            return;

        // 一次 pin + 按区间并行：Sigmoid/Erf 这类纯算术算子单核已经吃满，
        // 想再快只能横向摊开（yolo 里 78 个 Sigmoid 原本串行占 15%）。
        // 指针本身不能进 lambda（CS1764），转成 IntPtr 再在回调里还原。
        fixed (float* ps = sa)
        fixed (float* pd = sc)
        {
            if (n >= ParallelElementThreshold && ParallelRange.Threads > 1)
            {
                IntPtr pin = (IntPtr)ps;
                IntPtr pout = (IntPtr)pd;
                ParallelRange.For(n, (begin, end) => UnaryRange(op, (float*)pin, (float*)pout, begin, end));
            }
            else
            {
                UnaryRange(op, ps, pd, 0, n);
            }
        }
    }

    /// <summary>
    /// 一元算子的区间内核。<c>switch</c> 在区间外层，内层只有矢量化循环。
    /// </summary>
    private static unsafe void UnaryRange(UnaryOpKind op, float* sa, float* sc, int begin, int end)
    {
        int i = begin;

        switch (op)
        {
            case UnaryOpKind.Relu:
                if (Avx.IsSupported)
                {
                    for (; i <= end - 8; i += 8)
                        Avx.Store(sc + i, Avx.Max(Avx.LoadVector256(sa + i), Vector256<float>.Zero));
                }

                for (; i < end; i++) sc[i] = sa[i] > 0f ? sa[i] : 0f;
                break;

            case UnaryOpKind.Neg:
                if (Avx.IsSupported)
                {
                    var sign = Vector256.Create(-0.0f);
                    for (; i <= end - 8; i += 8)
                        Avx.Store(sc + i, Avx.Xor(Avx.LoadVector256(sa + i), sign));
                }

                for (; i < end; i++) sc[i] = -sa[i];
                break;

            case UnaryOpKind.Abs:
                if (Avx.IsSupported)
                {
                    var sign = Vector256.Create(-0.0f);
                    for (; i <= end - 8; i += 8)
                        Avx.Store(sc + i, Avx.AndNot(sign, Avx.LoadVector256(sa + i)));
                }

                for (; i < end; i++) sc[i] = MathF.Abs(sa[i]);
                break;

            case UnaryOpKind.Sqrt:
                if (Avx.IsSupported)
                {
                    for (; i <= end - 8; i += 8)
                        Avx.Store(sc + i, Avx.Sqrt(Avx.LoadVector256(sa + i)));
                }

                for (; i < end; i++) sc[i] = MathF.Sqrt(sa[i]);
                break;

            case UnaryOpKind.Exp:
                VectorMath.Exp(sa + begin, sc + begin, end - begin);
                break;

            case UnaryOpKind.Sigmoid:
                VectorMath.Sigmoid(sa + begin, sc + begin, end - begin);
                break;

            case UnaryOpKind.Erf:
                VectorMath.Erf(sa + begin, sc + begin, end - begin);
                break;

            case UnaryOpKind.Silu:
                VectorMath.Silu(sa + begin, sc + begin, end - begin);
                break;

            case UnaryOpKind.Sin:
                for (; i < end; i++) sc[i] = MathF.Sin(sa[i]);
                break;

            case UnaryOpKind.Cos:
                for (; i < end; i++) sc[i] = MathF.Cos(sa[i]);
                break;

            default:
                for (; i < end; i++) sc[i] = Apply(op, sa[i]);
                break;
        }
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

    /// <summary>
    /// 无广播的逐元素运算：switch 提到循环外、内层用 <see cref="Vector{T}"/>，
    /// 这样 JIT 才能向量化。把 <c>Apply(op, ...)</c> 放进循环里是之前的性能杀手。
    /// </summary>
    /// <summary>
    /// 二元运算的向量内核。用泛型静态抽象方法做分派：JIT 会为每个 struct 特化出一份循环，
    /// 热路径里只剩下 vaddps / vmulps 加一次 load、一次 store —— 没有 op 的 switch、
    /// 也没有 Span 的逐元素边界检查。原来是 128 位 Vector&lt;float&gt;，带宽只有 AVX2 的一半。
    /// </summary>
    private interface IBinaryKernel
    {
        static abstract Vector256<float> Apply(Vector256<float> a, Vector256<float> b);

        static abstract float Apply(float a, float b);
    }

    private readonly struct AddKernel : IBinaryKernel
    {
        public static Vector256<float> Apply(Vector256<float> a, Vector256<float> b) => Avx.Add(a, b);
        public static float Apply(float a, float b) => a + b;
    }

    private readonly struct SubKernel : IBinaryKernel
    {
        public static Vector256<float> Apply(Vector256<float> a, Vector256<float> b) => Avx.Subtract(a, b);
        public static float Apply(float a, float b) => a - b;
    }

    private readonly struct MulKernel : IBinaryKernel
    {
        public static Vector256<float> Apply(Vector256<float> a, Vector256<float> b) => Avx.Multiply(a, b);
        public static float Apply(float a, float b) => a * b;
    }

    private readonly struct DivKernel : IBinaryKernel
    {
        public static Vector256<float> Apply(Vector256<float> a, Vector256<float> b) => Avx.Divide(a, b);
        public static float Apply(float a, float b) => a / b;
    }

    private readonly struct MaxKernel : IBinaryKernel
    {
        public static Vector256<float> Apply(Vector256<float> a, Vector256<float> b) => Avx.Max(a, b);
        public static float Apply(float a, float b) => MathF.Max(a, b);
    }

    private readonly struct MinKernel : IBinaryKernel
    {
        public static Vector256<float> Apply(Vector256<float> a, Vector256<float> b) => Avx.Min(a, b);
        public static float Apply(float a, float b) => MathF.Min(a, b);
    }

    private static unsafe void ContiguousKernel<T>(
        ReadOnlySpan<float> x,
        ReadOnlySpan<float> y,
        Span<float> c,
        int n)
        where T : struct, IBinaryKernel
    {
        fixed (float* px = x)
        fixed (float* py = y)
        fixed (float* pc = c)
        {
            if (n >= ParallelElementThreshold && ParallelRange.Threads > 1)
            {
                IntPtr p1 = (IntPtr)px;
                IntPtr p2 = (IntPtr)py;
                IntPtr p3 = (IntPtr)pc;
                ParallelRange.For(n, (begin, end) => ContiguousRange<T>((float*)p1, (float*)p2, (float*)p3, begin, end));
            }
            else
            {
                ContiguousRange<T>(px, py, pc, 0, n);
            }
        }
    }

    private static unsafe void ContiguousRange<T>(float* px, float* py, float* pc, int begin, int end)
        where T : struct, IBinaryKernel
    {
        int i = begin;

        if (Avx.IsSupported)
        {
            for (; i <= end - 8; i += 8)
                Avx.Store(pc + i, T.Apply(Avx.LoadVector256(px + i), Avx.LoadVector256(py + i)));
        }

        for (; i < end; i++)
            pc[i] = T.Apply(px[i], py[i]);
    }

    private static void ApplyContiguous(
        BinaryOpKind op,
        ReadOnlySpan<float> left,
        ReadOnlySpan<float> right,
        Span<float> c)
    {
        int n = c.Length;

        switch (op)
        {
            case BinaryOpKind.Add: ContiguousKernel<AddKernel>(left, right, c, n); break;
            case BinaryOpKind.Sub: ContiguousKernel<SubKernel>(left, right, c, n); break;
            case BinaryOpKind.Mul: ContiguousKernel<MulKernel>(left, right, c, n); break;
            case BinaryOpKind.Div: ContiguousKernel<DivKernel>(left, right, c, n); break;
            case BinaryOpKind.Max: ContiguousKernel<MaxKernel>(left, right, c, n); break;
            case BinaryOpKind.Min: ContiguousKernel<MinKernel>(left, right, c, n); break;
            default:
                for (int i = 0; i < n; i++)
                    c[i] = Apply(op, left[i], right[i]);
                break;
        }
    }

    /// <summary>
    /// 标量广播：<c>x op s</c> 或 <c>s op x</c>（<paramref name="scalarIsLeft"/> 决定顺序）。
    /// </summary>
    private static unsafe void ScalarKernel<T>(
        ReadOnlySpan<float> x,
        float s,
        Span<float> c,
        int n,
        bool scalarIsLeft)
        where T : struct, IBinaryKernel
    {
        fixed (float* px = x)
        fixed (float* pc = c)
        {
            int i = 0;

            if (Avx.IsSupported)
            {
                var vs = Vector256.Create(s);
                if (scalarIsLeft)
                {
                    for (; i <= n - 8; i += 8)
                        Avx.Store(pc + i, T.Apply(vs, Avx.LoadVector256(px + i)));
                }
                else
                {
                    for (; i <= n - 8; i += 8)
                        Avx.Store(pc + i, T.Apply(Avx.LoadVector256(px + i), vs));
                }
            }

            if (scalarIsLeft)
            {
                for (; i < n; i++)
                    pc[i] = T.Apply(s, px[i]);
            }
            else
            {
                for (; i < n; i++)
                    pc[i] = T.Apply(px[i], s);
            }
        }
    }

    /// <summary>把 <paramref name="dims"/> 左补 1 到 <paramref name="rank"/>（广播对齐用）。</summary>
    private static int[] AlignLeft(ReadOnlySpan<int> dims, int rank)
    {
        var result = new int[rank];
        int offset = rank - dims.Length;
        for (int i = 0; i < dims.Length; i++)
            result[offset + i] = dims[i];
        for (int i = 0; i < offset; i++)
            result[i] = 1;
        return result;
    }

    /// <summary>
    /// 判断操作数是否「在最内连续段内恒定」，并给出段长。
    /// <para>
    /// <c>[1,97,1]</c> 对 <c>[1,97,320]</c> → 段长 320，每段一个标量（LayerNorm 的减均值、掩码相乘）；
    /// <c>[384]</c> 对 <c>[1,1024,384]</c> 是「段内也要变化」的情形，走 <see cref="ApplyBroadcast"/>。
    /// </para>
    /// </summary>
    private static bool IsInnerConstantBroadcast(TensorShape operand, TensorShape full, out int runLength)
    {
        runLength = 1;
        int rank = full.Rank;
        if (operand.Rank > rank)
            return false;

        var dims = AlignLeft(operand.Dims, rank);
        int axis = rank - 1;

        for (; axis >= 0; axis--)
        {
            if (dims[axis] != 1)
                break;
            runLength *= full.Dims[axis];
        }

        if (runLength <= 1 || full.Total() % runLength != 0)
            return false;

        for (int i = 0; i <= axis; i++)
        {
            if (dims[i] != 1 && dims[i] != full.Dims[i])
                return false;
        }

        return true;
    }

    /// <summary>每段一个标量的广播：段内复用向量化的 <see cref="ApplyScalar"/>。</summary>
    private static void ApplyInnerScalar(
        BinaryOpKind op,
        TensorShape fullShape,
        TensorShape operandShape,
        ReadOnlySpan<float> full,
        ReadOnlySpan<float> operand,
        Span<float> destination,
        int runLength,
        bool operandIsLeft)
    {
        int rank = fullShape.Rank;
        var fullDims = fullShape.ToArray();
        var aligned = AlignLeft(operandShape.Dims, rank);

        var operandStrides = new int[rank];
        int acc = 1;
        for (int axis = rank - 1; axis >= 0; axis--)
        {
            operandStrides[axis] = acc;
            acc *= aligned[axis];
        }

        // 段内恒定 ⇒ 这些轴在外层循环里不需要推进
        var counts = new int[rank];
        for (int axis = 0; axis < rank; axis++)
            counts[axis] = aligned[axis] == 1 && fullDims[axis] > 1 ? 1 : fullDims[axis];

        long offset = 0;
        int operandOffset = 0;
        long total = destination.Length;
        var coords = new int[rank];

        while (offset < total)
        {
            ApplyScalar(
                op,
                full.Slice((int)offset, runLength),
                operand[operandOffset],
                destination.Slice((int)offset, runLength),
                operandIsLeft);

            offset += runLength;

            for (int axis = rank - 1; axis >= 0; axis--)
            {
                if (counts[axis] == 1)
                    continue;

                coords[axis]++;
                operandOffset += operandStrides[axis];
                if (coords[axis] < counts[axis])
                    break;

                operandOffset -= operandStrides[axis] * counts[axis];
                coords[axis] = 0;
            }
        }
    }

    private static void ApplyScalar(
        BinaryOpKind op,
        ReadOnlySpan<float> x,
        float s,
        Span<float> c,
        bool scalarIsLeft)
    {
        int n = c.Length;

        switch (op)
        {
            case BinaryOpKind.Add: ScalarKernel<AddKernel>(x, s, c, n, scalarIsLeft); break;
            case BinaryOpKind.Sub: ScalarKernel<SubKernel>(x, s, c, n, scalarIsLeft); break;
            case BinaryOpKind.Mul: ScalarKernel<MulKernel>(x, s, c, n, scalarIsLeft); break;
            case BinaryOpKind.Div: ScalarKernel<DivKernel>(x, s, c, n, scalarIsLeft); break;
            case BinaryOpKind.Max: ScalarKernel<MaxKernel>(x, s, c, n, scalarIsLeft); break;
            case BinaryOpKind.Min: ScalarKernel<MinKernel>(x, s, c, n, scalarIsLeft); break;
            default:
                for (int i = 0; i < n; i++)
                    c[i] = scalarIsLeft ? Apply(op, s, x[i]) : Apply(op, x[i], s);
                break;
        }
    }

    /// <summary>
    /// 尾部广播：广播方每 <paramref name="inner"/> 个元素重复一次，
    /// 于是可以按 inner 分段，每段内部都是连续访问，照常向量化。
    /// </summary>
    private static void ApplyBroadcast(
        BinaryOpKind op,
        ReadOnlySpan<float> a,
        ReadOnlySpan<float> b,
        Span<float> c,
        int inner,
        bool leftIsFull)
    {
        for (int start = 0; start < c.Length; start += inner)
        {
            int length = Math.Min(inner, c.Length - start);
            var full = a.Slice(start, length);
            var repeat = b.Slice(0, length);
            var dst = c.Slice(start, length);

            if (leftIsFull)
                ApplyContiguous(op, full, repeat, dst);
            else
                ApplyContiguous(op, repeat, full, dst);
        }
    }

    /// <summary>
    /// <paramref name="x"/> 是否是 <paramref name="c"/> 的「尾部后缀」形状
    /// （右对齐后各维相等），此时 x 在 c 里就是按块重复。
    /// </summary>
    private static bool IsTailBroadcast(TensorShape x, TensorShape c, int targetCount, out int count)
    {
        count = 0;

        if (x.Rank > c.Rank)
            return false;

        int offset = c.Rank - x.Rank;
        for (int i = 0; i < x.Rank; i++)
        {
            if (x[i] != c[offset + i])
                return false;
        }

        int total = 1;
        for (int i = 0; i < x.Rank; i++)
            total *= x[i];

        // 等于全长时上面的「同形状」路径已经覆盖，这里只处理真正需要重复的情形
        if (total <= 0 || total >= targetCount)
            return false;

        count = total;
        return true;
    }

    private static float Apply(BinaryOpKind op, float a, float b) => op switch
    {
        BinaryOpKind.Add => a + b,
        BinaryOpKind.Sub => a - b,
        BinaryOpKind.Mul => a * b,
        BinaryOpKind.Div => a / b,
        BinaryOpKind.Max => MathF.Max(a, b),
        BinaryOpKind.Min => MathF.Min(a, b),
        BinaryOpKind.Pow => MathF.Pow(a, b),
        BinaryOpKind.Mod => a - b * MathF.Floor(a / b),
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
        UnaryOpKind.Erf => Erf(a),
        UnaryOpKind.Sin => MathF.Sin(a),
        UnaryOpKind.Cos => MathF.Cos(a),
        UnaryOpKind.Floor => MathF.Floor(a),
        UnaryOpKind.Ceil => MathF.Ceiling(a),
        UnaryOpKind.Log => MathF.Log(a),
        UnaryOpKind.Tanh => MathF.Tanh(a),
        UnaryOpKind.Reciprocal => 1f / a,
        UnaryOpKind.Sign => a > 0f ? 1f : a < 0f ? -1f : 0f,
        UnaryOpKind.Round => MathF.Round(a),
        UnaryOpKind.Silu => a * (1f / (1f + MathF.Exp(-a))),
        _ => throw new ArgumentOutOfRangeException(nameof(op), op, null),
    };
}
