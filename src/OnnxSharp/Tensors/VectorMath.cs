using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace OnnxSharp.Tensors;

/// <summary>
/// 超越函数（exp / erf / sigmoid）的 AVX2 向量实现。
/// <para>
/// .NET 里 <see cref="MathF.Exp"/> 是标量调用（每元素 20~30 拍），Erf/Sigmoid 更甚 ——
/// RF-DETR 里单是 Erf 就占了 16% 的前向时间。这里的多项式实现把误差压到 1e-7 量级
/// （F32 的 eps 是 1.2e-7），和标量版本的结果差异远小于模型本身的数值噪声。
/// </para>
/// </summary>
internal static unsafe class VectorMath
{
    /// <summary>本机是否具备走向量化路径的条件。</summary>
    public static bool Available => Avx2.IsSupported && Fma.IsSupported;

    /// <summary>exp(x)，相对误差约 1e-7；输入先夹到 [−87.3, 88.7] 避免 2^n 构造溢出。</summary>
    public static Vector256<float> Exp(Vector256<float> x)
    {
        x = Avx.Min(Avx.Max(x, Vector256.Create(-87.3f)), Vector256.Create(88.7f));

        // x = n·ln2 + r，|r| ≤ ln2/2；ln2 拆成 hi+lo 两段保证 r 的精度
        var n = Avx.RoundToNearestInteger(Avx.Multiply(x, Vector256.Create(1.4426950408889634f)));
        var r = Fma.MultiplyAdd(n, Vector256.Create(-0.693145751953125f), x);
        r = Fma.MultiplyAdd(n, Vector256.Create(-1.4286067653e-6f), r);

        // e^r 的 6 阶 Horner 展开：|r| ≤ 0.3466 → 截断误差 ~1.2e-7
        var p = Vector256.Create(1f / 720f);
        p = Fma.MultiplyAdd(p, r, Vector256.Create(1f / 120f));
        p = Fma.MultiplyAdd(p, r, Vector256.Create(1f / 24f));
        p = Fma.MultiplyAdd(p, r, Vector256.Create(1f / 6f));
        p = Fma.MultiplyAdd(p, r, Vector256.Create(0.5f));
        p = Fma.MultiplyAdd(p, r, Vector256<float>.One);
        p = Fma.MultiplyAdd(p, r, Vector256<float>.One);

        var exponent = Avx2.ShiftLeftLogical(
            Avx2.Add(Avx.ConvertToVector256Int32(n), Vector256.Create(127)), 23);

        return Avx.Multiply(p, exponent.AsSingle());
    }

    /// <summary>
    /// erf(x)，与标量版同样用 Abramowitz &amp; Stegun 7.1.26（绝对误差 &lt; 1.5e-7），
    /// 只是把 exp 换成向量版。
    /// </summary>
    public static Vector256<float> Erf(Vector256<float> x)
    {
        var signMask = Vector256.Create(-0.0f);
        var sign = Avx.And(x, signMask);
        var ax = Avx.AndNot(signMask, x);

        var t = Avx.Divide(Vector256<float>.One,
            Fma.MultiplyAdd(Vector256.Create(0.3275911f), ax, Vector256<float>.One));

        var poly = Fma.MultiplyAdd(Vector256.Create(1.061405429f), t, Vector256.Create(-1.453152027f));
        poly = Fma.MultiplyAdd(poly, t, Vector256.Create(1.421413741f));
        poly = Fma.MultiplyAdd(poly, t, Vector256.Create(-0.284496736f));
        poly = Fma.MultiplyAdd(poly, t, Vector256.Create(0.254829592f));
        poly = Avx.Multiply(poly, t);

        var e = Exp(Avx.Multiply(Avx.Multiply(ax, ax), Vector256.Create(-1f)));
        var magnitude = Fma.MultiplyAdd(Avx.Multiply(poly, e), Vector256.Create(-1f), Vector256<float>.One);

        return Avx.Or(magnitude, sign);
    }

    public static Vector256<float> Sigmoid(Vector256<float> x)
    {
        var e = Exp(Avx.Multiply(x, Vector256.Create(-1f)));
        return Avx.Divide(Vector256<float>.One, Avx.Add(Vector256<float>.One, e));
    }

    public static void Exp(ReadOnlySpan<float> src, Span<float> dst)
    {
        int n = src.Length;
        int i = 0;

        if (Available)
        {
            fixed (float* ps = src)
            fixed (float* pd = dst)
            {
                for (; i <= n - 8; i += 8)
                    Avx.Store(pd + i, Exp(Avx.LoadVector256(ps + i)));
            }
        }

        for (; i < n; i++)
            dst[i] = MathF.Exp(src[i]);
    }

    public static void Erf(ReadOnlySpan<float> src, Span<float> dst)
    {
        int n = src.Length;
        int i = 0;

        if (Available)
        {
            fixed (float* ps = src)
            fixed (float* pd = dst)
            {
                for (; i <= n - 8; i += 8)
                    Avx.Store(pd + i, Erf(Avx.LoadVector256(ps + i)));
            }
        }

        for (; i < n; i++)
            dst[i] = TensorOps.Erf(src[i]);
    }

    public static void Sigmoid(ReadOnlySpan<float> src, Span<float> dst)
    {
        int n = src.Length;
        int i = 0;

        if (Available)
        {
            fixed (float* ps = src)
            fixed (float* pd = dst)
            {
                for (; i <= n - 8; i += 8)
                    Avx.Store(pd + i, Sigmoid(Avx.LoadVector256(ps + i)));
            }
        }

        for (; i < n; i++)
            dst[i] = 1f / (1f + MathF.Exp(-src[i]));
    }

    /// <summary>
    /// 逐元素 <c>exp(x - bias)</c> 并求和 —— Softmax 的内层。放在一起是为了让
    /// 「求 exp」和「求 Σ」走同一遍寄存器数据，不再把 exp 结果写回再读出来。
    /// </summary>
    public static float SumExp(ReadOnlySpan<float> src, Span<float> dst, float bias)
    {
        int n = src.Length;
        int i = 0;

        if (Available)
        {
            var vsum = Vector256<float>.Zero;
            var vbias = Vector256.Create(bias);

            fixed (float* ps = src)
            fixed (float* pd = dst)
            {
                for (; i <= n - 8; i += 8)
                {
                    var e = Exp(Avx.Subtract(Avx.LoadVector256(ps + i), vbias));
                    Avx.Store(pd + i, e);
                    vsum = Avx.Add(vsum, e);
                }
            }

            float total = 0;
            for (int k = 0; k < 8; k++)
                total += vsum.GetElement(k);

            for (; i < n; i++)
            {
                float value = MathF.Exp(src[i] - bias);
                dst[i] = value;
                total += value;
            }

            return total;
        }
        else
        {
            float total = 0;
            for (; i < n; i++)
            {
                float value = MathF.Exp(src[i] - bias);
                dst[i] = value;
                total += value;
            }

            return total;
        }
    }

    /// <summary>逐元素乘以标量（Softmax 的 1/Σ 收尾）。</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Scale(ReadOnlySpan<float> src, Span<float> dst, float factor)
    {
        int n = src.Length;
        int i = 0;

        if (Available)
        {
            var vf = Vector256.Create(factor);

            fixed (float* ps = src)
            fixed (float* pd = dst)
            {
                for (; i <= n - 8; i += 8)
                    Avx.Store(pd + i, Avx.Multiply(Avx.LoadVector256(ps + i), vf));
            }
        }

        for (; i < n; i++)
            dst[i] = src[i] * factor;
    }
}
