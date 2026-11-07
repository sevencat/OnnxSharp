using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

/// <summary>
/// 纯寄存器 FMA 循环：用来量机器的**算力上限**（不碰内存）。
/// <para>
/// 累加器条数要够多才能把 FMA 延迟藏掉：Zen 4 的 FMA 延迟 4~5 拍、每拍能发 2 条，
/// 至少要 10 条独立链才不出现气泡；但 YMM 只有 16 个，还要留 2 个给常量，
/// 所以取 12 条 —— 每轮 12 条 FMA = 6 拍，大于延迟，正好形成稳态。
/// </para>
/// </summary>
internal static class FmaBench
{
    /// <summary>独立累加链条数（每条是一整个 <see cref="Vector256{T}"/>）。</summary>
    public const int Accumulators = 12;

    /// <summary>跑 <paramref name="inner"/> 轮，返回累加结果（返回值被调用方消费，防止被优化掉）。</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]
    public static float Run(int inner)
    {
        if (!Avx2.IsSupported || !Fma.IsSupported)
            return 0f;

        var a = Vector256.Create(1.0000001f);
        var b = Vector256.Create(0.9999999f);

        var c0 = Vector256<float>.Zero;
        var c1 = Vector256<float>.Zero;
        var c2 = Vector256<float>.Zero;
        var c3 = Vector256<float>.Zero;
        var c4 = Vector256<float>.Zero;
        var c5 = Vector256<float>.Zero;
        var c6 = Vector256<float>.Zero;
        var c7 = Vector256<float>.Zero;
        var c8 = Vector256<float>.Zero;
        var c9 = Vector256<float>.Zero;
        var c10 = Vector256<float>.Zero;
        var c11 = Vector256<float>.Zero;

        for (int i = 0; i < inner; i++)
        {
            var m = Vector256.Create((float)i);
            c0 = Fma.MultiplyAdd(a, b, c0);
            c1 = Fma.MultiplyAdd(a, m, c1);
            c2 = Fma.MultiplyAdd(a, b, c2);
            c3 = Fma.MultiplyAdd(a, m, c3);
            c4 = Fma.MultiplyAdd(a, b, c4);
            c5 = Fma.MultiplyAdd(a, m, c5);
            c6 = Fma.MultiplyAdd(a, b, c6);
            c7 = Fma.MultiplyAdd(a, m, c7);
            c8 = Fma.MultiplyAdd(a, b, c8);
            c9 = Fma.MultiplyAdd(a, m, c9);
            c10 = Fma.MultiplyAdd(a, b, c10);
            c11 = Fma.MultiplyAdd(a, m, c11);
        }

        var sum = c0 + c1 + c2 + c3;
        sum += c4 + c5 + c6 + c7;
        sum += c8 + c9 + c10 + c11;

        return Vector256.Sum(sum);
    }
}
