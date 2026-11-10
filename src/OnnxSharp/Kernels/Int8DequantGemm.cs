using System.Buffers;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using OnnxSharp.Tensors;

namespace OnnxSharp.Kernels;

/// <summary>
/// 「int8 权重存储 + F32 计算」的 GEMM：权重保持 u8（体积只有 F32 的 1/4），
/// 按**列块**在 scratch 里反量化成 F32 后走现有的 FMA 微内核。
/// <para>
/// <b>⚠️ 实测结论：这条路在 ASR 形状上是负收益，不要再当作主路径</b>（见 <c>misc.md</c> 坑 #76/#77）。
/// <c>96×320×1280</c>：10.88 ms vs FP32 1.07 ms（0.10×）。原因不是解包（`vpmovzxbw` 很快），
/// 而是「按列块调 <see cref="GemmKernel"/>.Multiply」把每次调用都要付的 B 面板打包与
/// <c>Parallel.For</c> 调度开销乘上了列块数（n/64）。同理「权重带宽是瓶颈」也不成立：
/// 同形状 FP32 GEMM 只有 73 GFLOPS（峰值 400+），瓶颈是小矩阵效率。
/// </para>
/// <para>
/// 代码保留的价值：① 作反例与回归基准（A/B 时直接给数字）；② 若将来换到大 `m`（≥512）或
/// 权重放不进缓存的形状，可以再测一次 —— 那时带宽可能真的成为瓶颈。
/// </para>
/// </summary>
public static class Int8DequantGemm
{
    /// <summary>列块宽度（F32 scratch = <c>k × 64 × 4 B</c>，k=1280 时正好 320 KB，留在 L2）。</summary>
    private const int BlockColumns = 64;

    /// <summary>
    /// <c>C[m,n] = (A[m,k] - a_zp)·a_scale · (B[k,n] - b_zp)·b_scale</c>，输出 F32。
    /// 权重与激活都是 per-tensor 量化参数（最常见的情形）。
    /// </summary>
    public static void Multiply(
        ReadOnlySpan<byte> a,
        int m,
        int k,
        float aScale,
        int aZeroPoint,
        ReadOnlySpan<byte> b,
        int n,
        float bScale,
        int bZeroPoint,
        Span<float> c,
        int ldc = 0)
    {
        if (m <= 0 || k <= 0 || n <= 0)
            throw new ArgumentException($"invalid shape [{m},{k}]x[{k},{n}]");
        if (a.Length < (long)m * k)
            throw new ArgumentException($"A has {a.Length} values, [{m},{k}] needed", nameof(a));
        if (b.Length < (long)k * n)
            throw new ArgumentException($"B has {b.Length} values, [{k},{n}] needed", nameof(b));

        if (ldc <= 0)
            ldc = n;
        if (c.Length < (long)(m - 1) * ldc + n)
            throw new ArgumentException($"C has {c.Length} values, [{m},{ldc}] needed", nameof(c));

        float[] aFloatBuffer = ArrayPool<float>.Shared.Rent(m * k);
        float[] blockBuffer = ArrayPool<float>.Shared.Rent(k * Math.Min(BlockColumns, n));

        try
        {
            var aFloat = aFloatBuffer.AsSpan(0, m * k);
            DequantizeActivation(a, aFloat, aScale, aZeroPoint);

            int blockCap = Math.Min(BlockColumns, n);
            var block = blockBuffer.AsSpan(0, k * blockCap);

            for (int columnStart = 0; columnStart < n; columnStart += BlockColumns)
            {
                int width = Math.Min(BlockColumns, n - columnStart);

                DequantizeWeights(b, k, n, columnStart, width, bScale, bZeroPoint, block);

                // 直接写到 C 的第 columnStart 列起，ldc = 完整行宽
                GemmKernel.Multiply(
                    aFloat,
                    block[..(k * width)],
                    c[columnStart..],
                    m, k, width,
                    ldc);
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(aFloatBuffer);
            ArrayPool<float>.Shared.Return(blockBuffer);
        }
    }

    /// <summary>把 u8 激活反量化成 F32（<c>(a - zp) * scale</c>）。</summary>
    private static void DequantizeActivation(ReadOnlySpan<byte> source, Span<float> destination, float scale, int zeroPoint)
    {
        int i = 0;

        if (Avx512BW.IsSupported)
        {
            var vScale = Vector512.Create(scale);
            var vZero = Vector512.Create((float)zeroPoint);

            for (; i + 32 <= source.Length; i += 32)
            {
                var bytes = Vector256.LoadUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(source[i..]));
                var words = Avx512BW.ConvertToVector512Int16(bytes);          // vpmovzxbw
                var low = Avx512BW.ConvertToVector512Int32(words.GetLower()); // i16 → i32（低 16 个）
                var high = Avx512BW.ConvertToVector512Int32(words.GetUpper());

                var lowF = Avx512F.Multiply(Avx512F.ConvertToVector512Single(low) - vZero, vScale);
                var highF = Avx512F.Multiply(Avx512F.ConvertToVector512Single(high) - vZero, vScale);

                lowF.StoreUnsafe(ref destination[i]);
                highF.StoreUnsafe(ref destination[i + 16]);
            }
        }
        else if (Avx2.IsSupported)
        {
            var vScale = Vector256.Create(scale);
            var vZero = Vector256.Create((float)zeroPoint);

            for (; i + 8 <= source.Length; i += 8)
            {
                var bytes = Vector128.LoadUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(source[i..]));
                var ints = Avx2.ConvertToVector256Int32(bytes);
                var value = Avx.Multiply(Avx.ConvertToVector256Single(ints) - vZero, vScale);
                value.StoreUnsafe(ref destination[i]);
            }
        }

        for (; i < source.Length; i++)
            destination[i] = (source[i] - zeroPoint) * scale;
    }

    /// <summary>把 B 的一块（k × width）从 u8 反量化成 F32，按列优先连续写入 <paramref name="destination"/>。</summary>
    private static void DequantizeWeights(
        ReadOnlySpan<byte> weights,
        int k,
        int n,
        int columnStart,
        int width,
        float scale,
        int zeroPoint,
        Span<float> destination)
    {
        for (int row = 0; row < k; row++)
        {
            var source = weights.Slice(row * n + columnStart, width);
            var target = destination.Slice(row * width, width);
            DequantizeActivation(source, target, scale, zeroPoint);
        }
    }
}
