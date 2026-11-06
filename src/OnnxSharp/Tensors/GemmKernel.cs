using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace OnnxSharp.Tensors;

/// <summary>
/// F32 矩阵乘内核：<c>C[m,n] = A[m,k] · B[k,n]</c>，供 MatMul / Gemm / 后续 Conv 复用。
/// <para>
/// 性能约定（对照 OpenCV <c>fastGEMM1T</c> / <c>conv_block</c> 的思路）：
/// </para>
/// <list type="bullet">
/// <item>循环顺序用 <b>i-k-j</b> 而不是朴素的 i-j-k：三路访存都顺序，
/// 避免 i-j-k 里 B 按列跨行进 cache miss。</item>
/// <item>内层 j 用 <see cref="Vector{T}"/> 向量化（AVX2 下 8 路 F32）。</item>
/// <item>大矩阵按行并行；每行内部仍是 t 递增累加，
/// 与朴素实现的点积顺序完全一致，<b>数值逐位不变</b>。</item>
/// </list>
/// <para>
/// 实现用指针而非 <see cref="Span{T}"/>：ref struct 不能进 lambda，而并行体需要它。
/// </para>
/// </summary>
internal static class GemmKernel
{
    /// <summary>低于这个乘加次数就不值得开线程（小算子开线程反而更慢）。</summary>
    private const long ParallelThreshold = 1L << 22;

    /// <summary>
    /// <c>c = a · b</c>。
    /// <para>
    /// <paramref name="ldc"/> 是 C 的行步长（元素数），0 表示紧凑 <c>n</c>。
    /// Conv 需要直接写进 NCHW 输出里某个 <c>[outPerGroup, outH*outW]</c> 的块，行步长不是 n。
    /// </para>
    /// <para>
    /// <paramref name="parallel"/> = false 供调用方自己控制并行（避免嵌套并行把线程池打爆）。
    /// </para>
    /// </summary>
    public static unsafe void Multiply(
        ReadOnlySpan<float> a,
        ReadOnlySpan<float> b,
        Span<float> c,
        int m,
        int k,
        int n,
        int ldc = 0,
        bool parallel = true)
    {
        if (m <= 0 || n <= 0)
            return;

        if (ldc <= 0)
            ldc = n;

        if (k <= 0)
        {
            for (int row = 0; row < m; row++)
                c.Slice(row * ldc, n).Clear();
            return;
        }

        fixed (float* pa = a)
        fixed (float* pb = b)
        fixed (float* pc = c)
        {
            if (parallel && (long)m * n * k >= ParallelThreshold && m >= 8)
            {
                // 按行切分：每行独立累加、互不重叠，结果与串行完全一致。
                float* a0 = pa;
                float* b0 = pb;
                float* c0 = pc;
                int step = ldc;

                Parallel.For(0, m, row =>
                {
                    float* crow = c0 + (long)row * step;
                    new Span<float>(crow, n).Clear();
                    MultiplyRow(a0 + (long)row * k, b0, crow, k, n, n);
                });
            }
            else
            {
                for (int row = 0; row < m; row++)
                {
                    float* crow = pc + (long)row * ldc;
                    new Span<float>(crow, n).Clear();
                    MultiplyRow(pa + (long)row * k, pb, crow, k, n, n);
                }
            }
        }
    }

    /// <summary>
    /// 单行的一列块：<c>c[0,width) += a[0,k) · b[k,width)</c>。
    /// <paramref name="bStride"/> 是 B 的真实行步长（列块起点已在 b 里偏移过）。
    /// <para>
    /// 主循环一次算 <b>4 个向量（32 个 float）</b>：这 4 个向量全程留在寄存器里累加，
    /// k 轮下来 C 只写一次。逐轮 load/store C 会让访存和 FMA 抢发射端口，是最容易踩的坑。
    /// </para>
    /// </summary>
    private static unsafe void MultiplyRow(float* a, float* b, float* c, int k, int width, int bStride)
    {
        int j = 0;

        // AVX2+FMA：256 位、8 路 F32，显式 FMA 融合。
        // Vector<float> 不保证被 JIT 融成 FMA，浮点语义上也不能随便融，所以这里手写。
        if (Avx2.IsSupported && Fma.IsSupported)
            j = MultiplyRowAvx2(a, b, c, k, width, bStride);

        int vw = Vector<float>.Count;

        // 剩余不足 4 个向量的部分：逐个向量处理（C 已清空，直接累加）
        int vectorEnd = width - vw;

        for (; j <= vectorEnd; j += vw)
        {
            var acc = Vector<float>.Zero;

            for (int t = 0; t < k; t++)
                acc += new Vector<float>(a[t]) * Unsafe.ReadUnaligned<Vector<float>>(b + (long)t * bStride + j);

            Unsafe.WriteUnaligned(c + j, acc);
        }

        for (; j < width; j++)
        {
            float sum = 0;
            for (int t = 0; t < k; t++)
                sum += a[t] * b[(long)t * bStride + j];
            c[j] += sum;
        }
    }

    /// <summary>
    /// AVX2+FMA 主循环：一次算 4 个 <see cref="Vector256{T}"/>（32 个 float），
    /// 累加器全在 ymm 寄存器里，k 轮下来 C 只写一次。返回已处理的列数。
    /// </summary>
    private static unsafe int MultiplyRowAvx2(float* a, float* b, float* c, int k, int width, int bStride)
    {
        const int vw = 8;   // Vector256<float>
        int j = 0;

        for (; j + 4 * vw <= width; j += 4 * vw)
        {
            var c0 = Vector256<float>.Zero;
            var c1 = Vector256<float>.Zero;
            var c2 = Vector256<float>.Zero;
            var c3 = Vector256<float>.Zero;

            for (int t = 0; t < k; t++)
            {
                var av = Vector256.Create(a[t]);
                float* brow = b + (long)t * bStride + j;

                c0 = Fma.MultiplyAdd(av, Avx.LoadVector256(brow), c0);
                c1 = Fma.MultiplyAdd(av, Avx.LoadVector256(brow + vw), c1);
                c2 = Fma.MultiplyAdd(av, Avx.LoadVector256(brow + 2 * vw), c2);
                c3 = Fma.MultiplyAdd(av, Avx.LoadVector256(brow + 3 * vw), c3);
            }

            Avx.Store(c + j, c0);
            Avx.Store(c + j + vw, c1);
            Avx.Store(c + j + 2 * vw, c2);
            Avx.Store(c + j + 3 * vw, c3);
        }

        return j;
    }

    /// <summary>
    /// 转置 <paramref name="src"/>（<c>rows×cols</c>），返回 <c>cols×rows</c> 的新数组。
    /// Gemm 的 transA/transB 走这条路径（权重规模远小于调用次数，可接受）。
    /// </summary>
    public static float[] Transpose(ReadOnlySpan<float> src, int rows, int cols)
    {
        var dst = new float[rows * cols];

        for (int i = 0; i < rows; i++)
        {
            var srow = src.Slice(i * cols, cols);
            for (int j = 0; j < cols; j++)
                dst[j * rows + i] = srow[j];
        }

        return dst;
    }
}
