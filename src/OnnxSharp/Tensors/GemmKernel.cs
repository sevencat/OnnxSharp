using System.Numerics;
using System.Runtime.CompilerServices;

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
    /// </summary>
    private static unsafe void MultiplyRow(float* a, float* b, float* c, int k, int width, int bStride)
    {
        int vecWidth = Vector<float>.Count;
        int vectorEnd = width - vecWidth;

        for (int t = 0; t < k; t++)
        {
            float av = a[t];
            float* brow = b + (long)t * bStride;
            var avv = new Vector<float>(av);

            int j = 0;
            for (; j <= vectorEnd; j += vecWidth)
            {
                var bv = Unsafe.ReadUnaligned<Vector<float>>(brow + j);
                var cv = Unsafe.ReadUnaligned<Vector<float>>(c + j);
                Unsafe.WriteUnaligned(c + j, cv + avv * bv);
            }

            for (; j < width; j++)
                c[j] += av * brow[j];
        }
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
