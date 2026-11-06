using System.Buffers;
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

        int nb = SelectColumnBlock(n);

        bool useParallel = parallel && (long)m * n * k >= ParallelThreshold && m > 1;
        int mb = useParallel ? SelectRowBlock(m, Environment.ProcessorCount) : m;

        fixed (float* pa = a)
        fixed (float* pb = b)
        fixed (float* pc = c)
        {
            if (useParallel && mb < m)
            {
                int rowBlocks = (m + mb - 1) / mb;

                // fixed 指针不能进 lambda，先取出裸副本
                float* a0 = pa;
                float* b0 = pb;
                float* c0 = pc;

                Parallel.For(0, rowBlocks, ib =>
                {
                    int row0 = ib * mb;
                    int row1 = Math.Min(row0 + mb, m);
                    MultiplyRowBlock(a0, b0, c0, row0, row1, k, n, ldc, nb);
                });
            }
            else
            {
                MultiplyRowBlock(pa, pb, pc, 0, m, k, n, ldc, nb);
            }
        }
    }

    /// <summary>
    /// 一个行块：列块在外、行在内。
    /// <para>
    /// 这是整个内核最关键的重排。原来的「行 → 列块 → k」会让 B 的 <c>[k, n]</c> 每行都完整重扫一遍：
    /// m=1024、B=768 KB 时光是 B 就要从 L2 拉 786 MB，实测正是 L2 带宽打满。
    /// 换成「列块 → k → 行」后，B 的一个 <c>k×nb</c> 面板（约 32 KB）留在 L1，
    /// 被本块的 <paramref name="row1"/>-<paramref name="row0"/> 行反复命中。
    /// </para>
    /// </summary>
    private static unsafe void MultiplyRowBlock(
        float* a,
        float* b,
        float* c,
        int row0,
        int row1,
        int k,
        int n,
        int ldc,
        int nb)
    {
        for (int row = row0; row < row1; row++)
            new Span<float>(c + (long)row * ldc, n).Clear();

        if (nb >= n)
        {
            // 只有一块：B 的行内已经连续（n ≤ nb 时步长最多 nb 个 float），直接算。
            for (int row = row0; row < row1; row++)
                MultiplyRow(a + (long)row * k, b, c + (long)row * ldc, k, n, n);
            return;
        }

        // 把列块打包成紧凑的 [k, width] 面板：B 原始行步长是 n（常见 512~4096 float），
        // 内层 k 循环每步跨一行 —— 4 KB 起步的步长在 L1 dTLB 上必然 thrash。
        // 打包后 k 方向是连续内存，panel 只有 k×width×4 字节，顺序读。
        var rented = ArrayPool<float>.Shared.Rent(k * nb);
        try
        {
            fixed (float* pack = rented)
            {
                for (int j0 = 0; j0 < n; j0 += nb)
                {
                    int width = n - j0 < nb ? n - j0 : nb;
                    PackColumns(b + j0, k, n, width, pack);

                    for (int row = row0; row < row1; row++)
                        MultiplyRow(a + (long)row * k, pack, c + (long)row * ldc + j0, k, width, width);
                }
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(rented);
        }
    }

    /// <summary>把 <c>b[t, 0..width)</c>（行步长 <paramref name="srcStride"/>）打包成连续的 <c>[k, width]</c>。</summary>
    private static unsafe void PackColumns(float* b, int k, int srcStride, int width, float* pack)
    {
        uint bytes = (uint)(width * sizeof(float));

        for (int t = 0; t < k; t++)
            Unsafe.CopyBlockUnaligned(pack + (long)t * width, b + (long)t * srcStride, bytes);
    }

    /// <summary>
    /// 列块宽度。取 64（8 个 AVX 向量）：Zen 4 的 FMA 延迟 4~5 拍、每拍能发 2 条，
    /// 4 条累加器链只能填到 1 条/拍，必须 8 条才能真正跑满。
    /// </summary>
    private static int SelectColumnBlock(int n) => n < 64 ? n : 64;

    /// <summary>行块高度：任务数至少是线程数的 2 倍，块高不小于 8 行且对齐到 8。</summary>
    private static int SelectRowBlock(int m, int threads)
    {
        if (threads < 1)
            threads = 1;

        int mb = (m + threads * 2 - 1) / (threads * 2);
        mb = (mb + 7) & ~7;
        if (mb < 8)
            mb = 8;
        return mb < m ? mb : m;
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

        // 8 条累加器链（64 列）：FMA 延迟 4~5 拍、每拍 2 条，4 条链只够填一半。
        for (; j + 8 * vw <= width; j += 8 * vw)
        {
            var c0 = Vector256<float>.Zero;
            var c1 = Vector256<float>.Zero;
            var c2 = Vector256<float>.Zero;
            var c3 = Vector256<float>.Zero;
            var c4 = Vector256<float>.Zero;
            var c5 = Vector256<float>.Zero;
            var c6 = Vector256<float>.Zero;
            var c7 = Vector256<float>.Zero;

            for (int t = 0; t < k; t++)
            {
                var av = Vector256.Create(a[t]);
                float* brow = b + (long)t * bStride + j;

                c0 = Fma.MultiplyAdd(av, Avx.LoadVector256(brow), c0);
                c1 = Fma.MultiplyAdd(av, Avx.LoadVector256(brow + vw), c1);
                c2 = Fma.MultiplyAdd(av, Avx.LoadVector256(brow + 2 * vw), c2);
                c3 = Fma.MultiplyAdd(av, Avx.LoadVector256(brow + 3 * vw), c3);
                c4 = Fma.MultiplyAdd(av, Avx.LoadVector256(brow + 4 * vw), c4);
                c5 = Fma.MultiplyAdd(av, Avx.LoadVector256(brow + 5 * vw), c5);
                c6 = Fma.MultiplyAdd(av, Avx.LoadVector256(brow + 6 * vw), c6);
                c7 = Fma.MultiplyAdd(av, Avx.LoadVector256(brow + 7 * vw), c7);
            }

            Avx.Store(c + j, c0);
            Avx.Store(c + j + vw, c1);
            Avx.Store(c + j + 2 * vw, c2);
            Avx.Store(c + j + 3 * vw, c3);
            Avx.Store(c + j + 4 * vw, c4);
            Avx.Store(c + j + 5 * vw, c5);
            Avx.Store(c + j + 6 * vw, c6);
            Avx.Store(c + j + 7 * vw, c7);
        }

        // 余下不足 8 个向量的部分：4 个向量一组（C 已清空）
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
