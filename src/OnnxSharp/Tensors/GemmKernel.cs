using System.Buffers;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using OnnxSharp.Threading;

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
        bool parallel = true,
        ReadOnlySpan<float> bias = default)
    {
        if (m <= 0 || n <= 0)
            return;

        if (ldc <= 0)
            ldc = n;

        bool hasBias = bias.Length >= m;

        if (k <= 0)
        {
            for (int row = 0; row < m; row++)
            {
                if (hasBias)
                    c.Slice(row * ldc, n).Fill(bias[row]);
                else
                    c.Slice(row * ldc, n).Clear();
            }

            return;
        }

        int nb = SelectColumnBlock(n);

        bool useParallel = parallel && (long)m * n * k >= ParallelThreshold && m > 1;
        int threads = ParallelRange.Threads;
        int columnBlocks = (n + nb - 1) / nb;

        fixed (float* pa = a)
        fixed (float* pb = b)
        fixed (float* pc = c)
        fixed (float* pbias = hasBias ? bias : ReadOnlySpan<float>.Empty)
        {
            // 点卷积形态：m（输出通道）小、n（空间位置）大。
            // 按行切分时每个行块都要重新打包一整份 B（k×n），行块数越多 B 的搬运就
            // 被放大得越厉害 —— 1×1 conv 上实测并行后和串行一样快（各自 16 份 B）。
            // 按列切分时每个线程只打包自己那段列，B 总共只搬一遍，A 虽被每个线程
            // 重读一次，但 A 只有 m×k 且留在 L1/L2。
            if (useParallel && m <= 256 && columnBlocks >= threads)
            {
                float* a0 = pa;
                float* b0 = pb;
                float* c0 = pc;
                float* bias0 = pbias;

                // 列块数一般不是线程数的整数倍，余数摊给前 rem 个线程，
                // 否则最后一批线程空跑（20×20 这种小空间层尤其明显）。
                int basePer = columnBlocks / threads;
                int rem = columnBlocks % threads;

                Parallel.For(0, threads, t =>
                {
                    int cb0 = t * basePer + Math.Min(t, rem);
                    int cb1 = cb0 + basePer + (t < rem ? 1 : 0);
                    if (cb0 >= cb1)
                        return;

                    var pack = ArrayPool<float>.Shared.Rent(k * nb);
                    try
                    {
                        fixed (float* pp = pack)
                        {
                            MultiplyColumnRange(a0, b0, c0, m, k, n, ldc, nb,
                                cb0 * nb, Math.Min(cb1 * nb, n), pp, bias0);
                        }
                    }
                    finally
                    {
                        ArrayPool<float>.Shared.Return(pack);
                    }
                });
            }
            else if (useParallel && SelectRowBlock(m, threads) < m)
            {
                int mb = SelectRowBlock(m, threads);
                int rowBlocks = (m + mb - 1) / mb;

                float* a0 = pa;
                float* b0 = pb;
                float* c0 = pc;
                float* bias0 = pbias;

                Parallel.For(0, rowBlocks, ib =>
                {
                    int row0 = ib * mb;
                    int row1 = Math.Min(row0 + mb, m);
                    MultiplyRowBlock(a0, b0, c0, row0, row1, k, n, ldc, nb, bias0);
                });
            }
            else
            {
                MultiplyRowBlock(pa, pb, pc, 0, m, k, n, ldc, nb, pbias);
            }
        }
    }

    /// <summary>
    /// 只算 C 的 <c>[jBegin, jEnd)</c> 这段列：每个线程打包自己那段 B 面板后，
    /// 对所有 m 行做微内核。
    /// </summary>
    private static unsafe void MultiplyColumnRange(
        float* a,
        float* b,
        float* c,
        int m,
        int k,
        int n,
        int ldc,
        int nb,
        int jBegin,
        int jEnd,
        float* pack,
        float* bias)
    {
        for (int j0 = jBegin; j0 < jEnd; j0 += nb)
        {
            int width = Math.Min(nb, jEnd - j0);
            PackColumns(b + j0, k, n, width, pack);

            int row = 0;
            for (; row + 4 <= m; row += 4)
            {
                Multiply4Rows(a + (long)row * k, k, pack, c + (long)row * ldc + j0, k, width, width, ldc,
                    bias is null ? null : bias + row);
            }

            for (; row < m; row++)
            {
                MultiplyRow(a + (long)row * k, pack, c + (long)row * ldc + j0, k, width, width,
                    bias is null ? 0f : bias[row]);
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
        int nb,
        float* bias)
    {
        if (nb >= n)
        {
            // 只有一块：B 的行内已经连续（n ≤ nb 时步长最多 nb 个 float），直接算。
            for (int row = row0; row < row1; row++)
                MultiplyRow(a + (long)row * k, b, c + (long)row * ldc, k, n, n,
                    bias is null ? 0f : bias[row]);
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

                    // 4 行一起算：同一次 B 的 load 供 4 行复用，L1 的 load 端口不再
                    // 被「每行各自把 B 拉一遍」占满。剩下不足 4 行的用单行内核收尾。
                    int row = row0;
                    for (; row + 4 <= row1; row += 4)
                    {
                        Multiply4Rows(a + (long)row * k, k, pack, c + (long)row * ldc + j0, k, width, width, ldc,
                            bias is null ? null : bias + row);
                    }

                    for (; row < row1; row++)
                    {
                        MultiplyRow(a + (long)row * k, pack, c + (long)row * ldc + j0, k, width, width,
                            bias is null ? 0f : bias[row]);
                    }
                }
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(rented);
        }
    }

    /// <summary>
    /// 把 <c>b[t, 0..width)</c>（行步长 <paramref name="srcStride"/>）打包成连续的 <c>[k, width]</c>。
    /// <para>
    /// 列块只有 16 列（64 字节），用 <c>CopyBlockUnaligned</c> 的话调用开销远大于拷贝本身
    /// （每行块要打 500 多次包），所以这里直接按 8-lane 向量搬。
    /// </para>
    /// </summary>
    private static unsafe void PackColumns(float* b, int k, int srcStride, int width, float* pack)
    {
        int j = 0;

        for (; j + 8 <= width; j += 8)
        {
            for (int t = 0; t < k; t++)
                Avx.Store(pack + (long)t * width + j, Avx.LoadVector256(b + (long)t * srcStride + j));
        }

        for (; j < width; j++)
        {
            for (int t = 0; t < k; t++)
                pack[(long)t * width + j] = b[(long)t * srcStride + j];
        }
    }

    /// <summary>
    /// 4 行共享一次 B 加载的微内核（一次 2 个 <see cref="Vector256{T}"/>，即 16 列）。
    /// <para>
    /// 单行 8 向量版本每个 k 需要 8 条 B load + 1 条 A 广播 = 9 条 load，而 Zen 4 每拍只发 2 条 load，
    /// 光 load 就要 4.5 拍 —— 已经超过 8 条 FMA 所需的 4 拍，等于被 load 端口卡住。
    /// 4 行共享时每条 B load 喂 4 条 FMA，load 指令数降到 2 + 4 条广播 = 6 条，
    /// 反过来让 8 条 FMA（4 拍）重新成为瓶颈。
    /// </para>
    /// </summary>
    private static unsafe void Multiply4Rows(
        float* a,
        int aStride,
        float* b,
        float* c,
        int k,
        int width,
        int bStride,
        int ldc,
        float* bias = null)
    {
        float* a1 = a + aStride;
        float* a2 = a + 2 * aStride;
        float* a3 = a + 3 * aStride;

        float* c1 = c + ldc;
        float* c2 = c + 2 * ldc;
        float* c3 = c + 3 * ldc;

        // 累加器直接以 bias 起头：微内核最后是整块 store（不是 +=），
        // 所以 bias 只能在这里融进去，写完再补一遍就是多一遍 C 的读写。
        var b0v = Vector256.Create(bias is null ? 0f : bias[0]);
        var b1v = Vector256.Create(bias is null ? 0f : bias[1]);
        var b2v = Vector256.Create(bias is null ? 0f : bias[2]);
        var b3v = Vector256.Create(bias is null ? 0f : bias[3]);

        int j = 0;

        for (; j + 16 <= width; j += 16)
        {
            var c00 = b0v;
            var c01 = b0v;
            var c10 = b1v;
            var c11 = b1v;
            var c20 = b2v;
            var c21 = b2v;
            var c30 = b3v;
            var c31 = b3v;

            float* brow = b + j;
            for (int t = 0; t < k; t++, brow += bStride)
            {
                var b0 = Avx.LoadVector256(brow);
                var b1 = Avx.LoadVector256(brow + 8);

                var va = Vector256.Create(a[t]);
                c00 = Fma.MultiplyAdd(va, b0, c00);
                c01 = Fma.MultiplyAdd(va, b1, c01);

                va = Vector256.Create(a1[t]);
                c10 = Fma.MultiplyAdd(va, b0, c10);
                c11 = Fma.MultiplyAdd(va, b1, c11);

                va = Vector256.Create(a2[t]);
                c20 = Fma.MultiplyAdd(va, b0, c20);
                c21 = Fma.MultiplyAdd(va, b1, c21);

                va = Vector256.Create(a3[t]);
                c30 = Fma.MultiplyAdd(va, b0, c30);
                c31 = Fma.MultiplyAdd(va, b1, c31);
            }

            Avx.Store(c + j, c00);
            Avx.Store(c + j + 8, c01);
            Avx.Store(c1 + j, c10);
            Avx.Store(c1 + j + 8, c11);
            Avx.Store(c2 + j, c20);
            Avx.Store(c2 + j + 8, c21);
            Avx.Store(c3 + j, c30);
            Avx.Store(c3 + j + 8, c31);
        }

        // 不足 16 列的尾巴：交回单行内核（只覆盖各自的列区间，不会和上面重叠）
        int rest = width - j;
        if (rest <= 0)
            return;

        MultiplyRow(a, b + j, c + j, k, rest, bStride, bias is null ? 0f : bias[0]);
        MultiplyRow(a1, b + j, c1 + j, k, rest, bStride, bias is null ? 0f : bias[1]);
        MultiplyRow(a2, b + j, c2 + j, k, rest, bStride, bias is null ? 0f : bias[2]);
        MultiplyRow(a3, b + j, c3 + j, k, rest, bStride, bias is null ? 0f : bias[3]);
    }



    /// <summary>
    /// 列块宽度。取 16（2 个 AVX 向量）：打包后的 B 面板是 <c>k×nb×4</c> 字节，
    /// k=256 时正好 16 KB，能整块待在 32 KB 的 L1 里；取 64 的话面板 64 KB 落到 L2，
    /// 每条 FMA 都要等 L2 带宽，实测只有峰值的一半。
    /// </summary>
    private static int SelectColumnBlock(int n) => n < 16 ? n : 16;

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
    private static unsafe void MultiplyRow(
        float* a, float* b, float* c, int k, int width, int bStride, float bias = 0f)
    {
        int j = 0;

        // AVX2+FMA：256 位、8 路 F32，显式 FMA 融合。
        // Vector<float> 不保证被 JIT 融成 FMA，浮点语义上也不能随便融，所以这里手写。
        if (Avx2.IsSupported && Fma.IsSupported)
            j = MultiplyRowAvx2(a, b, c, k, width, bStride, bias);

        int vw = Vector<float>.Count;

        // 剩余不足 4 个向量的部分：逐个向量处理（累加器带着 bias 起头，结果直接 store）
        int vectorEnd = width - vw;

        for (; j <= vectorEnd; j += vw)
        {
            var acc = new Vector<float>(bias);

            for (int t = 0; t < k; t++)
                acc += new Vector<float>(a[t]) * Unsafe.ReadUnaligned<Vector<float>>(b + (long)t * bStride + j);

            Unsafe.WriteUnaligned(c + j, acc);
        }

        for (; j < width; j++)
        {
            float sum = bias;
            for (int t = 0; t < k; t++)
                sum += a[t] * b[(long)t * bStride + j];
            c[j] = sum;
        }
    }

    /// <summary>
    /// AVX2+FMA 主循环：一次算 4 个 <see cref="Vector256{T}"/>（32 个 float），
    /// 累加器全在 ymm 寄存器里，k 轮下来 C 只写一次。返回已处理的列数。
    /// </summary>
    private static unsafe int MultiplyRowAvx2(float* a, float* b, float* c, int k, int width, int bStride, float bias = 0f)
    {
        const int vw = 8;   // Vector256<float>
        int j = 0;

        var bv = Vector256.Create(bias);

        // 8 条累加器链（64 列）：FMA 延迟 4~5 拍、每拍 2 条，4 条链只够填一半。
        for (; j + 8 * vw <= width; j += 8 * vw)
        {
            var c0 = bv;
            var c1 = bv;
            var c2 = bv;
            var c3 = bv;
            var c4 = bv;
            var c5 = bv;
            var c6 = bv;
            var c7 = bv;

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

        // 余下不足 8 个向量的部分：4 个向量一组
        for (; j + 4 * vw <= width; j += 4 * vw)
        {
            var c0 = bv;
            var c1 = bv;
            var c2 = bv;
            var c3 = bv;

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

        // 列块宽度 16 时只会走到这里（2 个向量一组）
        for (; j + 2 * vw <= width; j += 2 * vw)
        {
            var c0 = bv;
            var c1 = bv;

            for (int t = 0; t < k; t++)
            {
                var av = Vector256.Create(a[t]);
                float* brow = b + (long)t * bStride + j;

                c0 = Fma.MultiplyAdd(av, Avx.LoadVector256(brow), c0);
                c1 = Fma.MultiplyAdd(av, Avx.LoadVector256(brow + vw), c1);
            }

            Avx.Store(c + j, c0);
            Avx.Store(c + j + vw, c1);
        }

        for (; j + vw <= width; j += vw)
        {
            var c0 = bv;

            for (int t = 0; t < k; t++)
                c0 = Fma.MultiplyAdd(Vector256.Create(a[t]), Avx.LoadVector256(b + (long)t * bStride + j), c0);

            Avx.Store(c + j, c0);
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
