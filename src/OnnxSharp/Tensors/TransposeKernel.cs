using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace OnnxSharp.Tensors;

/// <summary>
/// N-D 转置内核。按 <c>perm</c> 的结构分三档，避免逐元素的多轴下标计算：
/// <list type="number">
/// <item><b>末尾维不变</b>：<c>perm[i] == i</c> 从末尾连续成立的那几维可以在一次拷贝里整块搬，
/// 每条拷贝就是一次 memcpy（attention 的 <c>(B,S,H,D) → (B,H,S,D)</c> 是 <c>inner = D</c> 的块拷贝）。</item>
/// <item><b>单轴旋转</b>：把某一维提到最前 / 挪到最后、其余顺序不变，等价于「批 × 矩阵转置」，
/// 走 8×8 AVX 分块转置（<c>(B,S,H*D) → (B,H*D,S)</c>、只交换最后两维都属于这一类）。</item>
/// <item>其余形状：通用进位循环兜底（每元素只做一次步长累加）。</item>
/// </list>
/// <para>
/// 与 <see cref="GemmKernel.Transpose"/> 的区别：那个是给 Gemm 的 transA/transB 用的 2D 小工具，
/// 这里是运行时算子的完整实现（任意 rank、任意 element size）。
/// </para>
/// </summary>
internal static class TransposeKernel
{
    /// <summary>块拷贝小于这个字节数就不值得走 memcpy 路径，改用分块转置。</summary>
    private const int MinBlockBytes = 32;

    /// <summary>2D 分块转置的缓存块边长。</summary>
    private const int BlockEdge = 32;

    public static unsafe void Transpose(
        ReadOnlySpan<byte> src,
        Span<byte> dst,
        ReadOnlySpan<int> inDims,
        ReadOnlySpan<int> perm,
        int elementSize)
    {
        int rank = inDims.Length;
        if (rank <= 1)
        {
            src.CopyTo(dst);
            return;
        }

        long total = 1;
        for (int i = 0; i < rank; i++)
            total *= inDims[i];
        if (total == 0)
            return;

        // 末尾有多少维「位置不变」：这些维在输入/输出里都是末端的连续段，可整块搬。
        int tail = 0;
        while (tail < rank && perm[rank - 1 - tail] == rank - 1 - tail)
            tail++;

        if (tail == rank)
        {
            src.CopyTo(dst);
            return;
        }

        if (tail >= 1)
        {
            long inner = 1;
            for (int i = rank - tail; i < rank; i++)
                inner *= inDims[i];

            if (inner * elementSize >= MinBlockBytes)
            {
                BlockCopy(src, dst, inDims, perm, rank, tail, inner, elementSize, total);
                return;
            }
        }

        if (TryMatrixTranspose(inDims, perm, out int rows, out int cols, out long batch))
        {
            MatrixTranspose(src, dst, rows, cols, batch, elementSize);
            return;
        }

        GenericCopy(src, dst, inDims, perm, elementSize, total);
    }

    /// <summary>行主序元素步长。</summary>
    private static void ComputeStrides(ReadOnlySpan<int> dims, Span<long> strides)
    {
        long run = 1;
        for (int i = dims.Length - 1; i >= 0; i--)
        {
            strides[i] = run;
            run *= dims[i];
        }
    }

    /// <summary>
    /// 末尾 <paramref name="tail"/> 维保持不变 → 外层逐组合拷贝长度为 <paramref name="innerElems"/> 的连续块。
    /// </summary>
    private static unsafe void BlockCopy(
        ReadOnlySpan<byte> src,
        Span<byte> dst,
        ReadOnlySpan<int> inDims,
        ReadOnlySpan<int> perm,
        int rank,
        int tail,
        long innerElems,
        int elementSize,
        long total)
    {
        int outerRank = rank - tail;
        long blockBytes = innerElems * elementSize;
        long outerCount = total / innerElems;

        Span<long> inStride = stackalloc long[rank];
        ComputeStrides(inDims, inStride);

        Span<long> step = stackalloc long[outerRank];   // 外层每维 +1 时输入字节偏移的增量
        Span<int> outerDims = stackalloc int[outerRank];
        for (int i = 0; i < outerRank; i++)
        {
            int axis = perm[i];
            outerDims[i] = inDims[axis];
            step[i] = inStride[axis] * elementSize;
        }

        Span<int> coord = stackalloc int[outerRank];

        fixed (byte* ps = src)
        fixed (byte* pd = dst)
        {
            long inOff = 0;
            long outOff = 0;

            for (long n = 0; n < outerCount; n++)
            {
                Unsafe.CopyBlockUnaligned(pd + outOff, ps + inOff, (uint)blockBytes);
                outOff += blockBytes;

                for (int i = outerRank - 1; i >= 0; i--)
                {
                    if (++coord[i] < outerDims[i])
                    {
                        inOff += step[i];
                        break;
                    }

                    inOff -= (long)(outerDims[i] - 1) * step[i];
                    coord[i] = 0;
                }
            }
        }
    }

    /// <summary>
    /// 判断 <paramref name="perm"/> 是否「把第 <paramref name="b"/> 维挪到最后、其余顺序不变」。
    /// </summary>
    private static bool MatchesRotateLeft(ReadOnlySpan<int> perm, int b)
    {
        int rank = perm.Length;
        for (int i = 0; i < rank; i++)
        {
            int expected = i < b ? i : (i == rank - 1 ? b : i + 1);
            if (perm[i] != expected)
                return false;
        }

        return true;
    }

    /// <summary>
    /// 判断 <paramref name="perm"/> 是否「把最后一维提到第 <paramref name="b"/> 位、其余顺序不变」。
    /// </summary>
    private static bool MatchesRotateRight(ReadOnlySpan<int> perm, int b)
    {
        int rank = perm.Length;
        for (int i = 0; i < rank; i++)
        {
            int expected = i < b ? i : (i == b ? rank - 1 : i - 1);
            if (perm[i] != expected)
                return false;
        }

        return true;
    }

    /// <summary>
    /// 若 <paramref name="perm"/> 是单轴旋转，则把整次转置化归为
    /// <c>batch</c> 个 <c>rows×cols</c> 矩阵转置 <c>→ cols×rows</c>。
    /// 多种划分等价时取 <c>batch</c> 最小的那个（每个块越大，分块转置越省）。
    /// </summary>
    private static bool TryMatrixTranspose(
        ReadOnlySpan<int> dims,
        ReadOnlySpan<int> perm,
        out int rows,
        out int cols,
        out long batch)
    {
        int rank = dims.Length;
        rows = 0;
        cols = 0;
        batch = 0;

        long bestBatch = long.MaxValue;

        for (int b = 0; b < rank; b++)
        {
            // 左移 b：输入 (B, R=dims[b], C=prod(dims[b+1..]))，输出 (B, C, R)
            if (b < rank - 1 && MatchesRotateLeft(perm, b))
            {
                long outer = 1;
                for (int i = 0; i < b; i++)
                    outer *= dims[i];

                if (outer < bestBatch)
                {
                    long inner = 1;
                    for (int i = b + 1; i < rank; i++)
                        inner *= dims[i];

                    bestBatch = outer;
                    rows = dims[b];
                    cols = checked((int)inner);
                    batch = outer;
                }
            }

            // 右移 b：输入 (B, R=prod(dims[b..rank-2]), C=dims[rank-1])，输出 (B, C, R)
            if (MatchesRotateRight(perm, b))
            {
                long outer = 1;
                for (int i = 0; i < b; i++)
                    outer *= dims[i];

                if (outer < bestBatch)
                {
                    long inner = 1;
                    for (int i = b; i < rank - 1; i++)
                        inner *= dims[i];

                    bestBatch = outer;
                    rows = checked((int)inner);
                    cols = dims[rank - 1];
                    batch = outer;
                }
            }
        }

        return batch != 0;
    }

    /// <summary>
    /// <paramref name="batch"/> 个独立的 <c>rows×cols</c> 矩阵转置（源、目的都是连续存放的批）。
    /// </summary>
    private static unsafe void MatrixTranspose(
        ReadOnlySpan<byte> src,
        Span<byte> dst,
        int rows,
        int cols,
        long batch,
        int elementSize)
    {
        long plane = (long)rows * cols * elementSize;

        fixed (byte* ps = src)
        fixed (byte* pd = dst)
        {
            if (elementSize == 4 && Avx.IsSupported)
            {
                for (long b = 0; b < batch; b++)
                    Transpose2DFloat((float*)(ps + b * plane), (float*)(pd + b * plane), rows, cols);
            }
            else
            {
                for (long b = 0; b < batch; b++)
                    Transpose2DScalar(ps + b * plane, pd + b * plane, rows, cols, elementSize);
            }
        }
    }

    /// <summary>
    /// F32 矩阵转置：<see cref="BlockEdge"/> 分块（控 cache 抖动）+ 8×8 AVX 块。
    /// </summary>
    private static unsafe void Transpose2DFloat(float* src, float* dst, int rows, int cols)
    {
        for (int i0 = 0; i0 < rows; i0 += BlockEdge)
        {
            int iMax = Math.Min(i0 + BlockEdge, rows);

            for (int j0 = 0; j0 < cols; j0 += BlockEdge)
            {
                int jMax = Math.Min(j0 + BlockEdge, cols);

                int i = i0;
                for (; i + 8 <= iMax; i += 8)
                {
                    int j = j0;
                    for (; j + 8 <= jMax; j += 8)
                        Block8x8(src + (long)i * cols + j, cols, dst + (long)j * rows + i, rows);

                    for (; j < jMax; j++)
                        for (int ii = i; ii < i + 8; ii++)
                            dst[(long)j * rows + ii] = src[(long)ii * cols + j];
                }

                for (; i < iMax; i++)
                    for (int j = j0; j < jMax; j++)
                        dst[(long)j * rows + i] = src[(long)i * cols + j];
            }
        }
    }

    /// <summary>
    /// 8×8 F32 转置（<paramref name="s"/> = 8 行、行步长 <paramref name="ss"/>；
    /// <paramref name="d"/> = 8 行、行步长 <paramref name="ds"/>）。
    /// <para>unpack → shuffle → permute2x128 三段，8 个 256 位寄存器全程在寄存器里换手。</para>
    /// </summary>
    private static unsafe void Block8x8(float* s, int ss, float* d, int ds)
    {
        var r0 = Avx.LoadVector256(s + 0 * ss);
        var r1 = Avx.LoadVector256(s + 1 * ss);
        var r2 = Avx.LoadVector256(s + 2 * ss);
        var r3 = Avx.LoadVector256(s + 3 * ss);
        var r4 = Avx.LoadVector256(s + 4 * ss);
        var r5 = Avx.LoadVector256(s + 5 * ss);
        var r6 = Avx.LoadVector256(s + 6 * ss);
        var r7 = Avx.LoadVector256(s + 7 * ss);

        var t0 = Avx.UnpackLow(r0, r1);
        var t1 = Avx.UnpackHigh(r0, r1);
        var t2 = Avx.UnpackLow(r2, r3);
        var t3 = Avx.UnpackHigh(r2, r3);
        var t4 = Avx.UnpackLow(r4, r5);
        var t5 = Avx.UnpackHigh(r4, r5);
        var t6 = Avx.UnpackLow(r6, r7);
        var t7 = Avx.UnpackHigh(r6, r7);

        var u0 = Avx.Shuffle(t0, t2, 0x44);
        var u1 = Avx.Shuffle(t0, t2, 0xEE);
        var u2 = Avx.Shuffle(t1, t3, 0x44);
        var u3 = Avx.Shuffle(t1, t3, 0xEE);
        var u4 = Avx.Shuffle(t4, t6, 0x44);
        var u5 = Avx.Shuffle(t4, t6, 0xEE);
        var u6 = Avx.Shuffle(t5, t7, 0x44);
        var u7 = Avx.Shuffle(t5, t7, 0xEE);

        Avx.Store(d + 0 * ds, Avx.Permute2x128(u0, u4, 0x20));
        Avx.Store(d + 1 * ds, Avx.Permute2x128(u1, u5, 0x20));
        Avx.Store(d + 2 * ds, Avx.Permute2x128(u2, u6, 0x20));
        Avx.Store(d + 3 * ds, Avx.Permute2x128(u3, u7, 0x20));
        Avx.Store(d + 4 * ds, Avx.Permute2x128(u0, u4, 0x31));
        Avx.Store(d + 5 * ds, Avx.Permute2x128(u1, u5, 0x31));
        Avx.Store(d + 6 * ds, Avx.Permute2x128(u2, u6, 0x31));
        Avx.Store(d + 7 * ds, Avx.Permute2x128(u3, u7, 0x31));
    }

    /// <summary>非 F32（或 AVX 不可用）时的分块转置。</summary>
    private static unsafe void Transpose2DScalar(byte* src, byte* dst, int rows, int cols, int elementSize)
    {
        for (int i0 = 0; i0 < rows; i0 += BlockEdge)
        {
            int iMax = Math.Min(i0 + BlockEdge, rows);

            for (int j0 = 0; j0 < cols; j0 += BlockEdge)
            {
                int jMax = Math.Min(j0 + BlockEdge, cols);

                for (int i = i0; i < iMax; i++)
                {
                    byte* srow = src + (long)i * cols * elementSize;
                    for (int j = j0; j < jMax; j++)
                        Unsafe.CopyBlockUnaligned(dst + ((long)j * rows + i) * elementSize,
                            srow + (long)j * elementSize, (uint)elementSize);
                }
            }
        }
    }

    /// <summary>兜底：任意 perm 的逐元素拷贝，输入偏移用进位法增量维护。</summary>
    private static unsafe void GenericCopy(
        ReadOnlySpan<byte> src,
        Span<byte> dst,
        ReadOnlySpan<int> inDims,
        ReadOnlySpan<int> perm,
        int elementSize,
        long total)
    {
        int rank = inDims.Length;

        Span<long> inStride = stackalloc long[rank];
        ComputeStrides(inDims, inStride);

        Span<int> outDims = stackalloc int[rank];
        Span<long> step = stackalloc long[rank];
        for (int i = 0; i < rank; i++)
        {
            outDims[i] = inDims[perm[i]];
            step[i] = inStride[perm[i]];
        }

        Span<int> coord = stackalloc int[rank];

        fixed (byte* ps = src)
        fixed (byte* pd = dst)
        {
            long inOff = 0;
            long outOff = 0;

            for (long n = 0; n < total; n++)
            {
                Unsafe.CopyBlockUnaligned(pd + outOff, ps + inOff * elementSize, (uint)elementSize);
                outOff += elementSize;

                for (int i = rank - 1; i >= 0; i--)
                {
                    if (++coord[i] < outDims[i])
                    {
                        inOff += step[i];
                        break;
                    }

                    inOff -= (long)(outDims[i] - 1) * step[i];
                    coord[i] = 0;
                }
            }
        }
    }
}
