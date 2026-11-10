using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

// AvxVnni 在 .NET 里仍标注为「预览」API（CA2252）；这里一律靠运行期 IsSupported 守卫，属有意使用。
#pragma warning disable CA2252

namespace OnnxSharp.Kernels;

/// <summary>int8 GEMM 的指令集档位（运行时探测）。</summary>
public enum Int8KernelTier
{
    /// <summary>纯标量：正确性 oracle，也是没有 int8 SIMD 时的兜底。</summary>
    Scalar = 0,

    /// <summary>
    /// <c>vpmovzxbw</c> + <c>vpmovsxbw</c> + <c>vpmaddwd</c>（512 位）：把 u8 激活与 s8 权重都扩到 i16
    /// 再点积。**精确、不饱和**（<c>vpmaddwd</c> 累加进 i32）；这是 OpenCV 无 VNNI 时的同款做法
    /// （<c>v_dotprod_expand(s8,s8)</c>）。每 64 个 MAC 约 5 条指令。
    /// </summary>
    Avx512Bw = 1,

    /// <summary><c>vpdpbusd</c>（.NET 的 <see cref="AvxVnni"/> 只暴露 256 位，且仅 Intel 12 代+）。</summary>
    AvxVnni = 2,
}

/// <summary>int8 内核的档位探测（静态构造里算一次）。</summary>
public static class Int8Kernel
{
    /// <summary>
    /// <c>AvxVnni</c>（Intel 12 代+）默认启用；**AVX-512BW 的 i16 展开路径默认关闭** ——
    /// 实测它比「权重展开成 F32 + F32 GEMM」慢 4 倍（`plan.md §M10-A`：146~225 vs 605~789 GFLOPS），
    /// 只在显式设置 <c>ONNXSHARP_INT8_TIER=avx512bw</c> 时启用（对照和回归用）。
    /// </summary>
    private static readonly Int8KernelTier s_tier = Detect();

    private static Int8KernelTier Detect()
    {
        if (AvxVnni.IsSupported)
            return Int8KernelTier.AvxVnni;

        string? requested = Environment.GetEnvironmentVariable("ONNXSHARP_INT8_TIER");
        if (Avx512BW.IsSupported && string.Equals(requested, "avx512bw", StringComparison.OrdinalIgnoreCase))
            return Int8KernelTier.Avx512Bw;

        return Int8KernelTier.Scalar;
    }

    public static Int8KernelTier Tier => s_tier;

    public static string Describe() => s_tier switch
    {
        Int8KernelTier.AvxVnni => "avx-vnni (vpdpbusd, 256-bit)",
        Int8KernelTier.Avx512Bw => "avx512bw (vpmovzx/sxbw + vpmaddwd, 512-bit)",
        _ => "scalar",
    };
}

/// <summary>
/// 打包后的 int8 权重（<see cref="Int8Gemm"/> 用）。
/// <para>
/// 布局：<c>[kBlock][nPadded][4]</c>，每列 4 个连续 k 值放一起 —— 正好喂
/// <c>vpdpbusd</c> 的「每 int32 lane 吃掉 4 个字节」语义。
/// </para>
/// <para>
/// <b>为什么存 <c>b - 128</c> 而不是 <c>b - b_zero_point</c></b>：<c>vpdpbusd</c> 的第二个操作数必须是
/// <b>s8</b>，而 ONNX 的量化权重通常是 <b>u8</b>（0..255）。先统一减 128 落在 [-128,127]（不丢精度），
/// 再在 epilogue 里用「行和 × 常数」把 128 与 zero_point 一起补回来：
/// <c>C = Σ a·b'' + (128 - bzp)·Σa − aza·Σb + aza·bzp·K</c>。
/// </para>
/// </summary>
public sealed class PackedInt8Weights
{
    private PackedInt8Weights(byte[] data, int[] columnSums, int[] bZeroPoints, int zeroPointOffset,
        int k, int n, int kPadded, int nPadded)
    {
        Data = data;
        ColumnSums = columnSums;
        BZeroPoints = bZeroPoints;
        ZeroPointOffset = zeroPointOffset;
        K = k;
        N = n;
        KPadded = kPadded;
        NPadded = nPadded;
    }

    /// <summary>打包后的 s8 字节（按 <c>byte</c> 存位模式，取用时 <c>AsSByte()</c>）。</summary>
    public byte[] Data { get; }

    /// <summary>每列 <c>Σ_k b</c>（原始 u8/s8 值，padding 为 0）。</summary>
    public int[] ColumnSums { get; }

    /// <summary>每列的 <c>b_zero_point</c>（per-tensor 时所有元素相同）。</summary>
    public int[] BZeroPoints { get; }

    /// <summary>打包时统一减掉的常数（u8 权重为 128，s8 权重为 0）。</summary>
    public int ZeroPointOffset { get; }

    public int K { get; }

    public int N { get; }

    public int KPadded { get; }

    public int NPadded { get; }

    /// <summary>unsigned（u8）权重：减 128 后落在 s8 范围。</summary>
    public static PackedInt8Weights Pack(ReadOnlySpan<byte> weights, int k, int n, ReadOnlySpan<int> zeroPoints)
        => PackCore(weights, k, n, zeroPoints, offset: 128, signed: false);

    /// <summary>signed（s8）权重：直接减 zero point。</summary>
    public static PackedInt8Weights PackSigned(ReadOnlySpan<sbyte> weights, int k, int n, ReadOnlySpan<int> zeroPoints)
    {
        var bytes = new byte[weights.Length];
        for (int i = 0; i < weights.Length; i++)
            bytes[i] = unchecked((byte)weights[i]);
        return PackCore(bytes, k, n, zeroPoints, offset: 0, signed: true);
    }

    private static PackedInt8Weights PackCore(
        ReadOnlySpan<byte> weights, int k, int n, ReadOnlySpan<int> zeroPoints, int offset, bool signed)
    {
        if (k <= 0 || n <= 0)
            throw new ArgumentException($"invalid weight shape [k={k}, n={n}]");
        if (weights.Length < (long)k * n)
            throw new ArgumentException($"weight buffer has {weights.Length} values, {k}x{n} needed");

        int kPadded = (k + 3) / 4 * 4;
        int nPadded = (n + 15) / 16 * 16;

        var data = new byte[(long)kPadded / 4 * nPadded * 4];
        var columnSums = new int[nPadded];
        var bZero = new int[nPadded];
        int lastZero = zeroPoints.IsEmpty ? 0 : zeroPoints[^1];

        for (int col = 0; col < nPadded; col++)
        {
            int zp = col < zeroPoints.Length ? zeroPoints[col] : lastZero;
            bZero[col] = zp;

            int sum = 0;
            for (int kb = 0; kb < kPadded / 4; kb++)
            {
                long baseIndex = ((long)kb * nPadded + col) * 4;
                for (int j = 0; j < 4; j++)
                {
                    int kk = kb * 4 + j;
                    int value = 0;
                    if (kk < k && col < n)
                    {
                        byte raw = weights[kk * n + col];
                        value = signed ? unchecked((sbyte)raw) : raw;
                        sum += value;
                    }

                    data[baseIndex + j] = unchecked((byte)(sbyte)(value - offset));
                }
            }

            columnSums[col] = sum;
        }

        return new PackedInt8Weights(data, columnSums, bZero, offset, k, n, kPadded, nPadded);
    }
}

/// <summary>
/// int8 矩阵乘：<c>C[m,n] = Σ_k (A[m,k] - a_zero_point) · (B[k,n] - b_zero_point)</c>，输出 int32。
/// <para>
/// A 是 u8 激活（<c>DynamicQuantizeLinear</c> 的输出），B 走 <see cref="PackedInt8Weights"/>。
/// 主路径用 <c>vpdpbusd</c>（VNNI），没有 VNNI 时退回标量实现 —— 两条路径的 int32 结果**逐位一致**。
/// </para>
/// <para>
/// <b>zero point 的处理</b>：主循环里只算 <c>Σ a·b''</c>（<c>b'' = b - 打包偏移</c>），
/// 其余三项在写出时一次补上：<c>(offset − bzp)·Σa − aza·Σb + aza·bzp·K</c>。
/// 这样内层循环只有一条 <c>vpdpbusd</c>，不需要逐元素减 zero point。
/// </para>
/// </summary>
public static class Int8Gemm
{
    /// <summary>并行阈值：m·k·n 超过该值时按行并行（与 GEMM 的口径一致）。</summary>
    private const long ParallelThreshold = 1L << 22;

    /// <summary>算 <c>C[m,n]</c>；按当前机器能力自动选 VNNI 或标量。</summary>
    public static void MultiplyInteger(
        ReadOnlySpan<byte> a,
        PackedInt8Weights packed,
        Span<int> c,
        int m,
        int aZeroPoint = 0)
    {
        ArgumentNullException.ThrowIfNull(packed);

        int k = packed.K;
        int n = packed.N;
        if (m < 0 || a.Length < (long)m * k)
            throw new ArgumentException($"A has {a.Length} values, [{m},{k}] needed", nameof(a));
        if (c.Length < (long)m * n)
            throw new ArgumentException($"C has {c.Length} values, [{m},{n}] needed", nameof(c));
        if (m == 0)
            return;

        if (Int8Kernel.Tier == Int8KernelTier.AvxVnni && AvxVnni.IsSupported)
            MultiplyVnni(a, packed, c, m, aZeroPoint);
        else if (Int8Kernel.Tier == Int8KernelTier.Avx512Bw && Avx512BW.IsSupported)
            MultiplyAvx512Bw(a, packed, c, m, aZeroPoint);
        else
            MultiplyIntegerScalar(a, packed, c, m, aZeroPoint);
    }

    /// <summary>标量参考实现（正确性 oracle，也是无 int8 SIMD 时的兜底）。</summary>
    public static void MultiplyIntegerScalar(
        ReadOnlySpan<byte> a,
        PackedInt8Weights packed,
        Span<int> c,
        int m,
        int aZeroPoint = 0)
    {
        int k = packed.K;
        int n = packed.N;

        for (int row = 0; row < m; row++)
        {
            var aRow = a.Slice(row * k, k);
            long cBase = (long)row * n;

            for (int col = 0; col < n; col++)
            {
                int zp = packed.BZeroPoints[col];
                long acc = 0;

                for (int kk = 0; kk < k; kk++)
                    acc += (aRow[kk] - aZeroPoint) * (WeightValue(packed, kk, col) - zp);

                c[(int)(cBase + col)] = (int)acc;
            }
        }
    }

    /// <summary>从打包布局里取回原始权重值（只给标量路径与测试用）。</summary>
    internal static int WeightValue(PackedInt8Weights packed, int k, int n)
    {
        long index = ((long)(k / 4) * packed.NPadded + n) * 4 + (k % 4);
        int packedValue = unchecked((sbyte)packed.Data[index]);
        return packedValue + packed.ZeroPointOffset;
    }

    private static unsafe void MultiplyVnni(
        ReadOnlySpan<byte> a,
        PackedInt8Weights packed,
        Span<int> c,
        int m,
        int aZeroPoint)
    {
        int k = packed.K;
        int n = packed.N;
        int kPadded = packed.KPadded;
        int nPadded = packed.NPadded;
        int kBlocks = kPadded / 4;

        byte[] data = packed.Data;
        int[] columnSums = packed.ColumnSums;
        int[] bZeroPoints = packed.BZeroPoints;
        int offset = packed.ZeroPointOffset;

        fixed (byte* aPointer = a)
        fixed (int* cPointer = c)
        fixed (byte* dataPointer = data)
        {
            nint aAddress = (nint)aPointer;
            nint cAddress = (nint)cPointer;
            nint dataAddress = (nint)dataPointer;

            void Row(int row)
            {
                byte* aRow = (byte*)aAddress + (long)row * k;
                int* cRow = (int*)cAddress + (long)row * n;
                byte* weights = (byte*)dataAddress;

                Span<byte> padded = kPadded <= 4096 ? stackalloc byte[kPadded] : new byte[kPadded];
                int rowSum = 0;
                for (int i = 0; i < k; i++)
                {
                    byte value = aRow[i];
                    padded[i] = value;
                    rowSum += value;
                }

                for (int i = k; i < kPadded; i++)
                    padded[i] = 0;

                for (int colBlock = 0; colBlock < n; colBlock += 8)
                {
                    var acc = Vector256<int>.Zero;

                    for (int kb = 0; kb < kBlocks; kb++)
                    {
                        uint dword = Unsafe.ReadUnaligned<uint>(ref padded[kb * 4]);
                        var aVec = Vector256.Create(dword).AsByte();

                        byte* bp = weights + ((long)kb * nPadded + colBlock) * 4;
                        var bVec = Vector256.LoadUnsafe(ref *bp).AsSByte();

                        acc = AvxVnni.MultiplyWideningAndAdd(acc, aVec, bVec);
                    }

                    int lanes = Math.Min(8, n - colBlock);
                    for (int j = 0; j < lanes; j++)
                    {
                        int col = colBlock + j;
                        int zp = bZeroPoints[col];
                        cRow[col] = acc.GetElement(j)
                                    + (offset - zp) * rowSum
                                    - aZeroPoint * columnSums[col]
                                    + aZeroPoint * zp * k;
                    }
                }
            }

            if ((long)m * k * n >= ParallelThreshold)
                Threading.ParallelRange.For(m, (begin, end) =>
                {
                    for (int row = begin; row < end; row++)
                        Row(row);
                });
            else
                for (int row = 0; row < m; row++)
                    Row(row);
        }
    }

    /// <summary>
    /// AVX-512BW 路径：<c>vpmovzxbw</c>(A) + <c>vpmovsxbw</c>(B) 扩到 i16，再 <c>vpmaddwd</c> 累加进 i32。
    /// <para>
    /// <b>为什么不用 <c>vpmaddubsw</c></b>：它把「一对乘积」累加进 i16，而
    /// <c>255×128×2 = 65280 &gt; 32767</c> 会饱和（OpenCV 也因此绕开它）。扩到 i16 再
    /// <c>vpmaddwd</c> 没这个问题 —— 累加器是 i32，结果**精确**。
    /// </para>
    /// <para>
    /// 布局沿用 <see cref="PackedInt8Weights"/> 的 <c>[kBlock][nPadded][4]</c>：一次读 64 字节
    /// = 16 列 × 4 个 k，正好两条 zmm（各 32 个 i16），A 的 4 个字节广播后两条共用。
    /// </para>
    /// </summary>
    private static unsafe void MultiplyAvx512Bw(
        ReadOnlySpan<byte> a,
        PackedInt8Weights packed,
        Span<int> c,
        int m,
        int aZeroPoint)
    {
        int k = packed.K;
        int n = packed.N;
        int kPadded = packed.KPadded;
        int nPadded = packed.NPadded;
        int kBlocks = kPadded / 4;

        byte[] data = packed.Data;
        int[] columnSums = packed.ColumnSums;
        int[] bZeroPoints = packed.BZeroPoints;
        int offset = packed.ZeroPointOffset;

        // 零点校正拆成「随行变化」与「每次调用一次」两部分：
        //   Σ(a-azp)(b-bzp) = Σa·b'' + (offset-bzp)·Σa - azp·Σb + azp·bzp·k
        var first = new int[nPadded];
        var second = new int[nPadded];
        for (int col = 0; col < nPadded; col++)
        {
            int bzp = bZeroPoints[col];
            first[col] = offset - bzp;
            second[col] = aZeroPoint * (bzp * k - columnSums[col]);
        }

        fixed (byte* aPointer = a)
        fixed (int* cPointer = c)
        fixed (byte* dataPointer = data)
        fixed (int* firstPointer = first)
        fixed (int* secondPointer = second)
        {
            nint aAddress = (nint)aPointer;
            nint cAddress = (nint)cPointer;
            nint dataAddress = (nint)dataPointer;
            nint firstAddress = (nint)firstPointer;
            nint secondAddress = (nint)secondPointer;

            // 一个线程负责 [colBegin, colEnd) 这些 16 列面板 × [rowBegin, rowEnd) 这些行。
            // 列切分（默认）让每个线程只读自己那段 B —— 按行切的话 16 个线程会把整块 B 各流一遍。
            void Compute(int colBegin, int colEnd, int rowBegin, int rowEnd)
            {
                Span<byte> padded = kPadded <= 4096 ? stackalloc byte[kPadded] : new byte[kPadded];
                Span<int> lanes = stackalloc int[16];

                fixed (byte* paddedPointer = padded)
                {
                    for (int row = rowBegin; row < rowEnd; row++)
                    {
                        byte* aRow = (byte*)aAddress + (long)row * k;
                        int* cRow = (int*)cAddress + (long)row * n + (long)colBegin * 16;

                        int rowSum = 0;
                        for (int i = 0; i < k; i++)
                        {
                            byte value = aRow[i];
                            padded[i] = value;
                            rowSum += value;
                        }

                        for (int i = k; i < kPadded; i++)
                            padded[i] = 0;

                        var rowSumVector = Vector512.Create(rowSum);

                        int* cBlock = cRow;
                        for (int colBlock = colBegin; colBlock < colEnd; colBlock++, cBlock += 16)
                        {
                            int column = colBlock * 16;
                            var accLow = Vector512<int>.Zero;
                            var accHigh = Vector512<int>.Zero;

                            for (int kb = 0; kb < kBlocks; kb++)
                            {
                                uint dword = Unsafe.ReadUnaligned<uint>(ref paddedPointer[kb * 4]);
                                var a16 = Avx512BW.ConvertToVector512Int16(Vector512.Create(dword).AsByte().GetLower());

                                byte* bp = (byte*)dataAddress + ((long)kb * nPadded + column) * 4;
                                var bLow = Avx512BW.ConvertToVector512Int16(Vector256.LoadUnsafe(ref *(sbyte*)bp));
                                var bHigh = Avx512BW.ConvertToVector512Int16(Vector256.LoadUnsafe(ref *(sbyte*)(bp + 32)));

                                accLow = Avx512F.Add(accLow, Avx512BW.MultiplyAddAdjacent(a16, bLow));
                                accHigh = Avx512F.Add(accHigh, Avx512BW.MultiplyAddAdjacent(a16, bHigh));
                            }

                            // 每列 4 个 i16 = 2 个 dword（k0k1 / k2k3）→ 折叠成每列 1 个和
                            var sums = Vector512.Create(FoldPairs(accLow), FoldPairs(accHigh));

                            var corrections = Avx512F.Add(
                                Avx512F.MultiplyLow(
                                    Vector512.LoadUnsafe(ref *(int*)(firstAddress + (long)column * sizeof(int))),
                                    rowSumVector),
                                Vector512.LoadUnsafe(ref *(int*)(secondAddress + (long)column * sizeof(int))));

                            var total = Avx512F.Add(sums, corrections);

                            int width = n - column < 16 ? n - column : 16;
                            if (width == 16)
                            {
                                total.StoreUnsafe(ref *cBlock);
                            }
                            else
                            {
                                total.CopyTo(lanes);
                                for (int j = 0; j < width; j++)
                                    cBlock[j] = lanes[j];
                            }
                        }
                    }
                }
            }

            int colBlocks = (n + 15) / 16;
            int threads = Threading.ParallelRange.Threads;

            if ((long)m * k * n < ParallelThreshold)
            {
                Compute(0, colBlocks, 0, m);
            }
            else if (colBlocks >= threads)
            {
                // 列切分：每个线程只读自己那段权重面板
                int basePer = colBlocks / threads;
                int rem = colBlocks % threads;

                Threading.ParallelRange.For(threads, (begin, end) =>
                {
                    for (int t = begin; t < end; t++)
                    {
                        int b0 = t * basePer + Math.Min(t, rem);
                        int b1 = b0 + basePer + (t < rem ? 1 : 0);
                        if (b0 < b1)
                            Compute(b0, b1, 0, m);
                    }
                });
            }
            else
            {
                // 列块不够分（n 很小）：退回按行并行
                Threading.ParallelRange.For(m, (begin, end) => Compute(0, colBlocks, begin, end));
            }
        }
    }

    // 交换相邻 dword（把「每列两个部分和」并到一起）与取偶数 lane
    private static readonly Vector512<int> SwapPairs =
        Vector512.Create(1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14);

    private static readonly Vector512<int> EvenLanes =
        Vector512.Create(0, 2, 4, 6, 8, 10, 12, 14, 0, 2, 4, 6, 8, 10, 12, 14);

    /// <summary>把「每列 2 个 dword」的累加器压成「8 列、每列一个和」（放在低 8 个 lane）。</summary>
    private static Vector256<int> FoldPairs(Vector512<int> accumulator)
    {
        var summed = Avx512F.Add(accumulator, Avx512F.PermuteVar16x32(accumulator, SwapPairs));
        return Avx512F.PermuteVar16x32(summed, EvenLanes).GetLower();
    }
}
