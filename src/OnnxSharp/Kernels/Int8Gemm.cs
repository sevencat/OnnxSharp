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
            MultiplyVnniDispatch(a, packed, c, m, aZeroPoint);
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

    /// <summary>
    /// VNNI 调度：按 4 行块跑 <see cref="MultiplyVnniFourRows"/>，剩下的 <c>m % 4</c> 行走单行内核。
    /// <para>
    /// 4 行块要求把 4 行 A 同时打包进栈（<c>4 × kPadded</c> 字节），超过 32 KB 时退回逐行。
    /// </para>
    /// </summary>
    private static void MultiplyVnniDispatch(
        ReadOnlySpan<byte> a,
        PackedInt8Weights packed,
        Span<int> c,
        int m,
        int aZeroPoint)
    {
        int blockRows = packed.KPadded <= 8192 ? 4 : 1;
        int blocks = blockRows == 4 ? m / 4 : 0;

        if (blocks == 0)
        {
            MultiplyVnni(a, packed, c, m, aZeroPoint);
            return;
        }

        MultiplyVnniFourRows(a, packed, c, m, blocks, aZeroPoint);

        int tailRows = m - blocks * 4;
        if (tailRows > 0)
        {
            MultiplyVnni(
                a.Slice(blocks * 4 * packed.K),
                packed,
                c.Slice(blocks * 4 * packed.N),
                tailRows,
                aZeroPoint);
        }
    }

    /// <summary>
    /// 4 行 × 24 列的输出块 —— 把 MLAS <c>QgemmU8X8KernelAvx512Core.asm</c> 的「6 行 × 48 列」按
    /// 256 位寄存器宽度缩下来的形状：12 个累加器 + 3 条 B + 1 条 A，刚好 16 个 ymm。
    /// <para>
    /// 一个 quad（4 个 k）：3 次 B 加载 + 4 次 A 广播 + 12 条 <c>vpdpbusd</c>，共 384 MAC →
    /// <b>20.2 MAC/指令</b>；而单行 32 列（4 加载 + 1 广播 + 4 dpbusd）只有 14.2。
    /// B 的加载对 4 行复用、A 的广播对 3 个列组复用，且每个 kBlock 内的 96 字节 B 是连续的；
    /// 12 条累加链彼此独立，足以盖住 <c>vpdpbusd</c> 约 4 cycle 的延迟，因此不需要再做 K 展开。
    /// </para>
    /// </summary>
    private static unsafe void MultiplyVnniFourRows(
        ReadOnlySpan<byte> a,
        PackedInt8Weights packed,
        Span<int> c,
        int m,
        int blocks,
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
        fixed (int* columnSumPointer = columnSums)
        fixed (int* bZeroPointPointer = bZeroPoints)
        {
            nint aAddress = (nint)aPointer;
            nint cAddress = (nint)cPointer;
            nint dataAddress = (nint)dataPointer;
            nint columnSumAddress = (nint)columnSumPointer;
            nint bZeroPointAddress = (nint)bZeroPointPointer;

            void Block(int rowBase)
            {
                byte* weights = (byte*)dataAddress;
                Span<byte> padded = stackalloc byte[kPadded * 4];
                Span<int> rowSums = stackalloc int[4];

                for (int r = 0; r < 4; r++)
                {
                    byte* source = (byte*)aAddress + (long)(rowBase + r) * k;
                    int start = r * kPadded;
                    int sum = 0;
                    for (int i = 0; i < k; i++)
                    {
                        byte value = source[i];
                        padded[start + i] = value;
                        sum += value;
                    }
                    for (int i = k; i < kPadded; i++)
                        padded[start + i] = 0;
                    rowSums[r] = sum;
                }

                fixed (byte* paddedPointer = padded)
                {
                    int colBlock = 0;
                    for (; colBlock + 24 <= n; colBlock += 24)
                    {
                        var a00 = Vector256<int>.Zero;
                        var a01 = Vector256<int>.Zero;
                        var a02 = Vector256<int>.Zero;
                        var a10 = Vector256<int>.Zero;
                        var a11 = Vector256<int>.Zero;
                        var a12 = Vector256<int>.Zero;
                        var a20 = Vector256<int>.Zero;
                        var a21 = Vector256<int>.Zero;
                        var a22 = Vector256<int>.Zero;
                        var a30 = Vector256<int>.Zero;
                        var a31 = Vector256<int>.Zero;
                        var a32 = Vector256<int>.Zero;

                        for (int kb = 0; kb < kBlocks; kb++)
                        {
                            byte* bp = weights + ((long)kb * nPadded + colBlock) * 4;
                            byte* q = paddedPointer + (long)kb * 4;

                            var b0 = Vector256.LoadUnsafe(ref *(sbyte*)bp);
                            var b1 = Vector256.LoadUnsafe(ref *(sbyte*)(bp + 32));
                            var b2 = Vector256.LoadUnsafe(ref *(sbyte*)(bp + 64));

                            var v = Vector256.Create(Unsafe.ReadUnaligned<uint>(ref *(byte*)q)).AsByte();
                            a00 = AvxVnni.MultiplyWideningAndAdd(a00, v, b0);
                            a01 = AvxVnni.MultiplyWideningAndAdd(a01, v, b1);
                            a02 = AvxVnni.MultiplyWideningAndAdd(a02, v, b2);

                            v = Vector256.Create(Unsafe.ReadUnaligned<uint>(ref *(byte*)(q + kPadded))).AsByte();
                            a10 = AvxVnni.MultiplyWideningAndAdd(a10, v, b0);
                            a11 = AvxVnni.MultiplyWideningAndAdd(a11, v, b1);
                            a12 = AvxVnni.MultiplyWideningAndAdd(a12, v, b2);

                            v = Vector256.Create(Unsafe.ReadUnaligned<uint>(ref *(byte*)(q + 2 * kPadded))).AsByte();
                            a20 = AvxVnni.MultiplyWideningAndAdd(a20, v, b0);
                            a21 = AvxVnni.MultiplyWideningAndAdd(a21, v, b1);
                            a22 = AvxVnni.MultiplyWideningAndAdd(a22, v, b2);

                            v = Vector256.Create(Unsafe.ReadUnaligned<uint>(ref *(byte*)(q + 3 * kPadded))).AsByte();
                            a30 = AvxVnni.MultiplyWideningAndAdd(a30, v, b0);
                            a31 = AvxVnni.MultiplyWideningAndAdd(a31, v, b1);
                            a32 = AvxVnni.MultiplyWideningAndAdd(a32, v, b2);
                        }

                        int* crow = (int*)cAddress + (long)rowBase * n + colBlock;

                        StoreVector(a00, crow, bZeroPointAddress, columnSumAddress, colBlock, rowSums[0], aZeroPoint, offset, k);
                        StoreVector(a01, crow + 8, bZeroPointAddress, columnSumAddress, colBlock + 8, rowSums[0], aZeroPoint, offset, k);
                        StoreVector(a02, crow + 16, bZeroPointAddress, columnSumAddress, colBlock + 16, rowSums[0], aZeroPoint, offset, k);

                        crow += n;
                        StoreVector(a10, crow, bZeroPointAddress, columnSumAddress, colBlock, rowSums[1], aZeroPoint, offset, k);
                        StoreVector(a11, crow + 8, bZeroPointAddress, columnSumAddress, colBlock + 8, rowSums[1], aZeroPoint, offset, k);
                        StoreVector(a12, crow + 16, bZeroPointAddress, columnSumAddress, colBlock + 16, rowSums[1], aZeroPoint, offset, k);

                        crow += n;
                        StoreVector(a20, crow, bZeroPointAddress, columnSumAddress, colBlock, rowSums[2], aZeroPoint, offset, k);
                        StoreVector(a21, crow + 8, bZeroPointAddress, columnSumAddress, colBlock + 8, rowSums[2], aZeroPoint, offset, k);
                        StoreVector(a22, crow + 16, bZeroPointAddress, columnSumAddress, colBlock + 16, rowSums[2], aZeroPoint, offset, k);

                        crow += n;
                        StoreVector(a30, crow, bZeroPointAddress, columnSumAddress, colBlock, rowSums[3], aZeroPoint, offset, k);
                        StoreVector(a31, crow + 8, bZeroPointAddress, columnSumAddress, colBlock + 8, rowSums[3], aZeroPoint, offset, k);
                        StoreVector(a32, crow + 16, bZeroPointAddress, columnSumAddress, colBlock + 16, rowSums[3], aZeroPoint, offset, k);
                    }

                    // 列尾巴：先按整条 8 列向量收（n 常见 320/1280，24 列块后正好剩 8），
                    // 最后不足 8 列才逐列。
                    for (; colBlock + 8 <= n; colBlock += 8)
                    {
                        var b0 = Vector256<int>.Zero;
                        var b1 = Vector256<int>.Zero;
                        var b2 = Vector256<int>.Zero;
                        var b3 = Vector256<int>.Zero;

                        for (int kb = 0; kb < kBlocks; kb++)
                        {
                            byte* bp = weights + ((long)kb * nPadded + colBlock) * 4;
                            byte* q = paddedPointer + (long)kb * 4;
                            var bv = Vector256.LoadUnsafe(ref *(sbyte*)bp);

                            var v = Vector256.Create(Unsafe.ReadUnaligned<uint>(ref *(byte*)q)).AsByte();
                            b0 = AvxVnni.MultiplyWideningAndAdd(b0, v, bv);
                            v = Vector256.Create(Unsafe.ReadUnaligned<uint>(ref *(byte*)(q + kPadded))).AsByte();
                            b1 = AvxVnni.MultiplyWideningAndAdd(b1, v, bv);
                            v = Vector256.Create(Unsafe.ReadUnaligned<uint>(ref *(byte*)(q + 2 * kPadded))).AsByte();
                            b2 = AvxVnni.MultiplyWideningAndAdd(b2, v, bv);
                            v = Vector256.Create(Unsafe.ReadUnaligned<uint>(ref *(byte*)(q + 3 * kPadded))).AsByte();
                            b3 = AvxVnni.MultiplyWideningAndAdd(b3, v, bv);
                        }

                        int* crow = (int*)cAddress + (long)rowBase * n + colBlock;
                        StoreVector(b0, crow, bZeroPointAddress, columnSumAddress, colBlock, rowSums[0], aZeroPoint, offset, k);
                        StoreVector(b1, crow + n, bZeroPointAddress, columnSumAddress, colBlock, rowSums[1], aZeroPoint, offset, k);
                        StoreVector(b2, crow + 2 * n, bZeroPointAddress, columnSumAddress, colBlock, rowSums[2], aZeroPoint, offset, k);
                        StoreVector(b3, crow + 3 * n, bZeroPointAddress, columnSumAddress, colBlock, rowSums[3], aZeroPoint, offset, k);
                    }

                    // 列尾巴（< 8 列）：逐列算，避免越界写。仍然用向量指令，只取 lane 0。
                    for (; colBlock < n; colBlock++)
                    {
                        int zp = bZeroPoints[colBlock];
                        int columnSum = columnSums[colBlock];
                        int bias = aZeroPoint * (zp * k - columnSum);

                        for (int r = 0; r < 4; r++)
                        {
                            byte* q = paddedPointer + (long)r * kPadded;
                            int acc = 0;
                            for (int kb = 0; kb < kBlocks; kb++)
                            {
                                byte* bp = weights + ((long)kb * nPadded + colBlock) * 4;
                                var bVec = Vector256.LoadUnsafe(ref *(sbyte*)bp);
                                var aVec = Vector256.Create(Unsafe.ReadUnaligned<uint>(ref *(byte*)(q + kb * 4))).AsByte();
                                acc += AvxVnni.MultiplyWideningAndAdd(Vector256<int>.Zero, aVec, bVec).GetElement(0);
                            }

                            int* crow = (int*)cAddress + (long)(rowBase + r) * n + colBlock;
                            crow[0] = acc + (offset - zp) * rowSums[r] + bias;
                        }
                    }
                }
            }

            if ((long)m * k * n >= ParallelThreshold && blocks > 1)
                Threading.ParallelRange.ForEach(blocks, index => Block(index * 4));
            else
                for (int index = 0; index < blocks; index++)
                    Block(index * 4);
        }
    }

    /// <summary>
    /// 把一条 8 列（i32 lane）的累加器补上 zero-point 修正并按向量写回：
    /// <c>acc + (offset − bzp)·Σa − azp·Σb + azp·bzр·K</c>，后半段合并成
    /// <c>azp·(bzp·K − Σb)</c>，整体两次 <c>vpmulld</c> + 两次 <c>vpaddd</c>。
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void StoreVector(
        Vector256<int> acc,
        int* destination,
        nint bZeroPointAddress,
        nint columnSumAddress,
        int column,
        int rowSum,
        int aZeroPoint,
        int zeroPointOffset,
        int k)
    {
        var zeroPoints = Vector256.LoadUnsafe(ref *(int*)(bZeroPointAddress + (long)column * sizeof(int)));
        var columnSums = Vector256.LoadUnsafe(ref *(int*)(columnSumAddress + (long)column * sizeof(int)));

        var correction = Avx2.Add(
            Avx2.MultiplyLow(Vector256.Subtract(Vector256.Create(zeroPointOffset), zeroPoints), Vector256.Create(rowSum)),
            Avx2.MultiplyLow(
                Vector256.Create(aZeroPoint),
                Avx2.Subtract(Avx2.MultiplyLow(zeroPoints, Vector256.Create(k)), columnSums)));

        Avx2.Add(acc, correction).StoreUnsafe(ref *destination);
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

                // 收尾：把一条 8 列（i32 lane）的累加器写回 cRow，并补 zero-point 修正。
                void Store8(Vector256<int> acc, int colStart)
                {
                    int lanes = Math.Min(8, n - colStart);
                    for (int j = 0; j < lanes; j++)
                    {
                        int col = colStart + j;
                        int zp = bZeroPoints[col];
                        cRow[col] = acc.GetElement(j)
                                    + (offset - zp) * rowSum
                                    - aZeroPoint * columnSums[col]
                                    + aZeroPoint * zp * k;
                    }
                }

                int colBlock = 0;
                // 4 组 × 8 列一起做：同一 kBlock 的 a 广播只做一次，喂给 4 条 dpbusd，
                // 且 4 个累加器互相独立（vpdpbusd 延迟 ~4 cycle，单链会把 IPC 压到 1/4）。
                for (; colBlock + 32 <= n; colBlock += 32)
                {
                    var acc0 = Vector256<int>.Zero;
                    var acc1 = Vector256<int>.Zero;
                    var acc2 = Vector256<int>.Zero;
                    var acc3 = Vector256<int>.Zero;

                    for (int kb = 0; kb < kBlocks; kb++)
                    {
                        var aVec = Vector256.Create(Unsafe.ReadUnaligned<uint>(ref padded[kb * 4])).AsByte();
                        byte* bp = weights + ((long)kb * nPadded + colBlock) * 4;

                        acc0 = AvxVnni.MultiplyWideningAndAdd(acc0, aVec, Vector256.LoadUnsafe(ref *(sbyte*)bp).AsSByte());
                        acc1 = AvxVnni.MultiplyWideningAndAdd(acc1, aVec, Vector256.LoadUnsafe(ref *(sbyte*)(bp + 32)).AsSByte());
                        acc2 = AvxVnni.MultiplyWideningAndAdd(acc2, aVec, Vector256.LoadUnsafe(ref *(sbyte*)(bp + 64)).AsSByte());
                        acc3 = AvxVnni.MultiplyWideningAndAdd(acc3, aVec, Vector256.LoadUnsafe(ref *(sbyte*)(bp + 96)).AsSByte());
                    }

                    Store8(acc0, colBlock);
                    Store8(acc1, colBlock + 8);
                    Store8(acc2, colBlock + 16);
                    Store8(acc3, colBlock + 24);
                }

                for (; colBlock < n; colBlock += 8)
                {
                    var acc = Vector256<int>.Zero;
                    for (int kb = 0; kb < kBlocks; kb++)
                    {
                        var aVec = Vector256.Create(Unsafe.ReadUnaligned<uint>(ref padded[kb * 4])).AsByte();
                        byte* bp = weights + ((long)kb * nPadded + colBlock) * 4;
                        acc = AvxVnni.MultiplyWideningAndAdd(acc, aVec, Vector256.LoadUnsafe(ref *(sbyte*)bp).AsSByte());
                    }
                    Store8(acc, colBlock);
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
