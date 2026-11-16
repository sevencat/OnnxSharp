using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using OnnxSharp.Tensors;
using OnnxSharp.Threading;

namespace OnnxSharp.Kernels;

/// <summary>
/// NCHWc16 布局的 3×3 / stride 2 直接卷积（`planv2 §P7.7` 阶段 1）。
/// <para>
/// 布局：输入 <c>[C/16][H+1][W+1][16]</c>（左上各留 1 像素 halo，pad 处为 0），
/// 输出 <c>[C/16][OH][OW][16]</c>，权重 <c>[OC/16][IC/16][9][icLane][ocLane]</c>。
/// 同一个像素的 16 个通道连续 —— 一次 <c>zmm</c> 载入就是「16 个通道」，
/// 于是 16 个累加器可以同时压在寄存器里，不必像 <see cref="PackedConv"/> 那样
/// 先把感受野摊平成 <c>[K][16]</c> 面板。
/// </para>
/// <para>
/// 为什么这样能快：<see cref="PackedConv"/> 在 s2 层上的时间大半花在
/// <c>PackPanel</c>（每个面板 K×16 次跨行取样，步长 2 时还要 <c>vpermd</c> 抽取）。
/// 本内核的「打包」在权重侧一次性完成、输入侧由一次 reorder 摊到整条 conv 链，
/// 稳态下每次 FMA 只对应 0.6 次载入，瓶颈回到 FMA 端口。
/// </para>
/// <para>阶段 1 的适用范围刻意很窄：groups=1、3×3、stride 2、pad 1、dil 1、ic%16=0、oc%32=0。</para>
/// </summary>
internal static class NchwcConv
{
    /// <summary>通道块宽：1 个 <c>zmm</c>。</summary>
    public const int Block = 16;

    /// <summary>一次并行的输出通道块数（2 块 = 32 通道 = 16 个累加器的另一半）。</summary>
    public const int OcGroup = 2;

    /// <summary>一次并行的输出像素数。</summary>
    public const int Pixels = 8;

    /// <summary>调试开关：<c>false</c> 交回既有路径（<c>ONNXSHARP_NCHWC=0</c>），用于 A/B。</summary>
    public static bool Enabled { get; set; } =
        Environment.GetEnvironmentVariable("ONNXSHARP_NCHWC") != "0";

    /// <summary>本机是否具备 AVX-512（32 个 <c>zmm</c> 寄存器是本内核的前提）。</summary>
    public static bool IsSupported => Avx512F.IsSupported;

    /// <summary>
    /// 分阶段计时开关（<c>ONNXSHARP_NCHWC_PROF=1</c>）。
    /// <para>
    /// 关掉时 <see cref="TryConv"/> 里只剩几个 <c>Profile</c> 判断。开的时候要把
    /// pack / 内核 / unpack 三段的真实开销分开量 —— 「每层都付的 reorder 到底吃掉多少」
    /// 只有这一条路能答（微基准里的 reorder 是热缓冲下的乐观值）。
    /// </para>
    /// </summary>
    internal static readonly bool Profile =
        Environment.GetEnvironmentVariable("ONNXSHARP_NCHWC_PROF") == "1";

    private static long _packTicks;
    private static long _kernelTicks;
    private static long _unpackTicks;
    private static int _profileCalls;

    /// <summary>清空累计值（预热之后、正式计时之前调用，免得预热帧混进「每帧」）。</summary>
    internal static void ResetProfile()
    {
        Interlocked.Exchange(ref _packTicks, 0);
        Interlocked.Exchange(ref _kernelTicks, 0);
        Interlocked.Exchange(ref _unpackTicks, 0);
        Interlocked.Exchange(ref _profileCalls, 0);
    }

    /// <summary>取走累计的分阶段耗时（毫秒）与调用次数，然后清零。</summary>
    internal static (double PackMs, double KernelMs, double UnpackMs, int Calls) TakeProfile()
    {
        double scale = 1000.0 / Stopwatch.Frequency;
        return (Interlocked.Exchange(ref _packTicks, 0) * scale,
                Interlocked.Exchange(ref _kernelTicks, 0) * scale,
                Interlocked.Exchange(ref _unpackTicks, 0) * scale,
                Interlocked.Exchange(ref _profileCalls, 0));
    }

    /// <summary>打包后的权重，挂在权重张量上缓存（与 <see cref="PackedConv"/> 同一做法）。</summary>
    private static readonly ConditionalWeakTable<Tensor, PackedWeights> Cache = new();

    private sealed class PackedWeights
    {
        public PackedWeights(float[] data, int ic, int oc)
        {
            Data = data;
            Ic = ic;
            Oc = oc;
        }

        public float[] Data { get; }

        public int Ic { get; }

        public int Oc { get; }
    }

    /// <summary>
    /// 形状/参数是否落在适用面内：3×3 / pad1 / dil1 / groups1，stride 1 或 2。
    /// <para>
    /// stride 2 时 <paramref name="h"/> / <paramref name="w"/> 必须为偶数：blocked 输入在左上留
    /// 1 像素 halo，最后一个输出像素的取样范围是 <c>[2*(o-1), 2*(o-1)+2]</c>；只有偶数边长才保证
    /// <c>2*ow == w</c>、<c>2*oh == h</c>，刚好落在 <c>[0, h] × [0, w]</c> 内。
    /// 奇数边长会多读一列/一行（实测 101×97 的 max|d| 直接是 24）。
    /// stride 1 需要四周各 1 像素 halo（padding 行/列显式清零），因此没有奇偶约束。
    /// </para>
    /// </summary>
    public static bool CanUse(
        int channels, int outChannels, int h, int w, int kernelH, int kernelW, int groups,
        int strideH, int strideW, int dilH, int dilW, int padT, int padL)
        => Enabled
           && IsSupported
           && groups == 1
           && kernelH == 3 && kernelW == 3
           && strideH == strideW && (strideH == 1 || strideH == 2)
           && dilH == 1 && dilW == 1
           && padT == 1 && padL == 1
           && (strideH == 1 || ((h & 1) == 0 && (w & 1) == 0))
           && channels % Block == 0
           && outChannels % (Block * OcGroup) == 0;

    /// <summary>
    /// 形状推断与内核分派共用的判据（<see cref="CanUse"/> + 输出尺寸关系）。
    /// <para>
    /// 两处必须用同一个函数：temp 缓冲是按这里算的大小申请的，判据一旦不一致，
    /// 就是「申请了但内核不用」或者相反的静默错位。
    /// </para>
    /// </summary>
    public static bool Dispatch(
        int channels, int outChannels, int inH, int inW, int outH, int outW,
        int kernelH, int kernelW, int groups,
        int strideH, int strideW, int dilH, int dilW, int padT, int padL)
    {
        if (!CanUse(channels, outChannels, inH, inW, kernelH, kernelW, groups,
                strideH, strideW, dilH, dilW, padT, padL))
            return false;

        // 取样范围必须恰好落在 padded 缓冲内（auto_pad 的各种组合都在这里兜住）
        return strideH == 2 ? outH * 2 == inH && outW * 2 == inW
                            : outH == inH && outW == inW;
    }

    /// <summary>halo 行/列数：stride 2 只需左上 1 像素，stride 1 四周各 1 像素。</summary>
    public static int HaloPad(int stride) => stride == 2 ? 1 : 2;

    /// <summary>blocked 输入的元素数（含 halo）。</summary>
    public static long BlockedInputCount(int channels, int h, int w, int stride)
    {
        int pad = HaloPad(stride);
        return (long)(channels / Block) * (h + pad) * (w + pad) * Block;
    }

    /// <summary>blocked 输出的元素数（与 NCHW 相同）。</summary>
    public static long BlockedOutputCount(int outChannels, int oh, int ow)
        => (long)outChannels * oh * ow;

    /// <summary>
    /// 算子入口：NCHW 进、NCHW 出，中间的 blocked 搬运由 <paramref name="blockedInput"/> /
    /// <paramref name="blockedOutput"/> 两块临时缓冲承载（由引擎的 temp 池按层预分配）。
    /// <para>
    /// 阶段 1 还没有做图级布局传播，所以每层都要付一次进出 reorder
    /// （实测 8 个 s2 层合计 0.45 ms，约占这些层 conv 时间的 7%）。
    /// </para>
    /// </summary>
    public static bool TryConv(
        Tensor input,
        Tensor weight,
        Tensor? bias,
        Tensor output,
        int groups,
        int strideH,
        int strideW,
        int dilH,
        int dilW,
        int padT,
        int padL,
        Tensor? blockedInput,
        Tensor? blockedOutput)
    {
        if (input.Shape.Rank != 4 || weight.Shape.Rank != 4 || output.Shape.Rank != 4)
            return false;

        int batches = input.Shape.Dims[0];
        int channels = input.Shape.Dims[1];
        int inH = input.Shape.Dims[2];
        int inW = input.Shape.Dims[3];

        int outChannels = output.Shape.Dims[1];
        int outH = output.Shape.Dims[2];
        int outW = output.Shape.Dims[3];

        // 单批次：blocked 缓冲按「一个样本」描述，多批次留给后续阶段
        if (batches != 1)
            return false;

        if (!Dispatch(channels, outChannels, inH, inW, outH, outW,
                weight.Shape.Dims[2], weight.Shape.Dims[3], groups,
                strideH, strideW, dilH, dilW, padT, padL))
            return false;

        if (blockedInput is null || blockedOutput is null)
            return false;
        if (blockedInput.Count < BlockedInputCount(channels, inH, inW, strideH)
            || blockedOutput.Count < BlockedOutputCount(outChannels, outH, outW))
            return false;

        var packed = Cache.GetValue(weight, w => new PackedWeights(
            PackWeights(w.AsSpan<float>(), channels, outChannels), channels, outChannels));

        if (packed.Ic != channels || packed.Oc != outChannels)
            return false;

        long t0 = Profile ? Stopwatch.GetTimestamp() : 0;

        int step = strideH;

        PackInput(input.AsSpan<float>(), blockedInput.AsSpan<float>(), channels, inH, inW, step);

        long t1 = Profile ? Stopwatch.GetTimestamp() : 0;

        Conv(blockedInput.AsSpan<float>(), packed.Data,
            bias is null ? default : bias.AsSpan<float>(),
            blockedOutput.AsSpan<float>(),
            channels, outChannels, inH, inW, outH, outW, step);

        long t2 = Profile ? Stopwatch.GetTimestamp() : 0;

        UnpackOutput(blockedOutput.AsSpan<float>(), output.AsSpan<float>(), outChannels, outH, outW);

        if (Profile)
        {
            long t3 = Stopwatch.GetTimestamp();
            Interlocked.Add(ref _packTicks, t1 - t0);
            Interlocked.Add(ref _kernelTicks, t2 - t1);
            Interlocked.Add(ref _unpackTicks, t3 - t2);
            Interlocked.Increment(ref _profileCalls);
        }

        return true;
    }

    /// <summary>
    /// 权重打包：<c>[oc][ic][3][3]</c> → <c>[oc/16][ic/16][ky*3+kx][icLane][ocLane]</c>。
    /// <para>一次性，挂在权重张量上缓存（与 <see cref="PackedConv"/> 同一做法）。</para>
    /// </summary>
    public static float[] PackWeights(ReadOnlySpan<float> src, int ic, int oc)
    {
        int ib = ic / Block;
        int ob = oc / Block;
        var dst = new float[(long)ob * ib * 9 * Block * Block];

        for (int o = 0; o < oc; o++)
        {
            int oBlock = o / Block;
            int oLane = o % Block;

            for (int i = 0; i < ic; i++)
            {
                int iBlock = i / Block;
                int iLane = i % Block;
                long srcBase = ((long)o * ic + i) * 9;

                for (int kp = 0; kp < 9; kp++)
                {
                    long index = (((long)oBlock * ib + iBlock) * 9 + kp) * Block * Block
                                 + (long)iLane * Block + oLane;
                    dst[index] = src[(int)(srcBase + kp)];
                }
            }
        }

        return dst;
    }

    /// <summary>reorder 的并行粒度：一次任务处理的行带高度。</summary>
    private const int BandRows = 32;

    /// <summary>
    /// NCHW → <c>[C/16][H+pad][W+pad][16]</c>（<paramref name="stride"/>=2 时 pad=1，stride=1 时 pad=2）。
    /// 原始像素落在 <c>[1, h] × [1, w]</c>，halo 位置一律清零：
    /// stride 2 只用到左上，stride 1 四周都用到（最后一行/列是卷积的底部/右侧 padding）。
    /// <para>
    /// 逐 lane 标量写会让每个像素的 64 字节块被扫 16 遍（写放大 16×，实测 320² 要 3.7 ms）。
    /// 这里改成 8×8 转置内核：一次搬 8 个通道 × 8 个像素，落盘是 8 个连续 32 字节半行，
    /// 两条 store 就填满一条 cache line。
    /// </para>
    /// </summary>
    public static unsafe void PackInput(
        ReadOnlySpan<float> src, Span<float> dst, int channels, int h, int w, int stride)
    {
        int cb = channels / Block;
        int pad = HaloPad(stride);
        int ihp = h + pad;
        int iwp = w + pad;
        int planeSrc = h * w;
        int bands = (h + BandRows - 1) / BandRows;
        long rowBytes = (long)iwp * Block;

        fixed (float* psrc = src)
        fixed (float* pdst = dst)
        {
            nint srcPtr = (nint)psrc;
            nint dstPtr = (nint)pdst;

            ParallelRange.For(cb * bands, (begin, end) =>
            {
                float* s0 = (float*)srcPtr;
                float* d0 = (float*)dstPtr;

                for (int task = begin; task < end; task++)
                {
                    int c = task / bands;
                    int band = task - c * bands;
                    int y0 = band * BandRows;
                    int yEnd = Math.Min(h, y0 + BandRows);

                    float* csrc = s0 + (long)c * Block * planeSrc;
                    float* plane = d0 + (long)c * rowBytes * ihp;

                    if (band == 0)
                    {
                        // 第 0 行整行是 halo：连续 20 KB，交给 Span.Clear 一次清掉
                        new Span<float>(plane, (int)rowBytes).Clear();

                        // stride 1 还要清最下面那一行 halo（原始第 h 行，越界行 → padded 行 h+1）
                        if (pad == 2)
                            new Span<float>(plane + (long)(h + 1) * rowBytes, (int)rowBytes).Clear();
                    }

                    for (int y = y0; y < yEnd; y++)
                    {
                        float* rowBase = plane + (long)(y + 1) * rowBytes;
                        float* dstRow = rowBase + Block;

                        // 左 halo 像素
                        Avx512F.Store(rowBase, Vector512<float>.Zero);

                        // 右 halo 像素（stride 1 才有；stride 2 的 pad 只读左上）
                        if (pad == 2)
                            Avx512F.Store(rowBase + (long)(w + 1) * Block, Vector512<float>.Zero);

                        int x = 0;

                        for (; x + 8 <= w; x += 8)
                        {
                            float* dpx = dstRow + (long)x * Block;

                            // 两个 lane 组落在同一条 cache line 的两半，紧挨着做完
                            for (int g = 0; g < 2; g++)
                                Transpose8(csrc + (long)g * 8 * planeSrc + (long)y * w + x, planeSrc,
                                    dpx + g * 8, Block);
                        }

                        for (; x < w; x++)
                        {
                            float* dpx = dstRow + (long)x * Block;
                            int srcIndex = y * w + x;

                            for (int lane = 0; lane < Block; lane++)
                                dpx[lane] = csrc[(long)lane * planeSrc + srcIndex];
                        }
                    }
                }
            });
        }
    }

    /// <summary><c>[OC/16][OH][OW][16]</c> → NCHW（<see cref="PackInput"/> 的逆变换）。</summary>
    public static unsafe void UnpackOutput(ReadOnlySpan<float> src, Span<float> dst, int oc, int oh, int ow)
    {
        int spatial = oh * ow;
        int ocb = oc / Block;

        fixed (float* psrc = src)
        fixed (float* pdst = dst)
        {
            nint srcPtr = (nint)psrc;
            nint dstPtr = (nint)pdst;

            // 任务 = (oc 块, lane 组, 输出行)：每行 ow 个像素连续，转置内核直接按行喂
            ParallelRange.For(ocb * 2 * oh, (begin, end) =>
            {
                float* s0 = (float*)srcPtr;
                float* d0 = (float*)dstPtr;

                for (int task = begin; task < end; task++)
                {
                    int row = task % oh;
                    int rest = task / oh;
                    int g = rest % 2;
                    int ob = rest / 2;

                    float* srow = s0 + ((long)ob * spatial + (long)row * ow) * Block + g * 8;
                    float* drow = d0 + ((long)ob * Block + g * 8) * spatial + (long)row * ow;

                    int x = 0;

                    for (; x + 8 <= ow; x += 8)
                        Transpose8(srow + (long)x * Block, Block, drow + x, spatial);

                    for (; x < ow; x++)
                    {
                        float* spx = srow + (long)x * Block;

                        for (int lane = 0; lane < 8; lane++)
                            drow[(long)lane * spatial + x] = spx[lane];
                    }
                }
            });
        }
    }

    /// <summary>
    /// 8×8 单精度转置：<paramref name="src"/> 的 8 行（行距 <paramref name="srcStride"/> 个 float）
    /// 转成结果写在 <paramref name="dst"/> 的 8 行（行距 <paramref name="dstStride"/>）。
    /// <para>打包与解包是同一个动作的两种行进方向，所以两边共用这一个内核。</para>
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void Transpose8(float* src, int srcStride, float* dst, int dstStride)
    {
        var r0 = Avx.LoadVector256(src + 0 * srcStride);
        var r1 = Avx.LoadVector256(src + 1 * srcStride);
        var r2 = Avx.LoadVector256(src + 2 * srcStride);
        var r3 = Avx.LoadVector256(src + 3 * srcStride);
        var r4 = Avx.LoadVector256(src + 4 * srcStride);
        var r5 = Avx.LoadVector256(src + 5 * srcStride);
        var r6 = Avx.LoadVector256(src + 6 * srcStride);
        var r7 = Avx.LoadVector256(src + 7 * srcStride);

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

        Avx.Store(dst + 0 * dstStride, Avx.Permute2x128(u0, u4, 0x20));
        Avx.Store(dst + 1 * dstStride, Avx.Permute2x128(u1, u5, 0x20));
        Avx.Store(dst + 2 * dstStride, Avx.Permute2x128(u2, u6, 0x20));
        Avx.Store(dst + 3 * dstStride, Avx.Permute2x128(u3, u7, 0x20));
        Avx.Store(dst + 4 * dstStride, Avx.Permute2x128(u0, u4, 0x31));
        Avx.Store(dst + 5 * dstStride, Avx.Permute2x128(u1, u5, 0x31));
        Avx.Store(dst + 6 * dstStride, Avx.Permute2x128(u2, u6, 0x31));
        Avx.Store(dst + 7 * dstStride, Avx.Permute2x128(u3, u7, 0x31));
    }

    /// <summary>
    /// 主体内核：输入/输出都是 blocked，权重已打包。单批次。
    /// </summary>
    public static unsafe void Conv(
        ReadOnlySpan<float> blockedInput,
        float[] packedWeights,
        ReadOnlySpan<float> bias,
        Span<float> blockedOutput,
        int channels,
        int outChannels,
        int h,
        int w,
        int oh,
        int ow,
        int step)
    {
        int icb = channels / Block;
        int ocb = outChannels / Block;
        int pad = HaloPad(step);
        int ihp = h + pad;
        int iwp = w + pad;
        int pairs = ocb / OcGroup;

        // span 不能进 lambda：先钉住，把指针以 nint 传进去
        fixed (float* pin = blockedInput)
        fixed (float* pw = packedWeights)
        fixed (float* pb = bias)
        fixed (float* pout = blockedOutput)
        {
            nint inPtr = (nint)pin;
            nint weightPtr = (nint)pw;
            nint biasPtr = (nint)pb;
            nint outPtr = (nint)pout;

            ParallelRange.For(pairs * oh, (begin, end) =>
            {
                for (int task = begin; task < end; task++)
                {
                    int row = task % oh;
                    int ob = task / oh * OcGroup;

                    Row((float*)inPtr, (float*)weightPtr, (float*)biasPtr, (float*)outPtr,
                        icb, ihp, iwp, oh, ow, step, row, ob);
                }
            });
        }
    }

    /// <summary>算一行（<see cref="Pixels"/> 个像素一组）：2 个 oc 块 × 8 个像素 = 16 个累加器。</summary>
    private static unsafe void Row(
        float* inBase,
        float* weights,
        float* bias,
        float* outBase,
        int icb,
        int ihp,
        int iwp,
        int oh,
        int ow,
        int step,
        int row,
        int ob)
    {
        var bias0 = bias is null ? Vector512<float>.Zero : Avx512F.LoadVector512(bias + (long)ob * Block);
        var bias1 = bias is null ? Vector512<float>.Zero : Avx512F.LoadVector512(bias + (long)(ob + 1) * Block);

        float* w0 = weights + (long)ob * icb * 9 * Block * Block;
        float* w1 = w0 + (long)icb * 9 * Block * Block;

        float* rowOut = outBase + ((long)ob * oh + row) * ow * Block;
        float* rowOut1 = rowOut + (long)oh * ow * Block;

        int groups = ow / Pixels;

        for (int g = 0; g < groups; g++)
        {
            var a00 = bias0;
            var a01 = bias0;
            var a02 = bias0;
            var a03 = bias0;
            var a04 = bias0;
            var a05 = bias0;
            var a06 = bias0;
            var a07 = bias0;
            var a10 = bias1;
            var a11 = bias1;
            var a12 = bias1;
            var a13 = bias1;
            var a14 = bias1;
            var a15 = bias1;
            var a16 = bias1;
            var a17 = bias1;

            // 相邻像素在 padded 缓冲里的间距：stride 2 → 128 字节，stride 1 → 64 字节
            int pix = step * Block;

            for (int ib = 0; ib < icb; ib++)
            {
                for (int ky = 0; ky < 3; ky++)
                {
                    float* rowIn = inBase + ((long)(ib * ihp + row * step + ky) * iwp + g * Pixels * step) * Block;

                    for (int kx = 0; kx < 3; kx++)
                    {
                        float* wp0 = w0 + (long)(ib * 9 + ky * 3 + kx) * Block * Block;
                        float* wp1 = w1 + (long)(ib * 9 + ky * 3 + kx) * Block * Block;
                        float* p = rowIn + kx * Block;

                        for (int lane = 0; lane < Block; lane++)
                        {
                            var wv0 = Avx512F.LoadVector512(wp0 + lane * Block);
                            var wv1 = Avx512F.LoadVector512(wp1 + lane * Block);
                            float* q = p + lane;

                            var x0 = Vector512.Create(q[0]);
                            a00 = Fma(x0, wv0, a00);
                            a10 = Fma(x0, wv1, a10);

                            var x1 = Vector512.Create(q[pix]);
                            a01 = Fma(x1, wv0, a01);
                            a11 = Fma(x1, wv1, a11);

                            var x2 = Vector512.Create(q[2 * pix]);
                            a02 = Fma(x2, wv0, a02);
                            a12 = Fma(x2, wv1, a12);

                            var x3 = Vector512.Create(q[3 * pix]);
                            a03 = Fma(x3, wv0, a03);
                            a13 = Fma(x3, wv1, a13);

                            var x4 = Vector512.Create(q[4 * pix]);
                            a04 = Fma(x4, wv0, a04);
                            a14 = Fma(x4, wv1, a14);

                            var x5 = Vector512.Create(q[5 * pix]);
                            a05 = Fma(x5, wv0, a05);
                            a15 = Fma(x5, wv1, a15);

                            var x6 = Vector512.Create(q[6 * pix]);
                            a06 = Fma(x6, wv0, a06);
                            a16 = Fma(x6, wv1, a16);

                            var x7 = Vector512.Create(q[7 * pix]);
                            a07 = Fma(x7, wv0, a07);
                            a17 = Fma(x7, wv1, a17);
                        }
                    }
                }

                // 每处理完一个 ic 块把累加器换回寄存器自由区不是必要的：16 个累加器 + 3 个临时永远够用
            }

            float* o0 = rowOut + (long)g * Pixels * Block;
            float* o1 = rowOut1 + (long)g * Pixels * Block;

            Avx512F.Store(o0 + 0 * Block, a00);
            Avx512F.Store(o0 + 1 * Block, a01);
            Avx512F.Store(o0 + 2 * Block, a02);
            Avx512F.Store(o0 + 3 * Block, a03);
            Avx512F.Store(o0 + 4 * Block, a04);
            Avx512F.Store(o0 + 5 * Block, a05);
            Avx512F.Store(o0 + 6 * Block, a06);
            Avx512F.Store(o0 + 7 * Block, a07);

            Avx512F.Store(o1 + 0 * Block, a10);
            Avx512F.Store(o1 + 1 * Block, a11);
            Avx512F.Store(o1 + 2 * Block, a12);
            Avx512F.Store(o1 + 3 * Block, a13);
            Avx512F.Store(o1 + 4 * Block, a14);
            Avx512F.Store(o1 + 5 * Block, a15);
            Avx512F.Store(o1 + 6 * Block, a16);
            Avx512F.Store(o1 + 7 * Block, a17);
        }

        // 行尾不足 8 个像素：逐像素收尾（yolo11n 只有 20×20 那层会走到）
        for (int col = groups * Pixels; col < ow; col++)
        {
            var a0 = bias0;
            var a1 = bias1;

            for (int ib = 0; ib < icb; ib++)
            {
                for (int ky = 0; ky < 3; ky++)
                {
                    for (int kx = 0; kx < 3; kx++)
                    {
                        float* wp0 = w0 + (long)(ib * 9 + ky * 3 + kx) * Block * Block;
                        float* wp1 = w1 + (long)(ib * 9 + ky * 3 + kx) * Block * Block;
                        float* p = inBase + ((long)(ib * ihp + row * step + ky) * iwp + col * step + kx) * Block;

                        for (int lane = 0; lane < Block; lane++)
                        {
                            var wv0 = Avx512F.LoadVector512(wp0 + lane * Block);
                            var wv1 = Avx512F.LoadVector512(wp1 + lane * Block);
                            var x = Vector512.Create(p[lane]);

                            a0 = Fma(x, wv0, a0);
                            a1 = Fma(x, wv1, a1);
                        }
                    }
                }
            }

            Avx512F.Store(rowOut + (long)col * Block, a0);
            Avx512F.Store(rowOut1 + (long)col * Block, a1);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector512<float> Fma(Vector512<float> a, Vector512<float> b, Vector512<float> c)
        => Avx512F.FusedMultiplyAdd(a, b, c);
}
