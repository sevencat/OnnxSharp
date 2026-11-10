using System.Buffers;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using OnnxSharp.Tensors;
using OnnxSharp.Threading;

namespace OnnxSharp.Kernels;

/// <summary>
/// 面板式直接卷积：不生成整幅 im2col 缓冲，只把当前 16 个输出像素的感受野打包成
/// <c>[K][16]</c>（正好一条 64 字节缓存行），立刻交给 4×16 FMA 微内核算出
/// <c>[outPerGroup][16]</c> 并直接写回 NCHW 输出。
/// <para>
/// 相比 <see cref="ConvKernels.Im2Col"/>：col 缓冲从 <c>K×空间</c> 降到 <c>K×16</c>，
/// 省掉「写一遍整幅 col、再按行块读回来并重新打包成列块」的两趟全量搬运；
/// 权重按 4 行交错预打包（与 SimdYolo 的 CpuPackedGemm、OpenCV dnn fastConv 的权重块同一思路），
/// 每个 K 步只剩固定行偏移，不再有逐行的地址乘法。
/// </para>
/// </summary>
internal static class PackedConv
{
    /// <summary>一个面板的列数：16 个 float 正好 64 字节，相邻面板不会共享缓存行。</summary>
    private const int Panel = 16;

    /// <summary>
    /// 微内核一次处理的输出通道行数（4 行 × 2 向量 = 8 个累加器 + 2 个 B + 1 个广播 = 11 个 YMM）。
    /// 试过 6 行（12 个累加器），x86-64 只有 16 个 YMM，实测会把累加器溢出到栈上，反而慢 25%。
    /// </summary>
    private const int RowBlock = 4;

    /// <summary>调试开关：<c>0</c> 关闭本路径，交回 im2col（用于 A/B 对比）。</summary>
    private static readonly bool Enabled = Environment.GetEnvironmentVariable("ONNXSHARP_PACKED_CONV") != "0";

    /// <summary>K 方向切片深度，0 表示不分片（调参用）。</summary>
    private static readonly int SliceDepth =
        int.TryParse(Environment.GetEnvironmentVariable("ONNXSHARP_PACKED_DEPTH"), out int depth) ? depth : 128;

    private static readonly ConditionalWeakTable<Tensor, PackedWeights> Cache = new();

    /// <summary>按 4 行交错的权重：<c>[group][block][k][4]</c>，缺少的行补零。</summary>
    private sealed class PackedWeights
    {
        public PackedWeights(float[] data, int groups, int rows, int blocks, int depth)
        {
            Data = data;
            Groups = groups;
            Rows = rows;
            Blocks = blocks;
            Depth = depth;
        }

        public float[] Data { get; }

        /// <summary>分组数。</summary>
        public int Groups { get; }

        /// <summary>每组输出通道数。</summary>
        public int Rows { get; }

        /// <summary>每组 4 行块的数量。</summary>
        public int Blocks { get; }

        /// <summary>归约深度 K。</summary>
        public int Depth { get; }
    }

    /// <summary>
    /// 尝试走面板式直接卷积，执行成功返回 true。
    /// <para>
    /// 1×1 卷积不进来：那种情况下 im2col 是恒等映射，直接当 GEMM 用比先复制一份更快。
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
        int padL)
    {
        if (!Enabled || !Avx2.IsSupported || !Fma.IsSupported)
            return false;

        if (input.Shape.Rank != 4 || weight.Shape.Rank != 4 || output.Shape.Rank != 4)
            return false;

        int batches = input.Shape.Dims[0];
        int channels = input.Shape.Dims[1];
        int inH = input.Shape.Dims[2];
        int inW = input.Shape.Dims[3];

        int outChannels = output.Shape.Dims[1];
        int outH = output.Shape.Dims[2];
        int outW = output.Shape.Dims[3];

        int kernelH = weight.Shape.Dims[2];
        int kernelW = weight.Shape.Dims[3];

        if (kernelH == 1 && kernelW == 1 && strideH == 1 && strideW == 1)
            return false;                       // 点卷积交给零拷贝的 GEMM 路径

        if (groups <= 0 || channels % groups != 0 || outChannels % groups != 0)
            return false;

        int inPerGroup = channels / groups;
        int outPerGroup = outChannels / groups;
        int kernelSize = inPerGroup * kernelH * kernelW;

        // 组内输出通道太少（depthwise 等）时 4 行微内核浪费太大，交回 im2col
        if (outPerGroup < 8)
            return false;

        long spatial = (long)outH * outW;
        int panels = (int)((spatial + Panel - 1) / Panel);

        // 大步长、大空间图的 s1 卷积上 im2col 的整幅 col 复用更好（实测 80²/160² 快 5~12%），
        // 这类层留给 im2col；面板路径专治「K 大而空间小」和「通道少、被 col 搬运拖死」的层
        if (strideH == 1 && strideW == 1 && spatial >= 4096)
            return false;

        // 面板数不够喂满线程时，交回 im2col（它会把输出宽度也切开）
        if ((long)batches * groups * panels < ParallelRange.Threads)
            return false;

        // K×16 的面板要留在 L1/L2 里，超过 128 KB 就不值得了
        if ((long)kernelSize * Panel > 32768)
            return false;

        var packed = Cache.GetValue(weight, w => PackWeights(w, groups, outPerGroup, kernelSize));

        if (packed.Groups != groups || packed.Rows != outPerGroup || packed.Depth != kernelSize)
            return false;

        Run(input, packed, bias, output, batches, groups, channels, inPerGroup, outChannels, outPerGroup,
            kernelH, kernelW, kernelSize, strideH, strideW, dilH, dilW, padT, padL,
            inH, inW, outH, outW, panels);

        return true;
    }

    /// <summary>把 <c>[oc][k]</c> 重排成 <c>[block][k][4]</c>，让 4 行共用一个 K 步偏移。</summary>
    private static PackedWeights PackWeights(Tensor weight, int groups, int outPerGroup, int kernelSize)
    {
        var src = weight.AsSpan<float>();
        int blocks = (outPerGroup + RowBlock - 1) / RowBlock;
        var data = new float[(long)groups * blocks * kernelSize * RowBlock];

        for (int g = 0; g < groups; g++)
        {
            for (int block = 0; block < blocks; block++)
            {
                long dstBase = ((long)g * blocks + block) * kernelSize * RowBlock;
                int first = g * outPerGroup + block * RowBlock;

                for (int p = 0; p < kernelSize; p++)
                {
                    long dst = dstBase + (long)p * RowBlock;

                    for (int r = 0; r < RowBlock; r++)
                    {
                        int oc = first + r;
                        data[dst + r] = oc < (g + 1) * outPerGroup ? src[oc * kernelSize + p] : 0f;
                    }
                }
            }
        }

        return new PackedWeights(data, groups, outPerGroup, blocks, kernelSize);
    }

    private static unsafe void Run(
        Tensor input,
        PackedWeights packed,
        Tensor? bias,
        Tensor output,
        int batches,
        int groups,
        int channels,
        int inPerGroup,
        int outChannels,
        int outPerGroup,
        int kernelH,
        int kernelW,
        int kernelSize,
        int strideH,
        int strideW,
        int dilH,
        int dilW,
        int padT,
        int padL,
        int inH,
        int inW,
        int outH,
        int outW,
        int panels)
    {
        int spatial = outH * outW;
        int tasks = batches * groups * panels;
        int weightStride = packed.Blocks * kernelSize * RowBlock;

        ParallelRange.For(tasks, (begin, end) =>
        {
            var src = input.AsSpan<float>();
            var dst = output.AsSpan<float>();
            var biases = bias is null ? Span<float>.Empty : bias.AsSpan<float>();

            float[] pool = ArrayPool<float>.Shared.Rent(kernelSize * Panel);

            try
            {
                fixed (float* psrc = src)
                fixed (float* pdst = dst)
                fixed (float* ppack = pool)
                fixed (float* pweights = packed.Data)
                fixed (float* pbias = biases)
                {
                    for (int task = begin; task < end; task++)
                    {
                        int panel = task % panels;
                        int rest = task / panels;
                        int g = rest % groups;
                        int b = rest / groups;

                        int start = panel * Panel;
                        int take = Math.Min(Panel, spatial - start);

                        float* inBase = psrc + ((long)b * channels + g * inPerGroup) * inH * inW;

                        PackPanel(inBase, ppack, inH, inW, inPerGroup, kernelH, kernelW,
                            strideH, strideW, dilH, dilW, padT, padL, outW, start, take);

                        Multiply(
                            pweights + (long)g * weightStride, kernelSize, ppack,
                            pdst + ((long)b * outChannels + g * outPerGroup) * spatial,
                            spatial, start, take,
                            pbias + g * outPerGroup, outPerGroup);
                    }
                }
            }
            finally
            {
                ArrayPool<float>.Shared.Return(pool);
            }
        });
    }

    /// <summary>
    /// 把一个面板（<see cref="Panel"/> 个连续输出像素）的感受野摊平成 <c>[K][16]</c>，越界写 0。
    /// </summary>
    private static unsafe void PackPanel(
        float* src,
        float* dst,
        int inH,
        int inW,
        int inPerGroup,
        int kernelH,
        int kernelW,
        int strideH,
        int strideW,
        int dilH,
        int dilW,
        int padT,
        int padL,
        int outW,
        int start,
        int take)
    {
        Span<int> oy = stackalloc int[Panel];
        Span<int> ox = stackalloc int[Panel];

        for (int j = 0; j < take; j++)
        {
            int index = start + j;
            int y = index / outW;
            oy[j] = y;
            ox[j] = index - y * outW;
        }

        // 面板整个落在同一输出行时，每个核位置都是一段规则读取：
        // 步长 1 直接搬，步长 2 用 AVX2 抽取偶数位（和 im2col 的 FillColumns 同一做法）
        bool singleRow = take == Panel && oy[0] == oy[Panel - 1];

        for (int ic = 0; ic < inPerGroup; ic++)
        {
            float* channel = src + (long)ic * inH * inW;

            for (int ky = 0; ky < kernelH; ky++)
            {
                for (int kx = 0; kx < kernelW; kx++)
                {
                    int row = (ic * kernelH + ky) * kernelW + kx;
                    float* target = dst + (long)row * Panel;

                    int y0 = oy[0] * strideH - padT + ky * dilH;
                    int x0 = ox[0] * strideW - padL + kx * dilW;

                    if (singleRow && (uint)y0 < (uint)inH)
                    {
                        float* source = channel + (long)y0 * inW + x0;

                        if (strideW == 1 && x0 >= 0 && x0 + Panel <= inW)
                        {
                            Avx.Store(target, Avx.LoadVector256(source));
                            Avx.Store(target + 8, Avx.LoadVector256(source + 8));
                            continue;
                        }

                        // 步长 2：一次读 32 个元素取出 16 个
                        if (strideW == 2 && x0 >= 0 && x0 + 2 * Panel <= inW)
                        {
                            var even = Vector256.Create(0, 2, 4, 6, 0, 0, 0, 0);

                            var a0 = Avx2.PermuteVar8x32(Avx.LoadVector256(source), even);
                            var a1 = Avx2.PermuteVar8x32(Avx.LoadVector256(source + 8), even);
                            Avx.Store(target, Avx2.Permute2x128(a0, a1, 0x20));

                            var b0 = Avx2.PermuteVar8x32(Avx.LoadVector256(source + 16), even);
                            var b1 = Avx2.PermuteVar8x32(Avx.LoadVector256(source + 24), even);
                            Avx.Store(target + 8, Avx2.Permute2x128(b0, b1, 0x20));
                            continue;
                        }
                    }

                    for (int j = 0; j < take; j++)
                    {
                        int y = oy[j] * strideH - padT + ky * dilH;
                        int x = ox[j] * strideW - padL + kx * dilW;
                        target[j] = (uint)y < (uint)inH && (uint)x < (uint)inW
                            ? channel[(long)y * inW + x]
                            : 0f;
                    }

                    // 池化缓冲区未清零，尾部必须显式写 0 才能让 GEMM 读到确定的数
                    for (int j = take; j < Panel; j++)
                        target[j] = 0f;
                }
            }
        }
    }

    /// <summary>
    /// <c>c[row][column..column+take) += Σ_p W[row][p] · b[p][0..take)</c>，
    /// 一次算 4 个输出通道，b 的每条载入供 4 行复用（load 端口不再是瓶颈）。
    /// </summary>
    private static unsafe void Multiply(
        float* w,
        int k,
        float* b,
        float* c,
        int ldc,
        int column,
        int take,
        float* bias,
        int rows)
    {
        // K 切片深度：0 表示不分片（整条 K 一次算完）。
        int depth = SliceDepth > 0 && take == Panel && k >= 256 && rows >= 32 ? SliceDepth : k;
        int blocks = (rows + RowBlock - 1) / RowBlock;

        for (int begin = 0; begin < k; begin += depth)
        {
            int end = Math.Min(k, begin + depth);

            for (int block = 0; block < blocks; block++)
            {
                int row = block * RowBlock;
                float* a = w + (long)block * k * RowBlock + (long)begin * RowBlock;
                float* bb = b + (long)begin * Panel;
                float* cc = c + (long)row * ldc + column;

                Vector256<float> c00, c01, c10, c11, c20, c21, c30, c31;

                if (begin == 0)
                {
                    // 注意：没有 bias 时 bias 指针是 null，四行都要判空（PaddleOCR 的 Conv+BN 就不带 bias）
                    c00 = c01 = Vector256.Create(bias is null ? 0f : bias[row]);
                    c10 = c11 = bias is null || row + 1 >= rows ? Vector256<float>.Zero : Vector256.Create(bias[row + 1]);
                    c20 = c21 = bias is null || row + 2 >= rows ? Vector256<float>.Zero : Vector256.Create(bias[row + 2]);
                    c30 = c31 = bias is null || row + 3 >= rows ? Vector256<float>.Zero : Vector256.Create(bias[row + 3]);
                }
                else
                {
                    c00 = Avx.LoadVector256(cc);
                    c01 = Avx.LoadVector256(cc + 8);
                    c10 = c11 = c20 = c21 = c30 = c31 = Vector256<float>.Zero;

                    if (row + 1 < rows) { c10 = Avx.LoadVector256(cc + ldc); c11 = Avx.LoadVector256(cc + ldc + 8); }
                    if (row + 2 < rows) { c20 = Avx.LoadVector256(cc + 2 * ldc); c21 = Avx.LoadVector256(cc + 2 * ldc + 8); }
                    if (row + 3 < rows) { c30 = Avx.LoadVector256(cc + 3 * ldc); c31 = Avx.LoadVector256(cc + 3 * ldc + 8); }
                }

                for (int p = begin; p < end; p++, a += RowBlock, bb += Panel)
                {
                    var b0 = Avx.LoadVector256(bb);
                    var b1 = Avx.LoadVector256(bb + 8);

                    var av = Avx.BroadcastScalarToVector256(a);
                    c00 = Fma.MultiplyAdd(av, b0, c00);
                    c01 = Fma.MultiplyAdd(av, b1, c01);

                    av = Avx.BroadcastScalarToVector256(a + 1);
                    c10 = Fma.MultiplyAdd(av, b0, c10);
                    c11 = Fma.MultiplyAdd(av, b1, c11);

                    av = Avx.BroadcastScalarToVector256(a + 2);
                    c20 = Fma.MultiplyAdd(av, b0, c20);
                    c21 = Fma.MultiplyAdd(av, b1, c21);

                    av = Avx.BroadcastScalarToVector256(a + 3);
                    c30 = Fma.MultiplyAdd(av, b0, c30);
                    c31 = Fma.MultiplyAdd(av, b1, c31);
                }

                Store(c00, c01, cc, take);
                if (row + 1 < rows) Store(c10, c11, cc + ldc, take);
                if (row + 2 < rows) Store(c20, c21, cc + 2 * ldc, take);
                if (row + 3 < rows) Store(c30, c31, cc + 3 * ldc, take);
            }
        }
    }

    /// <summary>写出最多 16 列结果，尾面板只写实际有效的列。</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void Store(Vector256<float> low, Vector256<float> high, float* target, int take)
    {
        if (take == Panel)
        {
            Avx.Store(target, low);
            Avx.Store(target + 8, high);
        }
        else
        {
            for (int j = 0; j < take; j++)
                target[j] = j < 8 ? low.GetElement(j) : high.GetElement(j - 8);
        }
    }
}
