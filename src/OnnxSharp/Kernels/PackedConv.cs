using System.Buffers;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
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
    /// 微内核一次处理的输出通道行数（8 行 × 2 向量 = 16 个累加器 + 2 个 B + 1 个广播 = 19 个寄存器）。
    /// 只有 16 个 YMM 的机器上 6 行（12 个累加器）就会溢出到栈上、实测反而慢 25%，所以那里保持 4 行；
    /// AVX-512VL 放开后 ymm16~31 可用，8 行才装得下 —— 每个 K 步的 2 条 B 载入摊到 8 行上，
    /// 载入端口不再和 FMA 抢（<c>ONNXSHARP_PACKED_ROWS=4|8</c> 可覆盖，用于 A/B）。
    /// </summary>
    private static readonly int MaxRowBlock = ResolveRowBlock();

    /// <summary>
    /// 一层用几行一块：AVX-512VL 上 oc ≥ 128 才上 8 行。oc 小的时候整层只有 2~4 个 8 行块，
    /// 每块的固定开销（8 次 bias 广播 + 16 次存回）和 K 短的计算量同量级，实测
    /// ic=3→16 的 640→320 从 1.23 退到 3.28 ms、ic=16→32 的 320→160 从 1.19 退到 5.88 ms、
    /// ic=32→64 的 160→80 从 0.93 退到 2.93 ms；换回 4 行都回来了。
    /// 8 行真正不可替代的地方是 oc=256 的 K=2304 层：只有直读（依赖 8 行内核）才不进 128 KB 面板缓冲。
    /// </summary>
    private static int RowsPerBlockFor(int outPerGroup)
        => MaxRowBlock == 8 && outPerGroup >= 128 ? 8 : 4;

    private static int ResolveRowBlock()
    {
        if (!Avx512F.VL.IsSupported)
            return 4;                       // 16 个 YMM 装不下 8 行的累加器

        if (int.TryParse(Environment.GetEnvironmentVariable("ONNXSHARP_PACKED_ROWS"), out int rows)
            && (rows == 4 || rows == 8))
        {
            return rows;
        }

        return 8;
    }

    /// <summary>调试开关：<c>0</c> 关闭本路径，交回 im2col（用于 A/B 对比）。</summary>
    private static readonly bool Enabled = Environment.GetEnvironmentVariable("ONNXSHARP_PACKED_CONV") != "0";

    /// <summary>调试开关：<c>1</c> 让「s1 且空间 ≥ 4096」的层也走面板式（用于复核上面那条经验）。</summary>
    private static readonly bool ForcePackedLargeSpatial =
        Environment.GetEnvironmentVariable("ONNXSHARP_PACKED_S1") is { Length: > 0 };

    /// <summary>
    /// 调试开关：<c>0</c> 关闭「面板直读」（不物化 <c>[K][16]</c>，内核就地读输入），
    /// 退回整块打包；用于 A/B。
    /// </summary>
    private static readonly bool Direct = Environment.GetEnvironmentVariable("ONNXSHARP_PACKED_DIRECT") != "0";

    /// <summary>K 方向切片深度，0 表示不分片（调参用）。</summary>
    private static readonly int SliceDepth =
        int.TryParse(Environment.GetEnvironmentVariable("ONNXSHARP_PACKED_DEPTH"), out int depth) ? depth : 128;

    private static readonly ConditionalWeakTable<Tensor, PackedWeights> Cache = new();

    /// <summary>按行块交错的权重：<c>[group][block][k][RowBlock]</c>，缺少的行补零。</summary>
    private sealed class PackedWeights
    {
        public PackedWeights(float[] data, int groups, int rows, int blocks, int depth, int rowBlock)
        {
            Data = data;
            Groups = groups;
            Rows = rows;
            Blocks = blocks;
            Depth = depth;
            RowBlock = rowBlock;
        }

        public float[] Data { get; }

        /// <summary>分组数。</summary>
        public int Groups { get; }

        /// <summary>每组输出通道数。</summary>
        public int Rows { get; }

        /// <summary>每组行块的数量。</summary>
        public int Blocks { get; }

        /// <summary>打包时用的行数（4 或 8），内核必须与之匹配。</summary>
        public int RowBlock { get; }

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

        // 大步长、大空间图的 s1 卷积上 im2col 的整幅 col 复用更好（历史实测 80²/160² 快 5~12%），
        // 这类层留给 im2col；面板路径专治「K 大而空间小」和「通道少、被 col 搬运拖死」的层。
        // ONNXSHARP_PACKED_S1=1 强制也走面板式：im2col 会物化 K×pixels 的 col
        // （160²/64→64 那层就是 576×25600×4 = 59 MB），值得定期复核这条经验。
        if (strideH == 1 && strideW == 1 && spatial >= 4096 && !ForcePackedLargeSpatial)
            return false;

        // 面板数不够喂满线程时，交回 im2col（它会把输出宽度也切开）
        if ((long)batches * groups * panels < ParallelRange.Threads)
            return false;

        // K×16 的面板要留在 L1/L2 里，超过 128 KB 就不值得了。
        // 直读上线后绝大多数面板不再物化面板缓冲（只有跨行/贴边的那几块回退到打包），
        // K=2304 这种 ic=256 的 3×3 也就能进来了 —— 它正是 yolo11n 里最慢的一层。
        int rowsPerBlock = RowsPerBlockFor(outPerGroup);
        bool directCapable = Direct && rowsPerBlock == 8 && kernelSize >= 64
            && (long)inPerGroup * inH * inW <= int.MaxValue;

        if (directCapable ? kernelSize > 4096 : (long)kernelSize * Panel > 32768)
            return false;

        var packed = Cache.GetValue(weight, w => PackWeights(w, groups, outPerGroup, kernelSize, rowsPerBlock));

        if (packed.Groups != groups || packed.Rows != outPerGroup || packed.Depth != kernelSize
            || packed.RowBlock != rowsPerBlock)
            return false;

        Run(input, packed, bias, output, batches, groups, channels, inPerGroup, outChannels, outPerGroup,
            kernelH, kernelW, kernelSize, strideH, strideW, dilH, dilW, padT, padL,
            inH, inW, outH, outW, panels);

        return true;
    }

    /// <summary>把 <c>[oc][k]</c> 重排成 <c>[block][k][rowsPerBlock]</c>，让同块的行共用一个 K 步偏移。</summary>
    private static PackedWeights PackWeights(
        Tensor weight, int groups, int outPerGroup, int kernelSize, int rowsPerBlock)
    {
        var src = weight.AsSpan<float>();
        int blocks = (outPerGroup + rowsPerBlock - 1) / rowsPerBlock;
        var data = new float[(long)groups * blocks * kernelSize * rowsPerBlock];

        for (int g = 0; g < groups; g++)
        {
            for (int block = 0; block < blocks; block++)
            {
                long dstBase = ((long)g * blocks + block) * kernelSize * rowsPerBlock;
                int first = g * outPerGroup + block * rowsPerBlock;

                for (int p = 0; p < kernelSize; p++)
                {
                    long dst = dstBase + (long)p * rowsPerBlock;

                    for (int r = 0; r < rowsPerBlock; r++)
                    {
                        int oc = first + r;
                        data[dst + r] = oc < (g + 1) * outPerGroup ? src[oc * kernelSize + p] : 0f;
                    }
                }
            }
        }

        return new PackedWeights(data, groups, outPerGroup, blocks, kernelSize, rowsPerBlock);
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
        int rowBlock = packed.RowBlock;
        int weightStride = packed.Blocks * kernelSize * rowBlock;

        ParallelRange.For(tasks, (begin, end) =>
        {
            var src = input.AsSpan<float>();
            var dst = output.AsSpan<float>();
            var biases = bias is null ? Span<float>.Empty : bias.AsSpan<float>();

            float[] pool = ArrayPool<float>.Shared.Rent(kernelSize * Panel);
            // 直读的下标表用 int（每个核位置一个），本组元素数必须装得下 int
            bool direct = Direct && rowBlock == 8 && kernelSize >= 64
                && (long)inPerGroup * inH * inW <= int.MaxValue;
            float[]? tapPool = direct ? ArrayPool<float>.Shared.Rent(kernelSize) : null;
            var taps = tapPool is null ? Span<int>.Empty : MemoryMarshal.Cast<float, int>(tapPool.AsSpan());

            try
            {
                fixed (float* psrc = src)
                fixed (float* pdst = dst)
                fixed (float* ppack = pool)
                fixed (float* pweights = packed.Data)
                fixed (float* pbias = biases)
                fixed (int* ptaps = taps)
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
                        float* weight = pweights + (long)g * weightStride;
                        float* outBase = pdst + ((long)b * outChannels + g * outPerGroup) * spatial;
                        float* biasGroup = pbias + g * outPerGroup;

                        // 直读只在 8 行版本上做：4 行版本要再复制一份内核，收益不值这份代码量
                        if (ptaps != null
                            && TryTapOffsets(inH, inW, inPerGroup, kernelH, kernelW,
                                strideH, strideW, dilH, dilW, padT, padL, outW, start, kernelSize, taps))
                        {
                            Multiply8Direct(weight, kernelSize, inBase, ptaps, outBase,
                                spatial, start, biasGroup, outPerGroup);
                            continue;
                        }

                        PackPanel(inBase, ppack, inH, inW, inPerGroup, kernelH, kernelW,
                            strideH, strideW, dilH, dilW, padT, padL, outW, start, take);

                        if (rowBlock == 8)
                        {
                            Multiply8(weight, kernelSize, ppack, outBase, spatial, start, take,
                                biasGroup, outPerGroup);
                        }
                        else
                        {
                            Multiply4(weight, kernelSize, ppack, outBase, spatial, start, take,
                                biasGroup, outPerGroup);
                        }
                    }
                }
            }
            finally
            {
                ArrayPool<float>.Shared.Return(pool);

                if (tapPool is not null)
                    ArrayPool<float>.Shared.Return(tapPool);
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
    private static unsafe void Multiply4(
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
        int blocks = (rows + 3) / 4;

        for (int begin = 0; begin < k; begin += depth)
        {
            int end = Math.Min(k, begin + depth);

            for (int block = 0; block < blocks; block++)
            {
                int row = block * 4;
                float* a = w + (long)block * k * 4 + (long)begin * 4;
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

                for (int p = begin; p < end; p++, a += 4, bb += Panel)
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

    /// <summary>
    /// 8 行版本（<c>RowBlock == 8</c> 时使用）：16 个累加器落在 ymm0~ymm15，
    /// 2 条 B 载入 + 广播仍在低寄存器，共 19 个 —— 只有 AVX-512VL 的 32 个 YMM 装得下。
    /// </summary>
    private static unsafe void Multiply8(
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
        int depth = SliceDepth > 0 && take == Panel && k >= 256 && rows >= 32 ? SliceDepth : k;
        int blocks = (rows + 7) / 8;

        for (int begin = 0; begin < k; begin += depth)
        {
            int end = Math.Min(k, begin + depth);

            for (int block = 0; block < blocks; block++)
            {
                int row = block * 8;
                float* a = w + (long)block * k * 8 + (long)begin * 8;
                float* bb = b + (long)begin * Panel;
                float* cc = c + (long)row * ldc + column;

                Vector256<float> c00, c01, c10, c11, c20, c21, c30, c31,
                                 c40, c41, c50, c51, c60, c61, c70, c71;

                if (begin == 0)
                {
                    c00 = c01 = Vector256.Create(bias is null ? 0f : bias[row]);
                    c10 = c11 = bias is null || row + 1 >= rows ? Vector256<float>.Zero : Vector256.Create(bias[row + 1]);
                    c20 = c21 = bias is null || row + 2 >= rows ? Vector256<float>.Zero : Vector256.Create(bias[row + 2]);
                    c30 = c31 = bias is null || row + 3 >= rows ? Vector256<float>.Zero : Vector256.Create(bias[row + 3]);
                    c40 = c41 = bias is null || row + 4 >= rows ? Vector256<float>.Zero : Vector256.Create(bias[row + 4]);
                    c50 = c51 = bias is null || row + 5 >= rows ? Vector256<float>.Zero : Vector256.Create(bias[row + 5]);
                    c60 = c61 = bias is null || row + 6 >= rows ? Vector256<float>.Zero : Vector256.Create(bias[row + 6]);
                    c70 = c71 = bias is null || row + 7 >= rows ? Vector256<float>.Zero : Vector256.Create(bias[row + 7]);
                }
                else
                {
                    c00 = Avx.LoadVector256(cc);
                    c01 = Avx.LoadVector256(cc + 8);
                    c10 = c11 = c20 = c21 = c30 = c31 = c40 = c41 = c50 = c51 = c60 = c61 = c70 = c71
                        = Vector256<float>.Zero;

                    if (row + 1 < rows) { c10 = Avx.LoadVector256(cc + ldc); c11 = Avx.LoadVector256(cc + ldc + 8); }
                    if (row + 2 < rows) { c20 = Avx.LoadVector256(cc + 2 * ldc); c21 = Avx.LoadVector256(cc + 2 * ldc + 8); }
                    if (row + 3 < rows) { c30 = Avx.LoadVector256(cc + 3 * ldc); c31 = Avx.LoadVector256(cc + 3 * ldc + 8); }
                    if (row + 4 < rows) { c40 = Avx.LoadVector256(cc + 4 * ldc); c41 = Avx.LoadVector256(cc + 4 * ldc + 8); }
                    if (row + 5 < rows) { c50 = Avx.LoadVector256(cc + 5 * ldc); c51 = Avx.LoadVector256(cc + 5 * ldc + 8); }
                    if (row + 6 < rows) { c60 = Avx.LoadVector256(cc + 6 * ldc); c61 = Avx.LoadVector256(cc + 6 * ldc + 8); }
                    if (row + 7 < rows) { c70 = Avx.LoadVector256(cc + 7 * ldc); c71 = Avx.LoadVector256(cc + 7 * ldc + 8); }
                }

                for (int p = begin; p < end; p++, a += 8, bb += Panel)
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

                    av = Avx.BroadcastScalarToVector256(a + 4);
                    c40 = Fma.MultiplyAdd(av, b0, c40);
                    c41 = Fma.MultiplyAdd(av, b1, c41);

                    av = Avx.BroadcastScalarToVector256(a + 5);
                    c50 = Fma.MultiplyAdd(av, b0, c50);
                    c51 = Fma.MultiplyAdd(av, b1, c51);

                    av = Avx.BroadcastScalarToVector256(a + 6);
                    c60 = Fma.MultiplyAdd(av, b0, c60);
                    c61 = Fma.MultiplyAdd(av, b1, c61);

                    av = Avx.BroadcastScalarToVector256(a + 7);
                    c70 = Fma.MultiplyAdd(av, b0, c70);
                    c71 = Fma.MultiplyAdd(av, b1, c71);
                }

                Store(c00, c01, cc, take);
                if (row + 1 < rows) Store(c10, c11, cc + ldc, take);
                if (row + 2 < rows) Store(c20, c21, cc + 2 * ldc, take);
                if (row + 3 < rows) Store(c30, c31, cc + 3 * ldc, take);
                if (row + 4 < rows) Store(c40, c41, cc + 4 * ldc, take);
                if (row + 5 < rows) Store(c50, c51, cc + 5 * ldc, take);
                if (row + 6 < rows) Store(c60, c61, cc + 6 * ldc, take);
                if (row + 7 < rows) Store(c70, c71, cc + 7 * ldc, take);
            }
        }
    }

    /// <summary>
    /// 8 行 + 直读：<c>offsets[p]</c> 给出第 p 个核位置在输入里的起始下标，16 个像素就地连续读，
    /// 不再先把 <c>[K][16]</c> 摊平到面板缓冲 —— 省掉「打包写一遍 + 内核读一遍」的两趟
    /// 面板搬运，而且重叠的核位置（kx 相邻）不再各存一份。
    /// <para>
    /// 前提由 <see cref="TryTapOffsets"/> 保证：面板整条落在同一输出行、<c>strideW == 1</c>、
    /// 每个核位置的 16 个像素都在界内，因此这里不需要任何越界判断。
    /// </para>
    /// </summary>
    private static unsafe void Multiply8Direct(
        float* w,
        int k,
        float* source,
        int* offsets,
        float* c,
        int ldc,
        int column,
        float* bias,
        int rows)
    {
        int depth = SliceDepth > 0 && k >= 256 && rows >= 32 ? SliceDepth : k;
        int blocks = (rows + 7) / 8;

        for (int begin = 0; begin < k; begin += depth)
        {
            int end = Math.Min(k, begin + depth);

            for (int block = 0; block < blocks; block++)
            {
                int row = block * 8;
                float* a = w + (long)block * k * 8 + (long)begin * 8;
                float* cc = c + (long)row * ldc + column;

                Vector256<float> c00, c01, c10, c11, c20, c21, c30, c31,
                                 c40, c41, c50, c51, c60, c61, c70, c71;

                if (begin == 0)
                {
                    c00 = c01 = Vector256.Create(bias is null ? 0f : bias[row]);
                    c10 = c11 = bias is null || row + 1 >= rows ? Vector256<float>.Zero : Vector256.Create(bias[row + 1]);
                    c20 = c21 = bias is null || row + 2 >= rows ? Vector256<float>.Zero : Vector256.Create(bias[row + 2]);
                    c30 = c31 = bias is null || row + 3 >= rows ? Vector256<float>.Zero : Vector256.Create(bias[row + 3]);
                    c40 = c41 = bias is null || row + 4 >= rows ? Vector256<float>.Zero : Vector256.Create(bias[row + 4]);
                    c50 = c51 = bias is null || row + 5 >= rows ? Vector256<float>.Zero : Vector256.Create(bias[row + 5]);
                    c60 = c61 = bias is null || row + 6 >= rows ? Vector256<float>.Zero : Vector256.Create(bias[row + 6]);
                    c70 = c71 = bias is null || row + 7 >= rows ? Vector256<float>.Zero : Vector256.Create(bias[row + 7]);
                }
                else
                {
                    c00 = Avx.LoadVector256(cc);
                    c01 = Avx.LoadVector256(cc + 8);
                    c10 = c11 = c20 = c21 = c30 = c31 = c40 = c41 = c50 = c51 = c60 = c61 = c70 = c71
                        = Vector256<float>.Zero;

                    if (row + 1 < rows) { c10 = Avx.LoadVector256(cc + ldc); c11 = Avx.LoadVector256(cc + ldc + 8); }
                    if (row + 2 < rows) { c20 = Avx.LoadVector256(cc + 2 * ldc); c21 = Avx.LoadVector256(cc + 2 * ldc + 8); }
                    if (row + 3 < rows) { c30 = Avx.LoadVector256(cc + 3 * ldc); c31 = Avx.LoadVector256(cc + 3 * ldc + 8); }
                    if (row + 4 < rows) { c40 = Avx.LoadVector256(cc + 4 * ldc); c41 = Avx.LoadVector256(cc + 4 * ldc + 8); }
                    if (row + 5 < rows) { c50 = Avx.LoadVector256(cc + 5 * ldc); c51 = Avx.LoadVector256(cc + 5 * ldc + 8); }
                    if (row + 6 < rows) { c60 = Avx.LoadVector256(cc + 6 * ldc); c61 = Avx.LoadVector256(cc + 6 * ldc + 8); }
                    if (row + 7 < rows) { c70 = Avx.LoadVector256(cc + 7 * ldc); c71 = Avx.LoadVector256(cc + 7 * ldc + 8); }
                }

                for (int p = begin; p < end; p++, a += 8)
                {
                    float* bb = source + offsets[p];
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

                    av = Avx.BroadcastScalarToVector256(a + 4);
                    c40 = Fma.MultiplyAdd(av, b0, c40);
                    c41 = Fma.MultiplyAdd(av, b1, c41);

                    av = Avx.BroadcastScalarToVector256(a + 5);
                    c50 = Fma.MultiplyAdd(av, b0, c50);
                    c51 = Fma.MultiplyAdd(av, b1, c51);

                    av = Avx.BroadcastScalarToVector256(a + 6);
                    c60 = Fma.MultiplyAdd(av, b0, c60);
                    c61 = Fma.MultiplyAdd(av, b1, c61);

                    av = Avx.BroadcastScalarToVector256(a + 7);
                    c70 = Fma.MultiplyAdd(av, b0, c70);
                    c71 = Fma.MultiplyAdd(av, b1, c71);
                }

                Store(c00, c01, cc, Panel);
                if (row + 1 < rows) Store(c10, c11, cc + ldc, Panel);
                if (row + 2 < rows) Store(c20, c21, cc + 2 * ldc, Panel);
                if (row + 3 < rows) Store(c30, c31, cc + 3 * ldc, Panel);
                if (row + 4 < rows) Store(c40, c41, cc + 4 * ldc, Panel);
                if (row + 5 < rows) Store(c50, c51, cc + 5 * ldc, Panel);
                if (row + 6 < rows) Store(c60, c61, cc + 6 * ldc, Panel);
                if (row + 7 < rows) Store(c70, c71, cc + 7 * ldc, Panel);
            }
        }
    }

    /// <summary>
    /// 判断当前面板能否直读：整条落在同一输出行、<c>strideW == 1</c>、每个核位置的
    /// <see cref="Panel"/> 个像素都在界内。可以时把每个核位置相对 <paramref name="inBase"/> 的
    /// 起始下标写进 <paramref name="offsets"/>（长度 ≥ <paramref name="kernelSize"/>）。
    /// </summary>
    private static bool TryTapOffsets(
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
        int kernelSize,
        Span<int> offsets)
    {
        int y = start / outW;
        int x = start - y * outW;

        if (offsets.Length < kernelSize)
            return false;

        // 只有「整条面板不跨行」+「16 个像素连续」才说明一次 64 字节读是合法的
        if (strideW != 1 || x + Panel > outW)
            return false;

        int y0 = y * strideH - padT;
        int x0 = x - padL;

        for (int api = 0, p = 0; api < inPerGroup; api++)
        {
            long channel = (long)api * inH * inW;

            for (int ky = 0; ky < kernelH; ky++)
            {
                int iy = y0 + ky * dilH;
                if ((uint)iy >= (uint)inH)
                    return false;

                long rowBase = channel + (long)iy * inW;

                for (int kx = 0; kx < kernelW; kx++, p++)
                {
                    int ix = x0 + kx * dilW;
                    if (ix < 0 || ix + Panel > inW)
                        return false;

                    offsets[p] = (int)(rowBase + ix);
                }
            }
        }

        return true;
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
