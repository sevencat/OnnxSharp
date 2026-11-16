using System.Buffers;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using OnnxSharp.Core;
using OnnxSharp.Tensors;
using OnnxSharp.Threading;

namespace OnnxSharp.Ops;

/// <summary>
/// 图融合产物：整条 multi-head attention 折成一层。
/// <code>
/// Qh[B,H,T,D] ─┐
/// Kt[B,H,D,T] ─┼→ S = scale·Qh·Kt (+mask) → Softmax(-1) → O = S·Vh → [B,T,H·D]
/// Vh[B,H,T,D] ─┘
/// </code>
/// <para>
/// 省掉的是：scores <c>[B,H,T,T]</c> 的物化（每块 46k 元素的两趟搬运）、输出
/// <c>Transpose+Reshape</c> 的整块拷贝，以及 6 次算子调度；softmax 与两次点积在同一份
/// 缓存友好的缓冲里连着做。
/// </para>
/// <para>
/// 输入布局有两种形态，由 <c>heads</c> 参数选择：
/// </para>
/// <list type="bullet">
/// <item><c>heads &gt; 0</c>：输入是<b>投影后的 3 维张量</b> <c>[B,T,C]</c>（<c>C = heads·D</c>），
/// 已把 <c>Reshape + Transpose</c> 折掉 —— 只在 head 数能从图上静态读出时用。</item>
/// <item><c>heads == 0</c>：输入已经在 head 空间（Q/V 是 <c>[B,H,T,D]</c>、K 是 <c>[B,H,D,T]</c>），
/// <c>heads/head_dim</c> 直接由形状给出 —— 动态形状的图（Paraformer 的 head-split 目标是
/// <c>Shape→Gather→Concat</c> 链）只能走这条。</item>
/// </list>
/// </summary>
internal static class AttentionOps
{
    /// <summary>
    /// 超过这个序列长度就改用 GEMM 分块内核（<see cref="ComputeBlocks"/>）。
    /// <para>
    /// <see cref="ComputeRows"/> 逐行做点积 + axpy，每行都要把整个 K/V 平面重读一遍，
    /// 只有在 K/V 平面小到能留在 L1/L2 且 hsum 开销被摊薄时（ASR 的 T=76）才划算；
    /// T=257 时它比图上的 MatMul+Softmax 慢一倍（66 ms vs 33 ms）。
    /// GEMM 分块把 S 块按 <c>[T][T]</c>（T=257 时 264 KB）算出来，两趟 GEMM 之间不落地，
    /// 于是既拿到 GEMM 的算术密度，又没有 6.3 MB 的 scores 帧缓冲往返。
    /// </para>
    /// <para><c>ONNXSHARP_ATTN_BLOCK_SEQ</c> 设默认值（0 = 全部走 GEMM 分块）。</para>
    /// <para>
    /// 可写（<c>ab</c> 子命令要在同一进程里 A/B 两套内核）；层在构造时<b>捕获</b>当时的取值，
    /// 于是两套配置建出来的两个引擎各自保留自己的内核选择 —— 建完图再改不影响已建的层。
    /// </para>
    /// </summary>
    internal static int BlockKernelMinSeq
    {
        get => _blockKernelMinSeq;
        set => _blockKernelMinSeq = value >= 0 ? value : 0;
    }

    private static int _blockKernelMinSeq =
        int.TryParse(Environment.GetEnvironmentVariable("ONNXSHARP_ATTN_BLOCK_SEQ"), out int blockSeq)
            && blockSeq >= 0
            ? blockSeq
            : 128;

    /// <summary>
    /// 分块内核里**每块分数矩阵 S 的预算字节数**，决定行块宽度
    /// （<c>rowBlock = clamp(budget / (T·4), 8, 256)</c>）。
    /// <para>
    /// 它不是一个「越大越好的缓存参数」：块本身在 L2 里，但**同时**还要装 Q 行块、
    /// K 平面与 V 平面（各 <c>T·D·4</c>）。rf-detr 的 T=1028 时
    /// <c>522 KB + 32 KB + 263 KB + 263 KB = 1.08 MB &gt; L2(1 MB/核)</c>，
    /// 已经在溢出点上，所以这个值要实测扫，不能推。
    /// </para>
    /// <para>
    /// <c>ONNXSHARP_ATTN_SCORE_BUDGET</c>（单位 KB）设默认；与
    /// <see cref="BlockKernelMinSeq"/> 一样是**建层时快照**，供 <c>ab</c> 同进程 A/B。
    /// </para>
    /// </summary>
    internal static int ScoreBudget
    {
        get => _scoreBudget;
        set => _scoreBudget = value > 0 ? value : 512 * 1024;
    }

    private static int _scoreBudget =
        int.TryParse(Environment.GetEnvironmentVariable("ONNXSHARP_ATTN_SCORE_BUDGET"), out int budgetKb)
            && budgetKb > 0
            ? budgetKb * 1024
            : 512 * 1024;

    /// <summary>
    /// 分块内核里把 softmax 的 <c>1/Σ</c> **折到 PV 之后**（乘在输出行的 D 个值上），
    /// 而不是先对整行 T 个分数做归一化再乘 V。
    /// <para>
    /// 数学等价：<c>Σ_j (e_j/Σ)·v_j</c> ≡ <c>(1/Σ)·Σ_j e_j·v_j</c>，只差浮点重结合（1e-7 级）。
    /// 省掉的是每块一次 522 KB 读 + 522 KB 写的 <c>Scale</c> 趟（T=1028 时占 soft 的三分之一）。
    /// </para>
    /// <para><c>ONNXSHARP_ATTN_LAZY_SOFTMAX=0</c> 可一键回退到就地归一化。</para>
    /// </summary>
    internal static bool LazySoftmax
    {
        get => _lazySoftmax;
        set => _lazySoftmax = value;
    }

    private static bool _lazySoftmax =
        Environment.GetEnvironmentVariable("ONNXSHARP_ATTN_LAZY_SOFTMAX") is not "0";

    /// <summary>
    /// 分块内核里把 <c>K^T</c> 与 <c>V</c> 的面板**按 <c>(b,h)</c> 只打一次**，
    /// 供该头的所有行块复用（原来是每块重打一遍）。
    /// <para>
    /// 并行区间是连续的 <c>index = bh·rowBlocks + block</c>，所以每个工作线程的区间天然
    /// 落在少数几个 <c>bh</c> 上 —— 用「记住上次的 bh」就能拿到几乎全部的复用，
    /// 不需要跨线程共享面板（那要同步，且 6 份面板会一起挤 L2）。
    /// </para>
    /// <para>
    /// 代价是每线程多两份面板（<c>T·D·4</c> × 2 ≈ 526 KB @ T=1028/D=64），会挤 L2 ——
    /// 与 <see cref="ScoreBudget"/> 互相影响，两者都要实测。
    /// </para>
    /// <para><c>ONNXSHARP_ATTN_PACK_REUSE=0</c> 可一键回退。</para>
    /// </summary>
    internal static bool PackReuse
    {
        get => _packReuse;
        set => _packReuse = value;
    }

    private static bool _packReuse =
        Environment.GetEnvironmentVariable("ONNXSHARP_ATTN_PACK_REUSE") is not "0";

    public static void Register()
        => LayerRegistry.Register("FusedAttention", p => new AttentionLayer(p));

    private sealed class AttentionLayer : Layer
    {
        private readonly float _scale;
        private readonly int _heads;
        private readonly bool _outputHeadSpace;

        /// <summary>本层构造时的 <see cref="AttentionOps.BlockKernelMinSeq"/>（引擎级快照）。</summary>
        private readonly int _blockMinSeq;

        /// <summary>本层构造时的 <see cref="AttentionOps.ScoreBudget"/>（引擎级快照）。</summary>
        private readonly int _scoreBudget;

        /// <summary>本层构造时的 <see cref="AttentionOps.LazySoftmax"/>（引擎级快照）。</summary>
        private readonly bool _lazySoftmax;

        /// <summary>本层构造时的 <see cref="AttentionOps.PackReuse"/>（引擎级快照）。</summary>
        private readonly bool _packReuse;

        public AttentionLayer(LayerParams parameters)
        {
            Type = parameters.Type;
            Name = parameters.Name;
            _scale = (float)parameters.GetFloat("scale", 1.0);
            _heads = (int)parameters.GetInt("heads", 0);
            _outputHeadSpace = parameters.GetInt("output_head_space", 0) != 0;
            _blockMinSeq = BlockKernelMinSeq;
            _scoreBudget = ScoreBudget;
            _lazySoftmax = LazySoftmax;
            _packReuse = PackReuse;
        }

        private (int B, int H, int T, int D) Resolve(IReadOnlyList<TensorShape> shapes)
        {
            var q = shapes[0];
            var k = shapes[1];
            var v = shapes[2];

            int b, h, t, d;

            if (_heads > 0)
            {
                if (q.Rank != 3 || k.Rank != 3 || v.Rank != 3)
                    throw new OnnxSharpException(
                        $"FusedAttention '{Name}': heads form needs rank-3 inputs ({q} x {k} x {v})");

                b = q[0];
                t = q[1];
                int c = q[2];
                if (c % _heads != 0)
                    throw new OnnxSharpException(
                        $"FusedAttention '{Name}': channels {c} not divisible by heads {_heads}");

                h = _heads;
                d = c / _heads;

                if (k[0] != b || k[1] != t || k[2] != c || v[0] != b || v[1] != t || v[2] != c)
                    throw new OnnxSharpException(
                        $"FusedAttention '{Name}': Q/K/V shapes disagree ({q} x {k} x {v})");
            }
            else
            {
                if (q.Rank != 4 || k.Rank != 4 || v.Rank != 4)
                    throw new OnnxSharpException(
                        $"FusedAttention '{Name}': head-space inputs must be rank 4 ({q} x {k} x {v})");

                b = q[0];
                h = q[1];
                t = q[2];
                d = q[3];

                if (k[0] != b || k[1] != h || k[2] != d || k[3] != t)
                    throw new OnnxSharpException(
                        $"FusedAttention '{Name}': K must be [B,H,D,T] ({q} x {k})");
                if (v[0] != b || v[1] != h || v[2] != t || v[3] != d)
                    throw new OnnxSharpException(
                        $"FusedAttention '{Name}': V must be [B,H,T,D] ({q} x {v})");
            }

            return (b, h, t, d);
        }

        public override void GetMemoryShapes(
            IReadOnlyList<TensorShape> inShapes,
            int requiredOutputs,
            List<TensorShape> outShapes,
            List<TensorShape> tempShapes)
        {
            var (b, h, t, d) = Resolve(inShapes);

            // output_head_space = 0：[B,T,H·D]（下游 Transpose+Reshape 也折掉了）
            // output_head_space = 1：[B,H,T,D]（下游还要用它的 shape 建 Reshape 目标）
            outShapes.Add(_outputHeadSpace
                ? new TensorShape([b, h, t, d])
                : new TensorShape([b, t, h * d]));
        }

        public override void GetTypes(
            IReadOnlyList<DType> inTypes,
            int requiredOutputs,
            int requiredTemps,
            List<DType> outTypes,
            List<DType> tempTypes)
        {
            outTypes.Clear();
            for (int i = 0; i < requiredOutputs; i++)
                outTypes.Add(DType.F32);

            tempTypes.Clear();
            for (int i = 0; i < requiredTemps; i++)
                tempTypes.Add(DType.F32);
        }

        public override void Forward(
            IReadOnlyList<Tensor> inputs,
            IReadOnlyList<Tensor> outputs,
            IReadOnlyList<Tensor> temps)
        {
            var q = inputs[0];
            var k = inputs[1];
            var v = inputs[2];
            var output = outputs[0];

            var (b, h, t, d) = Resolve([q.Shape, k.Shape, v.Shape]);
            if (b <= 0 || h <= 0 || t <= 0 || d <= 0)
                return;

            bool headsForm = _heads > 0;
            int width = h * d;

            // 每路张量在「某个 (b,h) 平面内」的起点步长
            int qStrideT = headsForm ? width : d;
            int kStrideT = headsForm ? width : 1;
            int kStrideD = headsForm ? 1 : t;
            int vStrideT = headsForm ? width : d;

            bool hasMask = Inputs.Count > 3 && Inputs[3].Idx != 0;
            Span<float> mask = default;
            int maskBatch = 0;
            int maskRow = 0;

            if (hasMask)
            {
                var m = inputs[3];
                var ms = m.Shape;
                if (ms.Rank != 4 || ms[0] != b || ms[3] != t || ms[1] != 1 || (ms[2] != 1 && ms[2] != t))
                    throw new OnnxSharpException(
                        $"FusedAttention '{Name}': mask {ms} cannot broadcast to [{b},1,1,{t}]");

                mask = m.AsSpan<float>();
                maskBatch = ms[1] * ms[2] * ms[3];
                maskRow = ms[2] == 1 ? 0 : t;
            }

            var sourceQ = q.AsSpan<float>();
            var sourceK = k.AsSpan<float>();
            var sourceV = v.AsSpan<float>();
            var destination = output.AsSpan<float>();

            // K 先转成 [B,H,T,D]（原始布局对点积不友好），之后每行都是连续读
            float[] keysBuffer = ArrayPool<float>.Shared.Rent(b * h * t * d);

            try
            {
                var keys = keysBuffer.AsSpan(0, b * h * t * d);

                unsafe
                {
                    fixed (float* pq = sourceQ)
                    fixed (float* pk = sourceK)
                    fixed (float* pv = sourceV)
                    fixed (float* po = destination)
                    fixed (float* pm = mask)
                    fixed (float* pkeys = keys)
                    {
                        // 固定指针不能进 lambda（CS1764），先拷成普通局部指针再捕获
                        float* keysPointer = pkeys;
                        float* keyPointer = pk;

                        ParallelRange.For(b * h, (begin, end) =>
                        {
                            for (int bh = begin; bh < end; bh++)
                            {
                                int bi = bh / h;
                                int hi = bh % h;
                                int kBase = headsForm ? bi * t * width + hi * d : bh * d * t;
                                float* plane = keysPointer + (long)bh * t * d;

                                for (int ti = 0; ti < t; ti++)
                                {
                                    int rowBase = kBase + ti * kStrideT;
                                    float* row = plane + (long)ti * d;
                                    for (int di = 0; di < d; di++)
                                        row[di] = keyPointer[rowBase + di * kStrideD];
                                }
                            }
                        });

                        if (t > _blockMinSeq)
                        {
                            ComputeBlocks(b, h, t, d, width, headsForm, qStrideT, vStrideT,
                                pm, maskBatch, maskRow, pq, pv, po, keysBuffer, b * t * width);
                        }
                        else
                        {
                            ComputeRows(b, h, t, d, width, headsForm, qStrideT, vStrideT,
                                pm, maskBatch, maskRow, pq, pv, po, keysPointer);
                        }
                    }
                }
            }
            finally
            {
                ArrayPool<float>.Shared.Return(keysBuffer);
            }
        }

        private unsafe void ComputeRows(
            int b,
            int h,
            int t,
            int d,
            int width,
            bool headsForm,
            int qStrideT,
            int vStrideT,
            float* mask,
            int maskBatch,
            int maskRow,
            float* sourceQ,
            float* sourceV,
            float* destination,
            float* keys)
        {
            bool hasMask = mask is not null;

            ParallelRange.For(b * h * t, (begin, end) =>
            {
                float[] scoreBuffer = ArrayPool<float>.Shared.Rent(t);
                float[] accBuffer = ArrayPool<float>.Shared.Rent(d);
                float[] rowBuffer = ArrayPool<float>.Shared.Rent(d);

                try
                {
                    fixed (float* scores = scoreBuffer)
                    fixed (float* accumulator = accBuffer)
                    fixed (float* queryRow = rowBuffer)
                    {
                        for (int index = begin; index < end; index++)
                        {
                            int ti = index % t;
                            int bh = index / t;
                            int bi = bh / h;
                            int hi = bh % h;

                            int planeBase = headsForm ? bi * t * width + hi * d : bh * t * d;

                            // 与图上一致：scale 乘在 Qh 上（Mul(Qh, 1/√D)），不是乘在 scores 上
                            int qBase = planeBase + ti * qStrideT;
                            for (int di = 0; di < d; di++)
                                queryRow[di] = sourceQ[qBase + di] * _scale;

                            float* keyPlane = keys + (long)bh * t * d;
                            for (int j = 0; j < t; j++)
                                scores[j] = Dot(queryRow, keyPlane + (long)j * d, d);

                            if (hasMask)
                            {
                                float* rowMask = mask + maskBatch * bi + maskRow * ti;
                                for (int j = 0; j < t; j++)
                                    scores[j] += rowMask[j];
                            }

                            VectorMath.SoftmaxRow(scores, scores, t);

                            for (int di = 0; di < d; di++)
                                accumulator[di] = 0f;

                            for (int j = 0; j < t; j++)
                            {
                                float weight = scores[j];
                                if (weight == 0f)
                                    continue;

                                Axpy(accumulator, sourceV + planeBase + j * vStrideT, d, weight);
                            }

                            float* row = _outputHeadSpace
                                ? destination + (long)(bi * h + hi) * t * d + (long)ti * d
                                : destination + (long)(bi * t + ti) * width + hi * d;
                            for (int di = 0; di < d; di++)
                                row[di] = accumulator[di];
                        }
                    }
                }
                finally
                {
                    ArrayPool<float>.Shared.Return(scoreBuffer);
                    ArrayPool<float>.Shared.Return(accBuffer);
                    ArrayPool<float>.Shared.Return(rowBuffer);
                }
            });
        }

        /// <summary>
        /// 大 T 的 GEMM 分块内核：把 <c>S = Q·K^T → Softmax → O = S·V</c> 按 **Br 行**切开，
        /// 一块 S 只有 <c>Br×T</c>（下面 budget 控制在 ~512 KB，落 L2），算完就地 softmax、
        /// 立刻乘 V，**不经过帧缓冲**。
        /// <para>
        /// 为什么按行切而不是整块：rf-detr 的 T 是 1028（512/16 的 32×32 加 4 个 register token），
        /// 整块 <c>S</c> 是 4.2 MB，超过 L2 一倍多 —— 那样只是把「25 MB 的 [B,H,T,T] 往返」
        /// 换成「每个 head 4.2 MB 往返」，白忙。按行切之后 S 块的写/读都在 L2 里，
        /// 剩下的成本只有 K/V 被每个行块重读一遍（<c>rows_per_block</c> 份）。
        /// </para>
        /// <para>
        /// ⚠ 不走 online-softmax：整行的 T 个分数都在 S 块里，max/sum 一遍就能算完，
        /// 不需要 rescale 累加器。代价是 K/V 要被重读，收益是 softmax 侧零额外状态。
        /// </para>
        /// </summary>
        private static readonly bool TraceBlocks =
            Environment.GetEnvironmentVariable("ONNXSHARP_ATTN_TRACE") == "1";

        private static long _traceCopy;
        private static long _traceQk;
        private static long _traceSoft;
        private static long _tracePv;
        private static long _traceCount;

        private unsafe void ComputeBlocks(
            int b,
            int h,
            int t,
            int d,
            int width,
            bool headsForm,
            int qStrideT,
            int vStrideT,
            float* mask,
            int maskBatch,
            int maskRow,
            float* sourceQ,
            float* sourceV,
            float* destination,
            float[] keyBuffer,
            int outputLength)
        {
            int plane = checked(t * d);
            int outputStride = _outputHeadSpace ? d : width;

            // 每块 S 的目标大小：默认 512 KB 给分数矩阵，剩下留给 K/V 平面（各 t·d·4）；
            // 溢出点上要实测扫，见 AttentionOps.ScoreBudget 的注释。
            int rowBlock = Math.Clamp(_scoreBudget / (t * sizeof(float)), 8, 256);
            int rowBlocks = (t + rowBlock - 1) / rowBlock;

            int qBlockCount = checked(rowBlock * d);
            int scoreBlockCount = checked(rowBlock * t);

            ParallelRange.For(b * h * rowBlocks, (begin, end) =>
            {
                float[] qBuffer = ArrayPool<float>.Shared.Rent(qBlockCount);
                float[] sBuffer = ArrayPool<float>.Shared.Rent(scoreBlockCount);
                float[] sumBuffer = ArrayPool<float>.Shared.Rent(rowBlock);
                float[] ktBuffer = ArrayPool<float>.Shared.Rent(GemmKernel.PackedPanelSize(d, t));
                float[] vBuffer = ArrayPool<float>.Shared.Rent(GemmKernel.PackedPanelSize(t, d));

                try
                {
                    fixed (float* pq = qBuffer)
                    fixed (float* ps = sBuffer)
                    fixed (float* pinv = sumBuffer)
                    fixed (float* pkey = keyBuffer)
                    {
                        long accCopy = 0, accQk = 0, accSoft = 0, accPv = 0, accBlocks = 0;

                        // 面板按 (b,h) 复用：区间在 index 上是连续的，所以只需记住上一次的 bh
                        GemmKernel.PackedB? ktPanel = null;
                        GemmKernel.PackedB? vPanel = null;
                        int cachedBh = -1;

                        for (int index = begin; index < end; index++)
                        {
                            int block = index % rowBlocks;
                            int bh = index / rowBlocks;
                            int bi = bh / h;
                            int hi = bh % h;
                            int r0 = block * rowBlock;
                            int rows = Math.Min(rowBlock, t - r0);

                            int planeBase = headsForm ? bi * t * width + hi * d : bh * t * d;

                            // 计时只在 ONNXSHARP_ATTN_TRACE=1 时取样：四次 rdtsc 本身很便宜，
                            // 但没必要在默认路径上留（0 常量会被 JIT 折掉）。
                            long t0 = TraceBlocks ? System.Diagnostics.Stopwatch.GetTimestamp() : 0;

                            // Q 行块紧凑化并乘 scale（与图上 Mul(Qh, 1/√D) 等价）
                            for (int r = 0; r < rows; r++)
                            {
                                float* src = sourceQ + planeBase + (long)(r0 + r) * qStrideT;
                                float* qRow = pq + (long)r * d;
                                for (int di = 0; di < d; di++)
                                    qRow[di] = src[di] * _scale;
                            }

                            long t1 = TraceBlocks ? System.Diagnostics.Stopwatch.GetTimestamp() : 0;

                            if (_packReuse && bh != cachedBh)
                            {
                                // K^T：[k=d][n=t]，源是 [T][D] 的 K 平面（行距 1、列距 d）
                                ktPanel = GemmKernel.PackFromSource(pkey + (long)bh * plane, d, 1, d, t, ktBuffer);

                                // V：[k=t][n=d]，源是 [T][D]（行距 vStrideT）
                                vPanel = GemmKernel.PackFromSource(sourceV + planeBase, t, vStrideT, 1, d, vBuffer);
                                cachedBh = bh;
                            }

                            // S 块 = Q块·K^T：[rows][t]
                            if (_packReuse)
                            {
                                GemmKernel.MultiplyPacked(
                                    new ReadOnlySpan<float>(pq, qBlockCount),
                                    ktPanel!,
                                    new Span<float>(ps, scoreBlockCount),
                                    rows,
                                    t,
                                    parallel: false);
                            }
                            else
                            {
                                GemmKernel.MultiplyFromSource(
                                    new ReadOnlySpan<float>(pq, qBlockCount),
                                    keyBuffer,
                                    bh * plane,
                                    d,
                                    1,
                                    d,
                                    t,
                                    new Span<float>(ps, scoreBlockCount),
                                    rows,
                                    t);
                            }

                            long t2 = TraceBlocks ? System.Diagnostics.Stopwatch.GetTimestamp() : 0;

                            if (mask is not null)
                            {
                                for (int r = 0; r < rows; r++)
                                {
                                    float* sRow = ps + (long)r * t;
                                    float* mRow = mask + (long)maskBatch * bi + (long)maskRow * (r0 + r);
                                    for (int j = 0; j < t; j++)
                                        sRow[j] += mRow[j];
                                }
                            }

                            for (int r = 0; r < rows; r++)
                            {
                                float* sRow = ps + (long)r * t;

                                if (_lazySoftmax)
                                {
                                    // 只出 exp 值与 Σ；1/Σ 等 PV 之后乘到输出行（D 个值）上
                                    float max = VectorMath.MaxRow(sRow, t);
                                    pinv[r] = 1f / VectorMath.SumExp(sRow, sRow, t, max);
                                }
                                else
                                {
                                    VectorMath.SoftmaxRow(sRow, sRow, t);
                                }
                            }

                            long t3 = TraceBlocks ? System.Diagnostics.Stopwatch.GetTimestamp() : 0;

                            // O = S块·V：[rows][d]
                            int offset = _outputHeadSpace
                                ? bh * t * d + r0 * d
                                : (bi * t + r0) * width + hi * d;

                            if (_packReuse)
                            {
                                GemmKernel.MultiplyPacked(
                                    new ReadOnlySpan<float>(ps, scoreBlockCount),
                                    vPanel!,
                                    new Span<float>(destination + offset, outputLength - offset),
                                    rows,
                                    outputStride,
                                    parallel: false);
                            }
                            else
                            {
                                GemmKernel.MultiplyFromSource(
                                    new ReadOnlySpan<float>(ps, scoreBlockCount),
                                    sourceV + planeBase,
                                    t,
                                    vStrideT,
                                    1,
                                    d,
                                    new Span<float>(destination + offset, outputLength - offset),
                                    rows,
                                    outputStride);
                            }

                            if (_lazySoftmax)
                            {
                                for (int r = 0; r < rows; r++)
                                {
                                    float* outRow = destination + offset + (long)r * outputStride;
                                    VectorMath.Scale(outRow, outRow, d, pinv[r]);
                                }
                            }

                            if (TraceBlocks)
                            {
                                long t4 = System.Diagnostics.Stopwatch.GetTimestamp();
                                accCopy += t1 - t0;
                                accQk += t2 - t1;
                                accSoft += t3 - t2;
                                accPv += t4 - t3;
                                accBlocks++;
                            }
                        }

                        if (TraceBlocks && accBlocks > 0)
                        {
                            Interlocked.Add(ref _traceCopy, accCopy);
                            Interlocked.Add(ref _traceQk, accQk);
                            Interlocked.Add(ref _traceSoft, accSoft);
                            Interlocked.Add(ref _tracePv, accPv);
                            long total = Interlocked.Add(ref _traceCount, accBlocks);
                            double k = 1000.0 / (System.Diagnostics.Stopwatch.Frequency * total);
                            Console.Error.WriteLine(
                                $"[attnblk] blocks={total} per-block ms: copy {_traceCopy * k:F3} " +
                                $"qk {_traceQk * k:F3} soft {_traceSoft * k:F3} pv {_tracePv * k:F3}");
                        }
                    }
                }
                finally
                {
                    ArrayPool<float>.Shared.Return(qBuffer);
                    ArrayPool<float>.Shared.Return(sBuffer);
                    ArrayPool<float>.Shared.Return(sumBuffer);
                    ArrayPool<float>.Shared.Return(ktBuffer);
                    ArrayPool<float>.Shared.Return(vBuffer);
                }
            });
        }

        private static unsafe float Dot(float* a, float* b, int n)
        {
            int i = 0;
            float sum = 0f;

            if (Fma.IsSupported)
            {
                var accumulator = Vector256<float>.Zero;
                for (; i + 8 <= n; i += 8)
                    accumulator = Fma.MultiplyAdd(Vector256.Load(a + i), Vector256.Load(b + i), accumulator);

                sum = HorizontalSum(accumulator);
            }

            for (; i < n; i++)
                sum += a[i] * b[i];

            return sum;
        }

        private static unsafe void Axpy(float* destination, float* x, int n, float factor)
        {
            int i = 0;

            if (Fma.IsSupported)
            {
                var vFactor = Vector256.Create(factor);
                for (; i + 8 <= n; i += 8)
                    Fma.MultiplyAdd(vFactor, Vector256.Load(x + i), Vector256.Load(destination + i))
                        .Store(destination + i);
            }

            for (; i < n; i++)
                destination[i] += factor * x[i];
        }

        private static float HorizontalSum(Vector256<float> value)
        {
            var lo = Avx.Add(value.GetLower(), value.GetUpper());
            lo = Vector128.Add(lo, Vector128.Shuffle(lo, Vector128.Create(2, 3, 0, 1)));
            lo = Vector128.Add(lo, Vector128.Shuffle(lo, Vector128.Create(1, 0, 3, 2)));
            return lo.GetElement(0);
        }
    }
}
