using System.Buffers;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using OnnxSharp.Core;
using OnnxSharp.Tensors;
using OnnxSharp.Threading;

namespace OnnxSharp.Kernels;

/// <summary>
/// 3×3 / stride 1 / pad 1 / group 1 卷积的 Winograd F(2×2, 3×3)。
/// <para>
/// 直接卷积算 2×2 个输出点要做 4×9 = 36 次乘加，Winograd 先把 4×4 输入块和 3×3 核
/// 各变换一次，再在变换域逐点相乘，只要 16 次乘法 —— 乘法量降到 4/9。
/// </para>
/// <para>
/// 变换后 16 个分量各自是一个独立的 <c>[oc, ic] × [ic, tiles]</c> 矩阵乘，
/// 直接复用 <see cref="GemmKernel"/>。权重是常量，变换结果按 Tensor 实例缓存。
/// </para>
/// </summary>
internal static unsafe class Winograd
{
    /// <summary>输出块边长（2×2）。</summary>
    private const int TileOut = 2;

    /// <summary>变换域边长（4×4）。</summary>
    private const int Alpha = 4;

    private const int Components = Alpha * Alpha;

    /// <summary>权重变换结果的上限（float 个数），超过就交回 im2col，免得白占几十 MB。</summary>
    private const long MaxTransformedWeights = 1L << 24;

    /// <summary>
    /// 允许走 Winograd 的最大输出空间像素数（默认 64×64）。超过就交回 im2col。
    /// 抬到 <see cref="long.MaxValue"/> 可以强制所有 3×3 层都走 Winograd（调参/基准用）。
    /// </summary>
    internal static long MaxSpatialPixels =
        long.TryParse(Environment.GetEnvironmentVariable("ONNXSHARP_WINO_MAX"), out long winoMax) ? winoMax : 4096;

    private static readonly ConditionalWeakTable<Tensor, float[]> WeightCache = new();

    /// <summary>一次并行处理的 tile 数：8 个 tile 正好是一条 256 位向量。</summary>
    private const int TileVector = 8;

    /// <summary>抽取偶数位（0,2,4,6）——取出 tile 的第 0、2 列。</summary>
    private static readonly Vector256<int> EvenLanes = Vector256.Create(0, 2, 4, 6, 0, 0, 0, 0);

    /// <summary>抽取奇数位（1,3,5,7）——取出 tile 的第 1、3 列。</summary>
    private static readonly Vector256<int> OddLanes = Vector256.Create(1, 3, 5, 7, 1, 1, 1, 1);

    /// <summary>整条向量左移一个 lane（右端空出的 lane 后面用 blend 补）。</summary>
    private static readonly Vector256<int> ShiftLanes = Vector256.Create(1, 2, 3, 4, 5, 6, 7, 7);

    /// <summary>判断这层能不能走 Winograd，能就执行并返回 true。</summary>
    public static bool TryConv3x3(
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
        if (groups != 1 || strideH != 1 || strideW != 1 || dilH != 1 || dilW != 1)
            return false;

        if (padT != 1 || padL != 1)
            return false;

        var inDims = input.Shape.Dims;
        var wDims = weight.Shape.Dims;

        if (inDims.Length != 4 || wDims.Length != 4)
            return false;

        if (wDims[2] != 3 || wDims[3] != 3)
            return false;

        int batches = inDims[0];
        int channels = inDims[1];
        int inH = inDims[2];
        int inW = inDims[3];
        int outChannels = wDims[0];

        if (wDims[1] != channels || channels < 8 || outChannels < 8)
            return false;

        var outDims = output.Shape.Dims;
        if (outDims[2] != inH || outDims[3] != inW)
            return false;               // pad 不是 1/1 时输出尺寸会变，交回 im2col

        if (inH < 6 || inW < 6)
            return false;

        // 输入/输出变换是标量散写循环，实测每个 tile·通道约 30 拍，摊到 GEMM 上
        // 相当于 30 个输出通道的成本。空间大时 im2col 的复用更好：
        // 160×160 的 32→32 上 Winograd 4.3 ms、im2col 1.7 ms；80×80 的 64→64 是 2.2 vs 1.3。
        // 到 40×40 以下 im2col 的并行度和复用一起塌掉，Winograd 反超 2~4 倍。
        if ((long)inH * inW > MaxSpatialPixels)
            return false;

        if ((long)outChannels * channels * Components > MaxTransformedWeights)
            return false;

        var transformed = WeightCache.GetValue(weight, w => TransformWeights(w, outChannels, channels));

        Run(input, transformed, bias, output, batches, channels, outChannels, inH, inW);
        return true;
    }

    /// <summary>G g Gᵀ，输出按 <c>[分量][oc][ic]</c> 排布（GEMM 的 A 需要 [m, k] = [oc, ic]）。</summary>
    private static float[] TransformWeights(Tensor weight, int outChannels, int inChannels)
    {
        var src = weight.AsSpan<float>();
        var dst = new float[Components * outChannels * inChannels];

        Span<float> s = stackalloc float[12];
        Span<float> u = stackalloc float[16];

        for (int oc = 0; oc < outChannels; oc++)
        {
            for (int ic = 0; ic < inChannels; ic++)
            {
                var g = src.Slice((oc * inChannels + ic) * 9, 9);

                // s = G g（4×3）
                for (int j = 0; j < 3; j++)
                {
                    float g0 = g[j], g1 = g[3 + j], g2 = g[6 + j];
                    s[j] = g0;
                    s[3 + j] = (g0 + g1 + g2) * 0.5f;
                    s[6 + j] = (g0 - g1 + g2) * 0.5f;
                    s[9 + j] = g2;
                }

                // u = s Gᵀ（4×4）
                for (int i = 0; i < 4; i++)
                {
                    float a = s[i * 3], b = s[i * 3 + 1], c = s[i * 3 + 2];
                    u[i * 4] = a;
                    u[i * 4 + 1] = (a + b + c) * 0.5f;
                    u[i * 4 + 2] = (a - b + c) * 0.5f;
                    u[i * 4 + 3] = c;
                }

                long plane = (long)outChannels * inChannels;
                for (int comp = 0; comp < Components; comp++)
                    dst[comp * plane + (long)oc * inChannels + ic] = u[comp];
            }
        }

        return dst;
    }

    private static void Run(
        Tensor input,
        float[] transformedWeights,
        Tensor? bias,
        Tensor output,
        int batches,
        int inChannels,
        int outChannels,
        int inH,
        int inW)
    {
        int tilesY = (inH + TileOut - 1) / TileOut;
        int tilesX = (inW + TileOut - 1) / TileOut;

        int threads = ParallelRange.Threads;

        // V 与 M 各占 16×max(ic, oc)×tiles 个 float，单块控制在一两 MB
        int budgetTiles = Math.Max(threads * 16, (1 << 21) / (4 * Components * Math.Max(inChannels, outChannels)));
        int rowBlock = Math.Max(1, Math.Min(tilesY, budgetTiles / Math.Max(1, tilesX)));

        var biasSpan = bias is null ? Span<float>.Empty : bias.AsSpan<float>();

        for (int b = 0; b < batches; b++)
        {
            for (int ty0 = 0; ty0 < tilesY; ty0 += rowBlock)
            {
                int tyCount = Math.Min(rowBlock, tilesY - ty0);
                int tiles = tyCount * tilesX;
                long vPlane = (long)inChannels * tiles;
                long mPlane = (long)outChannels * tiles;

                var poolV = ArrayPool<float>.Shared.Rent((int)(Components * vPlane));
                var poolM = ArrayPool<float>.Shared.Rent((int)(Components * mPlane));

                try
                {
                    fixed (float* v = poolV)
                    fixed (float* m = poolM)
                    fixed (float* u = transformedWeights)
                    fixed (float* src = input.AsSpan<float>())
                    fixed (float* dst = output.AsSpan<float>())
                    fixed (float* biasPtr = biasSpan)
                    {
                        TransformInput(src, v, b, inChannels, inH, inW, tilesX, ty0, tyCount, tiles);

                        MultiplyComponents(u, v, m, inChannels, outChannels, tiles, vPlane, mPlane);

                        TransformOutput(dst, m, biasPtr, b, outChannels, inH, inW, tilesX, ty0, tyCount, tiles);
                    }
                }
                finally
                {
                    ArrayPool<float>.Shared.Return(poolV);
                    ArrayPool<float>.Shared.Return(poolM);
                }
            }
        }
    }

    /// <summary>
    /// 16 个分量各做一次 <c>U[oc,ic] · V[ic,tiles]</c>。
    /// <para>
    /// 分量之间完全独立，外层按分量并行、内层 GEMM 串行即可吃满多核；
    /// 指针当参数传进来是因为 fixed 局部变量不能进 lambda。
    /// </para>
    /// </summary>
    private static void MultiplyComponents(
        float* u,
        float* v,
        float* m,
        int inChannels,
        int outChannels,
        int tiles,
        long vPlane,
        long mPlane)
    {
        long uPlane = (long)outChannels * inChannels;

        ParallelRange.For(Components, (begin, end) =>
        {
            for (int comp = begin; comp < end; comp++)
            {
                GemmKernel.Multiply(
                    new ReadOnlySpan<float>(u + comp * uPlane, (int)uPlane),
                    new ReadOnlySpan<float>(v + comp * vPlane, (int)vPlane),
                    new Span<float>(m + comp * mPlane, (int)mPlane),
                    outChannels, inChannels, tiles,
                    ldc: tiles, parallel: false);
            }
        });
    }

    /// <summary>V = Bᵀ d B，把每个 4×4 输入块（含 pad）展开成 16 个分量。</summary>
    private static void TransformInput(
        float* src,
        float* v,
        int batch,
        int inChannels,
        int inH,
        int inW,
        int tilesX,
        int ty0,
        int tyCount,
        int tiles)
    {
        long batchOffset = (long)batch * inChannels * inH * inW;
        long channelStride = (long)inH * inW;
        long compStride = (long)inChannels * tiles;

        ParallelRange.For(inChannels, (c0, c1) =>
        {
            float* d = stackalloc float[16];
            float* t = stackalloc float[16];

            for (int c = c0; c < c1; c++)
            {
                float* channel = src + batchOffset + c * channelStride;
                float* vBase = v + (long)c * tiles;

                for (int ty = 0; ty < tyCount; ty++)
                {
                    int oy = (ty0 + ty) * TileOut;
                    int y0 = oy - 1;
                    int tileRow = ty * tilesX;
                    int tx = 0;

                    // 一次 8 个 tile：4 行整块在界内时走向量化路径，边界与余数仍走标量
                    if (y0 >= 0 && y0 + Alpha <= inH && tilesX >= TileVector)
                    {
                        for (; tx <= tilesX - TileVector; tx += TileVector)
                        {
                            int vx = tx * TileOut - 1;

                            if (vx < 0 || vx + 2 * TileVector + 2 > inW)
                                break;

                            TransformTilesVector(channel, vBase, compStride, y0, vx, inW, tileRow + tx);
                        }
                    }

                    for (; tx < tilesX; tx++)
                    {
                        int ox = tx * TileOut;
                        int x0 = ox - 1;

                        if (y0 >= 0 && y0 + Alpha <= inH && x0 >= 0 && x0 + Alpha <= inW)
                        {
                            // 内部 tile：4 行 × 4 列全部在界内，直接连续读
                            for (int i = 0; i < Alpha; i++)
                            {
                                float* row = channel + (long)(y0 + i) * inW + x0;
                                d[i * 4] = row[0];
                                d[i * 4 + 1] = row[1];
                                d[i * 4 + 2] = row[2];
                                d[i * 4 + 3] = row[3];
                            }
                        }
                        else
                        {
                            for (int i = 0; i < Alpha; i++)
                            {
                                int y = y0 + i;
                                bool rowOk = (uint)y < (uint)inH;
                                for (int j = 0; j < Alpha; j++)
                                {
                                    int x = x0 + j;
                                    d[i * 4 + j] = rowOk && (uint)x < (uint)inW
                                        ? channel[(long)y * inW + x]
                                        : 0f;
                                }
                            }
                        }

                        // t = Bᵀ d（垂直 4 点一组）
                        for (int j = 0; j < Alpha; j++)
                        {
                            float d0 = d[j], d1 = d[4 + j], d2 = d[8 + j], d3 = d[12 + j];
                            t[j] = d0 - d2;
                            t[4 + j] = d1 + d2;
                            t[8 + j] = d2 - d1;
                            t[12 + j] = d1 - d3;
                        }

                        // v = t B（水平），顺手散写进 [分量][ic][tile]
                        int tile = tileRow + tx;
                        for (int i = 0; i < Alpha; i++)
                        {
                            float* r = t + i * Alpha;
                            float a = r[0], b = r[1], c2 = r[2], d3 = r[3];

                            vBase[(long)(i * Alpha) * compStride + tile] = a - c2;
                            vBase[(long)(i * Alpha + 1) * compStride + tile] = b + c2;
                            vBase[(long)(i * Alpha + 2) * compStride + tile] = c2 - b;
                            vBase[(long)(i * Alpha + 3) * compStride + tile] = b - d3;
                        }
                    }
                }
            }
        });
    }

    /// <summary>
    /// 一次变换 8 个连续 tile（<see cref="TileVector"/> 个）：
    /// 每行读两整条向量，用 <c>PermuteVar8x32</c> 把 4 列按 lane 拆开，
    /// 之后所有加减都是整条向量的运算，最后 16 条向量直接写进 16 个分量平面。
    /// <para>
    /// 调用方保证 <paramref name="y0"/> 与 <paramref name="y0"/>+3 在界内，
    /// 且 <paramref name="x0"/> ≥ 0、<paramref name="x0"/>+17 &lt; inW。
    /// </para>
    /// </summary>
    private static void TransformTilesVector(
        float* channel,
        float* vBase,
        long compStride,
        int y0,
        int x0,
        int inW,
        int tile)
    {
        Vector256<float>* rows = stackalloc Vector256<float>[Alpha * Alpha];
        Vector256<float>* t = stackalloc Vector256<float>[Alpha * Alpha];

        for (int i = 0; i < Alpha; i++)
        {
            float* p = channel + (long)(y0 + i) * inW + x0;
            var va = Avx.LoadVector256(p);
            var vb = Avx.LoadVector256(p + 8);

            // 偶数位 = tile 的第 0 列，奇数位 = 第 1 列
            var c0 = Avx2.Permute2x128(
                Avx2.PermuteVar8x32(va, EvenLanes),
                Avx2.PermuteVar8x32(vb, EvenLanes), 0x20);
            var c1 = Avx2.Permute2x128(
                Avx2.PermuteVar8x32(va, OddLanes),
                Avx2.PermuteVar8x32(vb, OddLanes), 0x20);

            // 第 2、3 列是同一序列右移一个 tile
            var c2 = Avx2.Blend(Avx2.PermuteVar8x32(c0, ShiftLanes), Vector256.Create(p[16]), 0x80);
            var c3 = Avx2.Blend(Avx2.PermuteVar8x32(c1, ShiftLanes), Vector256.Create(p[17]), 0x80);

            rows[i * Alpha] = c0;
            rows[i * Alpha + 1] = c1;
            rows[i * Alpha + 2] = c2;
            rows[i * Alpha + 3] = c3;
        }

        // t = Bᵀ d
        for (int j = 0; j < Alpha; j++)
        {
            var d0 = rows[j];
            var d1 = rows[Alpha + j];
            var d2 = rows[2 * Alpha + j];
            var d3 = rows[3 * Alpha + j];

            t[j] = Avx.Subtract(d0, d2);
            t[Alpha + j] = Avx.Add(d1, d2);
            t[2 * Alpha + j] = Avx.Subtract(d2, d1);
            t[3 * Alpha + j] = Avx.Subtract(d1, d3);
        }

        // v = t B
        for (int i = 0; i < Alpha; i++)
        {
            var a = t[i * Alpha];
            var b = t[i * Alpha + 1];
            var c2 = t[i * Alpha + 2];
            var d3 = t[i * Alpha + 3];

            Avx.Store(vBase + (long)(i * Alpha) * compStride + tile, Avx.Subtract(a, c2));
            Avx.Store(vBase + (long)(i * Alpha + 1) * compStride + tile, Avx.Add(b, c2));
            Avx.Store(vBase + (long)(i * Alpha + 2) * compStride + tile, Avx.Subtract(c2, b));
            Avx.Store(vBase + (long)(i * Alpha + 3) * compStride + tile, Avx.Subtract(b, d3));
        }
    }

    /// <summary>Y = Aᵀ M A，把 16 个分量还原成 2×2 输出块。</summary>
    private static void TransformOutput(
        float* dst,
        float* m,
        float* bias,
        int batch,
        int outChannels,
        int inH,
        int inW,
        int tilesX,
        int ty0,
        int tyCount,
        int tiles)
    {
        long batchOffset = (long)batch * outChannels * inH * inW;
        long channelStride = (long)inH * inW;
        long compStride = (long)outChannels * tiles;
        bool hasBias = bias != null;

        ParallelRange.For(outChannels, (c0, c1) =>
        {
            float* t = stackalloc float[8];

            for (int oc = c0; oc < c1; oc++)
            {
                float* channel = dst + batchOffset + oc * channelStride;
                float* mRow = m + (long)oc * tiles;
                float b = hasBias ? bias[oc] : 0f;

                for (int ty = 0; ty < tyCount; ty++)
                {
                    int oy = (ty0 + ty) * TileOut;
                    int tileRow = ty * tilesX;
                    int tx = 0;

                    // 一次 8 个 tile：输出正好是两行各 16 个连续像素
                    if (oy + TileOut <= inH && tilesX >= TileVector)
                    {
                        for (; tx <= tilesX - TileVector; tx += TileVector)
                        {
                            int vox = tx * TileOut;

                            if (vox + 2 * TileVector > inW)
                                break;

                            TransformOutputVector(channel, mRow, compStride, b, oy, vox, inW, tileRow + tx);
                        }
                    }

                    for (; tx < tilesX; tx++)
                    {
                        int ox = tx * TileOut;
                        int tile = tileRow + tx;

                        // Aᵀ M（2×4）：每个分量在 M 里是一条独立的 [oc][tile] 平面
                        for (int j = 0; j < Alpha; j++)
                        {
                            float m0 = mRow[(long)j * compStride + tile];
                            float m1 = mRow[(long)(4 + j) * compStride + tile];
                            float m2 = mRow[(long)(8 + j) * compStride + tile];
                            float m3 = mRow[(long)(12 + j) * compStride + tile];

                            t[j] = m0 + m1 + m2;
                            t[4 + j] = m1 - m2 - m3;
                        }

                        // Y = (Aᵀ M) A（2×2），bias 直接融进来
                        float y00 = t[0] + t[1] + t[2] + b;
                        float y01 = t[1] - t[2] - t[3] + b;
                        float y10 = t[4] + t[5] + t[6] + b;
                        float y11 = t[5] - t[6] - t[7] + b;

                        if ((uint)oy < (uint)inH)
                        {
                            float* row = channel + (long)oy * inW;
                            if ((uint)ox < (uint)inW)
                                row[ox] = y00;
                            if ((uint)(ox + 1) < (uint)inW)
                                row[ox + 1] = y01;
                        }

                        if ((uint)(oy + 1) < (uint)inH)
                        {
                            float* row = channel + (long)(oy + 1) * inW;
                            if ((uint)ox < (uint)inW)
                                row[ox] = y10;
                            if ((uint)(ox + 1) < (uint)inW)
                                row[ox + 1] = y11;
                        }
                    }
                }
            }
        });
    }

    /// <summary>
    /// 一次还原 8 个 tile（<see cref="TileVector"/> 个）：16 条分量向量做 Aᵀ M A，
    /// 再把 y00/y01 交错回 16 个连续像素，两行各写两条整向量。
    /// <para>调用方保证 <paramref name="oy"/>+1 &lt; inH 且 <paramref name="ox"/>+15 &lt; inW。</para>
    /// </summary>
    private static unsafe void TransformOutputVector(
        float* channel,
        float* mRow,
        long compStride,
        float bias,
        int oy,
        int ox,
        int inW,
        int tile)
    {
        var bv = Vector256.Create(bias);

        Vector256<float> t00, t01, t02, t03, t10, t11, t12, t13;

        {
            var m0 = Avx.LoadVector256(mRow + (long)0 * compStride + tile);
            var m1 = Avx.LoadVector256(mRow + (long)1 * compStride + tile);
            var m2 = Avx.LoadVector256(mRow + (long)2 * compStride + tile);
            var m3 = Avx.LoadVector256(mRow + (long)3 * compStride + tile);

            var n0 = Avx.LoadVector256(mRow + (long)4 * compStride + tile);
            var n1 = Avx.LoadVector256(mRow + (long)5 * compStride + tile);
            var n2 = Avx.LoadVector256(mRow + (long)6 * compStride + tile);
            var n3 = Avx.LoadVector256(mRow + (long)7 * compStride + tile);

            var o0 = Avx.LoadVector256(mRow + (long)8 * compStride + tile);
            var o1 = Avx.LoadVector256(mRow + (long)9 * compStride + tile);
            var o2 = Avx.LoadVector256(mRow + (long)10 * compStride + tile);
            var o3 = Avx.LoadVector256(mRow + (long)11 * compStride + tile);

            var p0 = Avx.LoadVector256(mRow + (long)12 * compStride + tile);
            var p1 = Avx.LoadVector256(mRow + (long)13 * compStride + tile);
            var p2 = Avx.LoadVector256(mRow + (long)14 * compStride + tile);
            var p3 = Avx.LoadVector256(mRow + (long)15 * compStride + tile);

            // Aᵀ M
            t00 = Avx.Add(Avx.Add(m0, n0), o0);
            t01 = Avx.Add(Avx.Add(m1, n1), o1);
            t02 = Avx.Add(Avx.Add(m2, n2), o2);
            t03 = Avx.Add(Avx.Add(m3, n3), o3);

            t10 = Avx.Subtract(Avx.Subtract(n0, o0), p0);
            t11 = Avx.Subtract(Avx.Subtract(n1, o1), p1);
            t12 = Avx.Subtract(Avx.Subtract(n2, o2), p2);
            t13 = Avx.Subtract(Avx.Subtract(n3, o3), p3);
        }

        // 运算顺序与标量路径保持一致，免得引入额外的舍入偏差
        var y00 = Avx.Add(Avx.Add(Avx.Add(t00, t01), t02), bv);
        var y01 = Avx.Add(Avx.Subtract(Avx.Subtract(t01, t02), t03), bv);
        var y10 = Avx.Add(Avx.Add(Avx.Add(t10, t11), t12), bv);
        var y11 = Avx.Add(Avx.Subtract(Avx.Subtract(t11, t12), t13), bv);

        float* top = channel + (long)oy * inW + ox;
        float* bottom = top + inW;

        var lo = Avx2.Permute2x128(Avx.UnpackLow(y00, y01), Avx.UnpackHigh(y00, y01), 0x20);
        var hi = Avx2.Permute2x128(Avx.UnpackLow(y00, y01), Avx.UnpackHigh(y00, y01), 0x31);
        Avx.Store(top, lo);
        Avx.Store(top + 8, hi);

        lo = Avx2.Permute2x128(Avx.UnpackLow(y10, y11), Avx.UnpackHigh(y10, y11), 0x20);
        hi = Avx2.Permute2x128(Avx.UnpackLow(y10, y11), Avx.UnpackHigh(y10, y11), 0x31);
        Avx.Store(bottom, lo);
        Avx.Store(bottom + 8, hi);
    }
}
