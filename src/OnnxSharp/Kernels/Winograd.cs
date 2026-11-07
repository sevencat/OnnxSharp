using System.Buffers;
using System.Runtime.CompilerServices;
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

    private static readonly ConditionalWeakTable<Tensor, float[]> WeightCache = new();

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
        if ((long)inH * inW > 4096)
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

        int threads = Environment.ProcessorCount;

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

                    for (int tx = 0; tx < tilesX; tx++)
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

                    for (int tx = 0; tx < tilesX; tx++)
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
}
