using OnnxSharp.Tensors;

namespace OnnxSharp.Tests;

/// <summary>
/// GEMM 微内核必须与朴素三重循环一致，尤其是 n（或 m）不是 16 的倍数的形状 ——
/// 列块有残块时最容易越界/漏算。这些形状直接来自 rf-detr 的 attention
/// （[257,64]×[64,257]、[257,257]×[257,64]）与 MLP（[257,384]×[384,1536]）。
/// </summary>
internal static class GemmKernelTests
{
    public static IEnumerable<TestCase> Cases()
    {
        yield return new("gemm: multiply matches naive on awkward shapes", MatchesNaive);
        yield return new("gemm: multiply with ldc and bias matches naive", MatchesNaiveWithLdcAndBias);
        yield return new("gemm: column bias added per column", ColumnBiasMatchesNaive);
        yield return new("gemm: packed panels match plain multiply", PackedMatchesPlain);
        yield return new("gemm: packed column scale and bias fold the epilogue", PackedScaleBiasMatchesEpilogue);
    }

    /// <summary>
    /// 预打包面板（常量权重复用）必须与 <see cref="GemmKernel.Multiply"/> 一致：
    /// 覆盖「n 不是 16 的倍数」（尾面板宽度不同）、m 不是 4 的倍数、以及小 n。
    /// </summary>
    private static void PackedMatchesPlain()
    {
        int[][] shapes =
        [
            [96, 320, 1280],
            [96, 1280, 320],
            [257, 384, 1536],
            [5, 17, 33],
            [3, 1, 7],
            [1, 300, 256],
            [9, 64, 17],
        ];

        foreach (var shape in shapes)
        {
            int m = shape[0], k = shape[1], n = shape[2];
            var random = new Random(m * 17 + k * 5 + n * 3);
            var a = new float[m * k];
            var b = new float[k * n];
            var plain = new float[m * n];
            var packedOutput = new float[m * n];

            Fill(a, random);
            Fill(b, random);

            GemmKernel.Multiply(a, b, plain, m, k, n, parallel: false);
            GemmKernel.MultiplyPacked(a, GemmKernel.Pack(b, k, n), packedOutput, m, parallel: false);

            CompareEqual(plain, packedOutput, $"packed m={m} k={k} n={n}");
        }
    }

    /// <summary>
    /// 融合的反量化 epilogue：<c>MultiplyPacked(columnScale, columnBias)</c> 等于
    /// 「先 GEMM、再逐列 <c>*scale + bias</c>」（只差 FMA 的一次舍入）。
    /// <para>
    /// 这条覆盖 ASR 的 <c>DynamicQuantizeMatMul</c> 快路径：它把 scale/bias 折进 GEMM 的写出，
    /// 省掉一整趟 m×n。
    /// </para>
    /// </summary>
    private static void PackedScaleBiasMatchesEpilogue()
    {
        int[][] shapes = [[96, 320, 1280], [96, 320, 320], [96, 1280, 320], [7, 33, 45], [4, 8, 3], [2, 96, 17]];

        foreach (var shape in shapes)
        {
            int m = shape[0], k = shape[1], n = shape[2];
            var random = new Random(m * 13 + k * 29 + n * 11);
            var a = new float[m * k];
            var b = new float[k * n];
            var scale = new float[n];
            var bias = new float[n];
            var plain = new float[m * n];
            var fused = new float[m * n];

            Fill(a, random);
            Fill(b, random);
            for (int j = 0; j < n; j++)
            {
                scale[j] = 0.0001f + (float)random.NextDouble() * 0.01f;
                bias[j] = (float)(random.NextDouble() * 2 - 1);
            }

            GemmKernel.Multiply(a, b, plain, m, k, n, parallel: false);
            GemmKernel.MultiplyPacked(a, GemmKernel.Pack(b, k, n), fused, m,
                parallel: false, columnBias: bias, columnScale: scale);

            for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
            {
                double expected = plain[i * n + j] * (double)scale[j] + bias[j];
                double diff = Math.Abs(fused[i * n + j] - expected);
                double magnitude = Math.Max(1.0, Math.Abs(expected));
                if (diff / magnitude > 1e-5)
                    throw new InvalidOperationException(
                        $"gemm fused epilogue m={m} k={k} n={n}: mismatch at ({i},{j}) " +
                        $"got {fused[i * n + j]:G9} expected {expected:G9}");
            }
        }
    }

    private static void Fill(float[] values, Random random)
    {
        for (int i = 0; i < values.Length; i++)
            values[i] = (float)(random.NextDouble() * 2 - 1);
    }

    private static void CompareEqual(float[] expected, float[] actual, string what)
    {
        for (int i = 0; i < expected.Length; i++)
        {
            double diff = Math.Abs(actual[i] - expected[i]);
            double magnitude = Math.Max(1.0, Math.Abs(expected[i]));
            if (diff / magnitude > 1e-5)
                throw new InvalidOperationException(
                    $"{what}: mismatch at flat {i}: got {actual[i]:G9} expected {expected[i]:G9}");
        }
    }

    private static void MatchesNaive()
    {
        int[][] shapes =
        [
            [257, 64, 257],
            [257, 257, 64],
            [64, 64, 64],
            [257, 384, 1536],
            [300, 256, 256],
            [1, 257, 64],
            [5, 17, 33],
            [17, 1, 257],
        ];

        foreach (var shape in shapes)
            Check(shape[0], shape[1], shape[2], ldc: 0);
    }

    private static void MatchesNaiveWithLdcAndBias()
    {
        Check(257, 257, 64, ldc: 384);
        Check(64, 64, 5, ldc: 9);
    }

    /// <summary>
    /// 按列 bias（<c>MatMul → Add(·, [n])</c> 折进 GEMM 用）：每一行的每一列都加上
    /// <c>bias[j]</c>，与「先算 GEMM 再单独做一次 Add」逐元素一致。
    /// </summary>
    private static void ColumnBiasMatchesNaive()
    {
        int[][] shapes = [[257, 384, 1536], [64, 64, 17], [1, 300, 256], [5, 7, 33]];

        foreach (var shape in shapes)
        {
            int m = shape[0], k = shape[1], n = shape[2];
            var random = new Random(m + k * 13 + n * 7);
            var a = new float[m * k];
            var b = new float[k * n];
            var bias = new float[n];
            var c = new float[m * n];

            for (int i = 0; i < a.Length; i++)
                a[i] = (float)(random.NextDouble() * 2 - 1);
            for (int i = 0; i < b.Length; i++)
                b[i] = (float)(random.NextDouble() * 2 - 1);
            for (int i = 0; i < bias.Length; i++)
                bias[i] = (float)(random.NextDouble() * 2 - 1);

            GemmKernel.Multiply(a, b, c, m, k, n, parallel: false, columnBias: bias);

            for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
            {
                double sum = bias[j];
                for (int p = 0; p < k; p++)
                    sum += (double)a[i * k + p] * b[p * n + j];

                double diff = Math.Abs(c[i * n + j] - sum);
                double scale = Math.Max(1.0, Math.Abs(sum));
                if (diff / scale > 1e-4)
                    throw new InvalidOperationException(
                        $"gemm column bias m={m} k={k} n={n}: mismatch at ({i},{j}) " +
                        $"got {c[i * n + j]:G9} expected {sum:G9}");
            }
        }
    }

    private static void Check(int m, int k, int n, int ldc)
    {
        var random = new Random(m * 31 + k * 7 + n);
        int stride = ldc > 0 ? ldc : n;

        var a = new float[m * k];
        var b = new float[k * n];
        var c = new float[m * stride];
        var expected = new float[m * stride];

        for (int i = 0; i < a.Length; i++)
            a[i] = (float)(random.NextDouble() * 2 - 1);
        for (int i = 0; i < b.Length; i++)
            b[i] = (float)(random.NextDouble() * 2 - 1);

        for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
        {
            double sum = 0;
            for (int p = 0; p < k; p++)
                sum += (double)a[i * k + p] * b[p * n + j];

            expected[i * stride + j] = (float)sum;
        }

        GemmKernel.Multiply(a, b, c, m, k, n, ldc: ldc, parallel: false);

        for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
        {
            int index = i * stride + j;
            double diff = Math.Abs(c[index] - expected[index]);
            double scale = Math.Max(1.0, Math.Abs(expected[index]));
            if (diff / scale > 1e-4)
                throw new InvalidOperationException(
                    $"gemm m={m} k={k} n={n} ldc={stride}: mismatch at ({i},{j}) " +
                    $"got {c[index]:G9} expected {expected[index]:G9}");
        }
    }
}
