using OnnxSharp.Kernels;
using OnnxSharp.Tensors;

namespace OnnxSharp.Tests;

/// <summary>
/// M10-01/02：int8 量化参数、饱和转换、int8 GEMM（VNNI 与标量必须逐位一致）。
/// </summary>
internal static class Int8Tests
{
    public static IEnumerable<TestCase> Cases()
    {
        yield return new("int8: quantize/dequantize honours scale, zero point and clamping", QuantizeDequantize);
        yield return new("int8: gemm matches the naive reference (ragged k/n, zero points)", GemmMatchesNaive);
        yield return new("int8: vnni and scalar paths agree bit-exactly", TierPathsAgree);
        yield return new("int8: signed weights pack and multiply", SignedWeights);
        yield return new("int8: dequant-weight gemm matches the reference", DequantGemmMatchesReference);
        yield return new("int8: dequant-weight gemm vs fp32 gemm throughput", DequantGemmThroughput);
        yield return new("int8: reports the selected kernel tier", TierReported);
    }

    private static void QuantizeDequantize()
    {
        float[] source = [0f, 0.1f, 0.25f, -0.5f, 10f, 1.5f];
        var quantized = new byte[source.Length];

        // scale 0.5、zp 128：0 → 128，0.25 → round(0.5)=0（half-to-even）→ 128，1.5 → round(3)=3 → 131
        Quantizer.QuantizeToUInt8(source, quantized, scale: 0.5f, zeroPoint: 128);

        Assert.Equal((byte)128, quantized[0], "0.0");
        Assert.Equal((byte)128, quantized[1], "0.1 → round(0.2)=0");
        Assert.Equal((byte)128, quantized[2], "0.25 → round(0.5)=0（half-to-even）");
        Assert.Equal((byte)127, quantized[3], "-0.5 → round(-1)=-1");
        Assert.Equal((byte)148, quantized[4], "10.0 / 0.5 = 20 → 148");
        Assert.Equal((byte)131, quantized[5], "1.5 → 131");

        // 饱和：1000 / 0.5 = 2000 → clamp 到 255
        var saturated = new byte[1];
        Quantizer.QuantizeToUInt8([1000f], saturated, scale: 0.5f, zeroPoint: 128);
        Assert.Equal((byte)255, saturated[0], "saturates at 255");

        var restored = new float[source.Length];
        Quantizer.DequantizeUInt8(quantized, restored, scale: 0.5f, zeroPoint: 128);
        Assert.Near(0f, restored[0], 1e-6f, "restored 0");
        Assert.Near(1.5f, restored[5], 1e-6f, "restored 1.5");

        var signed = new sbyte[source.Length];
        Quantizer.QuantizeToInt8(source, signed, scale: 0.5f, zeroPoint: 0);
        Assert.Equal((sbyte)0, signed[0], "s8 0");
        Assert.Equal((sbyte)-1, signed[3], "s8 -0.5");
        Assert.Equal((sbyte)3, signed[5], "s8 1.5");
    }

    private static void GemmMatchesNaive()
    {
        const int m = 3, k = 13, n = 11;      // 故意用非 4 / 非 8 的倍数
        var random = new Random(2026);

        var a = new byte[m * k];
        for (int i = 0; i < a.Length; i++)
            a[i] = (byte)random.Next(0, 256);

        var b = new byte[k * n];
        for (int i = 0; i < b.Length; i++)
            b[i] = (byte)random.Next(0, 256);

        var bZero = new int[n];
        for (int j = 0; j < n; j++)
            bZero[j] = 128;                    // 对称量化

        int aZero = 119;

        PackedInt8Weights packed = PackedInt8Weights.Pack(b, k, n, bZero);
        var fromKernel = new int[m * n];
        Int8Gemm.MultiplyInteger(a, packed, fromKernel, m, aZero);

        var fromScalar = new int[m * n];
        Int8Gemm.MultiplyIntegerScalar(a, packed, fromScalar, m, aZero);

        for (int row = 0; row < m; row++)
        {
            for (int col = 0; col < n; col++)
            {
                long expected = 0;
                for (int kk = 0; kk < k; kk++)
                    expected += (a[row * k + kk] - aZero) * (b[kk * n + col] - bZero[col]);

                Assert.Equal((int)expected, fromKernel[row * n + col], $"kernel [{row},{col}]");
                Assert.Equal((int)expected, fromScalar[row * n + col], $"scalar [{row},{col}]");
            }
        }
    }

    private static void TierPathsAgree()
    {
        const int m = 4, k = 64, n = 40;
        var random = new Random(7);

        var a = new byte[m * k];
        random.NextBytes(a);

        var b = new byte[k * n];
        random.NextBytes(b);

        var bZero = new int[1];
        bZero[0] = 127;

        PackedInt8Weights packed = PackedInt8Weights.Pack(b, k, n, bZero);

        var vnni = new int[m * n];
        var scalar = new int[m * n];
        Int8Gemm.MultiplyInteger(a, packed, vnni, m, aZeroPoint: 120);
        Int8Gemm.MultiplyIntegerScalar(a, packed, scalar, m, aZeroPoint: 120);

        for (int i = 0; i < vnni.Length; i++)
            Assert.Equal(scalar[i], vnni[i], $"element {i}");
    }

    private static void SignedWeights()
    {
        const int m = 2, k = 6, n = 9;
        var random = new Random(11);

        var a = new byte[m * k];
        random.NextBytes(a);

        var b = new sbyte[k * n];
        for (int i = 0; i < b.Length; i++)
            b[i] = (sbyte)random.Next(-128, 128);

        PackedInt8Weights packed = PackedInt8Weights.PackSigned(b, k, n, [0]);

        var actual = new int[m * n];
        Int8Gemm.MultiplyInteger(a, packed, actual, m, aZeroPoint: 0);

        for (int row = 0; row < m; row++)
        {
            for (int col = 0; col < n; col++)
            {
                long expected = 0;
                for (int kk = 0; kk < k; kk++)
                    expected += a[row * k + kk] * b[kk * n + col];

                Assert.Equal((int)expected, actual[row * n + col], $"[{row},{col}]");
            }
        }
    }

    private static void DequantGemmMatchesReference()
    {
        const int m = 5, k = 17, n = 13;      // 非 4/8/32 倍数，覆盖尾巴
        var random = new Random(99);

        var a = new byte[m * k];
        random.NextBytes(a);

        var b = new byte[k * n];
        random.NextBytes(b);

        const float aScale = 0.021f;
        const float bScale = 0.0078f;
        const int aZero = 121;
        const int bZero = 133;

        var actual = new float[m * n];
        Int8DequantGemm.Multiply(a, m, k, aScale, aZero, b, n, bScale, bZero, actual);

        for (int row = 0; row < m; row++)
        {
            for (int col = 0; col < n; col++)
            {
                double sum = 0;
                for (int kk = 0; kk < k; kk++)
                    sum += (a[row * k + kk] - aZero) * (b[kk * n + col] - bZero);

                float expected = (float)(sum * aScale * bScale);
                Assert.Near(expected, actual[row * n + col], MathF.Abs(expected) * 1e-4f + 1e-5f,
                    $"[{row},{col}]");
            }
        }
    }

    private static void DequantGemmThroughput()
    {
        // ASR encoder 的典型形状（Paraformer：token ≈ 96，隐藏维 320，FFN 放大到 1280）
        (int M, int K, int N)[] shapes = [(96, 320, 1280), (96, 320, 320)];
        var random = new Random(2026);

        foreach ((int m, int k, int n) in shapes)
        {
            var a = new byte[m * k];
            random.NextBytes(a);
            var b = new byte[k * n];
            random.NextBytes(b);

            var f32A = new float[m * k];
            for (int i = 0; i < f32A.Length; i++)
                f32A[i] = (a[i] - 128) * 0.0078f;

            var f32B = new float[k * n];
            for (int i = 0; i < f32B.Length; i++)
                f32B[i] = (b[i] - 128) * 0.0078f;

            var int8Result = new float[m * n];
            var fp32Result = new float[m * n];

            double int8Ms = Measure(() =>
                Int8DequantGemm.Multiply(a, m, k, 0.021f, 121, b, n, 0.0078f, 133, int8Result));

            double fp32Ms = Measure(() =>
                Tensors.GemmKernel.Multiply(f32A, f32B, fp32Result, m, k, n));

            double flops = 2.0 * m * k * n;
            Console.WriteLine(
                $"        [{m}x{k}x{n}] int8-weight {int8Ms:F2} ms ({flops / int8Ms / 1e6:F0} GFLOPS)  " +
                $"fp32 {fp32Ms:F2} ms ({flops / fp32Ms / 1e6:F0} GFLOPS)  ratio {fp32Ms / int8Ms:F2}x");
        }
    }

    private static double Measure(Action body, int warmup = 3, int iterations = 8)
    {
        for (int i = 0; i < warmup; i++)
            body();

        double best = double.MaxValue;
        var watch = System.Diagnostics.Stopwatch.StartNew();
        for (int i = 0; i < iterations; i++)
        {
            watch.Restart();
            body();
            watch.Stop();
            best = Math.Min(best, watch.Elapsed.TotalMilliseconds);
        }

        return best;
    }

    private static void TierReported()
    {
        Console.WriteLine(
            $"        int8 tier: {Int8Kernel.Tier} ({Int8Kernel.Describe()})   " +
            $"Avx2={System.Runtime.Intrinsics.X86.Avx2.IsSupported} " +
            $"Avx512BW={System.Runtime.Intrinsics.X86.Avx512BW.IsSupported} " +
            $"AvxVnni={System.Runtime.Intrinsics.X86.AvxVnni.IsSupported} " +
            $"AdvSimd={System.Runtime.Intrinsics.Arm.AdvSimd.IsSupported}");

        Assert.True(Enum.IsDefined(Int8Kernel.Tier), "tier is a known value");
    }
}
