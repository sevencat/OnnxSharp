using OnnxSharp.Core;
using OnnxSharp.Tensors;

namespace OnnxSharp.Tests;

/// <summary>
/// <c>ArgMax</c> 算子与 <c>MatMul + ArgMax</c> 融合（decoder / CTC 贪心解码的前置能力）。
/// 端到端金标准由 <c>tools/scripts/make_quant_model.py</c> 生成，缺失则 SKIP。
/// </summary>
internal static class MatMulArgMaxTests
{
    public static IEnumerable<TestCase> Cases()
    {
        yield return new("argmax: axis, keepdims, ties and select_last_index", ArgMaxSemantics);
        yield return new("argmax: fused MatMul+ArgMax matches onnxruntime", FusedMatchesOrt);
        yield return new("argmax: fused path beats materialising logits", FusedBeatsTwoPass);
    }

    private static void ArgMaxSemantics()
    {
        // [[1, 5, 5, 2], [9, 0, 1, 1]]
        var input = Tensor.FromArray([1f, 5f, 5f, 2f, 9f, 0f, 1f, 1f], new TensorShape(2, 4));

        // axis = -1, keepdims = 0：并列取首次出现
        var last = RunArgMax(input, axis: -1, keepDims: false, selectLast: false, out var shape);
        Assert.Equal(1, shape.Rank, "rank");
        Assert.Equal(1L, last[0], "[0] 并列取首个 5");
        Assert.Equal(0L, last[1], "[1]");

        // select_last_index = 1：并列取最后一次出现
        var lastOfLast = RunArgMax(input, axis: -1, keepDims: false, selectLast: true, out _);
        Assert.Equal(2L, lastOfLast[0], "[0] 并列取最后 5");

        // axis = 0, keepdims = 1 → [1, 4]
        var column = RunArgMax(input, axis: 0, keepDims: true, selectLast: false, out var columnShape);
        Assert.Equal(2, columnShape.Rank, "axis 0 keeps rank");
        Assert.Equal(1, columnShape.Dims[0], "keepdims");
        Assert.Equal(4, columnShape.Dims[1], "columns");
        Assert.Equal(1L, column[0], "col0 → row1 (9)");
        Assert.Equal(0L, column[3], "col3 → row0 (2)");
    }

    private static void FusedMatchesOrt()
    {
        string? directory = FindFixture();
        if (directory is null)
            throw new TestSkipException("argmax fixture not found (run tools/scripts/make_quant_model.py .tmp/quant)");

        foreach ((string tag, int rows) in new[] { ("decode", 1), ("ctc", 24) })
        {
            string modelPath = Path.Combine(directory, $"argmax_{tag}.onnx");
            if (!File.Exists(modelPath))
                continue;

            var net = Net.ReadModel(modelPath);
            Assert.True(
                net.MainGraph!.Prog.Any(op => op.Type == "MatMulArgMax"),
                $"{tag}: expected MatMul+ArgMax to be fused");

            byte[] inputBytes = File.ReadAllBytes(Path.Combine(directory, $"argmax_{tag}_x.f32"));
            var input = Tensor.Allocate(new TensorShape(rows, inputBytes.Length / sizeof(float) / rows), DType.F32);
            inputBytes.CopyTo(input.Buffer, 0);
            net.SetInput("x", input);

            var actual = net.Forward().AsSpan<long>();

            byte[] goldenBytes = File.ReadAllBytes(Path.Combine(directory, $"argmax_{tag}_golden.i64"));
            var golden = new long[goldenBytes.Length / sizeof(long)];
            Buffer.BlockCopy(goldenBytes, 0, golden, 0, goldenBytes.Length);

            Assert.Equal(golden.Length, actual.Length, $"{tag}: output length");
            for (int i = 0; i < golden.Length; i++)
                Assert.Equal(golden[i], actual[i], $"{tag}: argmax[{i}]");
        }
    }

    private static void FusedBeatsTwoPass()
    {
        // CTC 形状：m = 帧数、n = 词表；两趟写法必须物化 [m,n] 的 logits
        const int m = 24, k = 256, n = 8000;
        var random = new Random(7);

        var a = new float[m * k];
        for (int i = 0; i < a.Length; i++)
            a[i] = (float)(random.NextDouble() - 0.5);

        var b = new float[k * n];
        for (int i = 0; i < b.Length; i++)
            b[i] = (float)((random.NextDouble() - 0.5) * 0.1);

        var logits = new float[m * n];
        var fusedArgMax = new long[m];
        var twoPassArgMax = new long[m];

        double twoPass = Measure(() =>
        {
            Tensors.GemmKernel.Multiply(a, b, logits, m, k, n);
            ReferenceArgMax(logits, twoPassArgMax, m, n);
        });

        double fused = Measure(() =>
            Kernels.MatMulArgMaxKernel.Compute(a, b, default, m, k, n, fusedArgMax, default));

        Console.WriteLine(
            $"        ctc {m}x{k}x{n}: fused {fused:F2} ms vs matmul+argmax {twoPass:F2} ms ({twoPass / fused:F2}x)");

        for (int i = 0; i < m; i++)
            Assert.Equal(twoPassArgMax[i], fusedArgMax[i], $"[{i}] 两条路径的 argmax 必须一致");

        // 融合层内部对小 logits 用「GEMM 到池化缓冲 + 扫描」，所以只要求不慢于两趟
        // （机器负载会让这个比值抖动，别拿它当性能门槛）。
        Assert.True(fused <= twoPass * 1.25,
            $"fused ({fused:F2} ms) 不应明显慢于两趟 ({twoPass:F2} ms)");
    }

    private static void ReferenceArgMax(ReadOnlySpan<float> logits, Span<long> destination, int rows, int columns)
    {
        for (int row = 0; row < rows; row++)
        {
            int best = 0;
            for (int column = 1; column < columns; column++)
            {
                if (logits[row * columns + column] > logits[row * columns + best])
                    best = column;
            }

            destination[row] = best;
        }
    }

    private static double Measure(Action body, int warmup = 2, int iterations = 5)
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

    private static long[] RunArgMax(Tensor input, long axis, bool keepDims, bool selectLast, out TensorShape shape)
    {
        var parameters = new LayerParams { Type = "ArgMax", Name = "argmax" };
        parameters.Set("axis", axis);
        parameters.Set("keepdims", keepDims ? 1L : 0L);
        parameters.Set("select_last_index", selectLast ? 1L : 0L);

        var impl = new NetImpl();
        var arg = impl.Declare("x", ArgKind.Input);
        var graph = impl.NewGraph("g", new[] { arg }, isMain: true);

        var layer = LayerRegistry.Create("ArgMax", parameters);
        layer.Inputs.Add(arg);

        var output = impl.Declare("y", ArgKind.Temp);
        layer.Outputs.Add(output);
        graph.Append(layer);
        graph.SetOutputs(new[] { output });

        impl.SetInput("x", input);
        var result = impl.Forward();
        shape = result.Shape;
        return result.AsSpan<long>().ToArray();
    }

    private static string? FindFixture()
    {
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); dir is not null; dir = dir.Parent)
        {
            if (!File.Exists(Path.Combine(dir.FullName, "OnnxSharp.slnx")))
                continue;

            string candidate = Path.Combine(dir.FullName, ".tmp", "quant");
            return File.Exists(Path.Combine(candidate, "argmax_decode.onnx")) ? candidate : null;
        }

        return null;
    }
}
