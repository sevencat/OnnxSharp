using System.Diagnostics;
using OnnxSharp.Tensors;

/// <summary>
/// eltwisebench：逐元素算子的单算子微基准，输出 ms / GB/s / ns·元素。
/// <para>
/// 目的不是「跑分」，而是回答一个问题：**逐元素算子到底是撞了内存带宽，还是内核/调度有问题**。
/// 所以必须同时给出带宽参考 —— <c>copy</c>（纯 memmove，1 读 1 写）是这台机器在这个线程数下
/// 能达到的上限，其它内核的 GB/s 跟它比才有意义。绝对 ms 单独看没有信息量：
/// 同样是 0.2 ms，搬 100 KB 和搬 10 MB 完全是两件事。
/// </para>
/// <para>
/// 流量口径：一个 <c>F32</c> 输入按 4 字节/元素计，广播操作数按它的**实际**尺寸计。
/// 这就是为什么 <c>Add [4,257,384]+[384]</c> 只有约 2 趟流量而不是 3 趟。
/// </para>
/// </summary>
internal static class EltwiseBench
{
    public static int Run(CliOptions options)
    {
        int iters = Math.Max(1, options.GetInt("--iters", 20));

        var shapes = new List<int[]>();
        foreach (var spec in options.Shapes)
        {
            var p = spec.Split(new[] { 'x', 'X', ',' }, StringSplitOptions.RemoveEmptyEntries);
            var dims = new int[p.Length];
            for (int i = 0; i < p.Length; i++)
                dims[i] = int.Parse(p[i]);
            shapes.Add(dims);
        }

        if (shapes.Count == 0)
        {
            // rfdetr-small 的真实分布（`dump-graph --type Mul --inputs` 统计）：
            // [4,257,384] 出现 24 次（Mul/Add 的最大宗），[1,300,256] 恰好卡在
            // ParallelElementThreshold(65536) 之上一点，[1,300,2048] / [1,128,32,32]
            // 代表单个大张量，[1,1024,256] 是 Where 的形状。
            shapes.Add(new[] { 4, 257, 384 });
            shapes.Add(new[] { 1, 300, 256 });
            shapes.Add(new[] { 1, 300, 2048 });
            shapes.Add(new[] { 1, 128, 32, 32 });
            shapes.Add(new[] { 1, 1024, 256 });
        }

        Console.WriteLine($"threads : {OnnxSharp.Threading.ParallelRange.Threads}");
        Console.WriteLine($"iters   : {iters}（取 min；每项先预热 3 次）");
        Console.WriteLine();
        Console.WriteLine("copy   = 纯 memmove，1 读 1 写 —— 这是当前线程数下的带宽参考上限");
        Console.WriteLine("copy1  = 逐元素 4 字节 CopyTo —— 复刻 Where/GatherElements 的搬移方式");
        Console.WriteLine();

        var rng = new Random(1234);

        foreach (var dims in shapes)
        {
            var shape = new TensorShape(dims);
            long count = shape.Total();

            var a = Tensor.Allocate(shape, DType.F32);
            var b = Tensor.Allocate(shape, DType.F32);
            var c = Tensor.Allocate(shape, DType.F32);

            // 标量/尾广播的两个小操作数：[1] 与 [最后一维]
            var scalar = Tensor.Allocate(new TensorShape(1), DType.F32);
            scalar.AsSpan<float>()[0] = 0.5f;

            var tailShape = new TensorShape(new[] { dims[^1] });
            var tail = Tensor.Allocate(tailShape, DType.F32);

            Fill(a.AsSpan<float>(), rng);
            Fill(b.AsSpan<float>(), rng);
            Fill(tail.AsSpan<float>(), rng);

            long bytesCopy = count * 8;          // 读 a + 写 c
            long bytesFull = count * 12;         // 读 a + 读 b + 写 c
            long bytesScalar = count * 8 + 4;    // 读 a + 写 c，标量本身 4 字节
            long bytesTail = count * 8 + dims[^1] * 4;

            Console.WriteLine($"=== [{string.Join(",", dims)}]  {count:N0} 元素  " +
                              $"(a+b+c 三趟 = {bytesFull / 1024.0 / 1024.0:F2} MB)");

            Report("copy ", bytesCopy, count, Best(iters, () => a.AsSpan<float>().CopyTo(c.AsSpan<float>())));
            Report("copy1", bytesCopy, count, Best(iters, () => PerElementCopy(a, c, count)));
            Report("add  ", bytesFull, count, Best(iters, () => TensorOps.ElementWiseBinary(BinaryOpKind.Add, a, b, c)));
            Report("mul  ", bytesFull, count, Best(iters, () => TensorOps.ElementWiseBinary(BinaryOpKind.Mul, a, b, c)));
            Report("mul-s", bytesScalar, count, Best(iters, () => TensorOps.ElementWiseBinary(BinaryOpKind.Mul, a, scalar, c)));
            Report("mul-t", bytesTail, count, Best(iters, () => TensorOps.ElementWiseBinary(BinaryOpKind.Mul, a, tail, c)));
            Report("relu ", bytesCopy, count, Best(iters, () => TensorOps.ElementWiseUnary(UnaryOpKind.Relu, a, c)));
            Report("silu ", bytesCopy, count, Best(iters, () => TensorOps.ElementWiseUnary(UnaryOpKind.Silu, a, c)));

            Console.WriteLine();
        }

        return 0;
    }

    private static void Fill(Span<float> span, Random rng)
    {
        for (int i = 0; i < span.Length; i++)
            span[i] = (float)rng.NextDouble();
    }

    /// <summary>逐元素 4 字节搬移：<c>OpUtil.CopyBytes(..., 1)</c> 的做法，Where 就是这么写的。</summary>
    private static void PerElementCopy(Tensor src, Tensor dst, long count)
    {
        var s = src.Buffer.AsSpan(src.ByteOffset, src.ByteLength);
        var d = dst.Buffer.AsSpan(dst.ByteOffset, dst.ByteLength);
        for (int i = 0; i < count; i++)
            s.Slice(i * 4, 4).CopyTo(d.Slice(i * 4, 4));
    }

    /// <summary>预热 3 次，再跑 <paramref name="iters"/> 次取最小（避免被单次 GC/中断污染）。</summary>
    private static double Best(int iters, Action body)
    {
        for (int i = 0; i < 3; i++)
            body();

        double best = double.MaxValue;
        var watch = new Stopwatch();

        for (int i = 0; i < iters; i++)
        {
            watch.Restart();
            body();
            watch.Stop();
            best = Math.Min(best, watch.Elapsed.TotalMilliseconds);
        }

        return best;
    }

    private static void Report(string name, long bytes, long elements, double milliseconds)
    {
        double gbs = bytes / 1e6 / milliseconds;          // 字节/ns = GB/s
        double nsPerElement = milliseconds * 1e6 / elements;
        Console.WriteLine($"  {name}  {milliseconds,7:F3} ms  {gbs,6:F1} GB/s  {nsPerElement,6:F3} ns/元素");
    }
}
