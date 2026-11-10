using System.Buffers;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using OnnxSharp.Core;
using OnnxSharp.Tensors;
using OnnxSharp.Threading;

namespace OnnxSharp.Ops;

/// <summary>
/// 图融合产物：YOLO 的 DFL（Distribution Focal Loss 解码）折成一层。
/// <para>
/// 原始子图是 5 个算子：
/// </para>
/// <code>
/// x[C=S·B, N] → Reshape[B, S, bins, N] → Transpose[0,2,1,3] → Softmax(axis=1)
///             → Conv(1×1, 权重 = arange(bins)) → Reshape[B, S, N]
/// </code>
/// <para>
/// 其中「乘 arange」被导出成了一个权重为 <c>[0,1,…,bins-1]</c> 的 1×1 卷积。语义就是
/// <b>在每个 bin 上做跨通道 softmax，再按 bin 下标加权求和</b>：
/// <c>out[s, n] = Σ_b softmax_b(x[s·B+b, n]) · b</c>。
/// </para>
/// <para>
/// 一次遍历、不需要转置也不需要物化 softmax 结果；省掉 1 个 Softmax + 1 个 Conv + 3 个
/// Reshape/Transpose。yolo11n 上是 1.1 ms + 约 0.6 ms 的收益。
/// </para>
/// </summary>
internal static class DflOps
{
    public static void Register()
        => LayerRegistry.Register("DflSoftmax", p => new DflLayer(p));

    private sealed class DflLayer : Layer
    {
        /// <summary>每个方向的分箱数（yolo 是 16）。</summary>
        private readonly int _bins;

        /// <summary>每个任务处理的空间列数：256 列时每次行读 1 KB，预取器能跑满。</summary>
        private const int BlockColumns = 256;

        public DflLayer(LayerParams parameters)
        {
            Type = parameters.Type;
            Name = parameters.Name;
            _bins = (int)parameters.GetInt("bins", 16);
        }

        private (int Batches, int Sides, int Pixels) Resolve(IReadOnlyList<TensorShape> shapes)
        {
            var input = shapes[0];
            if (input.Rank != 3)
                throw new OnnxSharpException($"DflSoftmax '{Name}': input must be [B, sides·bins, N], got {input}");

            int bins = _bins;
            if (bins < 2 || input[1] % bins != 0)
                throw new OnnxSharpException(
                    $"DflSoftmax '{Name}': {input[1]} channels not divisible by bins={bins}");

            return (input[0], input[1] / bins, input[2]);
        }

        public override void GetMemoryShapes(
            IReadOnlyList<TensorShape> inShapes,
            int requiredOutputs,
            List<TensorShape> outShapes,
            List<TensorShape> tempShapes)
        {
            var (batches, sides, pixels) = Resolve(inShapes);
            outShapes.Add(new TensorShape(batches, sides, pixels));
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
            var source = inputs[0];
            var target = outputs[0];

            var (batches, sides, pixels) = Resolve([source.Shape]);
            if (batches <= 0 || sides <= 0 || pixels <= 0)
                return;

            var input = source.AsSpan<float>();
            var output = target.AsSpan<float>();

            int bins = _bins;
            const int block = BlockColumns;                       // 每任务 64 个空间位置：Exp 调用能摊薄
            int blocksPerSide = (pixels + block - 1) / block;
            int tasks = batches * sides * blocksPerSide;

            // Span / 固定指针都不能进 lambda：先固定成普通局部指针，再捕获。
            unsafe
            {
                fixed (float* sourcePointer = input)
                fixed (float* destinationPointer = output)
                {
                    float* src = sourcePointer;
                    float* dst = destinationPointer;

                    ParallelRange.For(tasks, (begin, end) =>
                    {
                        // 每任务一套 64 列缓冲：max / shifted / total / weighted / exp[bins][64]
                        float[] rented = ArrayPool<float>.Shared.Rent(4 * block + bins * block);
                        try
                        {
                            fixed (float* buffers = rented)
                            {
                                float* maximum = buffers;
                                float* shifted = buffers + block;
                                float* total = buffers + 2 * block;
                                float* weighted = buffers + 3 * block;
                                float* exponential = buffers + 4 * block;

                                for (int task = begin; task < end; task++)
                                {
                                    int blockIndex = task % blocksPerSide;
                                    int group = task / blocksPerSide;
                                    int side = group % sides;
                                    int batch = group / sides;

                                    int sourceOffset = (batch * sides + side) * bins * pixels;
                                    int destinationOffset = (batch * sides + side) * pixels;
                                    int column = blockIndex * block;
                                    int width = Math.Min(block, pixels - column);

                                    ApplyBlock(
                                        src + sourceOffset, dst + destinationOffset, column, width,
                                        bins, pixels, maximum, shifted, total, weighted, exponential);
                                }
                            }
                        }
                        finally
                        {
                            ArrayPool<float>.Shared.Return(rented);
                        }
                    });
                }
            }
        }

        /// <summary>
        /// 一块 <paramref name="width"/> 个空间位置（≤ 64）：跨 bin softmax + 按 bin 下标加权求和。
        /// <para>
        /// 全部按 8 lane 向量化，每个 bin 只调一次 <see cref="VectorMath.Exp(float*, float*, int)"/>。
        /// 早先按 8 列一块、每块调 16 次 Exp，光调用的固定开销就让这一层要 1.3 ms。
        /// </para>
        /// </summary>
        private static unsafe void ApplyBlock(
            float* source,
            float* destination,
            int column,
            int width,
            int bins,
            int pixels,
            float* maximum,
            float* shifted,
            float* total,
            float* weighted,
            float* exponential)
        {
            int vectorEnd = width & ~7;

            // 1) 逐 bin 求每一列的 max
            for (int j = 0; j < vectorEnd; j += 8)
                Avx.Store(maximum + j, Vector256.Create(float.NegativeInfinity));
            for (int j = vectorEnd; j < width; j++)
                maximum[j] = float.NegativeInfinity;

            for (int b = 0; b < bins; b++)
            {
                float* row = source + b * pixels + column;
                for (int j = 0; j < vectorEnd; j += 8)
                    Avx.Store(maximum + j, Avx.Max(Avx.LoadVector256(maximum + j), Avx.LoadVector256(row + j)));
                for (int j = vectorEnd; j < width; j++)
                    maximum[j] = MathF.Max(maximum[j], row[j]);
            }

            // 2) exp(x - max) 落盘，顺便累加 Σ
            for (int j = 0; j < vectorEnd; j += 8)
                Avx.Store(total + j, Vector256<float>.Zero);
            for (int j = vectorEnd; j < width; j++)
                total[j] = 0f;

            for (int b = 0; b < bins; b++)
            {
                float* row = source + b * pixels + column;
                for (int j = 0; j < vectorEnd; j += 8)
                    Avx.Store(shifted + j, Avx.Subtract(Avx.LoadVector256(row + j), Avx.LoadVector256(maximum + j)));
                for (int j = vectorEnd; j < width; j++)
                    shifted[j] = row[j] - maximum[j];

                float* slot = exponential + b * BlockColumns;
                VectorMath.Exp(shifted, slot, width);

                for (int j = 0; j < vectorEnd; j += 8)
                    Avx.Store(total + j, Avx.Add(Avx.LoadVector256(total + j), Avx.LoadVector256(slot + j)));
                for (int j = vectorEnd; j < width; j++)
                    total[j] += slot[j];
            }

            // 3) 与 Softmax 算子一致：先 1/Σ，再 p_b = e_b·(1/Σ)，最后 Σ p_b·b
            for (int j = 0; j < vectorEnd; j += 8)
                Avx.Store(total + j, Avx.Divide(Vector256.Create(1f), Avx.LoadVector256(total + j)));
            for (int j = vectorEnd; j < width; j++)
                total[j] = 1f / total[j];

            for (int j = 0; j < vectorEnd; j += 8)
                Avx.Store(weighted + j, Vector256<float>.Zero);
            for (int j = vectorEnd; j < width; j++)
                weighted[j] = 0f;

            for (int b = 0; b < bins; b++)
            {
                float* slot = exponential + b * BlockColumns;
                var bin = Vector256.Create((float)b);

                for (int j = 0; j < vectorEnd; j += 8)
                {
                    var probability = Avx.Multiply(Avx.LoadVector256(slot + j), Avx.LoadVector256(total + j));
                    Avx.Store(weighted + j, Fma.MultiplyAdd(probability, bin, Avx.LoadVector256(weighted + j)));
                }

                for (int j = vectorEnd; j < width; j++)
                    weighted[j] += slot[j] * total[j] * b;
            }

            for (int j = 0; j < vectorEnd; j += 8)
                Avx.Store(destination + column + j, Avx.LoadVector256(weighted + j));
            for (int j = vectorEnd; j < width; j++)
                destination[column + j] = weighted[j];
        }
    }
}
