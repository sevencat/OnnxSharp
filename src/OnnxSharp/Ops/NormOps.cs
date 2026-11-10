using System.Numerics;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using OnnxSharp.Core;
using OnnxSharp.Tensors;

namespace OnnxSharp.Ops;

/// <summary>
/// 归一化 / 透传类算子：<c>BatchNormalization</c> 与 <c>Identity</c>。
/// <para>
/// 这两个是 PaddleOCR（PP-OCRv6 rec 头）带出来的：rec 图里有 58 个 <c>Identity</c>
/// 与 4 个推理模式的 <c>BatchNormalization</c>，而 YOLO / RF-DETR 恰好都不需要它们。
/// BN 也可以在图级折叠成 Conv 的权重，但 rec 的 BN 挂在 HardSwish 之后、输入是运行时张量，
/// 直接实现成算子更省事（每通道一次仿射，代价可以忽略）。
/// </para>
/// </summary>
internal static class NormOps
{
    public static void Register()
    {
        LayerRegistry.Register("BatchNormalization", p => new BatchNormLayer(p));
        LayerRegistry.Register("Identity", p => new IdentityLayer(p));
    }

    // ---------------- BatchNormalization ----------------

    /// <summary>
    /// 推理模式批归一化：<c>y = (x - mean) / sqrt(var + eps) * scale + bias</c>（逐通道）。
    /// 输入 <c>X, scale, B, mean, var</c>；后四个必须是常量（ONNX 推理图的标准形态）。
    /// </summary>
    private sealed class BatchNormLayer : Layer
    {
        private readonly float _epsilon;

        public BatchNormLayer(LayerParams parameters)
        {
            Type = parameters.Type;
            Name = parameters.Name;
            _epsilon = (float)parameters.GetFloat("epsilon", 1e-5);
        }

        public override void GetMemoryShapes(
            IReadOnlyList<TensorShape> inShapes,
            int requiredOutputs,
            List<TensorShape> outShapes,
            List<TensorShape> tempShapes)
        {
            if (inShapes.Count == 0 || inShapes[0].Rank < 2)
                throw new OnnxSharpException($"BatchNormalization '{Name}': input must be rank >= 2");
            outShapes.Add(inShapes[0]);
        }

        public override void Forward(
            IReadOnlyList<Tensor> inputs,
            IReadOnlyList<Tensor> outputs,
            IReadOnlyList<Tensor> temps)
        {
            if (inputs.Count < 5)
                throw new OnnxSharpException(
                    $"BatchNormalization '{Name}': training mode needs 5 inputs (X, scale, B, mean, var), got {inputs.Count}");

            var input = inputs[0];
            var output = outputs[0];
            var shape = input.Shape;

            int batches = shape.Dims[0];
            int channels = shape.Dims[1];
            long inner = 1;
            for (int i = 2; i < shape.Rank; i++)
                inner *= shape.Dims[i];

            var scale = inputs[1].AsSpan<float>();
            var bias = inputs[2].AsSpan<float>();
            var mean = inputs[3].AsSpan<float>();
            var variance = inputs[4].AsSpan<float>();

            if (scale.Length < channels || bias.Length < channels || mean.Length < channels || variance.Length < channels)
                throw new OnnxSharpException(
                    $"BatchNormalization '{Name}': parameter length < channels ({channels})");

            var src = input.AsSpan<float>();
            var dst = output.AsSpan<float>();

            // 每通道预算成一次仿射：y = x * a + b
            for (int c = 0; c < channels; c++)
            {
                float invStd = 1f / MathF.Sqrt(variance[c] + _epsilon);
                float a = scale[c] * invStd;
                float b = bias[c] - mean[c] * a;

                for (int n = 0; n < batches; n++)
                {
                    long start = ((long)n * channels + c) * inner;
                    int count = checked((int)inner);
                    int i = 0;

                    if (Fma.IsSupported)
                    {
                        var va = Vector256.Create(a);
                        var vb = Vector256.Create(b);
                        for (; i + Vector256<float>.Count <= count; i += Vector256<float>.Count)
                        {
                            var x = Vector256.LoadUnsafe(ref src[(int)start + i]);
                            Vector256.StoreUnsafe(Fma.MultiplyAdd(x, va, vb), ref dst[(int)start + i]);
                        }
                    }

                    for (; i < count; i++)
                        dst[(int)start + i] = src[(int)start + i] * a + b;
                }
            }
        }
    }

    // ---------------- Identity ----------------

    /// <summary>
    /// 原样透传（PaddleOCR 的 rec 图里成串出现）。常量输入会被 <see cref="ConstFolder"/> 直接折掉，
    /// 走到这里的都是运行时张量，按字节整块拷贝即可。
    /// </summary>
    private sealed class IdentityLayer : Layer
    {
        public IdentityLayer(LayerParams parameters)
        {
            Type = parameters.Type;
            Name = parameters.Name;
        }

        public override void GetMemoryShapes(
            IReadOnlyList<TensorShape> inShapes,
            int requiredOutputs,
            List<TensorShape> outShapes,
            List<TensorShape> tempShapes)
            => outShapes.Add(inShapes[0]);

        public override void Forward(
            IReadOnlyList<Tensor> inputs,
            IReadOnlyList<Tensor> outputs,
            IReadOnlyList<Tensor> temps)
        {
            var src = inputs[0];
            var dst = outputs[0];

            // 缓冲池把输入输出分到同一槽时 src/dst 可能是同一个对象，不需要（也不能）自拷贝。
            if (ReferenceEquals(src, dst) || src.ByteLength == 0)
                return;

            if (dst.ByteLength < src.ByteLength)
                throw new OnnxSharpException(
                    $"Identity '{Name}': output buffer {dst.ByteLength} B < input {src.ByteLength} B");

            src.Buffer.AsSpan(src.ByteOffset, src.ByteLength)
                .CopyTo(dst.Buffer.AsSpan(dst.ByteOffset, src.ByteLength));
        }
    }
}
