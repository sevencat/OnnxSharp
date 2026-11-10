using System.Buffers;
using OnnxSharp.Core;
using OnnxSharp.Tensors;

namespace OnnxSharp.Ops;

/// <summary>
/// <c>QLinearAdd</c> / <c>QLinearSigmoid</c>（<c>com.microsoft</c> 域，ORT 的量化逐元素算子）。
/// <para>
/// 语义都是「反量化 → 算 → 重量化」：
/// <c>QLinearAdd: y = saturate(((a-azp)·as + (b-bzp)·bs) / ys + yzp)</c>、
/// <c>QLinearSigmoid: y = saturate(sigmoid((x-xzp)·xs) / ys + yzp)</c>。
/// 图里出现次数很少（YuNet 是 2 + 6），走 F32 中转足够。
/// </para>
/// </summary>
internal static class QLinearElementOps
{
    public static void Register()
    {
        LayerRegistry.Register("QLinearAdd", p => new QLinearAddLayer(p));
        LayerRegistry.Register("QLinearSigmoid", p => new QLinearSigmoidLayer(p));
    }

    // ---------------- QLinearAdd ----------------

    private sealed class QLinearAddLayer : Layer
    {
        public QLinearAddLayer(LayerParams parameters)
        {
            Type = parameters.Type;
            Name = parameters.Name;
        }

        public override void GetMemoryShapes(
            IReadOnlyList<TensorShape> inShapes,
            int requiredOutputs,
            List<TensorShape> outShapes,
            List<TensorShape> tempShapes)
            => outShapes.Add(new TensorShape(OpUtil.BroadcastDims([inShapes[0], inShapes[3]])));

        public override void GetTypes(
            IReadOnlyList<DType> inTypes,
            int requiredOutputs,
            int requiredTemps,
            List<DType> outTypes,
            List<DType> tempTypes)
        {
            var quantized = inTypes.Count > 7 && inTypes[7] == DType.I8 ? DType.I8 : DType.U8;

            outTypes.Clear();
            for (int i = 0; i < requiredOutputs; i++)
                outTypes.Add(quantized);

            tempTypes.Clear();
            for (int i = 0; i < requiredTemps; i++)
                tempTypes.Add(DType.F32);
        }

        public override void Forward(
            IReadOnlyList<Tensor> inputs,
            IReadOnlyList<Tensor> outputs,
            IReadOnlyList<Tensor> temps)
        {
            var a = inputs[0];
            float aScale = inputs[1].AsSpan<float>()[0];
            int aZeroPoint = OpUtil.ReadInts(inputs[2])[0];

            var b = inputs[3];
            float bScale = inputs[4].AsSpan<float>()[0];
            int bZeroPoint = OpUtil.ReadInts(inputs[5])[0];

            float yScale = inputs[6].AsSpan<float>()[0];
            int yZeroPoint = OpUtil.ReadInts(inputs[7])[0];

            var output = outputs[0];
            int count = (int)output.Count;

            float[] aBuffer = ArrayPool<float>.Shared.Rent(count);
            float[] bBuffer = ArrayPool<float>.Shared.Rent(count);

            try
            {
                var left = aBuffer.AsSpan(0, count);
                var right = bBuffer.AsSpan(0, count);

                Expand(a, left, aScale, aZeroPoint);
                Expand(b, right, bScale, bZeroPoint);

                for (int i = 0; i < count; i++)
                    left[i] += right[i];

                DequantizeOps.QuantizeValues(left, output, yScale, yZeroPoint);
            }
            finally
            {
                ArrayPool<float>.Shared.Return(aBuffer);
                ArrayPool<float>.Shared.Return(bBuffer);
            }
        }

        /// <summary>按输出形状广播展开成 F32（同形、标量；其它形状明确报错）。</summary>
        private static void Expand(Tensor source, Span<float> destination, float scale, int zeroPoint)
        {
            int count = destination.Length;

            if (source.Count == count)
            {
                Dequantize(source, destination, scale, zeroPoint);
                return;
            }

            if (source.Count == 1)
            {
                float value = source.DType == DType.I8
                    ? (source.AsSpan<sbyte>()[0] - zeroPoint) * scale
                    : (source.AsSpan<byte>()[0] - zeroPoint) * scale;

                destination.Fill(value);
                return;
            }

            throw new OnnxSharpException(
                $"QLinearAdd: broadcasting {source.Shape} to {count} values is not supported");
        }

        internal static void Dequantize(Tensor source, Span<float> destination, float scale, int zeroPoint)
        {
            if (source.DType == DType.I8)
            {
                var values = source.AsSpan<sbyte>();
                for (int i = 0; i < destination.Length; i++)
                    destination[i] = (values[i] - zeroPoint) * scale;
                return;
            }

            var bytes = source.AsSpan<byte>();
            for (int i = 0; i < destination.Length; i++)
                destination[i] = (bytes[i] - zeroPoint) * scale;
        }
    }

    // ---------------- QLinearSigmoid ----------------

    private sealed class QLinearSigmoidLayer : Layer
    {
        public QLinearSigmoidLayer(LayerParams parameters)
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

        public override void GetTypes(
            IReadOnlyList<DType> inTypes,
            int requiredOutputs,
            int requiredTemps,
            List<DType> outTypes,
            List<DType> tempTypes)
        {
            var quantized = inTypes.Count > 4 && inTypes[4] == DType.I8 ? DType.I8 : DType.U8;

            outTypes.Clear();
            for (int i = 0; i < requiredOutputs; i++)
                outTypes.Add(quantized);

            tempTypes.Clear();
            for (int i = 0; i < requiredTemps; i++)
                tempTypes.Add(DType.F32);
        }

        public override void Forward(
            IReadOnlyList<Tensor> inputs,
            IReadOnlyList<Tensor> outputs,
            IReadOnlyList<Tensor> temps)
        {
            var x = inputs[0];
            float xScale = inputs[1].AsSpan<float>()[0];
            int xZeroPoint = OpUtil.ReadInts(inputs[2])[0];

            float yScale = inputs[3].AsSpan<float>()[0];
            int yZeroPoint = OpUtil.ReadInts(inputs[4])[0];

            var output = outputs[0];
            int count = (int)output.Count;

            float[] buffer = ArrayPool<float>.Shared.Rent(count);
            try
            {
                var values = buffer.AsSpan(0, count);
                QLinearAddLayer.Dequantize(x, values, xScale, xZeroPoint);

                for (int i = 0; i < count; i++)
                    values[i] = 1f / (1f + MathF.Exp(-values[i]));

                DequantizeOps.QuantizeValues(values, output, yScale, yZeroPoint);
            }
            finally
            {
                ArrayPool<float>.Shared.Return(buffer);
            }
        }
    }
}
