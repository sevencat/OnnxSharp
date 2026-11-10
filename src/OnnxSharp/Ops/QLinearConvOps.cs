using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using OnnxSharp.Core;
using OnnxSharp.Kernels;
using OnnxSharp.Tensors;

namespace OnnxSharp.Ops;

/// <summary>
/// <c>QLinearConv</c>（ONNX QOperator）：量化输入 + 量化权重 + int32 bias 的 2D 卷积。
/// <para>
/// 计算路径 = <b>反量化 → 现有 F32 卷积 → 重量化</b>：
/// <c>x</c> 按 <c>(x - x_zp)·x_scale</c> 展开，常量权重按其 per-channel <c>w_scale/w_zp</c>
/// **只展开一次并缓存**，bias 预先乘上 <c>x_scale·w_scale[m]</c>，结果按 <c>y_scale/y_zp</c> 饱和回 u8/i8。
/// </para>
/// <para>
/// 为什么不做整数卷积：本机（Zen4 + .NET）拿不到可用的 int8 SIMD —— 实测 i16 展开内核比 F32 慢 4×
/// （<c>plan.md §M10-A</c>）；ORT 在没有 VNNI 的机器上走的也是「反量化成 F32 再算」这条路。
/// 将来有 VNNI 时可以把这里换成整数版，接口不变。
/// </para>
/// <para>
/// 只实现 2D（NCHW）：YuNet-int8 / MobileNet 这类目标检测模型用不到 1D/3D 量化卷积，真遇到会明确报错。
/// </para>
/// </summary>
internal static class QLinearConvOps
{
    public static void Register() => LayerRegistry.Register("QLinearConv", p => new QLinearConvLayer(p));

    private sealed class QLinearConvLayer : Layer
    {
        private readonly int _group;
        private readonly int[] _kernelShape;
        private readonly int[] _strides;
        private readonly int[] _pads;
        private readonly int[] _dilations;
        private readonly string _autoPad;

        // 常量权重的 F32 版本与缩放后的 bias（量化图的权重在推理期不变，只算一次）
        private Tensor? _floatWeights;
        private Tensor? _weightSource;
        private int[]? _weightZeroPoints;

        private Tensor? _floatBias;
        private float _biasScale = float.NaN;
        private int _biasLength = -1;

        public QLinearConvLayer(LayerParams parameters)
        {
            Type = parameters.Type;
            Name = parameters.Name;
            _group = (int)parameters.GetInt("group", 1);
            _kernelShape = ToInts(parameters.GetInts("kernel_shape"));
            _strides = ToInts(parameters.GetInts("strides"));
            _pads = ToInts(parameters.GetInts("pads"));
            _dilations = ToInts(parameters.GetInts("dilations"));
            _autoPad = parameters.GetString("auto_pad", "NOTSET");
        }

        private static int[] ToInts(long[]? values)
        {
            if (values is null)
                return [];

            var result = new int[values.Length];
            for (int i = 0; i < values.Length; i++)
                result[i] = (int)values[i];
            return result;
        }

        public override void GetMemoryShapes(
            IReadOnlyList<TensorShape> inShapes,
            int requiredOutputs,
            List<TensorShape> outShapes,
            List<TensorShape> tempShapes)
        {
            var input = inShapes[0];
            var weight = inShapes[3];

            if (input.Rank != 4)
                throw new OnnxSharpException($"QLinearConv '{Name}': only 2-D NCHW input is supported ({input})");
            if (weight.Rank != 4)
                throw new OnnxSharpException($"QLinearConv '{Name}': weight must be 4-D ({weight})");

            int channels = input[1];
            if (_group <= 0 || channels % _group != 0)
                throw new OnnxSharpException($"QLinearConv '{Name}': channels {channels} not divisible by group {_group}");

            var dims = new int[4];
            dims[0] = input[0];
            dims[1] = weight[0];

            for (int i = 0; i < 2; i++)
            {
                int kernel = _kernelShape.Length > i ? _kernelShape[i] : weight[2 + i];
                int stride = _strides.Length > i ? _strides[i] : 1;
                int dilation = _dilations.Length > i ? _dilations[i] : 1;

                int padBegin = _pads.Length > i ? _pads[i] : 0;
                int padEnd = _pads.Length > 2 + i ? _pads[2 + i] : padBegin;

                if (_autoPad is "SAME_UPPER" or "SAME_LOWER" or "VALID")
                    (padBegin, padEnd) = AutoPad(_autoPad, input[2 + i], kernel, stride, dilation);

                int extent = (input[2 + i] + padBegin + padEnd - dilation * (kernel - 1) - 1) / stride + 1;
                if (extent <= 0)
                    throw new OnnxSharpException(
                        $"QLinearConv '{Name}': spatial dim {i} becomes {extent} (input {input}, kernel {kernel})");
                dims[2 + i] = extent;
            }

            outShapes.Add(new TensorShape(dims));

            // 两个 F32 中转缓冲（反量化后的输入、卷积输出）交给引擎复用，避免每帧 106 次分配
            tempShapes.Add(input);
            tempShapes.Add(new TensorShape(dims));
        }

        public override void GetTypes(
            IReadOnlyList<DType> inTypes,
            int requiredOutputs,
            int requiredTemps,
            List<DType> outTypes,
            List<DType> tempTypes)
        {
            // 输出 dtype 跟 y_zero_point 一致（u8 或 i8）
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
            var x = inputs[0];
            float xScale = inputs[1].AsSpan<float>()[0];
            int xZeroPoint = OpUtil.ReadInts(inputs[2])[0];

            var w = inputs[3];
            var wScale = OpUtil.ReadFloats(inputs[4]);
            var wZeroPoint = Inputs.Count > 5 && Inputs[5].Idx != 0 ? OpUtil.ReadInts(inputs[5]) : [0];

            float yScale = inputs[6].AsSpan<float>()[0];
            int yZeroPoint = OpUtil.ReadInts(inputs[7])[0];

            var biasInt = Inputs.Count > 8 && Inputs[8].Idx != 0 ? OpUtil.ReadInts(inputs[8]) : [];
            var output = outputs[0];

            var weights = GetFloatWeights(w, wScale, wZeroPoint);
            var bias = GetFloatBias(biasInt, xScale, wScale, w.Shape.Dims[0]);

            Resolve(x.Shape, out int[] strides, out int[] dilations, out int[] pads);

            var inputFloat = temps.Count > 0 ? temps[0] : Tensor.Allocate(x.Shape, DType.F32);
            var outputFloat = temps.Count > 1 ? temps[1] : Tensor.Allocate(output.Shape, DType.F32);

            Dequantize(x, inputFloat, xScale, xZeroPoint);

            if (!PackedConv.TryConv(
                    inputFloat, weights, bias, outputFloat,
                    _group, strides[0], strides[1], dilations[0], dilations[1], pads[0], pads[1])
                && !Winograd.TryConv3x3(
                    inputFloat, weights, bias, outputFloat,
                    _group, strides[0], strides[1], dilations[0], dilations[1], pads[0], pads[1]))
            {
                ConvKernels.Im2Col(
                    inputFloat, weights, bias, outputFloat,
                    _group, strides[0], strides[1], dilations[0], dilations[1], pads[0], pads[1]);
            }

            Requantize(outputFloat, output, yScale, yZeroPoint);
        }

        private void Resolve(TensorShape input, out int[] strides, out int[] dilations, out int[] pads)
        {
            strides = [_strides.Length > 0 ? _strides[0] : 1, _strides.Length > 1 ? _strides[1] : 1];
            dilations = [_dilations.Length > 0 ? _dilations[0] : 1, _dilations.Length > 1 ? _dilations[1] : 1];

            int padTop = _pads.Length > 0 ? _pads[0] : 0;
            int padLeft = _pads.Length > 1 ? _pads[1] : 0;
            pads =
            [
                padTop,
                padLeft,
                _pads.Length > 2 ? _pads[2] : padTop,
                _pads.Length > 3 ? _pads[3] : padLeft,
            ];

            if (_autoPad is not ("SAME_UPPER" or "SAME_LOWER" or "VALID"))
                return;

            for (int i = 0; i < 2; i++)
            {
                int kernel = _kernelShape.Length > i ? _kernelShape[i] : 0;
                if (kernel <= 0)
                    continue;

                var (begin, end) = AutoPad(_autoPad, input[2 + i], kernel, strides[i], dilations[i]);
                pads[i] = begin;
                pads[2 + i] = end;
            }
        }

        private Tensor GetFloatWeights(Tensor weight, ReadOnlySpan<float> scales, int[] zeroPoints)
        {
            if (_floatWeights is not null && ReferenceEquals(_weightSource, weight) && SameInts(_weightZeroPoints, zeroPoints))
                return _floatWeights;

            var dims = weight.Shape.ToArray();
            int outChannels = dims[0];
            int inner = 1;
            for (int i = 1; i < dims.Length; i++)
                inner *= dims[i];

            var tensor = Tensor.Allocate(weight.Shape, DType.F32);
            var target = tensor.AsSpan<float>();

            if (weight.DType == DType.I8)
            {
                var source = weight.AsSpan<sbyte>();
                for (int c = 0; c < outChannels; c++)
                {
                    float scale = scales[Math.Min(c, scales.Length - 1)];
                    int zp = zeroPoints[Math.Min(c, zeroPoints.Length - 1)];
                    int baseIndex = c * inner;
                    for (int i = 0; i < inner; i++)
                        target[baseIndex + i] = (source[baseIndex + i] - zp) * scale;
                }
            }
            else
            {
                var source = weight.AsSpan<byte>();
                for (int c = 0; c < outChannels; c++)
                {
                    float scale = scales[Math.Min(c, scales.Length - 1)];
                    int zp = zeroPoints[Math.Min(c, zeroPoints.Length - 1)];
                    int baseIndex = c * inner;
                    for (int i = 0; i < inner; i++)
                        target[baseIndex + i] = (source[baseIndex + i] - zp) * scale;
                }
            }

            _floatWeights = tensor;
            _weightSource = weight;
            _weightZeroPoints = zeroPoints;
            return tensor;
        }

        /// <summary>
        /// ONNX 的 int32 bias 是按 <c>x_scale·w_scale[m]</c> 量化的，换回 F32 时要乘回去；
        /// 没有 bias 时返回 null（卷积内核按 0 处理）。
        /// </summary>
        private Tensor? GetFloatBias(int[] bias, float xScale, ReadOnlySpan<float> wScale, int outChannels)
        {
            if (bias.Length == 0)
                return null;

            if (_floatBias is not null && _biasScale.Equals(xScale) && _biasLength == bias.Length)
                return _floatBias;

            var tensor = Tensor.Allocate(new TensorShape(outChannels), DType.F32);
            var target = tensor.AsSpan<float>();
            int count = Math.Min(outChannels, bias.Length);

            for (int c = 0; c < count; c++)
                target[c] = bias[c] * xScale * wScale[Math.Min(c, wScale.Length - 1)];

            for (int c = count; c < outChannels; c++)
                target[c] = 0f;

            _floatBias = tensor;
            _biasScale = xScale;
            _biasLength = bias.Length;
            return tensor;
        }

        private static void Dequantize(Tensor source, Tensor destination, float scale, int zeroPoint)
        {
            var target = destination.AsSpan<float>();
            var vScale = Vector256.Create(scale);
            var vZero = Vector256.Create((float)zeroPoint);

            if (source.DType == DType.I8)
            {
                var values = source.AsSpan<sbyte>();
                int i = 0;
                if (Avx2.IsSupported)
                {
                    for (; i + 8 <= values.Length; i += 8)
                    {
                        var raw = Vector128.LoadUnsafe(ref MemoryMarshal.GetReference(values[i..]));
                        var floats = Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(raw));
                        Avx.Multiply(Avx.Subtract(floats, vZero), vScale).StoreUnsafe(ref target[i]);
                    }
                }

                for (; i < values.Length; i++)
                    target[i] = (values[i] - zeroPoint) * scale;
                return;
            }

            {
                var values = source.AsSpan<byte>();
                int i = 0;
                if (Avx2.IsSupported)
                {
                    for (; i + 8 <= values.Length; i += 8)
                    {
                        var raw = Vector128.LoadUnsafe(ref MemoryMarshal.GetReference(values[i..]));
                        var floats = Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(raw));
                        Avx.Multiply(Avx.Subtract(floats, vZero), vScale).StoreUnsafe(ref target[i]);
                    }
                }

                for (; i < values.Length; i++)
                    target[i] = (values[i] - zeroPoint) * scale;
            }
        }

        private static void Requantize(Tensor source, Tensor destination, float scale, int zeroPoint)
        {
            var values = source.AsSpan<float>();
            float inverse = 1f / scale;

            bool signed = destination.DType == DType.I8;
            int min = signed ? sbyte.MinValue : 0;
            int max = signed ? sbyte.MaxValue : 255;

            var quantized = destination.AsSpan<byte>();
            var vInverse = Vector256.Create(inverse);
            var vZero = Vector256.Create((float)zeroPoint);
            var vMin = Vector256.Create((float)min);
            var vMax = Vector256.Create((float)max);

            Span<int> lanes = stackalloc int[8];
            int i = 0;

            if (Avx.IsSupported)
            {
                for (; i + 8 <= values.Length; i += 8)
                {
                    var scaled = Avx.Multiply(Vector256.LoadUnsafe(ref MemoryMarshal.GetReference(values[i..])), vInverse);
                    var rounded = Avx.Add(Avx.RoundToNearestInteger(scaled), vZero);
                    rounded = Avx.Min(vMax, Avx.Max(vMin, rounded));

                    var ints = Avx.ConvertToVector256Int32(rounded);
                    ints.StoreUnsafe(ref lanes[0]);

                    for (int k = 0; k < 8; k++)
                        quantized[i + k] = unchecked((byte)(sbyte)lanes[k]);
                }
            }

            for (; i < values.Length; i++)
            {
                float q = MathF.Round(values[i] * inverse, MidpointRounding.ToEven) + zeroPoint;
                quantized[i] = unchecked((byte)(sbyte)Math.Clamp((int)q, min, max));
            }
        }

        private static bool SameInts(int[]? left, int[] right)
        {
            if (left is null || left.Length != right.Length)
                return false;

            for (int i = 0; i < left.Length; i++)
            {
                if (left[i] != right[i])
                    return false;
            }

            return true;
        }

        /// <summary>与 <c>Conv</c> 同一套 auto_pad 口径（SAME_* 把多出来的 pixel 摊在头/尾）。</summary>
        private static (int Begin, int End) AutoPad(string mode, int input, int kernel, int stride, int dilation)
        {
            if (mode is "VALID")
                return (0, 0);

            int extent = dilation * (kernel - 1) + 1;
            int output = (input + stride - 1) / stride;
            int total = Math.Max(0, (output - 1) * stride + extent - input);
            int begin = total / 2;

            return mode is "SAME_LOWER" ? (total - begin, begin) : (begin, total - begin);
        }
    }
}
