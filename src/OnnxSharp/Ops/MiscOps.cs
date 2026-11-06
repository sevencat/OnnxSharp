using OnnxSharp.Core;
using OnnxSharp.Tensors;

namespace OnnxSharp.Ops;

/// <summary>
/// 类型转换与归约类算子：Cast / ConstantOfShape / ReduceSum / ReduceMax。
/// <para>对应 OpenCV 的 <c>elementwise_layer.cpp</c>（Cast）与 <c>reduce_layer.cpp</c>。</para>
/// </summary>
internal static class MiscOps
{
    public static void Register()
    {
        LayerRegistry.Register("Cast", p => new CastLayer(p));
        LayerRegistry.Register("ConstantOfShape", p => new ConstantOfShapeLayer(p));
        LayerRegistry.Register("ReduceSum", p => new ReduceLayer(p, ReduceKind.Sum));
        LayerRegistry.Register("ReduceMax", p => new ReduceLayer(p, ReduceKind.Max));
    }

    internal enum ReduceKind
    {
        Sum,
        Max,
    }

    // ---------------- Cast ----------------

    private sealed class CastLayer : Layer
    {
        private readonly DType _target;

        public CastLayer(LayerParams parameters)
        {
            Type = parameters.Type;
            Name = parameters.Name;
            _target = ToDType((int)parameters.GetInt("to", 1));
        }

        private static DType ToDType(int onnxType) => onnxType switch
        {
            1 => DType.F32,
            2 => DType.U8,
            3 => DType.I8,
            6 => DType.I32,
            7 => DType.I64,
            9 => DType.Bool,
            10 => DType.F16,
            _ => throw new OnnxSharpException($"Cast: unsupported target type {onnxType}"),
        };

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
            outTypes.Clear();
            for (int i = 0; i < requiredOutputs; i++)
                outTypes.Add(_target);

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
            var output = outputs[0];

            if (source.DType == _target && source.Count == output.Count)
            {
                OpUtil.CopyBytes(source, 0, output, 0, source.Count);
                return;
            }

            for (long i = 0; i < output.Count; i++)
                SetValue(output, i, GetValue(source, i));
        }

        private static double GetValue(Tensor tensor, long index) => tensor.DType switch
        {
            DType.F32 => tensor.AsSpan<float>()[(int)index],
            DType.I64 => tensor.AsSpan<long>()[(int)index],
            DType.I32 => tensor.AsSpan<int>()[(int)index],
            DType.I8 => tensor.AsSpan<sbyte>()[(int)index],
            DType.U8 => tensor.AsSpan<byte>()[(int)index],
            DType.Bool => tensor.AsSpan<byte>()[(int)index] != 0 ? 1 : 0,
            _ => throw new OnnxSharpException($"Cast: unsupported source dtype {tensor.DType}"),
        };

        private static void SetValue(Tensor tensor, long index, double value)
        {
            switch (tensor.DType)
            {
                case DType.F32:
                    tensor.AsSpan<float>()[(int)index] = (float)value;
                    break;
                case DType.I64:
                    tensor.AsSpan<long>()[(int)index] = (long)value;
                    break;
                case DType.I32:
                    tensor.AsSpan<int>()[(int)index] = (int)value;
                    break;
                case DType.I8:
                    tensor.AsSpan<sbyte>()[(int)index] = (sbyte)value;
                    break;
                case DType.U8:
                    tensor.AsSpan<byte>()[(int)index] = (byte)value;
                    break;
                case DType.Bool:
                    tensor.AsSpan<byte>()[(int)index] = value != 0 ? (byte)1 : (byte)0;
                    break;
                default:
                    throw new OnnxSharpException($"Cast: unsupported target dtype {tensor.DType}");
            }
        }
    }

    // ---------------- ConstantOfShape ----------------

    private sealed class ConstantOfShapeLayer : Layer
    {
        private readonly Tensor? _value;

        public ConstantOfShapeLayer(LayerParams parameters)
        {
            Type = parameters.Type;
            Name = parameters.Name;
            _value = parameters.TryGet("value")?.Tensor;
        }

        private DType OutputDType => _value?.DType ?? DType.F32;

        public override void GetMemoryShapes(
            IReadOnlyList<TensorShape> inShapes,
            int requiredOutputs,
            List<TensorShape> outShapes,
            List<TensorShape> tempShapes)
            => outShapes.Add(new TensorShape(OpUtil.ReadInts(OpUtil.RequireConstInput(this, 0))));

        public override void GetTypes(
            IReadOnlyList<DType> inTypes,
            int requiredOutputs,
            int requiredTemps,
            List<DType> outTypes,
            List<DType> tempTypes)
        {
            outTypes.Clear();
            for (int i = 0; i < requiredOutputs; i++)
                outTypes.Add(OutputDType);

            tempTypes.Clear();
            for (int i = 0; i < requiredTemps; i++)
                tempTypes.Add(DType.F32);
        }

        public override void Forward(
            IReadOnlyList<Tensor> inputs,
            IReadOnlyList<Tensor> outputs,
            IReadOnlyList<Tensor> temps)
        {
            var output = outputs[0];

            switch (output.DType)
            {
                case DType.F32:
                    output.AsSpan<float>().Fill(_value is null ? 0f : _value.AsSpan<float>()[0]);
                    break;
                case DType.I64:
                    output.AsSpan<long>().Fill(_value is null ? 0L : _value.AsSpan<long>()[0]);
                    break;
                case DType.I32:
                    output.AsSpan<int>().Fill(_value is null ? 0 : _value.AsSpan<int>()[0]);
                    break;
                case DType.Bool:
                case DType.U8:
                    output.AsSpan<byte>().Fill(_value is null ? (byte)0 : _value.AsSpan<byte>()[0]);
                    break;
                default:
                    throw new OnnxSharpException(
                        $"ConstantOfShape '{Name}': unsupported dtype {output.DType}");
            }
        }
    }

    // ---------------- ReduceSum / ReduceMax ----------------

    private sealed class ReduceLayer : Layer
    {
        private readonly ReduceKind _kind;
        private readonly int[] _axesAttr;
        private readonly bool _keepDims;
        private readonly bool _noopWithEmptyAxes;

        public ReduceLayer(LayerParams parameters, ReduceKind kind)
        {
            Type = parameters.Type;
            Name = parameters.Name;
            _kind = kind;
            _axesAttr = ShapeOps.ToInts(parameters.GetInts("axes"));
            _keepDims = parameters.GetInt("keepdims", 1) != 0;
            _noopWithEmptyAxes = parameters.GetInt("noop_with_empty_axes", 0) != 0;
        }

        private bool[] ResolveMask(int rank)
        {
            var raw = OpUtil.Axes(this, _axesAttr);
            var reduced = new bool[rank];

            if (raw.Length == 0)
            {
                // 空 axes：默认「归约所有轴」（除非 noop_with_empty_axes=1，此时整体不变）
                if (!_noopWithEmptyAxes)
                    Array.Fill(reduced, true);
                return reduced;
            }

            foreach (var axis in OpUtil.NormalizeAxes(raw, rank, $"'{Type}' '{Name}'"))
                reduced[axis] = true;

            return reduced;
        }

        public override void GetMemoryShapes(
            IReadOnlyList<TensorShape> inShapes,
            int requiredOutputs,
            List<TensorShape> outShapes,
            List<TensorShape> tempShapes)
        {
            var input = inShapes[0];
            var reduced = ResolveMask(input.Rank);

            var dims = new List<int>(input.Rank);
            for (int i = 0; i < input.Rank; i++)
            {
                if (reduced[i])
                {
                    if (_keepDims)
                        dims.Add(1);
                }
                else
                {
                    dims.Add(input[i]);
                }
            }

            outShapes.Add(new TensorShape(dims.ToArray()));
        }

        public override void Forward(
            IReadOnlyList<Tensor> inputs,
            IReadOnlyList<Tensor> outputs,
            IReadOnlyList<Tensor> temps)
        {
            var input = inputs[0];
            var output = outputs[0];

            var dims = input.Shape.ToArray();
            int rank = dims.Length;
            var reduced = ResolveMask(rank);

            if (output.Count == input.Count)
            {
                // noop_with_empty_axes 命中：原样拷贝
                OpUtil.CopyBytes(input, 0, output, 0, input.Count);
                return;
            }

            var inStrides = OpUtil.Strides(dims);
            var outStrides = OpUtil.Strides(output.Shape.ToArray());

            // 每个输入轴对「输出线性偏移」的贡献步长；被归约的轴贡献 0
            var map = new int[rank];
            int outAxis = 0;
            for (int i = 0; i < rank; i++)
            {
                if (reduced[i])
                {
                    map[i] = 0;
                }
                else
                {
                    map[i] = outStrides[outAxis++];
                }
            }

            if (input.DType == DType.I64)
            {
                ReduceInt64(input.AsSpan<long>(), output.AsSpan<long>(), dims, inStrides, map);
                return;
            }

            if (input.DType != DType.F32)
                throw new OnnxSharpException($"'{Type}' '{Name}': only F32/I64 is supported");

            var src = input.AsSpan<float>();
            var dst = output.AsSpan<float>();

            if (_kind == ReduceKind.Max)
                dst.Fill(float.NegativeInfinity);
            else
                dst.Clear();

            var coords = new int[rank];

            for (int n = 0; n < src.Length; n++)
            {
                int rest = n;
                long target = 0;

                for (int d = 0; d < rank; d++)
                {
                    coords[d] = rest / inStrides[d];
                    rest -= coords[d] * inStrides[d];
                    target += (long)coords[d] * map[d];
                }

                float value = src[n];

                if (_kind == ReduceKind.Max)
                {
                    if (value > dst[(int)target])
                        dst[(int)target] = value;
                }
                else
                {
                    dst[(int)target] += value;
                }
            }
        }

        /// <summary>形状链里的整数归约（如 ReduceSum(shape, ...)）。</summary>
        private void ReduceInt64(
            ReadOnlySpan<long> src,
            Span<long> dst,
            int[] dims,
            int[] inStrides,
            int[] map)
        {
            if (_kind == ReduceKind.Max)
                dst.Fill(long.MinValue);
            else
                dst.Clear();

            var coords = new int[dims.Length];

            for (int n = 0; n < src.Length; n++)
            {
                int rest = n;
                long target = 0;

                for (int d = 0; d < dims.Length; d++)
                {
                    coords[d] = rest / inStrides[d];
                    rest -= coords[d] * inStrides[d];
                    target += (long)coords[d] * map[d];
                }

                long value = src[n];

                if (_kind == ReduceKind.Max)
                {
                    if (value > dst[(int)target])
                        dst[(int)target] = value;
                }
                else
                {
                    dst[(int)target] += value;
                }
            }
        }
    }
}
