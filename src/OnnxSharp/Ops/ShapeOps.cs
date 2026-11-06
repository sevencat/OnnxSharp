using OnnxSharp.Core;
using OnnxSharp.Tensors;

namespace OnnxSharp.Ops;

/// <summary>
/// 纯形状/数据搬运类算子：Reshape / Concat / Split / Transpose / Slice。
/// 对应 OpenCV 的 <c>reshape_layer.cpp</c> / <c>concat_layer.cpp</c> / <c>slice_layer.cpp</c>。
/// </summary>
internal static class ShapeOps
{
    public static void Register()
    {
        LayerRegistry.Register("Reshape", p => new ReshapeLayer(p));
        LayerRegistry.Register("Concat", p => new ConcatLayer(p));
        LayerRegistry.Register("Split", p => new SplitLayer(p));
        LayerRegistry.Register("Transpose", p => new TransposeLayer(p));
        LayerRegistry.Register("Slice", p => new SliceLayer(p));
    }

    private static int[] ToInts(long[]? values)
    {
        if (values is null || values.Length == 0)
            return Array.Empty<int>();
        return Array.ConvertAll(values, v => checked((int)v));
    }

    private static int[] Fill(int count, int value)
    {
        var result = new int[count];
        Array.Fill(result, value);
        return result;
    }

    // ---------------- Reshape ----------------

    private sealed class ReshapeLayer : Layer
    {
        private readonly bool _allowZero;
        private readonly int[] _shapeAttr;

        public ReshapeLayer(LayerParams parameters)
        {
            Type = parameters.Type;
            Name = parameters.Name;
            _allowZero = parameters.GetInt("allowzero", 0) != 0;
            _shapeAttr = ToInts(parameters.GetInts("shape"));
        }

        public override void GetMemoryShapes(
            IReadOnlyList<TensorShape> inShapes,
            int requiredOutputs,
            List<TensorShape> outShapes,
            List<TensorShape> tempShapes)
        {
            var input = inShapes[0];
            int[] target = Inputs.Count > 1
                ? OpUtil.ReadInts(OpUtil.RequireConstInput(this, 1))
                : _shapeAttr;

            if (target.Length == 0)
                throw new OnnxSharpException($"Reshape '{Name}': no target shape");

            outShapes.Add(Resolve(input, target));
        }

        private TensorShape Resolve(TensorShape input, int[] target)
        {
            var dims = new int[target.Length];
            long known = 1;
            int inferred = -1;

            for (int i = 0; i < target.Length; i++)
            {
                int d = target[i];
                if (d == 0 && !_allowZero)
                {
                    if (i >= input.Rank)
                        throw new OnnxSharpException(
                            $"Reshape '{Name}': cannot copy dim #{i} from rank-{input.Rank} input");
                    dims[i] = input.Dims[i];
                    known = checked(known * dims[i]);
                }
                else if (d == -1)
                {
                    if (inferred >= 0)
                        throw new OnnxSharpException($"Reshape '{Name}': multiple -1 in target shape");
                    inferred = i;
                    dims[i] = 1;
                }
                else
                {
                    dims[i] = d;
                    known = checked(known * d);
                }
            }

            if (inferred >= 0)
            {
                long total = input.Total();
                if (known == 0 || total % known != 0)
                    throw new OnnxSharpException(
                        $"Reshape '{Name}': cannot infer dim from {input} to [{string.Join(",", target)}]");
                dims[inferred] = checked((int)(total / known));
            }

            var result = new TensorShape(dims);
            if (result.Total() != input.Total())
                throw new OnnxSharpException($"Reshape '{Name}': element count changed {input} to {result}");
            return result;
        }

        public override void Forward(
            IReadOnlyList<Tensor> inputs,
            IReadOnlyList<Tensor> outputs,
            IReadOnlyList<Tensor> temps)
            => OpUtil.CopyBytes(inputs[0], 0, outputs[0], 0, inputs[0].Count);
    }

    // ---------------- Concat ----------------

    private sealed class ConcatLayer : Layer
    {
        private readonly long _axis;

        public ConcatLayer(LayerParams parameters)
        {
            Type = parameters.Type;
            Name = parameters.Name;
            _axis = parameters.GetInt("axis", 0);
        }

        public override void GetMemoryShapes(
            IReadOnlyList<TensorShape> inShapes,
            int requiredOutputs,
            List<TensorShape> outShapes,
            List<TensorShape> tempShapes)
        {
            if (inShapes.Count == 0)
                throw new OnnxSharpException($"Concat '{Name}': no inputs");

            var first = inShapes[0];
            int axis = OpUtil.NormalizeAxis(_axis, first.Rank, $"Concat '{Name}'");

            var dims = first.ToArray();
            dims[axis] = 0;

            foreach (var shape in inShapes)
            {
                if (shape.Rank != first.Rank)
                    throw new OnnxSharpException($"Concat '{Name}': rank mismatch {first} vs {shape}");
                for (int i = 0; i < dims.Length; i++)
                {
                    if (i == axis)
                        dims[i] = checked(dims[i] + shape.Dims[i]);
                    else if (dims[i] != shape.Dims[i])
                        throw new OnnxSharpException($"Concat '{Name}': dim #{i} mismatch {first} vs {shape}");
                }
            }

            outShapes.Add(new TensorShape(dims));
        }

        public override void Forward(
            IReadOnlyList<Tensor> inputs,
            IReadOnlyList<Tensor> outputs,
            IReadOnlyList<Tensor> temps)
        {
            var output = outputs[0];
            int axis = OpUtil.NormalizeAxis(_axis, output.Shape.Rank, $"Concat '{Name}'");

            var outDims = output.Shape.ToArray();
            int outer = OpUtil.Product(outDims.AsSpan(0, axis));
            int inner = axis + 1 < outDims.Length
                ? OpUtil.Product(outDims.AsSpan(axis + 1))
                : 1;
            long outBlock = (long)outDims[axis] * inner;

            long offset = 0;
            foreach (var input in inputs)
            {
                var dims = input.Shape.ToArray();
                long srcBlock = (long)dims[axis] * inner;
                for (int o = 0; o < outer; o++)
                    OpUtil.CopyBytes(input, o * srcBlock, output, o * outBlock + offset, srcBlock);
                offset += srcBlock;
            }
        }
    }

    // ---------------- Split ----------------

    private sealed class SplitLayer : Layer
    {
        private readonly long _axis;
        private readonly int[] _splitAttr;
        private readonly int _numOutputs;

        public SplitLayer(LayerParams parameters)
        {
            Type = parameters.Type;
            Name = parameters.Name;
            _axis = parameters.GetInt("axis", 0);
            _splitAttr = ToInts(parameters.GetInts("split"));
            _numOutputs = (int)parameters.GetInt("num_outputs", 0);
        }

        private int[] ResolveSizes(TensorShape input, int axis)
        {
            if (_splitAttr.Length > 0)
                return _splitAttr;

            if (Inputs.Count > 1 && IsConstInput(1))
                return OpUtil.ReadInts(In(1));

            int outputs = Math.Max(_numOutputs, Outputs.Count);
            if (outputs <= 0)
                throw new OnnxSharpException($"Split '{Name}': cannot determine number of outputs");

            int dim = input.Dims[axis];
            if (dim % outputs != 0)
                throw new OnnxSharpException($"Split '{Name}': dim {dim} is not divisible by {outputs}");

            return Fill(outputs, dim / outputs);
        }

        public override void GetMemoryShapes(
            IReadOnlyList<TensorShape> inShapes,
            int requiredOutputs,
            List<TensorShape> outShapes,
            List<TensorShape> tempShapes)
        {
            var input = inShapes[0];
            int axis = OpUtil.NormalizeAxis(_axis, input.Rank, $"Split '{Name}'");
            var sizes = ResolveSizes(input, axis);

            long sum = 0;
            foreach (var size in sizes)
            {
                sum += size;
                var dims = input.ToArray();
                dims[axis] = size;
                outShapes.Add(new TensorShape(dims));
            }

            if (sum != input.Dims[axis])
                throw new OnnxSharpException(
                    $"Split '{Name}': sizes sum to {sum}, expected {input.Dims[axis]}");
        }

        public override void Forward(
            IReadOnlyList<Tensor> inputs,
            IReadOnlyList<Tensor> outputs,
            IReadOnlyList<Tensor> temps)
        {
            var input = inputs[0];
            int axis = OpUtil.NormalizeAxis(_axis, input.Shape.Rank, $"Split '{Name}'");

            var dims = input.Shape.ToArray();
            int outer = OpUtil.Product(dims.AsSpan(0, axis));
            int inner = axis + 1 < dims.Length ? OpUtil.Product(dims.AsSpan(axis + 1)) : 1;
            long inBlock = (long)dims[axis] * inner;

            long offset = 0;
            foreach (var output in outputs)
            {
                long block = (long)output.Shape.Dims[axis] * inner;
                for (int o = 0; o < outer; o++)
                    OpUtil.CopyBytes(input, o * inBlock + offset, output, o * block, block);
                offset += block;
            }
        }
    }

    // ---------------- Transpose ----------------

    private sealed class TransposeLayer : Layer
    {
        private readonly int[] _perm;

        public TransposeLayer(LayerParams parameters)
        {
            Type = parameters.Type;
            Name = parameters.Name;
            _perm = ToInts(parameters.GetInts("perm"));
        }

        private int[] ResolvePerm(int rank)
        {
            if (_perm.Length == 0)
            {
                var reversed = new int[rank];
                for (int i = 0; i < rank; i++)
                    reversed[i] = rank - 1 - i;
                return reversed;
            }

            if (_perm.Length != rank)
                throw new OnnxSharpException(
                    $"Transpose '{Name}': perm has {_perm.Length} entries, rank is {rank}");
            return _perm;
        }

        public override void GetMemoryShapes(
            IReadOnlyList<TensorShape> inShapes,
            int requiredOutputs,
            List<TensorShape> outShapes,
            List<TensorShape> tempShapes)
        {
            var input = inShapes[0];
            var perm = ResolvePerm(input.Rank);

            var dims = new int[input.Rank];
            for (int i = 0; i < dims.Length; i++)
                dims[i] = input.Dims[perm[i]];

            outShapes.Add(new TensorShape(dims));
        }

        public override void Forward(
            IReadOnlyList<Tensor> inputs,
            IReadOnlyList<Tensor> outputs,
            IReadOnlyList<Tensor> temps)
        {
            var input = inputs[0];
            var output = outputs[0];
            var perm = ResolvePerm(input.Shape.Rank);

            var inStrides = OpUtil.Strides(input.Shape.Dims);
            var outDims = output.Shape.ToArray();
            var coords = new int[outDims.Length];

            int elementSize = input.DType.SizeOf();
            var src = input.Buffer.AsSpan(input.ByteOffset, input.ByteLength);
            var dst = output.Buffer.AsSpan(output.ByteOffset, output.ByteLength);

            long total = output.Shape.Total();
            for (long n = 0; n < total; n++)
            {
                long source = 0;
                for (int i = 0; i < coords.Length; i++)
                    source += (long)coords[i] * inStrides[perm[i]];

                src.Slice(checked((int)(source * elementSize)), elementSize)
                   .CopyTo(dst.Slice(checked((int)(n * elementSize)), elementSize));

                for (int i = coords.Length - 1; i >= 0; i--)
                {
                    if (++coords[i] < outDims[i])
                        break;
                    coords[i] = 0;
                }
            }
        }
    }

    // ---------------- Slice ----------------

    private sealed class SliceLayer : Layer
    {
        private readonly int[] _startsAttr;
        private readonly int[] _endsAttr;
        private readonly int[] _axesAttr;
        private readonly int[] _stepsAttr;

        public SliceLayer(LayerParams parameters)
        {
            Type = parameters.Type;
            Name = parameters.Name;
            _startsAttr = ToInts(parameters.GetInts("starts"));
            _endsAttr = ToInts(parameters.GetInts("ends"));
            _axesAttr = ToInts(parameters.GetInts("axes"));
            _stepsAttr = ToInts(parameters.GetInts("steps"));
        }

        private bool IsEmptyInput(int index)
            => index >= Inputs.Count || Inputs[index].Idx == 0;

        private (int[] Starts, int[] Ends, int[] Axes, int[] Steps) Resolve(int rank)
        {
            int[] starts, ends, axes, steps;

            if (Inputs.Count > 1)
            {
                starts = OpUtil.ReadInts(OpUtil.RequireConstInput(this, 1));
                ends = OpUtil.ReadInts(OpUtil.RequireConstInput(this, 2));
                axes = IsEmptyInput(3) ? DefaultAxes(starts.Length) : OpUtil.ReadInts(OpUtil.RequireConstInput(this, 3));
                steps = IsEmptyInput(4) ? Fill(starts.Length, 1) : OpUtil.ReadInts(OpUtil.RequireConstInput(this, 4));
            }
            else
            {
                starts = _startsAttr;
                ends = _endsAttr;
                axes = _axesAttr.Length > 0 ? _axesAttr : DefaultAxes(starts.Length);
                steps = _stepsAttr.Length > 0 ? _stepsAttr : Fill(starts.Length, 1);
            }

            if (starts.Length != ends.Length)
                throw new OnnxSharpException(
                    $"Slice '{Name}': starts({starts.Length}) / ends({ends.Length}) mismatch");
            if (axes.Length != starts.Length || steps.Length != starts.Length)
                throw new OnnxSharpException(
                    $"Slice '{Name}': starts/ends/axes/steps must have the same length");

            var normalized = new int[axes.Length];
            for (int i = 0; i < axes.Length; i++)
            {
                normalized[i] = OpUtil.NormalizeAxis(axes[i], rank, $"Slice '{Name}'");
                if (steps[i] == 0)
                    throw new OnnxSharpException($"Slice '{Name}': step must not be 0");
            }

            return (starts, ends, normalized, steps);
        }

        private static int[] DefaultAxes(int count)
        {
            var axes = new int[count];
            for (int i = 0; i < count; i++)
                axes[i] = i;
            return axes;
        }

        private static int SliceLength(int dim, int start, int end, int step)
        {
            if (step > 0)
            {
                start = Clamp(start < 0 ? start + dim : start, 0, dim);
                end = Clamp(end < 0 ? end + dim : end, 0, dim);
                return start >= end ? 0 : (end - start + step - 1) / step;
            }

            start = Clamp(start < 0 ? start + dim : start, 0, dim - 1);
            end = Clamp(end < 0 ? end + dim : end, -1, dim - 1);
            return start <= end ? 0 : (start - end - step - 1) / -step;
        }

        private static int Clamp(int value, int min, int max) => Math.Min(Math.Max(value, min), max);

        public override void GetMemoryShapes(
            IReadOnlyList<TensorShape> inShapes,
            int requiredOutputs,
            List<TensorShape> outShapes,
            List<TensorShape> tempShapes)
        {
            var input = inShapes[0];
            var (starts, ends, axes, steps) = Resolve(input.Rank);

            var dims = input.ToArray();
            for (int i = 0; i < axes.Length; i++)
                dims[axes[i]] = SliceLength(dims[axes[i]], starts[i], ends[i], steps[i]);

            outShapes.Add(new TensorShape(dims));
        }

        public override void Forward(
            IReadOnlyList<Tensor> inputs,
            IReadOnlyList<Tensor> outputs,
            IReadOnlyList<Tensor> temps)
        {
            var input = inputs[0];
            var output = outputs[0];
            var (starts, _, axes, steps) = Resolve(input.Shape.Rank);

            var inDims = input.Shape.ToArray();
            var outDims = output.Shape.ToArray();
            var inStrides = OpUtil.Strides(inDims);

            var begin = new int[inDims.Length];
            var axisStep = new int[inDims.Length];
            for (int i = 0; i < inDims.Length; i++)
                axisStep[i] = 1;

            for (int i = 0; i < axes.Length; i++)
            {
                int axis = axes[i];
                int dim = inDims[axis];
                int start = starts[i] < 0 ? starts[i] + dim : starts[i];
                begin[axis] = steps[i] > 0 ? Clamp(start, 0, dim) : Clamp(start, 0, dim - 1);
                axisStep[axis] = steps[i];
            }

            int elementSize = input.DType.SizeOf();
            var src = input.Buffer.AsSpan(input.ByteOffset, input.ByteLength);
            var dst = output.Buffer.AsSpan(output.ByteOffset, output.ByteLength);
            var coords = new int[outDims.Length];

            long total = output.Shape.Total();
            for (long n = 0; n < total; n++)
            {
                long source = 0;
                for (int i = 0; i < coords.Length; i++)
                    source += (long)(begin[i] + coords[i] * axisStep[i]) * inStrides[i];

                src.Slice(checked((int)(source * elementSize)), elementSize)
                   .CopyTo(dst.Slice(checked((int)(n * elementSize)), elementSize));

                for (int i = coords.Length - 1; i >= 0; i--)
                {
                    if (++coords[i] < outDims[i])
                        break;
                    coords[i] = 0;
                }
            }
        }
    }
}
