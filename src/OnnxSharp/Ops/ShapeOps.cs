using System.Runtime.CompilerServices;
using OnnxSharp.Core;
using OnnxSharp.Tensors;
using OnnxSharp.Threading;

namespace OnnxSharp.Ops;

/// <summary>
/// 纯形状/数据搬运类算子：Reshape / Concat / Split / Transpose / Slice。
/// 对应 OpenCV 的 <c>reshape_layer.cpp</c> / <c>concat_layer.cpp</c> / <c>slice_layer.cpp</c>。
/// </summary>
internal static class ShapeOps
{
    /// <summary>Concat 低于该元素数时不开线程（拷贝太小，调度开销盖过收益）。</summary>
    private const int ParallelConcatThreshold = 1 << 16;

    public static void Register()
    {
        LayerRegistry.Register("Reshape", p => new ReshapeLayer(p));
        LayerRegistry.Register("Flatten", p => new FlattenLayer(p));
        LayerRegistry.Register("Concat", p => new ConcatLayer(p));
        LayerRegistry.Register("Split", p => new SplitLayer(p));
        LayerRegistry.Register("Transpose", p => new TransposeLayer(p));
        LayerRegistry.Register("Slice", p => new SliceLayer(p));
        LayerRegistry.Register("Pad", p => new PadLayer(p));
    }

    internal static int[] ToInts(long[]? values)
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
            int[] target = Inputs.Count > 1 && Inputs[1].Idx != 0
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

        /// <summary>
        /// 目标形状来自运行时（Paraformer 的 <c>Reshape(Shape→Gather→Concat)</c>）时，
        /// 交给 <see cref="Forward"/> 自己算形状并 Fit；常量形状仍然静态推断 + 常量折叠。
        /// </summary>
        public override bool DynamicOutputShapes()
            => Inputs.Count > 1 && Inputs[1].Idx != 0 && !IsConstInput(1);

        public override void Forward(
            IReadOnlyList<Tensor> inputs,
            IReadOnlyList<Tensor> outputs,
            IReadOnlyList<Tensor> temps)
        {
            if (DynamicOutputShapes())
                outputs[0].Fit(Resolve(inputs[0].Shape, OpUtil.ReadInts(inputs[1])), inputs[0].DType);

            OpUtil.CopyBytes(inputs[0], 0, outputs[0], 0, inputs[0].Count);
        }
    }

    // ---------------- Flatten ----------------

    /// <summary>
    /// 把 [0, axis) 折成第 0 维、[axis, rank) 折成第 1 维。纯布局变换，数据不动。
    /// yolov10 / RT-DETR 的检测头前面都会来一层。
    /// </summary>
    private sealed class FlattenLayer : Layer
    {
        private readonly long _axis;

        public FlattenLayer(LayerParams parameters)
        {
            Type = parameters.Type;
            Name = parameters.Name;
            _axis = parameters.GetInt("axis", 1);
        }

        public override void GetMemoryShapes(
            IReadOnlyList<TensorShape> inShapes,
            int requiredOutputs,
            List<TensorShape> outShapes,
            List<TensorShape> tempShapes)
        {
            var input = inShapes[0];

            // ONNX Flatten 的 axis 允许等于 rank（[-r, r]），表示「整张量拍平成 [total, 1]」；
            // OpUtil.NormalizeAxis 只接受 [-r, r-1]，所以这里自己归一化。
            int axis = _axis < 0 ? (int)_axis + input.Rank : (int)_axis;
            if (axis < 0 || axis > input.Rank)
                throw new OnnxSharpException($"Flatten '{Name}': axis {_axis} out of range for rank {input.Rank}");

            long outer = 1;
            for (int i = 0; i < axis; i++)
                outer *= input.Dims[i];

            long inner = 1;
            for (int i = axis; i < input.Rank; i++)
                inner *= input.Dims[i];

            outShapes.Add(new TensorShape(new[] { (int)outer, (int)inner }));
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

            // yolo 里的 Concat 都是 axis=1、outer=1、3~4 个输入，串行版就是一条
            // 单线程 memcpy 搬 1~2 MB。各输入的落点互不重叠，直接按输入摊开。
            if (outer == 1
                && inputs.Count > 1
                && output.Count >= ParallelConcatThreshold
                && ParallelRange.Threads > 1)
            {
                var blocks = new long[inputs.Count + 1];
                for (int i = 0; i < inputs.Count; i++)
                    blocks[i + 1] = blocks[i] + (long)inputs[i].Shape.Dims[axis] * inner;

                ParallelRange.For(inputs.Count, (begin, end) =>
                {
                    for (int i = begin; i < end; i++)
                        OpUtil.CopyBytes(inputs[i], 0, output, blocks[i], blocks[i + 1] - blocks[i]);
                });

                return;
            }

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

    /// <summary>
    /// 图融合用：读取 <c>Transpose</c> 的 <c>perm</c>（未指定时为空数组，表示「反转维度」）。
    /// </summary>
    internal interface ITransposePerm
    {
        int[] Perm { get; }
    }

    private sealed class TransposeLayer : Layer, ITransposePerm
    {
        private readonly int[] _perm;

        public TransposeLayer(LayerParams parameters)
        {
            Type = parameters.Type;
            Name = parameters.Name;
            _perm = ToInts(parameters.GetInts("perm"));
        }

        public int[] Perm => _perm;

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

            var src = input.Buffer.AsSpan(input.ByteOffset, input.ByteLength);
            var dst = output.Buffer.AsSpan(output.ByteOffset, output.ByteLength);

            TransposeKernel.Transpose(src, dst, input.Shape.Dims, perm, input.DType.SizeOf());
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

            return Normalize(starts, ends, axes, steps, rank);
        }
        private (int[] Starts, int[] Ends, int[] Axes, int[] Steps) ResolveRuntime(
            IReadOnlyList<Tensor> inputs, int rank)
        {
            if (Inputs.Count <= 1)
                return Resolve(rank);

            var starts = OpUtil.ReadInts(inputs[1]);
            var ends = Inputs.Count > 2 && Inputs[2].Idx != 0 ? OpUtil.ReadInts(inputs[2]) : _endsAttr;
            var axes = IsEmptyInput(3) ? DefaultAxes(starts.Length) : OpUtil.ReadInts(inputs[3]);
            var steps = IsEmptyInput(4) ? Fill(starts.Length, 1) : OpUtil.ReadInts(inputs[4]);

            return Normalize(starts, ends, axes, steps, rank);
        }

        private (int[] Starts, int[] Ends, int[] Axes, int[] Steps) Normalize(
            int[] starts, int[] ends, int[] axes, int[] steps, int rank)
        {
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

        private bool HasRuntimeSliceInputs()
            => (Inputs.Count > 1 && Inputs[1].Idx != 0 && !IsConstInput(1))
            || (Inputs.Count > 2 && Inputs[2].Idx != 0 && !IsConstInput(2))
            || (Inputs.Count > 3 && Inputs[3].Idx != 0 && !IsConstInput(3))
            || (Inputs.Count > 4 && Inputs[4].Idx != 0 && !IsConstInput(4));

        /// <summary>starts/ends 来自运行时：形状要在 <see cref="Forward"/> 里算完再 Fit。</summary>
        public override bool DynamicOutputShapes() => HasRuntimeSliceInputs();

        private TensorShape ShapeOf(TensorShape input, int[] starts, int[] ends, int[] axes, int[] steps)
        {
            var dims = input.ToArray();
            for (int i = 0; i < axes.Length; i++)
                dims[axes[i]] = SliceLength(dims[axes[i]], starts[i], ends[i], steps[i]);

            return new TensorShape(dims);
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
            outShapes.Add(ShapeOf(input, starts, ends, axes, steps));
        }

        public override unsafe void Forward(
            IReadOnlyList<Tensor> inputs,
            IReadOnlyList<Tensor> outputs,
            IReadOnlyList<Tensor> temps)
        {
            var input = inputs[0];
            var output = outputs[0];

            int rank = input.Shape.Rank;
            var (starts, ends, axes, steps) = HasRuntimeSliceInputs()
                ? ResolveRuntime(inputs, rank)
                : Resolve(rank);

            if (DynamicOutputShapes())
                output.Fit(ShapeOf(input.Shape, starts, ends, axes, steps), input.DType);

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

            // 尾部连续可整段拷贝的维数：从最内维往外，只要 step == 1 就能并进同一段；
            // 遇到「被切过的维」时把它的长度也并进来再停。Slice 绝大多数是切
            // 最外几维、保留末维连续（切 qkv / 切坐标），这样能把上百万次逐元素
            // CopyTo 变成几十次 memcpy。
            int firstInner = inDims.Length;
            long runLength = 1;
            for (int axis = inDims.Length - 1; axis >= 0; axis--)
            {
                if (axisStep[axis] != 1)
                    break;

                runLength *= outDims[axis];
                firstInner = axis;

                if (begin[axis] != 0 || outDims[axis] != inDims[axis])
                    break;
            }

            long innerBase = 0;
            for (int axis = firstInner; axis < inDims.Length; axis++)
                innerBase += (long)begin[axis] * inStrides[axis];

            int innerBytes = checked((int)(runLength * elementSize));
            var coords = new int[firstInner];

            fixed (byte* src = input.Buffer.AsSpan(input.ByteOffset, input.ByteLength))
            fixed (byte* dst = output.Buffer.AsSpan(output.ByteOffset, output.ByteLength))
            {
                long total = output.Shape.Total();
                long written = 0;

                while (written < total)
                {
                    long source = innerBase;
                    for (int i = 0; i < firstInner; i++)
                        source += (long)(begin[i] + coords[i] * axisStep[i]) * inStrides[i];

                    Unsafe.CopyBlockUnaligned(dst + written * elementSize, src + source * elementSize, (uint)innerBytes);
                    written += runLength;

                    for (int i = firstInner - 1; i >= 0; i--)
                    {
                        if (++coords[i] < outDims[i])
                            break;
                        coords[i] = 0;
                    }
                }
            }
        }
    }
    // ---------------- Pad ----------------

    /// <summary>
    /// Pad：opset ≥ 11 用输入 <c>pads</c> / <c>constant_value</c> / <c>axes</c>，更早用属性。
    /// 支持 constant / edge / reflect；<c>pads</c> 允许为负数（相当于裁剪）。
    /// <para>Paraformer 的 <c>pad_fn</c> 用它把注意力的 K/V 补齐到固定长度。</para>
    /// </summary>
    private sealed class PadLayer : Layer
    {
        private readonly string _mode;
        private readonly int[] _padsAttr;
        private readonly double _valueAttr;

        public PadLayer(LayerParams parameters)
        {
            Type = parameters.Type;
            Name = parameters.Name;
            _mode = parameters.GetString("mode", "constant");
            _padsAttr = ToInts(parameters.GetInts("pads"));
            _valueAttr = parameters.GetFloat("value", 0.0);
        }

        private static bool HasInput(IReadOnlyList<Arg> inputs, int index)
            => index < inputs.Count && inputs[index].Idx != 0;

        /// <summary>返回长度 <c>2*rank</c> 的 pads（前 rank 个是 begin，后 rank 个是 end）。</summary>
        private int[] ResolvePads(int rank)
        {
            int[] pads = HasInput(Inputs, 1) ? OpUtil.ReadInts(OpUtil.RequireConstInput(this, 1)) : _padsAttr;

            if (pads.Length == 0 || pads.Length % 2 != 0)
                throw new OnnxSharpException($"Pad '{Name}': invalid pads ({pads.Length} values)");

            int count = pads.Length / 2;
            int[]? axes = HasInput(Inputs, 3) ? OpUtil.ReadInts(OpUtil.RequireConstInput(this, 3)) : null;

            var result = new int[2 * rank];
            if (axes is null)
            {
                if (count != rank)
                    throw new OnnxSharpException($"Pad '{Name}': pads has {count} axes, input rank is {rank}");

                for (int i = 0; i < rank; i++)
                {
                    result[i] = pads[i];
                    result[rank + i] = pads[count + i];
                }

                return result;
            }

            if (axes.Length != count)
                throw new OnnxSharpException($"Pad '{Name}': axes({axes.Length}) / pads({count}) mismatch");

            for (int i = 0; i < count; i++)
            {
                int axis = OpUtil.NormalizeAxis(axes[i], rank, $"Pad '{Name}'");
                result[axis] = pads[i];
                result[rank + axis] = pads[count + i];
            }

            return result;
        }

        public override void GetMemoryShapes(
            IReadOnlyList<TensorShape> inShapes,
            int requiredOutputs,
            List<TensorShape> outShapes,
            List<TensorShape> tempShapes)
        {
            var input = inShapes[0];
            var pads = ResolvePads(input.Rank);
            var dims = input.ToArray();

            for (int axis = 0; axis < dims.Length; axis++)
            {
                dims[axis] += pads[axis] + pads[dims.Length + axis];
                if (dims[axis] < 0)
                    throw new OnnxSharpException($"Pad '{Name}': axis {axis} becomes negative ({dims[axis]})");
            }

            outShapes.Add(new TensorShape(dims));
        }

        public override void Forward(
            IReadOnlyList<Tensor> inputs,
            IReadOnlyList<Tensor> outputs,
            IReadOnlyList<Tensor> temps)
        {
            var input = inputs[0];
            var output = outputs[0];
            int rank = input.Shape.Rank;
            var inDims = input.Shape.ToArray();
            var outDims = output.Shape.ToArray();
            var pads = ResolvePads(rank);
            int elementSize = input.DType.SizeOf();

            var source = input.Buffer.AsSpan(input.ByteOffset, input.ByteLength);
            var target = output.Buffer.AsSpan(output.ByteOffset, output.ByteLength);

            if (_mode == "constant")
            {
                Fill(target, elementSize, ReadConstantValue(inputs));
                CopyStraight(source, target, inDims, outDims, pads, elementSize);
            }
            else
            {
                CopyReplicated(source, target, inDims, outDims, pads, elementSize, _mode == "reflect");
            }
        }

        private double ReadConstantValue(IReadOnlyList<Tensor> inputs)
        {
            if (!HasInput(Inputs, 2))
                return _valueAttr;

            var tensor = inputs[2];
            return tensor.DType switch
            {
                DType.F32 => tensor.AsSpan<float>()[0],
                DType.I32 => tensor.AsSpan<int>()[0],
                DType.I64 => tensor.AsSpan<long>()[0],
                DType.I8 => tensor.AsSpan<sbyte>()[0],
                DType.U8 => tensor.AsSpan<byte>()[0],
                _ => _valueAttr,
            };
        }

        /// <summary>constant 模式：整块填常数后，把输入按「行」拷进内区（每行内部仍连续）。</summary>
        private static void CopyStraight(
            ReadOnlySpan<byte> source,
            Span<byte> target,
            int[] inDims,
            int[] outDims,
            int[] pads,
            int elementSize)
        {
            int rank = inDims.Length;
            var inStrides = OpUtil.Strides(inDims);
            var outStrides = OpUtil.Strides(outDims);

            int last = rank - 1;
            int runElements = inDims[last];
            if (runElements <= 0)
                return;

            int runBytes = runElements * elementSize;
            var coords = new int[Math.Max(rank - 1, 1)];

            // 负 pads 相当于裁剪：首/末轴都要按输出范围夹紧，否则会越界写
            int lastPad = pads[last];
            int outLast = outDims[last];
            int runStart = Math.Max(0, -lastPad);
            int runEnd = Math.Min(runElements, outLast - lastPad);
            if (runEnd <= runStart)
                return;

            int validElements = runEnd - runStart;
            int sourceStart = runStart;
            int destinationStart = runStart + lastPad;

            long rows = 1;
            for (int axis = 0; axis < last; axis++)
                rows *= inDims[axis];

            for (long row = 0; row < rows; row++)
            {
                long sourceOffset = sourceStart;
                long targetOffset = destinationStart;
                bool inside = true;

                for (int axis = 0; axis < last; axis++)
                {
                    int outputIndex = coords[axis] + pads[axis];
                    if ((uint)outputIndex >= (uint)outDims[axis])
                    {
                        inside = false;
                        break;
                    }

                    sourceOffset += (long)coords[axis] * inStrides[axis];
                    targetOffset += (long)outputIndex * outStrides[axis];
                }

                if (inside)
                {
                    source.Slice((int)sourceOffset * elementSize, validElements * elementSize)
                          .CopyTo(target.Slice((int)targetOffset * elementSize));
                }

                for (int axis = last - 1; axis >= 0; axis--)
                {
                    if (++coords[axis] < inDims[axis])
                        break;
                    coords[axis] = 0;
                }
            }
        }
        /// <summary>edge / reflect：逐元素按轴回映。</summary>
        private static void CopyReplicated(
            ReadOnlySpan<byte> source,
            Span<byte> target,
            int[] inDims,
            int[] outDims,
            int[] pads,
            int elementSize,
            bool reflect)
        {
            int rank = inDims.Length;
            var inStrides = OpUtil.Strides(inDims);
            var coords = new int[rank];

            long total = 1;
            for (int axis = 0; axis < rank; axis++)
                total *= outDims[axis];

            for (long index = 0; index < total; index++)
            {
                long sourceOffset = 0;
                for (int axis = 0; axis < rank; axis++)
                    sourceOffset += (long)MapIndex(coords[axis] - pads[axis], inDims[axis], reflect) * inStrides[axis];

                int from = (int)sourceOffset * elementSize;
                int to = (int)index * elementSize;
                source.Slice(from, elementSize).CopyTo(target.Slice(to, elementSize));

                for (int axis = rank - 1; axis >= 0; axis--)
                {
                    if (++coords[axis] < outDims[axis])
                        break;
                    coords[axis] = 0;
                }
            }
        }

        private static int MapIndex(int index, int dim, bool reflect)
        {
            if (dim <= 1)
                return 0;
            if (!reflect)
                return Math.Clamp(index, 0, dim - 1);

            while (index < 0 || index >= dim)
                index = index < 0 ? -index : 2 * dim - 2 - index;
            return index;
        }

        private static void Fill(Span<byte> target, int elementSize, double value)
        {
            Span<byte> pattern = stackalloc byte[8];
            switch (elementSize)
            {
                case 1: pattern[0] = (byte)(long)value; break;
                case 2: BitConverter.TryWriteBytes(pattern, (Half)(float)value); break;
                case 4: BitConverter.TryWriteBytes(pattern, (float)value); break;
                default: BitConverter.TryWriteBytes(pattern, (long)value); break;
            }

            var block = new byte[64];
            for (int i = 0; i + elementSize <= block.Length; i += elementSize)
                pattern[..elementSize].CopyTo(block.AsSpan(i));

            int written = 0;
            while (written + block.Length <= target.Length)
            {
                block.CopyTo(target[written..]);
                written += block.Length;
            }

            while (written + elementSize <= target.Length)
            {
                pattern[..elementSize].CopyTo(target[written..]);
                written += elementSize;
            }
        }

    }
}

