using OnnxSharp.Core;
using OnnxSharp.Tensors;

namespace OnnxSharp.Ops;

/// <summary>
/// 索引 / 形状类算子：Shape / Gather / GatherElements / Unsqueeze / Squeeze。
/// <para>对应 OpenCV 的 <c>gather_layer.cpp</c> 与 reshape 家族。</para>
/// </summary>
internal static class IndexOps
{
    public static void Register()
    {
        LayerRegistry.Register("Shape", p => new ShapeLayer(p));
        LayerRegistry.Register("Gather", p => new GatherLayer(p));
        LayerRegistry.Register("GatherElements", p => new GatherElementsLayer(p));
        LayerRegistry.Register("GatherND", p => new GatherNDLayer(p));
        LayerRegistry.Register("Unsqueeze", p => new UnsqueezeLayer(p));
        LayerRegistry.Register("Squeeze", p => new SqueezeLayer(p));
    }

    // ---------------- Shape ----------------

    private sealed class ShapeLayer : Layer
    {
        /// <summary>opset 15+ 的 <c>start</c> / <c>end</c>（裁剪输出的维度区间）。</summary>
        private readonly long _start;
        private readonly long _end;

        public ShapeLayer(LayerParams parameters)
        {
            Type = parameters.Type;
            Name = parameters.Name;
            _start = parameters.GetInt("start", 0);
            _end = parameters.GetInt("end", long.MaxValue);
        }

        /// <summary>输出的维度个数（1-D I64）。</summary>
        private static int Count(TensorShape shape, long start, long end)
        {
            int from = Clamp(start, shape.Rank);
            int to = Clamp(end, shape.Rank);
            return Math.Max(to - from, 0);
        }

        private static int Clamp(long value, int rank)
        {
            long v = value < 0 ? value + rank : value;
            return v <= 0 ? 0 : v >= rank ? rank : (int)v;
        }

        /// <summary>输出是 1-D I64：[dimCount]。</summary>
        public override void GetMemoryShapes(
            IReadOnlyList<TensorShape> inShapes,
            int requiredOutputs,
            List<TensorShape> outShapes,
            List<TensorShape> tempShapes)
            => outShapes.Add(new TensorShape(Count(inShapes[0], _start, _end)));

        public override void GetTypes(
            IReadOnlyList<DType> inTypes,
            int requiredOutputs,
            int requiredTemps,
            List<DType> outTypes,
            List<DType> tempTypes)
        {
            outTypes.Clear();
            for (int i = 0; i < requiredOutputs; i++)
                outTypes.Add(DType.I64);

            tempTypes.Clear();
            for (int i = 0; i < requiredTemps; i++)
                tempTypes.Add(DType.F32);
        }

        /// <summary>形状静态已知时，<c>Shape</c> 的结果就是常量（RF-DETR 的形状链靠这条折叠掉）。</summary>
        internal override bool TryFoldShapeOnly(IReadOnlyList<TensorShape> inShapes, out Tensor? result)
        {
            var shape = inShapes[0];
            int count = Count(shape, _start, _end);

            var tensor = Tensor.Allocate(new TensorShape(count), DType.I64);
            var target = tensor.AsSpan<long>();

            int from = Clamp(_start, shape.Rank);
            for (int i = 0; i < count; i++)
                target[i] = shape[from + i];

            result = tensor;
            return true;
        }

        public override void Forward(
            IReadOnlyList<Tensor> inputs,
            IReadOnlyList<Tensor> outputs,
            IReadOnlyList<Tensor> temps)
        {
            var dims = inputs[0].Shape.Dims;
            var target = outputs[0].AsSpan<long>();

            int from = Clamp(_start, dims.Length);
            for (int i = 0; i < target.Length; i++)
                target[i] = dims[from + i];
        }
    }

    // ---------------- Gather ----------------

    private sealed class GatherLayer : Layer
    {
        private readonly long _axis;

        public GatherLayer(LayerParams parameters)
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
            var data = inShapes[0];
            var indices = inShapes[1];
            int axis = OpUtil.NormalizeAxis(_axis, data.Rank, $"Gather '{Name}'");

            var dims = new List<int>(data.Rank + indices.Rank);
            for (int i = 0; i < axis; i++)
                dims.Add(data[i]);
            for (int i = 0; i < indices.Rank; i++)
                dims.Add(indices[i]);
            for (int i = axis + 1; i < data.Rank; i++)
                dims.Add(data[i]);

            outShapes.Add(new TensorShape(dims.ToArray()));
        }

        public override void Forward(
            IReadOnlyList<Tensor> inputs,
            IReadOnlyList<Tensor> outputs,
            IReadOnlyList<Tensor> temps)
        {
            var data = inputs[0];
            var indices = OpUtil.ReadInts(inputs[1]);
            var output = outputs[0];

            var dataDims = data.Shape.ToArray();
            int axis = OpUtil.NormalizeAxis(_axis, dataDims.Length, $"Gather '{Name}'");
            int axisDim = dataDims[axis];

            long outer = 1;
            for (int i = 0; i < axis; i++)
                outer *= dataDims[i];

            long inner = 1;
            for (int i = axis + 1; i < dataDims.Length; i++)
                inner *= dataDims[i];

            // 每个「索引」取一整块连续数据，按块拷贝（dtype 无关）
            for (long o = 0; o < outer; o++)
            {
                for (int i = 0; i < indices.Length; i++)
                {
                    int k = indices[i];
                    if (k < 0)
                        k += axisDim;
                    if ((uint)k >= (uint)axisDim)
                        throw new OnnxSharpException(
                            $"Gather '{Name}': index {indices[i]} out of range for axis dim {axisDim}");

                    long source = (o * axisDim + k) * inner;
                    long target = (o * indices.Length + i) * inner;
                    OpUtil.CopyBytes(data, source, output, target, inner);
                }
            }
        }
    }

    // ---------------- GatherElements ----------------

    private sealed class GatherElementsLayer : Layer
    {
        private readonly long _axis;

        public GatherElementsLayer(LayerParams parameters)
        {
            Type = parameters.Type;
            Name = parameters.Name;
            _axis = parameters.GetInt("axis", 0);
        }

        /// <summary>输出形状 = indices 形状。</summary>
        public override void GetMemoryShapes(
            IReadOnlyList<TensorShape> inShapes,
            int requiredOutputs,
            List<TensorShape> outShapes,
            List<TensorShape> tempShapes)
            => outShapes.Add(inShapes[1]);

        public override void Forward(
            IReadOnlyList<Tensor> inputs,
            IReadOnlyList<Tensor> outputs,
            IReadOnlyList<Tensor> temps)
        {
            var data = inputs[0];
            var indices = inputs[1];
            var output = outputs[0];

            if (data.Shape.Rank != indices.Shape.Rank)
                throw new OnnxSharpException($"GatherElements '{Name}': data/indices rank mismatch");

            var dims = indices.Shape.ToArray();
            int axis = OpUtil.NormalizeAxis(_axis, dims.Length, $"GatherElements '{Name}'");
            int axisDim = data.Shape[axis];

            var dataStrides = OpUtil.Strides(data.Shape.ToArray());
            var indexStrides = OpUtil.Strides(dims);
            var index = OpUtil.ReadInts(indices);
            var coords = new int[dims.Length];

            for (int n = 0; n < index.Length; n++)
            {
                int rest = n;
                for (int d = 0; d < dims.Length; d++)
                {
                    coords[d] = rest / indexStrides[d];
                    rest -= coords[d] * indexStrides[d];
                }

                int k = index[n];
                if (k < 0)
                    k += axisDim;
                if ((uint)k >= (uint)axisDim)
                    throw new OnnxSharpException(
                        $"GatherElements '{Name}': index {index[n]} out of range for axis dim {axisDim}");

                long source = 0;
                for (int d = 0; d < dims.Length; d++)
                    source += (long)(d == axis ? k : coords[d]) * dataStrides[d];

                OpUtil.CopyBytes(data, source, output, n, 1);
            }
        }
    }

    // ---------------- GatherND ----------------

    /// <summary>
    /// <c>GatherND(data, indices)</c>：indices 最后一维是坐标元组长度 r，
    /// 输出形状 = indices.shape[:-1] + data.shape[r:]。
    /// </summary>
    private sealed class GatherNDLayer : Layer
    {
        public GatherNDLayer(LayerParams parameters)
        {
            Type = parameters.Type;
            Name = parameters.Name;
        }

        private int ResolveRank(TensorShape data, TensorShape indices)
        {
            int r = indices.Dims[indices.Rank - 1];
            if (r <= 0 || r > data.Rank)
                throw new OnnxSharpException(
                    $"GatherND '{Name}': indices last dim {r} out of range for data rank {data.Rank}");
            return r;
        }

        public override void GetMemoryShapes(
            IReadOnlyList<TensorShape> inShapes,
            int requiredOutputs,
            List<TensorShape> outShapes,
            List<TensorShape> tempShapes)
        {
            var data = inShapes[0];
            var indices = inShapes[1];
            int r = ResolveRank(data, indices);

            int outer = indices.Rank - 1;
            int tail = data.Rank - r;

            var dims = new int[outer + tail];
            for (int i = 0; i < outer; i++)
                dims[i] = indices.Dims[i];
            for (int i = 0; i < tail; i++)
                dims[outer + i] = data.Dims[r + i];

            outShapes.Add(new TensorShape(dims));
        }

        public override void Forward(
            IReadOnlyList<Tensor> inputs,
            IReadOnlyList<Tensor> outputs,
            IReadOnlyList<Tensor> temps)
        {
            var data = inputs[0];
            var indices = inputs[1];
            var output = outputs[0];

            int r = ResolveRank(data.Shape, indices.Shape);

            var strides = new long[r];
            long stride = 1;
            for (int i = data.Shape.Rank - 1; i >= r; i--)
                stride *= data.Shape.Dims[i];

            long slice = stride;
            for (int i = r - 1; i >= 0; i--)
            {
                strides[i] = stride;
                stride *= data.Shape.Dims[i];
            }

            var idx = indices.DType == DType.I64
                ? indices.AsSpan<long>()
                : throw new OnnxSharpException($"GatherND '{Name}': indices must be int64, got {indices.DType}");

            long tuples = idx.Length / r;

            for (long t = 0; t < tuples; t++)
            {
                long offset = 0;
                for (int i = 0; i < r; i++)
                {
                    long v = idx[(int)(t * r + i)];
                    int dim = data.Shape.Dims[i];
                    if (v < 0)
                        v += dim;
                    if ((ulong)v >= (ulong)dim)
                        throw new OnnxSharpException($"GatherND '{Name}': index {v} out of range for dim {dim}");

                    offset += v * strides[i];
                }

                OpUtil.CopyBytes(data, offset, output, t * slice, slice);
            }
        }
    }

    // ---------------- Unsqueeze ----------------

    private sealed class UnsqueezeLayer : Layer
    {
        private readonly int[] _axesAttr;

        public UnsqueezeLayer(LayerParams parameters)
        {
            Type = parameters.Type;
            Name = parameters.Name;
            _axesAttr = ShapeOps.ToInts(parameters.GetInts("axes"));
        }

        public override void GetMemoryShapes(
            IReadOnlyList<TensorShape> inShapes,
            int requiredOutputs,
            List<TensorShape> outShapes,
            List<TensorShape> tempShapes)
        {
            var input = inShapes[0];
            var axes = OpUtil.Axes(this, _axesAttr);
            int outRank = input.Rank + axes.Length;

            // Unsqueeze 的负轴按「输出 rank」归一化
            var normalized = OpUtil.NormalizeAxes(axes, outRank, $"Unsqueeze '{Name}'");
            if (normalized.Length != axes.Length)
                throw new OnnxSharpException($"Unsqueeze '{Name}': duplicate axes");

            var dims = new int[outRank];
            int source = 0;
            int cursor = 0;

            for (int i = 0; i < outRank; i++)
            {
                if (cursor < normalized.Length && normalized[cursor] == i)
                {
                    dims[i] = 1;
                    cursor++;
                }
                else
                {
                    dims[i] = input[source++];
                }
            }

            outShapes.Add(new TensorShape(dims));
        }

        public override void Forward(
            IReadOnlyList<Tensor> inputs,
            IReadOnlyList<Tensor> outputs,
            IReadOnlyList<Tensor> temps)
            => OpUtil.CopyBytes(inputs[0], 0, outputs[0], 0, inputs[0].Count);
    }

    // ---------------- Squeeze ----------------

    private sealed class SqueezeLayer : Layer
    {
        private readonly int[] _axesAttr;

        public SqueezeLayer(LayerParams parameters)
        {
            Type = parameters.Type;
            Name = parameters.Name;
            _axesAttr = ShapeOps.ToInts(parameters.GetInts("axes"));
        }

        public override void GetMemoryShapes(
            IReadOnlyList<TensorShape> inShapes,
            int requiredOutputs,
            List<TensorShape> outShapes,
            List<TensorShape> tempShapes)
        {
            var input = inShapes[0];
            var raw = OpUtil.Axes(this, _axesAttr);

            var keep = new bool[input.Rank];

            if (raw.Length == 0)
            {
                // 未给 axes：压掉所有长度为 1 的维
                for (int i = 0; i < input.Rank; i++)
                    keep[i] = input[i] != 1;
            }
            else
            {
                Array.Fill(keep, true);

                foreach (var axis in OpUtil.NormalizeAxes(raw, input.Rank, $"Squeeze '{Name}'"))
                {
                    if (input[axis] != 1)
                        throw new OnnxSharpException(
                            $"Squeeze '{Name}': axis {axis} has dim {input[axis]}, cannot squeeze");
                    keep[axis] = false;
                }
            }

            var dims = new List<int>(input.Rank);
            for (int i = 0; i < input.Rank; i++)
            {
                if (keep[i])
                    dims.Add(input[i]);
            }

            outShapes.Add(new TensorShape(dims.ToArray()));
        }

        public override void Forward(
            IReadOnlyList<Tensor> inputs,
            IReadOnlyList<Tensor> outputs,
            IReadOnlyList<Tensor> temps)
            => OpUtil.CopyBytes(inputs[0], 0, outputs[0], 0, inputs[0].Count);
    }
}
