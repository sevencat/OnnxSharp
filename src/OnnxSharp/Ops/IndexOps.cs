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
        LayerRegistry.Register("Unsqueeze", p => new UnsqueezeLayer(p));
        LayerRegistry.Register("Squeeze", p => new SqueezeLayer(p));
    }

    // ---------------- Shape ----------------

    private sealed class ShapeLayer : Layer
    {
        public ShapeLayer(LayerParams parameters)
        {
            Type = parameters.Type;
            Name = parameters.Name;
        }

        /// <summary>输出是 1-D I64：[rank]。</summary>
        public override void GetMemoryShapes(
            IReadOnlyList<TensorShape> inShapes,
            int requiredOutputs,
            List<TensorShape> outShapes,
            List<TensorShape> tempShapes)
            => outShapes.Add(new TensorShape(inShapes[0].Rank));

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

        public override void Forward(
            IReadOnlyList<Tensor> inputs,
            IReadOnlyList<Tensor> outputs,
            IReadOnlyList<Tensor> temps)
        {
            var dims = inputs[0].Shape.Dims;
            var target = outputs[0].AsSpan<long>();

            for (int i = 0; i < dims.Length; i++)
                target[i] = dims[i];
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
