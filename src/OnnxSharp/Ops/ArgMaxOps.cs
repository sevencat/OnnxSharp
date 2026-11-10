using OnnxSharp.Core;
using OnnxSharp.Tensors;

namespace OnnxSharp.Ops;

/// <summary>
/// ONNX <c>ArgMax</c>（我们之前没有，decoder / CTC 贪心解码都要用）。
/// </summary>
internal static class ArgMaxOps
{
    public static void Register() => LayerRegistry.Register("ArgMax", p => new ArgMaxLayer(p));

    /// <summary>
    /// 图融合用：只有「沿最后一维、<c>keepdims = 0</c>、取首次出现」的 ArgMax 才能被
    /// <see cref="MatMulArgMaxOps"/> 吸收（那条路上不需要物化 logits）。
    /// </summary>
    internal interface ILastAxisArgMax
    {
        bool IsLastAxisNoKeep { get; }
    }

    private sealed class ArgMaxLayer : Layer, ILastAxisArgMax
    {
        private readonly long _axis;
        private readonly bool _keepDims;
        private readonly bool _selectLast;

        public ArgMaxLayer(LayerParams parameters)
        {
            Type = parameters.Type;
            Name = parameters.Name;
            _axis = parameters.GetInt("axis", 0);
            _keepDims = parameters.GetInt("keepdims", 1) != 0;
            _selectLast = parameters.GetInt("select_last_index", 0) != 0;
        }

        public bool IsLastAxisNoKeep => !_keepDims && !_selectLast && _axis is -1;
        public bool KeepDims => _keepDims;
        public long Axis => _axis;

        public override void GetMemoryShapes(
            IReadOnlyList<TensorShape> inShapes,
            int requiredOutputs,
            List<TensorShape> outShapes,
            List<TensorShape> tempShapes)
        {
            var input = inShapes[0];
            int axis = OpUtil.NormalizeAxis(_axis, input.Rank, $"ArgMax '{Name}'");
            var dims = input.ToArray();

            if (_keepDims)
            {
                dims[axis] = 1;
            }
            else
            {
                var reduced = new int[dims.Length - 1];
                Array.Copy(dims, 0, reduced, 0, axis);
                Array.Copy(dims, axis + 1, reduced, axis, dims.Length - axis - 1);
                dims = reduced;
            }

            outShapes.Add(new TensorShape(dims));
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
            var input = inputs[0];
            var target = outputs[0].AsSpan<long>();

            int rank = input.Shape.Rank;
            int axis = OpUtil.NormalizeAxis(_axis, rank, $"ArgMax '{Name}'");
            var dims = input.Shape.ToArray();

            long axisLength = dims[axis];
            long inner = 1;
            for (int i = axis + 1; i < rank; i++)
                inner *= dims[i];

            long outer = 1;
            for (int i = 0; i < axis; i++)
                outer *= dims[i];

            // 三种受支持的输入类型各自一条路径（都在 [outer, axisLength, inner] 视角下遍历）
            if (input.DType == DType.F32)
                Reduce(input.AsSpan<float>(), target, outer, axisLength, inner, _selectLast);
            else if (input.DType == DType.I32)
                Reduce(input.AsSpan<int>(), target, outer, axisLength, inner, _selectLast);
            else if (input.DType == DType.I64)
                Reduce(input.AsSpan<long>(), target, outer, axisLength, inner, _selectLast);
            else
                throw new OnnxSharpException($"ArgMax '{Name}': unsupported dtype {input.DType}");
        }

        private static void Reduce<T>(ReadOnlySpan<T> source, Span<long> target, long outer, long axisLength, long inner, bool selectLast)
            where T : struct, IComparable<T>
        {
            for (long o = 0; o < outer; o++)
            {
                for (long i = 0; i < inner; i++)
                {
                    long baseIndex = o * axisLength * inner + i;
                    T best = source[(int)baseIndex];
                    long bestIndex = 0;

                    for (long a = 1; a < axisLength; a++)
                    {
                        T value = source[(int)(baseIndex + a * inner)];
                        int comparison = value.CompareTo(best);

                        // select_last_index = 1 时相等也更新，取最后一次出现
                        if (comparison > 0 || (selectLast && comparison == 0))
                        {
                            best = value;
                            bestIndex = a;
                        }
                    }

                    target[(int)(o * inner + i)] = bestIndex;
                }
            }
        }
    }
}
