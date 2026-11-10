using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
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
        LayerRegistry.Register("ReduceMean", p => new ReduceLayer(p, ReduceKind.Mean));
        LayerRegistry.Register("Range", p => new RangeLayer(p));
    }

    internal enum ReduceKind
    {
        Sum,
        Max,

        /// <summary>先按 Sum 归约，最后统一除以窗口大小。</summary>
        Mean,
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

        /// <summary>
        /// 形状来自运行时（Paraformer 的 CIF 用它造零初始状态）时由 <see cref="Forward"/> 自己 Fit；
        /// 常量形状仍然走静态推断 + 常量折叠。
        /// </summary>
        public override bool DynamicOutputShapes() => !IsConstInput(0);

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

            if (DynamicOutputShapes())
                output.Fit(new TensorShape(OpUtil.ReadInts(inputs[0])), OutputDType);

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

    // ---------------- Range ----------------

    /// <summary>
    /// <c>Range(start, limit, delta)</c> → 一维等差序列，长度 = max(ceil((limit-start)/delta), 0)。
    /// 三个输入在生成位置编码时都是常量，所以长度能在形状推断阶段算出来，也能被常量折叠。
    /// </summary>
    private sealed class RangeLayer : Layer
    {
        public RangeLayer(LayerParams parameters)
        {
            Type = parameters.Type;
            Name = parameters.Name;
        }

        private static double ReadScalar(Tensor t, string what, string name) => t.DType switch
        {
            DType.F32 => t.AsSpan<float>()[0],
            DType.I64 => t.AsSpan<long>()[0],
            DType.I32 => t.AsSpan<int>()[0],
            _ => throw new OnnxSharpException($"Range '{name}': unsupported {what} dtype {t.DType}"),
        };

        /// <summary>
        /// 常量输入的 Range（位置编码）走静态形状 + 常量折叠；只有 <c>limit</c> 来自运行时
        /// （Paraformer 的 <c>make_pad_mask</c>：<c>Range(0, ReduceMax(speech_lengths))</c>）
        /// 时才转成动态输出形状，由 <see cref="Forward"/> 自己 Fit。
        /// </summary>
        public override bool DynamicOutputShapes()
            => !IsConstInput(0) || !IsConstInput(1) || !IsConstInput(2);

        private static long CountOf(double start, double limit, double delta, string name)
        {
            if (delta == 0)
                throw new OnnxSharpException($"Range '{name}': delta must not be zero");

            double n = Math.Ceiling((limit - start) / delta);
            return n <= 0 ? 0 : (long)n;
        }

        private long Count()
            => CountOf(
                ReadScalar(OpUtil.RequireConstInput(this, 0), "start", Name),
                ReadScalar(OpUtil.RequireConstInput(this, 1), "limit", Name),
                ReadScalar(OpUtil.RequireConstInput(this, 2), "delta", Name),
                Name);

        public override void GetMemoryShapes(
            IReadOnlyList<TensorShape> inShapes,
            int requiredOutputs,
            List<TensorShape> outShapes,
            List<TensorShape> tempShapes)
            => outShapes.Add(new TensorShape(new[] { (int)Count() }));

        public override void GetTypes(
            IReadOnlyList<DType> inTypes,
            int requiredOutputs,
            int requiredTemps,
            List<DType> outTypes,
            List<DType> tempTypes)
        {
            outTypes.Clear();
            for (int i = 0; i < requiredOutputs; i++)
                outTypes.Add(inTypes.Count > 0 ? inTypes[0] : DType.F32);

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

            double start = ReadScalar(inputs[0], "start", Name);
            double limit = ReadScalar(inputs[1], "limit", Name);
            double delta = ReadScalar(inputs[2], "delta", Name);

            if (DynamicOutputShapes())
            {
                int length = (int)CountOf(start, limit, delta, Name);
                output.Fit(new TensorShape(new[] { length }), inputs[0].DType);
            }

            if (output.DType == DType.I64)
            {
                var d = output.AsSpan<long>();
                for (int i = 0; i < d.Length; i++)
                    d[i] = (long)(start + i * delta);
                return;
            }

            var f = output.AsSpan<float>();
            for (int i = 0; i < f.Length; i++)
                f[i] = (float)(start + i * delta);
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
                if (_kind == ReduceKind.Mean)
                    throw new OnnxSharpException($"'ReduceMean' '{Name}': only F32 is supported");

                ReduceInt64(input.AsSpan<long>(), output.AsSpan<long>(), dims, inStrides, map);
                return;
            }

            if (input.DType != DType.F32)
                throw new OnnxSharpException($"'{Type}' '{Name}': only F32/I64 is supported");

            var src = input.AsSpan<float>();
            var dst = output.AsSpan<float>();

            // 绝大多数归约（reduce 最后几维 / 中间一段连续维）都可以拆成
            // (outer, mid, inner)，既能向量化，也不用每个元素再做 rank 次除法取坐标。
            if (TryContiguousReduction(dims, reduced, out int outer, out int mid, out int inner))
            {
                ReduceBlocks(src, dst, outer, mid, inner);
                ScaleIfMean(dst, input.Count);
                return;
            }

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

            ScaleIfMean(dst, input.Count);
        }

        /// <summary>ReduceMean = 先按 Sum 累加完再整体乘 1/N，避免内层每次都做除法。</summary>
        private void ScaleIfMean(Span<float> dst, long inputCount)
        {
            if (_kind != ReduceKind.Mean || dst.Length == 0 || inputCount == 0)
                return;

            float inv = (float)dst.Length / inputCount;
            for (int i = 0; i < dst.Length; i++)
                dst[i] *= inv;
        }

        /// <summary>
        /// 被归约的轴如果正好是连续的一段 [r0, r1]，整个归约就能拆成
        /// <c>outer × mid × inner</c> 三层（inner 是 r1 之后不动的那几维）。
        /// </summary>
        private static bool TryContiguousReduction(
            int[] dims,
            bool[] reduced,
            out int outer,
            out int mid,
            out int inner)
        {
            int r0 = -1;
            int r1 = -1;
            for (int i = 0; i < dims.Length; i++)
            {
                if (!reduced[i])
                    continue;

                if (r0 < 0)
                    r0 = i;
                r1 = i;
            }

            outer = mid = inner = 0;
            if (r0 < 0)
                return false;

            for (int i = r0; i <= r1; i++)
            {
                if (!reduced[i])
                    return false;
            }

            outer = 1;
            for (int i = 0; i < r0; i++)
                outer *= dims[i];

            mid = 1;
            for (int i = r0; i <= r1; i++)
                mid *= dims[i];

            inner = 1;
            for (int i = r1 + 1; i < dims.Length; i++)
                inner *= dims[i];

            return true;
        }

        private unsafe void ReduceBlocks(ReadOnlySpan<float> src, Span<float> dst, int outer, int mid, int inner)
        {
            bool isMax = _kind == ReduceKind.Max;
            float identity = isMax ? float.NegativeInfinity : 0f;

            for (int o = 0; o < outer; o++)
            {
                var block = src.Slice(o * mid * inner, mid * inner);
                var result = dst.Slice(o * inner, inner);

                if (mid == 1)
                {
                    block.Slice(0, inner).CopyTo(result);
                    continue;
                }

                fixed (float* pBlock = block)
                fixed (float* pResult = result)
                {
                    if (inner == 1)
                    {
                        // 归约最后一维：源连续，直接 8 路累加
                        int i = 0;
                        float acc = identity;

                        if (Avx.IsSupported && mid >= 8)
                        {
                            var v = Vector256.Create(identity);
                            for (; i <= mid - 8; i += 8)
                            {
                                var x = Avx.LoadVector256(pBlock + i);
                                v = isMax ? Avx.Max(v, x) : Avx.Add(v, x);
                            }

                            acc = isMax ? MaxLanes(v) : SumLanes(v);
                        }

                        for (; i < mid; i++)
                            acc = isMax ? MathF.Max(acc, pBlock[i]) : acc + pBlock[i];

                        pResult[0] = acc;
                        continue;
                    }

                    int j = 0;
                    if (Avx.IsSupported && inner >= 8)
                    {
                        for (; j <= inner - 8; j += 8)
                        {
                            var acc = Vector256.Create(identity);
                            for (int m = 0; m < mid; m++)
                            {
                                var x = Avx.LoadVector256(pBlock + (long)m * inner + j);
                                acc = isMax ? Avx.Max(acc, x) : Avx.Add(acc, x);
                            }

                            Avx.Store(pResult + j, acc);
                        }
                    }

                    for (; j < inner; j++)
                    {
                        float acc = identity;
                        for (int m = 0; m < mid; m++)
                        {
                            float value = pBlock[(long)m * inner + j];
                            acc = isMax ? MathF.Max(acc, value) : acc + value;
                        }

                        pResult[j] = acc;
                    }
                }
            }
        }

        private static float SumLanes(Vector256<float> v)
            => Vector128.Sum(Avx.Add(v.GetLower(), v.GetUpper()));

        private static float MaxLanes(Vector256<float> v)
        {
            var lo = Avx.Max(v.GetLower(), v.GetUpper());
            lo = Vector128.Max(lo, Vector128.Shuffle(lo, Vector128.Create(2, 3, 0, 1)));
            return MathF.Max(lo.GetElement(0), lo.GetElement(1));
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
