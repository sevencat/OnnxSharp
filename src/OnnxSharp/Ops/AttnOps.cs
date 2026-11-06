using OnnxSharp.Core;
using OnnxSharp.Tensors;

namespace OnnxSharp.Ops;

/// <summary>
/// 检测 / Transformer 常用算子：Gemm / LayerNormalization / GridSample / TopK。
/// <para>
/// 对应 OpenCV 的 <c>gemm_layer.cpp</c>、<c>layer_norm_layer.cpp</c>、
/// <c>grid_sample_layer.cpp</c>、<c>topk_layer.cpp</c>。
/// </para>
/// </summary>
internal static class AttnOps
{
    public static void Register()
    {
        LayerRegistry.Register("Gemm", p => new GemmLayer(p));
        LayerRegistry.Register("LayerNormalization", p => new LayerNormLayer(p));
        LayerRegistry.Register("GridSample", p => new GridSampleLayer(p));
        LayerRegistry.Register("TopK", p => new TopKLayer(p));
    }

    // ---------------- Gemm ----------------

    private sealed class GemmLayer : Layer
    {
        private readonly float _alpha;
        private readonly float _beta;
        private readonly bool _transA;
        private readonly bool _transB;

        public GemmLayer(LayerParams parameters)
        {
            Type = parameters.Type;
            Name = parameters.Name;
            _alpha = (float)parameters.GetFloat("alpha", 1.0);
            _beta = (float)parameters.GetFloat("beta", 1.0);
            _transA = parameters.GetInt("transA", 0) != 0;
            _transB = parameters.GetInt("transB", 0) != 0;
        }

        private (int m, int k, int n) Resolve(IReadOnlyList<TensorShape> inShapes)
        {
            var a = inShapes[0];
            var b = inShapes[1];

            if (a.Rank != 2 || b.Rank != 2)
                throw new OnnxSharpException($"Gemm '{Name}': only 2-D A/B are supported");

            int m = _transA ? a[1] : a[0];
            int k = _transA ? a[0] : a[1];
            int n = _transB ? b[0] : b[1];

            int bK = _transB ? b[1] : b[0];
            if (bK != k)
                throw new OnnxSharpException(
                    $"Gemm '{Name}': K mismatch (A {( _transA ? "T" : "N")}{a} / B {( _transB ? "T" : "N")}{b})");

            return (m, k, n);
        }

        public override void GetMemoryShapes(
            IReadOnlyList<TensorShape> inShapes,
            int requiredOutputs,
            List<TensorShape> outShapes,
            List<TensorShape> tempShapes)
        {
            var (m, _, n) = Resolve(inShapes);
            outShapes.Add(new TensorShape(m, n));
        }

        public override void Forward(
            IReadOnlyList<Tensor> inputs,
            IReadOnlyList<Tensor> outputs,
            IReadOnlyList<Tensor> temps)
        {
            var a = inputs[0];
            var b = inputs[1];
            var output = outputs[0];

            Tensor? c = Inputs.Count > 2 && Inputs[2].Idx != 0 && inputs[2].Count > 0 ? inputs[2] : null;

            var shapes = new List<TensorShape>(2) { a.Shape, b.Shape };
            var (m, k, n) = Resolve(shapes);

            var sa = a.AsSpan<float>();
            var sb = b.AsSpan<float>();
            var dst = output.AsSpan<float>();
            var sc = c is null ? Span<float>.Empty : c.AsSpan<float>();

            for (int i = 0; i < m; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    float sum = 0;

                    for (int t = 0; t < k; t++)
                    {
                        float av = _transA ? sa[t * m + i] : sa[i * k + t];
                        float bv = _transB ? sb[j * k + t] : sb[t * n + j];
                        sum += av * bv;
                    }

                    float value = _alpha * sum;

                    if (c is not null && sc.Length > 0)
                    {
                        // C 可以是 [M,N]、[1,N] 或 [N]（甚至标量）
                        float cv = sc.Length == 1 ? sc[0]
                            : sc.Length >= (i + 1) * n ? sc[i * n + j]
                            : sc[j % sc.Length];
                        value += _beta * cv;
                    }

                    dst[i * n + j] = value;
                }
            }
        }
    }

    // ---------------- LayerNormalization ----------------

    private sealed class LayerNormLayer : Layer
    {
        private readonly long _axis;
        private readonly float _epsilon;

        public LayerNormLayer(LayerParams parameters)
        {
            Type = parameters.Type;
            Name = parameters.Name;
            _axis = parameters.GetInt("axis", -1);
            _epsilon = (float)parameters.GetFloat("epsilon", 1e-5);
        }

        private (int outer, int inner) Resolve(TensorShape input)
        {
            int axis = OpUtil.NormalizeAxis(_axis, input.Rank, $"LayerNormalization '{Name}'");

            int outer = 1;
            for (int i = 0; i < axis; i++)
                outer *= input[i];

            int inner = 1;
            for (int i = axis; i < input.Rank; i++)
                inner *= input[i];

            return (outer, inner);
        }

        public override void GetMemoryShapes(
            IReadOnlyList<TensorShape> inShapes,
            int requiredOutputs,
            List<TensorShape> outShapes,
            List<TensorShape> tempShapes)
        {
            var input = inShapes[0];
            var (outer, _) = Resolve(input);

            outShapes.Add(input);

            // 可选的 mean / inv_std_dev 输出：形状 = 去掉归一化轴后的部分
            for (int i = 1; i < requiredOutputs; i++)
                outShapes.Add(new TensorShape(outer));
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
                outTypes.Add(DType.F32);

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
            var output = outputs[0];

            if (x.DType != DType.F32)
                throw new OnnxSharpException($"LayerNormalization '{Name}': only F32 is supported");

            var (outer, inner) = Resolve(x.Shape);

            var scale = Inputs.Count > 1 && Inputs[1].Idx != 0 && inputs[1].Count > 0
                ? inputs[1].AsSpan<float>()
                : Span<float>.Empty;

            var bias = Inputs.Count > 2 && Inputs[2].Idx != 0 && inputs[2].Count > 0
                ? inputs[2].AsSpan<float>()
                : Span<float>.Empty;

            var sx = x.AsSpan<float>();
            var dst = output.AsSpan<float>();

            for (int o = 0; o < outer; o++)
            {
                int baseIndex = o * inner;

                double sum = 0;
                for (int i = 0; i < inner; i++)
                    sum += sx[baseIndex + i];

                float mean = (float)(sum / inner);

                double variance = 0;
                for (int i = 0; i < inner; i++)
                {
                    double d = sx[baseIndex + i] - mean;
                    variance += d * d;
                }

                float invStd = 1f / MathF.Sqrt((float)(variance / inner) + _epsilon);

                for (int i = 0; i < inner; i++)
                {
                    float value = (sx[baseIndex + i] - mean) * invStd;
                    if (scale.Length > 0)
                        value *= scale[i % scale.Length];
                    if (bias.Length > 0)
                        value += bias[i % bias.Length];
                    dst[baseIndex + i] = value;
                }
            }
        }
    }

    // ---------------- GridSample ----------------

    private sealed class GridSampleLayer : Layer
    {
        private readonly bool _alignCorners;
        private readonly bool _nearest;
        private readonly bool _border;

        public GridSampleLayer(LayerParams parameters)
        {
            Type = parameters.Type;
            Name = parameters.Name;
            _alignCorners = parameters.GetInt("align_corners", 0) != 0;
            _nearest = parameters.GetString("mode", "linear") == "nearest";
            _border = parameters.GetString("padding_mode", "zeros") == "border";
        }

        public override void GetMemoryShapes(
            IReadOnlyList<TensorShape> inShapes,
            int requiredOutputs,
            List<TensorShape> outShapes,
            List<TensorShape> tempShapes)
        {
            var x = inShapes[0];
            var grid = inShapes[1];

            if (x.Rank != 4 || grid.Rank != 4)
                throw new OnnxSharpException($"GridSample '{Name}': only 2-D NCHW input is supported");

            outShapes.Add(new TensorShape(x[0], x[1], grid[1], grid[2]));
        }

        /// <summary>把 [-1,1] 的归一化坐标映射到像素坐标（与 PyTorch 一致）。</summary>
        private float Unnormalize(float value, int size)
        {
            if (_alignCorners)
                return (value + 1f) * 0.5f * (size - 1);

            float result = ((value + 1f) * size - 1f) * 0.5f;
            return _border ? Math.Clamp(result, 0f, size - 1) : result;
        }

        private float Fetch(ReadOnlySpan<float> source, int channelBase, int height, int width, int y, int x)
        {
            if (_border)
            {
                y = Math.Clamp(y, 0, height - 1);
                x = Math.Clamp(x, 0, width - 1);
            }
            else if ((uint)y >= (uint)height || (uint)x >= (uint)width)
            {
                return 0f;   // padding_mode = zeros
            }

            return source[channelBase + y * width + x];
        }

        public override void Forward(
            IReadOnlyList<Tensor> inputs,
            IReadOnlyList<Tensor> outputs,
            IReadOnlyList<Tensor> temps)
        {
            var x = inputs[0];
            var grid = inputs[1];
            var output = outputs[0];

            int batches = x.Shape[0];
            int channels = x.Shape[1];
            int height = x.Shape[2];
            int width = x.Shape[3];

            int outH = grid.Shape[1];
            int outW = grid.Shape[2];

            var sx = x.AsSpan<float>();
            var sg = grid.AsSpan<float>();
            var dst = output.AsSpan<float>();

            for (int n = 0; n < batches; n++)
            {
                for (int oy = 0; oy < outH; oy++)
                {
                    for (int ox = 0; ox < outW; ox++)
                    {
                        int gridIndex = ((n * outH + oy) * outW + ox) * 2;
                        float gx = Unnormalize(sg[gridIndex], width);
                        float gy = Unnormalize(sg[gridIndex + 1], height);

                        for (int c = 0; c < channels; c++)
                        {
                            int channelBase = (n * channels + c) * height * width;
                            float value;

                            if (_nearest)
                            {
                                int ix = (int)MathF.Round(gx);
                                int iy = (int)MathF.Round(gy);
                                value = Fetch(sx, channelBase, height, width, iy, ix);
                            }
                            else
                            {
                                int x0 = (int)MathF.Floor(gx);
                                int y0 = (int)MathF.Floor(gy);
                                float fx = gx - x0;
                                float fy = gy - y0;

                                float v00 = Fetch(sx, channelBase, height, width, y0, x0);
                                float v01 = Fetch(sx, channelBase, height, width, y0, x0 + 1);
                                float v10 = Fetch(sx, channelBase, height, width, y0 + 1, x0);
                                float v11 = Fetch(sx, channelBase, height, width, y0 + 1, x0 + 1);

                                float top = v00 + (v01 - v00) * fx;
                                float bottom = v10 + (v11 - v10) * fx;
                                value = top + (bottom - top) * fy;
                            }

                            dst[((n * channels + c) * outH + oy) * outW + ox] = value;
                        }
                    }
                }
            }
        }
    }

    // ---------------- TopK ----------------

    private sealed class TopKLayer : Layer
    {
        private readonly long _axis;
        private readonly bool _largest;

        public TopKLayer(LayerParams parameters)
        {
            Type = parameters.Type;
            Name = parameters.Name;
            _axis = parameters.GetInt("axis", -1);
            _largest = parameters.GetInt("largest", 1) != 0;
        }

        private int ResolveK()
        {
            // opset 10+：K 在第 2 个输入；opset 1：k 属性
            if (Inputs.Count > 1 && Inputs[1].Idx != 0)
                return OpUtil.ReadInts(OpUtil.RequireConstInput(this, 1))[0];
            return 0;
        }

        public override void GetMemoryShapes(
            IReadOnlyList<TensorShape> inShapes,
            int requiredOutputs,
            List<TensorShape> outShapes,
            List<TensorShape> tempShapes)
        {
            var input = inShapes[0];
            int axis = OpUtil.NormalizeAxis(_axis, input.Rank, $"TopK '{Name}'");
            int k = ResolveK();

            if (k <= 0)
                throw new OnnxSharpException($"TopK '{Name}': k must be positive (got {k})");

            var dims = input.ToArray();
            dims[axis] = k;

            for (int i = 0; i < requiredOutputs; i++)
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
            if (requiredOutputs > 0)
                outTypes.Add(inTypes[0]);
            for (int i = 1; i < requiredOutputs; i++)
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
            var values = outputs[0];
            var indices = outputs.Count > 1 ? outputs[1] : null;

            var dims = input.Shape.ToArray();
            int axis = OpUtil.NormalizeAxis(_axis, dims.Length, $"TopK '{Name}'");
            int axisDim = dims[axis];
            int k = values.Shape[axis];

            long outer = 1;
            for (int i = 0; i < axis; i++)
                outer *= dims[i];

            long inner = 1;
            for (int i = axis + 1; i < dims.Length; i++)
                inner *= dims[i];

            var src = input.AsSpan<float>();
            var dst = values.AsSpan<float>();
            var idx = indices is null ? Span<long>.Empty : indices.AsSpan<long>();

            var buffer = new (float value, int index)[axisDim];

            for (long o = 0; o < outer; o++)
            {
                for (long i = 0; i < inner; i++)
                {
                    for (int t = 0; t < axisDim; t++)
                        buffer[t] = (src[(int)((o * axisDim + t) * inner + i)], t);

                    Array.Sort(buffer, (x, y) => _largest
                        ? y.value.CompareTo(x.value)
                        : x.value.CompareTo(y.value));

                    for (int t = 0; t < k; t++)
                    {
                        long target = (o * k + t) * inner + i;
                        dst[(int)target] = buffer[t].value;
                        if (idx.Length > 0)
                            idx[(int)target] = buffer[t].index;
                    }
                }
            }
        }
    }
}
