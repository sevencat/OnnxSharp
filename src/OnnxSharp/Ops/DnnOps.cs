using System.Numerics;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using OnnxSharp.Core;
using OnnxSharp.Tensors;
using OnnxSharp.Threading;

namespace OnnxSharp.Ops;

/// <summary>
/// 计算密集型算子：Conv / MaxPool / Resize / Softmax / MatMul。
/// M1 阶段先保证 <see cref="LayerInfo.GetMemoryShapes"/> 与 onnx.shape_inference 一致；
/// Conv 的 kernel 在 M2 落地（见 plan.md）。
/// </summary>
internal static class DnnOps
{
    public static void Register()
    {
        LayerRegistry.Register("Conv", p => new ConvLayer(p));
        LayerRegistry.Register("MaxPool", p => new MaxPoolLayer(p));
        LayerRegistry.Register("Resize", p => new ResizeLayer(p));
        LayerRegistry.Register("Softmax", p => new SoftmaxLayer(p));
        LayerRegistry.Register("MatMul", p => new MatMulLayer(p));
    }

    private static int[] ToInts(long[]? values)
        => values is null || values.Length == 0 ? Array.Empty<int>() : Array.ConvertAll(values, v => checked((int)v));

    /// <summary>解析 auto_pad / pads / kernel / stride / dilation，返回输出尺寸。</summary>
    private static int SpatialOut(
        int input,
        int kernel,
        int stride,
        int dilation,
        int padBegin,
        int padEnd,
        bool ceilMode)
    {
        int effective = (kernel - 1) * dilation + 1;
        int numerator = input + padBegin + padEnd - effective;
        if (ceilMode)
            return (int)Math.Ceiling((double)numerator / stride) + 1;
        return numerator / stride + 1;
    }

    private static (int Begin, int End) AutoPad(string mode, int input, int kernel, int stride, int dilation, int index)
    {
        int effective = (kernel - 1) * dilation + 1;
        int output = (int)Math.Ceiling((double)input / stride);

        switch (mode)
        {
            case "SAME_UPPER":
            case "SAME_LOWER":
                {
                    int total = Math.Max(0, (output - 1) * stride + effective - input);
                    int begin = mode == "SAME_UPPER" ? total / 2 : total - total / 2;
                    // index 0/1 只影响“总数是否已按轴拆分”；这里每次调用只算一根轴
                    return index < 0 ? (begin, total - begin) : (begin, total - begin);
                }
            case "VALID":
                return (0, 0);
            default:
                return (-1, -1); // 调用方改用 pads
        }
    }

    // ---------------- Conv ----------------

    private sealed class ConvLayer : Layer
    {
        private readonly int _group;
        private readonly int[] _kernelShape;
        private readonly int[] _strides;
        private readonly int[] _pads;
        private readonly int[] _dilations;
        private readonly string _autoPad;

        public ConvLayer(LayerParams parameters)
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

        public override void GetMemoryShapes(
            IReadOnlyList<TensorShape> inShapes,
            int requiredOutputs,
            List<TensorShape> outShapes,
            List<TensorShape> tempShapes)
        {
            if (inShapes.Count < 2)
                throw new OnnxSharpException($"Conv '{Name}': needs data and weights");

            var input = inShapes[0];
            var weight = inShapes[1];   // [M, C/group, kH, kW]
            int spatial = input.Rank - 2;

            if (weight.Rank != input.Rank)
                throw new OnnxSharpException($"Conv '{Name}': weight rank {weight.Rank} != input rank {input.Rank}");

            int channels = input.Dims[1];
            if (_group <= 0 || channels % _group != 0)
                throw new OnnxSharpException($"Conv '{Name}': channels {channels} not divisible by group {_group}");
            if (weight.Dims[1] != channels / _group)
                throw new OnnxSharpException(
                    $"Conv '{Name}': weight in-channels {weight.Dims[1]} != {channels / _group}");

            var dims = new int[input.Rank];
            dims[0] = input.Dims[0];
            dims[1] = weight.Dims[0];

            for (int i = 0; i < spatial; i++)
            {
                int kernel = _kernelShape.Length > i ? _kernelShape[i] : weight.Dims[2 + i];
                int stride = _strides.Length > i ? _strides[i] : 1;
                int dilation = _dilations.Length > i ? _dilations[i] : 1;

                int padBegin = _pads.Length > i ? _pads[i] : 0;
                int padEnd = _pads.Length > 2 * spatial ? _pads[spatial + i] : padBegin;

                if (_autoPad is "SAME_UPPER" or "SAME_LOWER" or "VALID")
                    (padBegin, padEnd) = AutoPad(_autoPad, input.Dims[2 + i], kernel, stride, dilation, -1);

                dims[2 + i] = SpatialOut(input.Dims[2 + i], kernel, stride, dilation, padBegin, padEnd, false);
            }

            outShapes.Add(new TensorShape(dims));
        }

        /// <summary>解析 stride / dilation / pads（含 auto_pad），与 <see cref="GetMemoryShapes"/> 同源。</summary>
        private void Resolve(
            TensorShape input,
            out int[] strides,
            out int[] dilations,
            out int[] pads)
        {
            int spatial = input.Rank - 2;
            strides = new int[spatial];
            dilations = new int[spatial];
            pads = new int[spatial * 2];

            for (int i = 0; i < spatial; i++)
            {
                int kernel = _kernelShape.Length > i ? _kernelShape[i] : 1;
                strides[i] = _strides.Length > i ? _strides[i] : 1;
                dilations[i] = _dilations.Length > i ? _dilations[i] : 1;

                if (_autoPad is "SAME_UPPER" or "SAME_LOWER" or "VALID")
                {
                    var (begin, end) = AutoPad(_autoPad, input.Dims[2 + i], kernel, strides[i], dilations[i], -1);
                    pads[i] = begin;
                    pads[spatial + i] = end;
                }
                else
                {
                    pads[i] = _pads.Length > i ? _pads[i] : 0;
                    pads[spatial + i] = _pads.Length > spatial + i ? _pads[spatial + i] : pads[i];
                }
            }
        }

        public override void Forward(
            IReadOnlyList<Tensor> inputs,
            IReadOnlyList<Tensor> outputs,
            IReadOnlyList<Tensor> temps)
        {
            var input = inputs[0];
            var weight = inputs[1];
            var output = outputs[0];

            if (input.Shape.Rank != 4)
                throw new OnnxSharpException($"Conv '{Name}': only 2D convolution (NCHW) is implemented");

            Resolve(input.Shape, out var strides, out var dilations, out var pads);

            Tensor? bias = Inputs.Count > 2 && Inputs[2].Idx != 0 && inputs[2].Count > 0 ? inputs[2] : null;

            // 3×3 / s1 / pad1 优先走 Winograd F(2,3)（乘法量只有 4/9），不适用再回退 im2col
            if (!Kernels.Winograd.TryConv3x3(
                    input, weight, bias, output,
                    _group, strides[0], strides[1], dilations[0], dilations[1], pads[0], pads[1]))
            {
                Kernels.ConvKernels.Im2Col(
                    input, weight, bias, output,
                    _group, strides[0], strides[1], dilations[0], dilations[1], pads[0], pads[1]);
            }
        }
    }

    // ---------------- MaxPool ----------------

    private sealed class MaxPoolLayer : Layer
    {
        private readonly int[] _kernelShape;
        private readonly int[] _strides;
        private readonly int[] _pads;
        private readonly int[] _dilations;
        private readonly string _autoPad;
        private readonly bool _ceilMode;

        public MaxPoolLayer(LayerParams parameters)
        {
            Type = parameters.Type;
            Name = parameters.Name;
            _kernelShape = ToInts(parameters.GetInts("kernel_shape"));
            _strides = ToInts(parameters.GetInts("strides"));
            _pads = ToInts(parameters.GetInts("pads"));
            _dilations = ToInts(parameters.GetInts("dilations"));
            _autoPad = parameters.GetString("auto_pad", "NOTSET");
            _ceilMode = parameters.GetInt("ceil_mode", 0) != 0;
        }

        private (int[] Pads, int[] Strides, int[] Kernel, int[] Dilations) Resolve(int spatial, TensorShape input)
        {
            var kernel = new int[spatial];
            var strides = new int[spatial];
            var pads = new int[spatial * 2];
            var dilations = new int[spatial];

            if (_kernelShape.Length != spatial)
                throw new OnnxSharpException($"MaxPool '{Name}': kernel_shape must have {spatial} entries");

            for (int i = 0; i < spatial; i++)
            {
                kernel[i] = _kernelShape[i];
                strides[i] = _strides.Length > i ? _strides[i] : 1;
                dilations[i] = _dilations.Length > i ? _dilations[i] : 1;
            }

            for (int i = 0; i < spatial; i++)
            {
                if (_autoPad is "SAME_UPPER" or "SAME_LOWER" or "VALID")
                {
                    var (begin, end) = AutoPad(_autoPad, input.Dims[2 + i], kernel[i], strides[i], dilations[i], -1);
                    pads[i] = begin;
                    pads[spatial + i] = end;
                }
                else
                {
                    pads[i] = _pads.Length > i ? _pads[i] : 0;
                    pads[spatial + i] = _pads.Length > spatial + i ? _pads[spatial + i] : pads[i];
                }
            }

            return (pads, strides, kernel, dilations);
        }

        public override void GetMemoryShapes(
            IReadOnlyList<TensorShape> inShapes,
            int requiredOutputs,
            List<TensorShape> outShapes,
            List<TensorShape> tempShapes)
        {
            var input = inShapes[0];
            int spatial = input.Rank - 2;
            var (pads, strides, kernel, dilations) = Resolve(spatial, input);

            var dims = input.ToArray();
            for (int i = 0; i < spatial; i++)
                dims[2 + i] = SpatialOut(input.Dims[2 + i], kernel[i], strides[i], dilations[i],
                    pads[i], pads[spatial + i], _ceilMode);

            outShapes.Add(new TensorShape(dims));

            // MaxPool 的第二个输出是 Indices（yolo 不用），形状与第一个一致
            if (requiredOutputs > 1)
                outShapes.Add(new TensorShape(dims));
        }

        public override void Forward(
            IReadOnlyList<Tensor> inputs,
            IReadOnlyList<Tensor> outputs,
            IReadOnlyList<Tensor> temps)
        {
            var input = inputs[0];
            var output = outputs[0];
            int spatial = input.Shape.Rank - 2;
            var (pads, strides, kernel, dilations) = Resolve(spatial, input.Shape);

            if (spatial != 2)
                throw new OnnxSharpException($"MaxPool '{Name}': only 2D pooling is implemented");

            int batches = input.Shape.Dims[0];
            int channels = input.Shape.Dims[1];
            int inH = input.Shape.Dims[2];
            int inW = input.Shape.Dims[3];
            int outH = output.Shape.Dims[2];
            int outW = output.Shape.Dims[3];

            int kH = kernel[0], kW = kernel[1];
            int sH = strides[0], sW = strides[1];
            int dH = dilations[0], dW = dilations[1];
            int pT = pads[0], pL = pads[1];

            // 按 (batch, channel) 并行：SPPF 的三个 5×5 池化原来是整段串行，
            // 内层还要对每个 tap 做一次边界判断。
            ParallelRange.For(batches * channels, (begin, end) =>
            {
                var s = input.AsSpan<float>();
                var d = output.AsSpan<float>();

                for (int t = begin; t < end; t++)
                {
                    long inBase = (long)t * inH * inW;
                    long outBase = (long)t * outH * outW;

                    for (int oy = 0; oy < outH; oy++)
                    {
                        int y0 = oy * sH - pT;

                        for (int ox = 0; ox < outW; ox++)
                        {
                            int x0 = ox * sW - pL;

                            // 把左右越界的 tap 直接裁掉，内层循环里就没有分支了
                            int kx0 = x0 >= 0 ? 0 : (-x0 + dW - 1) / dW;
                            int limit = inW - x0;
                            int kx1 = limit <= 0 ? 0 : Math.Min(kW, (limit + dW - 1) / dW);
                            if (kx1 < kx0)
                                kx1 = kx0;

                            float best = float.NegativeInfinity;
                            for (int ky = 0; ky < kH; ky++)
                            {
                                int y = y0 + ky * dH;
                                if ((uint)y >= (uint)inH)
                                    continue;

                                long rowBase = inBase + (long)y * inW;
                                for (int kx = kx0; kx < kx1; kx++)
                                {
                                    float v = s[(int)(rowBase + x0 + kx * dW)];
                                    if (v > best)
                                        best = v;
                                }
                            }

                            d[(int)(outBase + (long)oy * outW + ox)] = best;
                        }
                    }
                }
            });
        }
    }

    // ---------------- Resize ----------------

    private sealed class ResizeLayer : Layer
    {
        private readonly string _mode;
        private readonly string _coordinateMode;
        private readonly string _nearestMode;
        private readonly double[] _scalesAttr;

        public ResizeLayer(LayerParams parameters)
        {
            Type = parameters.Type;
            Name = parameters.Name;
            _mode = parameters.GetString("mode", "nearest");
            _coordinateMode = parameters.GetString("coordinate_transformation_mode", "half_pixel");
            _nearestMode = parameters.GetString("nearest_mode", "round_prefer_floor");
            _scalesAttr = parameters.GetFloats("scales") ?? Array.Empty<double>();
        }

        private (int[] Dims, double[] Scales) Resolve(TensorShape input)
        {
            // 输入：X, roi(可选), scales(可选), sizes(可选)
            if (Inputs.Count > 3 && Inputs[3].Idx != 0 && IsConstInput(3))
            {
                var sizes = OpUtil.ReadInts(In(3));
                var dims = new int[input.Rank];
                for (int i = 0; i < dims.Length; i++)
                    dims[i] = i < sizes.Length ? sizes[i] : input.Dims[i];

                var scales = new double[dims.Length];
                for (int i = 0; i < dims.Length; i++)
                    scales[i] = (double)dims[i] / input.Dims[i];
                return (dims, scales);
            }

            double[] scaleValues;
            if (Inputs.Count > 2 && Inputs[2].Idx != 0 && IsConstInput(2))
                scaleValues = Array.ConvertAll(OpUtil.ReadFloats(In(2)), v => (double)v);
            else
                scaleValues = _scalesAttr;

            if (scaleValues.Length == 0)
            {
                var inputs = new List<string>(Inputs.Count);
                for (int i = 0; i < Inputs.Count; i++)
                {
                    inputs.Add(Inputs[i].Idx == 0
                        ? "#<empty>"
                        : $"#{Net?.GetArgName(Inputs[i])}({(IsConstInput(i) ? "const" : "runtime")})");
                }
                throw new OnnxSharpException(
                    $"Resize '{Name}': neither scales nor sizes provided (inputs: {string.Join(", ", inputs)})");
            }

            var outDims = new int[input.Rank];
            for (int i = 0; i < outDims.Length; i++)
            {
                double scale = i < scaleValues.Length ? scaleValues[i] : 1;
                outDims[i] = (int)Math.Floor(input.Dims[i] * scale);
            }

            return (outDims, scaleValues);
        }

        public override void GetMemoryShapes(
            IReadOnlyList<TensorShape> inShapes,
            int requiredOutputs,
            List<TensorShape> outShapes,
            List<TensorShape> tempShapes)
        {
            var (dims, _) = Resolve(inShapes[0]);
            outShapes.Add(new TensorShape(dims));
        }

        private double SourceCoordinate(int dst, int inSize, int outSize)
        {
            double scale = (double)outSize / inSize;
            return _coordinateMode switch
            {
                "align_corners" => outSize > 1 ? dst * (double)(inSize - 1) / (outSize - 1) : 0,
                "asymmetric" => dst / scale,
                "pytorch_half_pixel" => outSize > 1 ? (dst + 0.5) / scale - 0.5 : 0,
                "half_pixel" => (dst + 0.5) / scale - 0.5,
                _ => dst / scale,
            };
        }

        private int NearestIndex(double coordinate, int size)
        {
            double value = _nearestMode switch
            {
                "floor" => Math.Floor(coordinate),
                "ceil" => Math.Ceiling(coordinate),
                "round_prefer_ceil" => Math.Round(coordinate, MidpointRounding.AwayFromZero),
                _ => Math.Round(coordinate, MidpointRounding.ToNegativeInfinity),
            };
            return Math.Clamp((int)value, 0, size - 1);
        }

        public override void Forward(
            IReadOnlyList<Tensor> inputs,
            IReadOnlyList<Tensor> outputs,
            IReadOnlyList<Tensor> temps)
        {
            var input = inputs[0];
            var output = outputs[0];

            if (input.Shape.Rank != 4)
                throw new OnnxSharpException($"Resize '{Name}': only NCHW (rank 4) is implemented");

            int batches = input.Shape.Dims[0];
            int channels = input.Shape.Dims[1];
            int inH = input.Shape.Dims[2];
            int inW = input.Shape.Dims[3];
            int outH = output.Shape.Dims[2];
            int outW = output.Shape.Dims[3];

            bool linear = _mode != "nearest";

            // 映射表只依赖输出坐标，先算一遍：原来每个输出像素都要做两次 double 除法
            // 加 Math.Round（YOLO 的两个 2× 上采样共 40 万个输出点），这一步就是 7.8 ms。
            var mapY = new int[outH];
            var mapX = new int[outW];
            var wyTab = linear ? new double[outH] : null;
            var wxTab = linear ? new double[outW] : null;

            for (int oy = 0; oy < outH; oy++)
            {
                double sy = SourceCoordinate(oy, inH, outH);
                mapY[oy] = linear
                    ? Math.Clamp((int)Math.Floor(Math.Clamp(sy, 0, inH - 1)), 0, inH - 1)
                    : NearestIndex(sy, inH);
                if (linear)
                    wyTab![oy] = Math.Clamp(sy, 0, inH - 1) - mapY[oy];
            }

            for (int ox = 0; ox < outW; ox++)
            {
                double sx = SourceCoordinate(ox, inW, outW);
                mapX[ox] = linear
                    ? Math.Clamp((int)Math.Floor(Math.Clamp(sx, 0, inW - 1)), 0, inW - 1)
                    : NearestIndex(sx, inW);
                if (linear)
                    wxTab![ox] = Math.Clamp(sx, 0, inW - 1) - mapX[ox];
            }

            ParallelRange.For(batches * channels, (begin, end) =>
            {
                var s = input.AsSpan<float>();
                var d = output.AsSpan<float>();

                for (int t = begin; t < end; t++)
                {
                    long inBase = (long)t * inH * inW;
                    long outBase = (long)t * outH * outW;

                    if (!linear)
                    {
                        for (int oy = 0; oy < outH; oy++)
                        {
                            long rowIn = inBase + (long)mapY[oy] * inW;
                            long rowOut = outBase + (long)oy * outW;

                            for (int ox = 0; ox < outW; ox++)
                                d[(int)(rowOut + ox)] = s[(int)(rowIn + mapX[ox])];
                        }

                        continue;
                    }

                    for (int oy = 0; oy < outH; oy++)
                    {
                        int y0 = mapY[oy];
                        int y1 = Math.Min(y0 + 1, inH - 1);
                        double wy = wyTab![oy];
                        double iwy = 1 - wy;
                        long topBase = inBase + (long)y0 * inW;
                        long bottomBase = inBase + (long)y1 * inW;
                        long rowOut = outBase + (long)oy * outW;

                        for (int ox = 0; ox < outW; ox++)
                        {
                            int x0 = mapX[ox];
                            int x1 = Math.Min(x0 + 1, inW - 1);
                            double wx = wxTab![ox];
                            double iwx = 1 - wx;

                            double top = s[(int)(topBase + x0)] * iwx + s[(int)(topBase + x1)] * wx;
                            double bottom = s[(int)(bottomBase + x0)] * iwx + s[(int)(bottomBase + x1)] * wx;

                            d[(int)(rowOut + ox)] = (float)(top * iwy + bottom * wy);
                        }
                    }
                }
            });
        }
    }

    // ---------------- Softmax ----------------

    private sealed class SoftmaxLayer : Layer
    {
        private readonly long _axis;

        public SoftmaxLayer(LayerParams parameters)
        {
            Type = parameters.Type;
            Name = parameters.Name;
            _axis = parameters.GetInt("axis", -1);
        }

        public override void GetMemoryShapes(
            IReadOnlyList<TensorShape> inShapes,
            int requiredOutputs,
            List<TensorShape> outShapes,
            List<TensorShape> tempShapes)
        {
            OpUtil.NormalizeAxis(_axis, inShapes[0].Rank, $"Softmax '{Name}'");
            outShapes.Add(inShapes[0]);
        }

        public override void Forward(
            IReadOnlyList<Tensor> inputs,
            IReadOnlyList<Tensor> outputs,
            IReadOnlyList<Tensor> temps)
        {
            var input = inputs[0];
            var output = outputs[0];
            int axis = OpUtil.NormalizeAxis(_axis, input.Shape.Rank, $"Softmax '{Name}'");

            var dims = input.Shape.ToArray();
            int outer = OpUtil.Product(dims.AsSpan(0, axis));
            int inner = axis + 1 < dims.Length ? OpUtil.Product(dims.AsSpan(axis + 1)) : 1;
            int length = dims[axis];

            if (outer <= 0 || length <= 0)
                return;

            // 按 outer（attention 里就是 batch*heads*rows）并行；每行内部保持顺序累加，数值不变。
            ParallelRange.For(outer, (begin, end) =>
            {
                var src = input.AsSpan<float>();
                var dst = output.AsSpan<float>();

                for (int o = begin; o < end; o++)
                {
                    if (inner == 1)
                    {
                        SoftmaxVector(src.Slice(o * length, length), dst.Slice(o * length, length));
                        continue;
                    }

                    for (int i = 0; i < inner; i++)
                    {
                        long baseIndex = (long)o * length * inner + i;

                        float max = float.NegativeInfinity;
                        for (int k = 0; k < length; k++)
                            max = MathF.Max(max, src[(int)(baseIndex + (long)k * inner)]);

                        float sum = 0;
                        for (int k = 0; k < length; k++)
                            sum += MathF.Exp(src[(int)(baseIndex + (long)k * inner)] - max);

                        float inverse = 1f / sum;
                        for (int k = 0; k < length; k++)
                            dst[(int)(baseIndex + (long)k * inner)] =
                                MathF.Exp(src[(int)(baseIndex + (long)k * inner)] - max) * inverse;
                    }
                }
            });
        }

        /// <summary>
        /// 末轴 Softmax（连续内存）：exp 只算一次并直接写入输出，再乘倒数归一化。
        /// 原实现算了两遍 exp（求和一次、回填一次），是 attention 里的主要开销之一。
        /// </summary>
        private static void SoftmaxVector(ReadOnlySpan<float> src, Span<float> dst)
        {
            float max = MaxOf(src);

            // exp 与求和合并成一遍（向量版 exp），再乘倒数归一化
            float sum = VectorMath.SumExp(src, dst, max);
            VectorMath.Scale(dst, dst, 1f / sum);
        }

        private static unsafe float MaxOf(ReadOnlySpan<float> values)
        {
            int n = values.Length;
            int i = 0;
            float max = float.NegativeInfinity;

            // DFL 的 Softmax 是 33600 组 16 元素，每组都调这个函数：
            // 用 256 位一次吃 8 个，横向归约也压到 3 条指令。
            if (Avx.IsSupported && n >= 8)
            {
                fixed (float* p = values)
                {
                    var acc = Avx.LoadVector256(p);
                    for (i = 8; i <= n - 8; i += 8)
                        acc = Avx.Max(acc, Avx.LoadVector256(p + i));

                    var lo = Avx.Max(acc.GetLower(), acc.GetUpper());
                    lo = Vector128.Max(lo, Vector128.Shuffle(lo, Vector128.Create(2, 3, 0, 1)));
                    max = MathF.Max(lo.GetElement(0), lo.GetElement(1));
                }
            }
            else
            {
                int width = Vector<float>.Count;
                var accumulator = new Vector<float>(float.NegativeInfinity);

                for (; i <= n - width; i += width)
                    accumulator = Vector.Max(accumulator, new Vector<float>(values.Slice(i)));

                for (int lane = 0; lane < width; lane++)
                    max = MathF.Max(max, accumulator[lane]);
            }

            for (; i < n; i++)
                max = MathF.Max(max, values[i]);

            return max;
        }
    }

    // ---------------- MatMul ----------------

    private sealed class MatMulLayer : Layer
    {
        public MatMulLayer(LayerParams parameters)
        {
            Type = parameters.Type;
            Name = parameters.Name;
        }

        public override void GetMemoryShapes(
            IReadOnlyList<TensorShape> inShapes,
            int requiredOutputs,
            List<TensorShape> outShapes,
            List<TensorShape> tempShapes)
        {
            var a = inShapes[0];
            var b = inShapes[1];

            if (a.Rank < 1 || b.Rank < 1)
                throw new OnnxSharpException($"MatMul '{Name}': inputs must have rank >= 1");

            if (a.Rank == 1 && b.Rank == 1)
            {
                if (a.Dims[0] != b.Dims[0])
                    throw new OnnxSharpException($"MatMul '{Name}': {a} x {b} mismatch");
                outShapes.Add(new TensorShape(Array.Empty<int>()));
                return;
            }

            // 1D 输入按「补 1 再撤」处理，与 numpy 一致
            var aDims = a.Rank == 1 ? new[] { 1, a.Dims[0] } : a.ToArray();
            var bDims = b.Rank == 1 ? new[] { b.Dims[0], 1 } : b.ToArray();

            int k = aDims[^1];
            if (k != bDims[^2])
                throw new OnnxSharpException($"MatMul '{Name}': {a} x {b} inner dim mismatch");

            int batchRank = Math.Max(aDims.Length - 2, bDims.Length - 2);
            var aBatch = OpUtil.AlignLeft(aDims.AsSpan(0, aDims.Length - 2), batchRank);
            var bBatch = OpUtil.AlignLeft(bDims.AsSpan(0, bDims.Length - 2), batchRank);

            var dims = new int[batchRank + 2];
            for (int i = 0; i < batchRank; i++)
            {
                if (aBatch[i] != bBatch[i] && aBatch[i] != 1 && bBatch[i] != 1)
                    throw new OnnxSharpException($"MatMul '{Name}': batch dim #{i} mismatch {a} x {b}");
                dims[i] = Math.Max(aBatch[i], bBatch[i]);
            }

            dims[batchRank] = aDims[^2];
            dims[batchRank + 1] = bDims[^1];

            var shape = new TensorShape(dims);
            if (a.Rank == 1)
                shape = Squeeze(shape, batchRank);       // 去掉 M
            if (b.Rank == 1)
                shape = Squeeze(shape, shape.Rank - 1);  // 去掉 N

            outShapes.Add(shape);
        }

        private static TensorShape Squeeze(TensorShape shape, int axis)
        {
            var dims = shape.ToArray();
            return new TensorShape(dims.Where((_, i) => i != axis).ToArray());
        }

        public override void Forward(
            IReadOnlyList<Tensor> inputs,
            IReadOnlyList<Tensor> outputs,
            IReadOnlyList<Tensor> temps)
        {
            var a = inputs[0];
            var b = inputs[1];
            var output = outputs[0];

            var aDims = a.Shape.Rank == 1 ? new[] { 1, a.Shape.Dims[0] } : a.Shape.ToArray();
            var bDims = b.Shape.Rank == 1 ? new[] { b.Shape.Dims[0], 1 } : b.Shape.ToArray();

            int m = aDims[^2];
            int k = aDims[^1];
            int n = bDims[^1];

            int batchRank = Math.Max(aDims.Length - 2, bDims.Length - 2);
            var aBatch = OpUtil.AlignLeft(aDims.AsSpan(0, aDims.Length - 2), batchRank);
            var bBatch = OpUtil.AlignLeft(bDims.AsSpan(0, bDims.Length - 2), batchRank);

            var outBatch = new int[batchRank];
            for (int i = 0; i < batchRank; i++)
                outBatch[i] = Math.Max(aBatch[i], bBatch[i]);

            long batches = outBatch.Length == 0 ? 1 : OpUtil.Product(outBatch);
            var aStrides = OpUtil.Strides(aBatch);
            var bStrides = OpUtil.Strides(bBatch);
            var coords = new int[batchRank];

            var srcA = a.AsSpan<float>();
            var srcB = b.AsSpan<float>();
            var dst = output.AsSpan<float>();

            for (long batch = 0; batch < batches; batch++)
            {
                long aOffset = 0;
                long bOffset = 0;
                for (int i = 0; i < batchRank; i++)
                {
                    int coord = aBatch[i] == 1 ? 0 : coords[i];
                    aOffset += (long)coord * aStrides[i] * m * k;
                    int coordB = bBatch[i] == 1 ? 0 : coords[i];
                    bOffset += (long)coordB * bStrides[i] * k * n;
                }

                // 单个 batch 的 [m,k] × [k,n] 交给共享内核（i-k-j + 向量化 + 大矩阵并行）
                GemmKernel.Multiply(
                    srcA.Slice((int)aOffset, m * k),
                    srcB.Slice((int)bOffset, k * n),
                    dst.Slice((int)(batch * m * n), m * n),
                    m, k, n);

                for (int i = batchRank - 1; i >= 0; i--)
                {
                    if (++coords[i] < outBatch[i])
                        break;
                    coords[i] = 0;
                }
            }
        }
    }
}
