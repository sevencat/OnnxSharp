using System.Buffers;
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
        LayerRegistry.Register("AveragePool", p => new AveragePoolLayer(p));
        LayerRegistry.Register("GlobalAveragePool", p => new AveragePoolLayer(p));
        LayerRegistry.Register("GlobalMaxPool", p => new MaxPoolLayer(p));
        LayerRegistry.Register("ConvTranspose", p => new ConvTransposeLayer(p));
        LayerRegistry.Register("Einsum", p => new EinsumLayer(p));
        LayerRegistry.Register("RoiAlign", p => new RoiAlignLayer(p));
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

    private sealed class ConvLayer : Layer, IActivationFusable
    {
        /// <inheritdoc />
        public UnaryOpKind? FusedActivation { get; set; }

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

            // 3×3 / s1 / pad1 优先走 Winograd F(2,3)（乘法量只有 4/9），
            // 其次走面板式直接卷积（不物化整幅 im2col），都不适用再回退 im2col
            if (!Kernels.Winograd.TryConv3x3(
                    input, weight, bias, output,
                    _group, strides[0], strides[1], dilations[0], dilations[1], pads[0], pads[1])
                && !Kernels.PackedConv.TryConv(
                    input, weight, bias, output,
                    _group, strides[0], strides[1], dilations[0], dilations[1], pads[0], pads[1]))
            {
                Kernels.ConvKernels.Im2Col(
                    input, weight, bias, output,
                    _group, strides[0], strides[1], dilations[0], dilations[1], pads[0], pads[1]);
            }

            // 图级融合进来的激活（yolo 里每个卷积后面都跟着 SiLU）：
            // 输出刚写完还在 L1/L2 里，就地再过一遍比单独跑一个算子的开销小。
            //
            // 试过把激活塞进 GEMM 微内核的 epilogue（累加器直接激活再 store），
            // 结果更慢 26ms：多出来的调用把 16 个累加器 YMM 全挤出寄存器，
            // 主循环退化。所以这里保持在 GEMM 之外、但紧贴着的就地一趟。
            if (FusedActivation is { } activation)
                TensorOps.ElementWiseUnary(activation, output, output);
        }
    }

    // ---------------- ConvTranspose ----------------

    /// <summary>
    /// 转置卷积（NCHW，2D）。权重形状 <c>[Cin, Cout/groups, kH, kW]</c>。
    /// 用「输入驱动 scatter-accumulate」实现：每个输入点把 v*w 累加到它覆盖的输出窗口，
    /// 复杂度与正向卷积同阶，且不需要 col 缓冲。yolov8-seg / v11-seg 的 proto 分支要用。
    /// </summary>
    private sealed class ConvTransposeLayer : Layer
    {
        private readonly int[] _kernelShape;
        private readonly int[] _strides;
        private readonly int[] _pads;
        private readonly int[] _outputPads;
        private readonly int[] _dilations;
        private readonly int _groups;
        private readonly string _autoPad;

        /// <summary>重排后的权重（缓存），顺序 <c>[groups, kH, kW, Cin/groups, Cout/groups]</c>。</summary>
        private float[]? _packed;
        private Tensor? _packedSource;

        public ConvTransposeLayer(LayerParams parameters)
        {
            Type = parameters.Type;
            Name = parameters.Name;
            _kernelShape = ToInts(parameters.GetInts("kernel_shape"));
            _strides = ToInts(parameters.GetInts("strides"));
            _pads = ToInts(parameters.GetInts("pads"));
            _outputPads = ToInts(parameters.GetInts("output_padding"));
            _dilations = ToInts(parameters.GetInts("dilations"));
            _groups = (int)parameters.GetInt("group", 1);
            _autoPad = parameters.GetString("auto_pad", "NOTSET");
        }

        private (int[] Pads, int[] Strides, int[] Kernel, int[] Dilations, int[] OutputPads) Resolve(
            int spatial, TensorShape input, TensorShape weight)
        {
            var kernel = new int[spatial];
            var strides = new int[spatial];
            var pads = new int[spatial * 2];
            var dilations = new int[spatial];
            var outPads = new int[spatial];

            if (_kernelShape.Length > 0)
            {
                if (_kernelShape.Length != spatial)
                    throw new OnnxSharpException($"ConvTranspose '{Name}': kernel_shape must have {spatial} entries");

                for (int i = 0; i < spatial; i++)
                    kernel[i] = _kernelShape[i];
            }
            else
            {
                for (int i = 0; i < spatial; i++)
                    kernel[i] = weight.Dims[2 + i];
            }

            for (int i = 0; i < spatial; i++)
            {
                strides[i] = _strides.Length > i ? _strides[i] : 1;
                dilations[i] = _dilations.Length > i ? _dilations[i] : 1;
                outPads[i] = _outputPads.Length > i ? _outputPads[i] : 0;

                if (_autoPad is "SAME_UPPER" or "SAME_LOWER")
                {
                    int total = (input.Dims[2 + i] - 1) * strides[i] + dilations[i] * (kernel[i] - 1) + 1
                                + outPads[i] - input.Dims[2 + i] * strides[i];
                    if (total < 0)
                        total = 0;

                    pads[i] = _autoPad == "SAME_UPPER" ? total / 2 : total - total / 2;
                    pads[spatial + i] = total - pads[i];
                }
                else
                {
                    pads[i] = _pads.Length > i ? _pads[i] : 0;
                    pads[spatial + i] = _pads.Length > spatial + i ? _pads[spatial + i] : pads[i];
                }
            }

            return (pads, strides, kernel, dilations, outPads);
        }

        public override void GetMemoryShapes(
            IReadOnlyList<TensorShape> inShapes,
            int requiredOutputs,
            List<TensorShape> outShapes,
            List<TensorShape> tempShapes)
        {
            var input = inShapes[0];
            var weight = inShapes[1];
            int spatial = input.Rank - 2;
            var (pads, strides, kernel, dilations, outPads) = Resolve(spatial, input, weight);

            var dims = input.ToArray();
            dims[1] = weight.Dims[1] * _groups;

            for (int i = 0; i < spatial; i++)
            {
                long size = (long)(input.Dims[2 + i] - 1) * strides[i]
                            - pads[i] - pads[spatial + i]
                            + dilations[i] * (kernel[i] - 1) + outPads[i] + 1;

                dims[2 + i] = (int)size;
            }

            outShapes.Add(new TensorShape(dims));
        }

        public override unsafe void Forward(
            IReadOnlyList<Tensor> inputs,
            IReadOnlyList<Tensor> outputs,
            IReadOnlyList<Tensor> temps)
        {
            var input = inputs[0];
            var weight = inputs[1];
            var output = outputs[0];

            int spatial = input.Shape.Rank - 2;
            if (spatial != 2)
                throw new OnnxSharpException($"ConvTranspose '{Name}': only 2D is implemented");

            var (pads, strides, kernel, dilations, _) = Resolve(spatial, input.Shape, weight.Shape);

            int batches = input.Shape.Dims[0];
            int inChannels = input.Shape.Dims[1];
            int inH = input.Shape.Dims[2];
            int inW = input.Shape.Dims[3];

            int outChannels = output.Shape.Dims[1];
            int outH = output.Shape.Dims[2];
            int outW = output.Shape.Dims[3];

            int kH = kernel[0], kW = kernel[1];
            int sH = strides[0], sW = strides[1];
            int dH = dilations[0], dW = dilations[1];
            int pT = pads[0], pL = pads[1];

            if (_groups <= 0 || inChannels % _groups != 0 || outChannels % _groups != 0)
                throw new OnnxSharpException($"ConvTranspose '{Name}': channels are not divisible by group {_groups}");

            int inPerGroup = inChannels / _groups;
            int outPerGroup = outChannels / _groups;

            var src = input.AsSpan<float>();
            var dst = output.AsSpan<float>();
            var packed = GetPacked(weight, inPerGroup, outPerGroup, kH, kW);

            Tensor? bias = Inputs.Count > 2 && Inputs[2].Idx != 0 && inputs.Count > 2 && inputs[2].Count > 0
                ? inputs[2]
                : null;
            float[]? biasArray = bias is null ? null : bias.AsSpan<float>().ToArray();

            long outPlane = (long)outH * outW;
            long inPlane = (long)inH * inW;
            long groupKernel = (long)kH * kW * inPerGroup * outPerGroup;

            // 输出侧 gather：每个输出像素只在「合法相位」的 kH/sH × kW/sW 个 tap 上累加，
            // 内层按 co（连续打包）向量化，写回每个像素只做一次。
            // 并行维度是「输出行块」——scatter 方案只能按 batch*group 并行，
            // seg 的 proto 层 batch=group=1 会退化成单线程（1.4 s → 单核跑 838 MMAC）。
            int threads = ParallelRange.Threads;
            int blocks = Math.Min(outH, Math.Max(1, threads * 2));
            int rowsPerBlock = Math.Max(1, (outH + blocks - 1) / blocks);
            blocks = (outH + rowsPerBlock - 1) / rowsPerBlock;
            int tasks = batches * _groups * blocks;

            fixed (float* px = src)
            fixed (float* pp = packed)
            fixed (float* py = dst)
            {
                IntPtr pin = (IntPtr)px;
                IntPtr ppk = (IntPtr)pp;
                IntPtr pout = (IntPtr)py;

                ParallelRange.For(tasks, (begin, end) =>
                {
                    float* x0 = (float*)pin;
                    float* w0 = (float*)ppk;
                    float* y0 = (float*)pout;

                    float[] accBuf = ArrayPool<float>.Shared.Rent(outPerGroup);
                    Span<int> kyList = stackalloc int[16];
                    Span<int> iyList = stackalloc int[16];
                    var accSpan = new Span<float>(accBuf, 0, outPerGroup);

                    try
                    {
                        fixed (float* acc = accBuf)
                        {
                            for (int task = begin; task < end; task++)
                            {
                                int block = task % blocks;
                                int rest = task / blocks;
                                int g = rest % _groups;
                                int b = rest / _groups;

                                int oyBegin = block * rowsPerBlock;
                                int oyEnd = Math.Min(outH, oyBegin + rowsPerBlock);

                                float* wGroup = w0 + (long)g * groupKernel;
                                long outGroupBase = (long)b * outChannels * outPlane + (long)g * outPerGroup * outPlane;

                                for (int oy = oyBegin; oy < oyEnd; oy++)
                                {
                                    int kyCount = 0;
                                    for (int ky = 0; ky < kH && kyCount < 16; ky++)
                                    {
                                        int t = oy + pT - ky * dH;
                                        if (t < 0 || t % sH != 0)
                                            continue;

                                        int iy = t / sH;
                                        if ((uint)iy >= (uint)inH)
                                            continue;

                                        kyList[kyCount] = ky;
                                        iyList[kyCount] = iy;
                                        kyCount++;
                                    }

                                    long outRow = outGroupBase + (long)oy * outW;

                                    for (int ox = 0; ox < outW; ox++)
                                    {
                                        if (biasArray is null)
                                        {
                                            accSpan.Clear();
                                        }
                                        else
                                        {
                                            biasArray.AsSpan(g * outPerGroup, outPerGroup).CopyTo(accSpan);
                                        }

                                        for (int k = 0; k < kyCount; k++)
                                        {
                                            int ky = kyList[k];
                                            int iy = iyList[k];

                                            for (int kx = 0; kx < kW; kx++)
                                            {
                                                int t = ox + pL - kx * dW;
                                                if (t < 0 || t % sW != 0)
                                                    continue;

                                                int ix = t / sW;
                                                if ((uint)ix >= (uint)inW)
                                                    continue;

                                                float* wTap = wGroup + ((long)(ky * kW + kx) * inPerGroup) * outPerGroup;
                                                float* xPlane = x0 + ((long)b * inChannels + g * inPerGroup) * inPlane
                                                                + (long)iy * inW + ix;

                                                for (int ci = 0; ci < inPerGroup; ci++)
                                                {
                                                    float v = xPlane[(long)ci * inPlane];
                                                    if (v == 0f)
                                                        continue;

                                                    float* wp = wTap + (long)ci * outPerGroup;
                                                    int co = 0;

                                                    if (Avx2.IsSupported)
                                                    {
                                                        var vv = Vector256.Create(v);
                                                        for (; co + Vector256<float>.Count <= outPerGroup; co += Vector256<float>.Count)
                                                        {
                                                            var a = Avx.LoadVector256(acc + co);
                                                            var wv = Avx.LoadVector256(wp + co);
                                                            Avx.Store(acc + co, Avx.Add(a, Avx.Multiply(vv, wv)));
                                                        }
                                                    }

                                                    for (; co < outPerGroup; co++)
                                                        acc[co] += v * wp[co];
                                                }
                                            }
                                        }

                                        for (int co = 0; co < outPerGroup; co++)
                                            y0[outRow + (long)co * outPlane + ox] = acc[co];
                                    }
                                }
                            }
                        }
                    }
                    finally
                    {
                        ArrayPool<float>.Shared.Return(accBuf);
                    }
                });
            }
        }

        /// <summary>
        /// 把 <c>[Cin, Cout/groups, kH, kW]</c> 重排为 <c>[groups, kH, kW, Cin/groups, Cout/groups]</c>，
        /// 让 gather 内核内层的 co 维连续（可直接 8 路向量累加）。结果按权重张量实例缓存。
        /// </summary>
        private float[] GetPacked(Tensor weight, int inPerGroup, int outPerGroup, int kH, int kW)
        {
            if (_packed is not null && ReferenceEquals(_packedSource, weight))
                return _packed;

            var w = weight.AsSpan<float>();
            var packed = new float[(long)_groups * kH * kW * inPerGroup * outPerGroup];
            int kernelArea = kH * kW;

            for (int g = 0; g < _groups; g++)
            {
                for (int ci = 0; ci < inPerGroup; ci++)
                {
                    int ic = g * inPerGroup + ci;

                    for (int co = 0; co < outPerGroup; co++)
                    {
                        long srcBase = ((long)ic * outPerGroup + co) * kernelArea;
                        long dstBase = ((long)g * kernelArea * inPerGroup + ci) * outPerGroup + co;

                        for (int t = 0; t < kernelArea; t++)
                            packed[dstBase + (long)t * inPerGroup * outPerGroup] = w[(int)(srcBase + t)];
                    }
                }
            }

            _packed = packed;
            _packedSource = weight;
            return packed;
        }
    }

    // ---------------- MaxPool ----------------

    /// <summary>MaxPool / AveragePool 共用的属性归一化（默认值、auto_pad 推导、pads 补齐）。</summary>
    private static (int[] Pads, int[] Strides, int[] Kernel, int[] Dilations) ResolvePool(
        string type,
        string name,
        int[] kernelShape,
        int[] stridesAttr,
        int[] padsAttr,
        int[] dilationsAttr,
        string autoPad,
        int spatial,
        TensorShape input)
    {
        var kernel = new int[spatial];
        var strides = new int[spatial];
        var pads = new int[spatial * 2];
        var dilations = new int[spatial];

        // Global*Pool：核覆盖整个空间，等价于 kernel = 输入尺寸、stride = kernel。
        if (type is "GlobalAveragePool" or "GlobalMaxPool")
        {
            for (int i = 0; i < spatial; i++)
            {
                kernel[i] = input.Dims[2 + i];
                strides[i] = kernel[i];
                dilations[i] = 1;
            }

            return (pads, strides, kernel, dilations);
        }

        if (kernelShape.Length != spatial)
            throw new OnnxSharpException($"{type} '{name}': kernel_shape must have {spatial} entries");

        for (int i = 0; i < spatial; i++)
        {
            kernel[i] = kernelShape[i];
            strides[i] = stridesAttr.Length > i ? stridesAttr[i] : 1;
            dilations[i] = dilationsAttr.Length > i ? dilationsAttr[i] : 1;
        }

        for (int i = 0; i < spatial; i++)
        {
            if (autoPad is "SAME_UPPER" or "SAME_LOWER" or "VALID")
            {
                var (begin, end) = AutoPad(autoPad, input.Dims[2 + i], kernel[i], strides[i], dilations[i], -1);
                pads[i] = begin;
                pads[spatial + i] = end;
            }
            else
            {
                pads[i] = padsAttr.Length > i ? padsAttr[i] : 0;
                pads[spatial + i] = padsAttr.Length > spatial + i ? padsAttr[spatial + i] : pads[i];
            }
        }

        return (pads, strides, kernel, dilations);
    }

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
            => ResolvePool(Type, Name, _kernelShape, _strides, _pads, _dilations, _autoPad, spatial, input);

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

                    // 步长 1、无横向膨胀时可以一次比较 8 个输出（SPPF 的 5×5/9×9/13×13 全走这条）
                    bool vectorizable = dW == 1 && sW == 1;

                    // 只有整块窗口都落在真实输入内才能向量化，含 pad 的边界列交回标量
                    int vStart = vectorizable ? Math.Max(0, pL) : 0;
                    int vLimit = vectorizable ? Math.Min(outW, inW - kW - (Vector256<float>.Count - 1) + pL) : 0;

                    for (int oy = 0; oy < outH; oy++)
                    {
                        int y0 = oy * sH - pT;

                        for (int ox = 0; ox < outW; ox++)
                        {
                            if (vectorizable && ox >= vStart && ox + Vector256<float>.Count <= vLimit)
                            {
                                int vx = ox - pL;
                                var acc = Vector256.Create(float.NegativeInfinity);

                                for (int ky = 0; ky < kH; ky++)
                                {
                                    int y = y0 + ky * dH;
                                    if ((uint)y >= (uint)inH)
                                        continue;

                                    long rowBase = inBase + (long)y * inW + vx;
                                    for (int kx = 0; kx < kW; kx++)
                                        acc = Vector256.Max(acc, Vector256.LoadUnsafe(ref s[(int)(rowBase + kx)]));
                                }

                                Vector256.StoreUnsafe(acc, ref d[(int)(outBase + (long)oy * outW + ox)]);
                                ox += Vector256<float>.Count - 1;
                                continue;
                            }

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

    // ---------------- AveragePool ----------------

    /// <summary>
    /// 均值池化。count_include_pad=1（默认）时 pad 出来的 0 也计入分母，但 ceil_mode
    /// 撑到 pad 边界之外的部分不计；=0 时只统计与真实输入重叠的 tap。
    /// RT-DETR 的 hybrid encoder、yolov9 的 SPP/SPPF 都用它。
    /// </summary>
    private sealed class AveragePoolLayer : Layer
    {
        private readonly int[] _kernelShape;
        private readonly int[] _strides;
        private readonly int[] _pads;
        private readonly int[] _dilations;
        private readonly string _autoPad;
        private readonly bool _ceilMode;
        private readonly bool _countIncludePad;

        public AveragePoolLayer(LayerParams parameters)
        {
            Type = parameters.Type;
            Name = parameters.Name;
            _kernelShape = ToInts(parameters.GetInts("kernel_shape"));
            _strides = ToInts(parameters.GetInts("strides"));
            _pads = ToInts(parameters.GetInts("pads"));
            _dilations = ToInts(parameters.GetInts("dilations"));
            _autoPad = parameters.GetString("auto_pad", "NOTSET");
            _ceilMode = parameters.GetInt("ceil_mode", 0) != 0;
            _countIncludePad = parameters.GetInt("count_include_pad", 1) != 0;
        }

        private (int[] Pads, int[] Strides, int[] Kernel, int[] Dilations) Resolve(int spatial, TensorShape input)
            => ResolvePool(Type, Name, _kernelShape, _strides, _pads, _dilations, _autoPad, spatial, input);

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
        }

        /// <summary>窗口里落在 [0, paddedSize) 内的 tap 个数（padded 坐标系）。</summary>
        private static int PaddedTaps(int start, int kernel, int dilation, int paddedSize)
        {
            int k0 = start >= 0 ? 0 : (-start + dilation - 1) / dilation;
            int k1 = (int)Math.Floor((paddedSize - 1 - start) / (double)dilation) + 1;
            if (k1 > kernel)
                k1 = kernel;
            return k1 > k0 ? k1 - k0 : 0;
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
                throw new OnnxSharpException($"AveragePool '{Name}': only 2D pooling is implemented");

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
            int padH = inH + pads[0] + pads[2];
            int padW = inW + pads[1] + pads[3];
            bool includePad = _countIncludePad;

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

                        int ky0 = y0 >= 0 ? 0 : (-y0 + dH - 1) / dH;
                        int yLimit = inH - y0;
                        int ky1 = yLimit <= 0 ? 0 : Math.Min(kH, (yLimit + dH - 1) / dH);
                        if (ky1 < ky0)
                            ky1 = ky0;

                        int cntH = includePad ? PaddedTaps(y0 + pT, kH, dH, padH) : ky1 - ky0;

                        for (int ox = 0; ox < outW; ox++)
                        {
                            int x0 = ox * sW - pL;

                            int kx0 = x0 >= 0 ? 0 : (-x0 + dW - 1) / dW;
                            int xLimit = inW - x0;
                            int kx1 = xLimit <= 0 ? 0 : Math.Min(kW, (xLimit + dW - 1) / dW);
                            if (kx1 < kx0)
                                kx1 = kx0;

                            int cntW = includePad ? PaddedTaps(x0 + pL, kW, dW, padW) : kx1 - kx0;
                            int count = cntH * cntW;

                            float sum = 0;
                            for (int ky = ky0; ky < ky1; ky++)
                            {
                                long rowBase = inBase + (long)(y0 + ky * dH) * inW + x0;
                                for (int kx = kx0; kx < kx1; kx++)
                                    sum += s[(int)(rowBase + kx * dW)];
                            }

                            d[(int)(outBase + (long)oy * outW + ox)] = count > 0 ? sum / count : 0f;
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
            var src = input.AsSpan<float>();
            var dst = output.AsSpan<float>();

            if (inner == 1)
            {
                // DFL 是 33600 组 16 元素：pin 一次、并行区间直接算，省掉每组一次
                // Span.Slice + fixed + 函数调用的固定开销（原来 12 ns/元素）。
                unsafe
                {
                    fixed (float* ps = src)
                    fixed (float* pd = dst)
                    {
                        int stride = length;
                        long total = (long)outer * length;

                        IntPtr pin = (IntPtr)ps;
                        IntPtr pout = (IntPtr)pd;

                        if (total >= 1 << 16 && ParallelRange.Threads > 1)
                        {
                            ParallelRange.For(outer, (begin, end) =>
                            {
                                float* s = (float*)pin;
                                float* d = (float*)pout;
                                for (int o = begin; o < end; o++)
                                    VectorMath.SoftmaxRow(s + (long)o * stride, d + (long)o * stride, stride);
                            });
                        }
                        else
                        {
                            for (int o = 0; o < outer; o++)
                                VectorMath.SoftmaxRow(ps + (long)o * stride, pd + (long)o * stride, stride);
                        }
                    }
                }

                return;
            }

            // inner > 1（DFL 的 [1,4,16,8400] 沿 16 归一化）：按列 8 个一组向量化，
            // 多线程按 (outer × 列段) 摊开。
            unsafe
            {
                fixed (float* ps = src)
                fixed (float* pd = dst)
                {
                    IntPtr pin = (IntPtr)ps;
                    IntPtr pout = (IntPtr)pd;

                    int perItem = 64;
                    int perOuter = (inner + perItem - 1) / perItem;
                    int items = outer * perOuter;
                    long total = (long)outer * length * inner;

                    if (total >= 1 << 16 && ParallelRange.Threads > 1)
                    {
                        ParallelRange.For(items, (begin, end) =>
                        {
                            float* s = (float*)pin;
                            float* d = (float*)pout;
                            long block = (long)length * inner;

                            for (int item = begin; item < end; item++)
                            {
                                int o = item / perOuter;
                                int c0 = (item % perOuter) * perItem;
                                int c1 = Math.Min(c0 + perItem, inner);
                                VectorMath.SoftmaxColumns(s + o * block, d + o * block, length, inner, c0, c1);
                            }
                        });
                    }
                    else
                    {
                        long block = (long)length * inner;
                        for (int o = 0; o < outer; o++)
                            VectorMath.SoftmaxColumns(ps + o * block, pd + o * block, length, inner, 0, inner);
                    }
                }
            }
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

            // B 没有 batch 维、A 也不做广播时，(batch, m) 可以合并成一个大 GEMM：
            // 两者拼起来正好是 [batches*m, k] × [k, n]，A/C 本来就是连续存放的。
            // 典型例子是 DETR decoder 的 q/k/v 投影，运行时形状 [300,1,256] × [256,256]：
            // 按 batch 逐次调用会让内核把 256 KB 的权重重新打包 300 遍（实测单个算子 4.06 ms），
            // 合并成一次 300×256×256 只要 0.13 ms —— 这是当前最大的单点浪费。
            bool bBatched = false;
            bool aBroadcast = false;
            for (int i = 0; i < batchRank; i++)
            {
                if (bBatch[i] != 1)
                {
                    bBatched = true;
                    break;
                }
            }

            if (!bBatched)
            {
                for (int i = 0; i < batchRank; i++)
                {
                    if (aBatch[i] != outBatch[i])
                    {
                        aBroadcast = true;
                        break;
                    }
                }
            }

            if (!bBatched && !aBroadcast)
            {
                int rows = checked((int)(batches * m));
                GemmKernel.Multiply(
                    srcA.Slice(0, rows * k),
                    srcB.Slice(0, k * n),
                    dst.Slice(0, rows * n),
                    rows, k, n);
                return;
            }

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

    // ---------------- Einsum ----------------

    /// <summary>
    /// 爱因斯坦求和。**只实现 SAM3 用到的那一种**：<c>bqc,bchw-&gt;bqhw</c>
    /// （mask embedder 的 <c>[B,Q,C]</c> 与特征图 <c>[B,C,H,W]</c> 做逐批矩阵乘）。
    /// <para>
    /// 其它方程一律抛 <see cref="OnnxSharpException"/> 并报出方程原文 ——
    /// 通用 einsum 需要下标解析 + 路径规划，收益不成正比；真需要时按方程逐个加特化。
    /// </para>
    /// </summary>
    private sealed class EinsumLayer : Layer
    {
        private readonly string _equation;

        public EinsumLayer(LayerParams parameters)
        {
            Type = parameters.Type;
            Name = parameters.Name;
            _equation = parameters.GetString("equation", string.Empty);
        }

        private bool IsSupported => _equation.Replace(" ", string.Empty) == "bqc,bchw->bqhw";

        public override void GetMemoryShapes(
            IReadOnlyList<TensorShape> inShapes,
            int requiredOutputs,
            List<TensorShape> outShapes,
            List<TensorShape> tempShapes)
        {
            if (!IsSupported)
                throw new OnnxSharpException(
                    $"Einsum '{Name}': only \"bqc,bchw->bqhw\" is implemented (got \"{_equation}\")");

            var a = inShapes[0];
            var b = inShapes[1];
            if (a.Rank != 3 || b.Rank != 4 || a[2] != b[1])
                throw new OnnxSharpException(
                    $"Einsum '{Name}': expected [B,Q,C]x[B,C,H,W], got {a}x{b}");

            outShapes.Add(new TensorShape(a[0], a[1], b[2], b[3]));
        }

        public override void Forward(
            IReadOnlyList<Tensor> inputs,
            IReadOnlyList<Tensor> outputs,
            IReadOnlyList<Tensor> temps)
        {
            var a = inputs[0];
            var b = inputs[1];
            var output = outputs[0];

            int batches = a.Shape[0];
            int queries = a.Shape[1];
            int channels = a.Shape[2];
            int height = b.Shape[2];
            int width = b.Shape[3];
            int plane = height * width;

            var sa = a.AsSpan<float>();
            var sb = b.AsSpan<float>();
            var dst = output.AsSpan<float>();

            // 每个 query 都是 [C, H*W] 的加权和：标量广播 + SIMD 累加（200×256×5184 ≈ 265 MMAC）。
            for (int batch = 0; batch < batches; batch++)
            {
                for (int q = 0; q < queries; q++)
                {
                    var target = dst.Slice((batch * queries + q) * plane, plane);
                    target.Clear();

                    for (int c = 0; c < channels; c++)
                    {
                        float weight = sa[(batch * queries + q) * channels + c];
                        if (weight == 0f)
                            continue;

                        var source = sb.Slice((batch * channels + c) * plane, plane);
                        int i = 0;

                        if (Fma.IsSupported)
                        {
                            var vw = Vector256.Create(weight);
                            for (; i + Vector256<float>.Count <= plane; i += Vector256<float>.Count)
                            {
                                var acc = Vector256.LoadUnsafe(ref target[i]);
                                var v = Vector256.LoadUnsafe(ref source[i]);
                                Vector256.StoreUnsafe(Fma.MultiplyAdd(v, vw, acc), ref target[i]);
                            }
                        }

                        for (; i < plane; i++)
                            target[i] += source[i] * weight;
                    }
                }
            }
        }
    }

    // ---------------- RoiAlign ----------------

    /// <summary>
    /// 兴趣区域对齐（SAM3 的 geometry-encoder：把一个框对应的特征采样成 7×7）。
    /// <para>
    /// 按 ONNX 语义实现：<c>avg</c> / <c>max</c>、<c>spatial_scale</c>、
    /// <c>coordinate_transformation_mode</c>（<c>half_pixel</c> 偏移 0.5；<c>output_half_pixel</c>
    /// 不偏移、宽高最小取 1）、<c>sampling_ratio = 0</c> 时自适应采样数。
    /// 双线性采样把坐标夹到 <c>[0, H-1]</c>（与 ORT 一致）。
    /// </para>
    /// </summary>
    private sealed class RoiAlignLayer : Layer
    {
        private readonly string _mode;
        private readonly string _coordinateMode;
        private readonly int _outputHeight;
        private readonly int _outputWidth;
        private readonly int _samplingRatio;
        private readonly float _spatialScale;

        public RoiAlignLayer(LayerParams parameters)
        {
            Type = parameters.Type;
            Name = parameters.Name;
            _mode = parameters.GetString("mode", "avg");
            _coordinateMode = parameters.GetString("coordinate_transformation_mode", "half_pixel");
            _outputHeight = (int)parameters.GetInt("output_height", 1);
            _outputWidth = (int)parameters.GetInt("output_width", 1);
            _samplingRatio = (int)parameters.GetInt("sampling_ratio", 0);
            _spatialScale = (float)parameters.GetFloat("spatial_scale", 1.0);
        }

        public override void GetMemoryShapes(
            IReadOnlyList<TensorShape> inShapes,
            int requiredOutputs,
            List<TensorShape> outShapes,
            List<TensorShape> tempShapes)
        {
            var x = inShapes[0];
            var rois = inShapes[1];
            if (x.Rank != 4 || rois.Rank != 2 || rois[1] != 4)
                throw new OnnxSharpException(
                    $"RoiAlign '{Name}': expected X [N,C,H,W] and rois [num,4], got {x}x{rois}");

            outShapes.Add(new TensorShape(rois[0], x[1], _outputHeight, _outputWidth));
        }

        public override void Forward(
            IReadOnlyList<Tensor> inputs,
            IReadOnlyList<Tensor> outputs,
            IReadOnlyList<Tensor> temps)
        {
            var x = inputs[0];
            var output = outputs[0];

            int batches = x.Shape[0];
            int channels = x.Shape[1];
            int height = x.Shape[2];
            int width = x.Shape[3];

            int numRois = output.Shape[0];
            int outHeight = output.Shape[2];
            int outWidth = output.Shape[3];

            var rois = inputs[1].AsSpan<float>();
            int[] batchIndices = OpUtil.ReadInts(inputs[2]);

            var src = x.AsSpan<float>();
            var dst = output.AsSpan<float>();

            bool average = !_mode.Equals("max", StringComparison.OrdinalIgnoreCase);
            bool halfPixel = !_coordinateMode.Equals("output_half_pixel", StringComparison.OrdinalIgnoreCase);
            double offset = halfPixel ? 0.5 : 0.0;

            for (int r = 0; r < numRois; r++)
            {
                int batch = batchIndices.Length > r ? batchIndices[r] : 0;
                if ((uint)batch >= (uint)batches)
                    continue;

                double x1 = rois[r * 4 + 0] * _spatialScale - offset;
                double y1 = rois[r * 4 + 1] * _spatialScale - offset;
                double x2 = rois[r * 4 + 2] * _spatialScale - offset;
                double y2 = rois[r * 4 + 3] * _spatialScale - offset;

                double roiWidth = x2 - x1;
                double roiHeight = y2 - y1;
                if (!halfPixel)
                {
                    roiWidth = Math.Max(roiWidth, 1.0);
                    roiHeight = Math.Max(roiHeight, 1.0);
                }

                double binWidth = roiWidth / outWidth;
                double binHeight = roiHeight / outHeight;

                int gridWidth = _samplingRatio > 0 ? _samplingRatio : (int)Math.Ceiling(roiWidth / outWidth);
                int gridHeight = _samplingRatio > 0 ? _samplingRatio : (int)Math.Ceiling(roiHeight / outHeight);
                gridWidth = Math.Max(gridWidth, 1);
                gridHeight = Math.Max(gridHeight, 1);
                double sampleCount = gridWidth * gridHeight;

                for (int c = 0; c < channels; c++)
                {
                    var plane = src.Slice((batch * channels + c) * height * width, height * width);
                    long dstBase = ((long)r * channels + c) * outHeight * outWidth;

                    for (int ph = 0; ph < outHeight; ph++)
                    {
                        for (int pw = 0; pw < outWidth; pw++)
                        {
                            double sum = 0;
                            double best = double.NegativeInfinity;

                            for (int iy = 0; iy < gridHeight; iy++)
                            {
                                double y = y1 + ph * binHeight + (iy + 0.5) * binHeight / gridHeight;

                                for (int ix = 0; ix < gridWidth; ix++)
                                {
                                    double sampleX = x1 + pw * binWidth + (ix + 0.5) * binWidth / gridWidth;
                                    double value = SampleBilinear(plane, height, width, y, sampleX);

                                    sum += value;
                                    if (value > best)
                                        best = value;
                                }
                            }

                            dst[(int)(dstBase + (long)ph * outWidth + pw)] =
                                (float)(average ? sum / sampleCount : best);
                        }
                    }
                }
            }
        }

        /// <summary>双线性采样；坐标先夹到 [0, size-1]（ONNX RoiAlign 的边界约定）。</summary>
        private static double SampleBilinear(ReadOnlySpan<float> plane, int height, int width, double y, double x)
        {
            double cy = Math.Clamp(y, 0.0, height - 1);
            double cx = Math.Clamp(x, 0.0, width - 1);

            int y0 = (int)Math.Floor(cy);
            int x0 = (int)Math.Floor(cx);
            int y1 = Math.Min(y0 + 1, height - 1);
            int x1 = Math.Min(x0 + 1, width - 1);

            double ly = cy - y0;
            double lx = cx - x0;

            double top = plane[y0 * width + x0] + (plane[y0 * width + x1] - plane[y0 * width + x0]) * lx;
            double bottom = plane[y1 * width + x0] + (plane[y1 * width + x1] - plane[y1 * width + x0]) * lx;
            return top + (bottom - top) * ly;
        }
    }
}
