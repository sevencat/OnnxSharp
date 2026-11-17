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
    /// <summary>
    /// 池化的几个形状专用内核（<c>ONNXSHARP_POOL_FAST=0</c> 关闭）。
    /// 供 <c>OnnxSharp.Cli ab</c> 同进程交替 A/B：两条路径输出应当逐位一致。
    /// </summary>
    public static bool FastPool { get; set; } = EnvFlag("ONNXSHARP_POOL_FAST");

    /// <summary>
    /// nearest 2× 上采样专用内核（<c>ONNXSHARP_RESIZE_FAST=0</c> 关闭），同样只为一件事：
    /// 能和通用查表路径同进程对照。
    /// </summary>
    public static bool FastResize2x { get; set; } = EnvFlag("ONNXSHARP_RESIZE_FAST");

    private static bool EnvFlag(string name)
        => !string.Equals(Environment.GetEnvironmentVariable(name), "0", StringComparison.Ordinal);

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
        LayerRegistry.Register("LogSoftmax", p => new LogSoftmaxLayer(p));
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

    /// <summary>
    /// 图融合用：这个卷积是不是「1×1、s1、pad 0、dilation 1、group 1」。
    /// </summary>
    internal interface IPointwiseConv
    {
        bool IsPointwiseUngrouped { get; }
    }

    private sealed class ConvLayer : Layer, IActivationFusable, IPointwiseConv
    {
        /// <inheritdoc />
        public UnaryOpKind? FusedActivation { get; set; }

        /// <inheritdoc />
        public bool IsPointwiseUngrouped
            => _group == 1
               && _autoPad is "NOTSET"
               && _kernelShape is [1, 1]
               && (_strides is [] or [1, 1])
               && (_pads is [] or [0, 0] or [0, 0, 0, 0])
               && (_dilations is [] or [1, 1]);

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

            if (input.Rank == 3)
            {
                outShapes.Add(Conv1DShape(input, weight));
                return;
            }

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

            // NCHWc16（planv2 §P7.7）：命中 3×3/pad1、stride 1 或 2 且通道对齐时，
            // 额外申请 blocked 输入/输出两块缓冲。判定条件与 Forward 里的 TryConv 同源（Dispatch）。
            if (input.Rank == 4 && input.Dims[0] == 1)
            {
                Resolve(input, out var nchwcStrides, out var nchwcDilations, out var nchwcPads);

                if (Kernels.NchwcConv.Dispatch(channels, dims[1], input.Dims[2], input.Dims[3], dims[2], dims[3],
                        weight.Dims[2], weight.Dims[3], _group,
                        nchwcStrides[0], nchwcStrides[1], nchwcDilations[0], nchwcDilations[1],
                        nchwcPads[0], nchwcPads[1]))
                {
                    tempShapes.Add(new TensorShape((int)Kernels.NchwcConv.BlockedInputCount(
                        channels, input.Dims[2], input.Dims[3], nchwcStrides[0])));
                    tempShapes.Add(new TensorShape(
                        (int)Kernels.NchwcConv.BlockedOutputCount(dims[1], dims[2], dims[3])));
                }
            }
        }

        /// <summary>
        /// 1-D 卷积（<c>[N,C,T]</c> × <c>[M,C/group,K]</c>）：FSMN 记忆块用它做时间轴卷积。
        /// 朴素实现就够（单节点约 60 万次乘加），2-D 主路径完全不受影响。
        /// </summary>
        private TensorShape Conv1DShape(TensorShape input, TensorShape weight)
        {
            if (weight.Rank != 3)
                throw new OnnxSharpException($"Conv '{Name}': 1-D input needs a 3-D weight, got {weight}");

            int channels = input.Dims[1];
            if (_group <= 0 || channels % _group != 0)
                throw new OnnxSharpException($"Conv '{Name}': channels {channels} not divisible by group {_group}");

            var (kernel, stride, dilation, padBegin, padEnd) = Resolve1D(input.Dims[2], weight.Dims[2]);
            int length = SpatialOut(input.Dims[2], kernel, stride, dilation, padBegin, padEnd, false);
            return new TensorShape(new[] { input.Dims[0], weight.Dims[0], length });
        }

        private (int Kernel, int Stride, int Dilation, int PadBegin, int PadEnd) Resolve1D(int length, int weightKernel)
        {
            int kernel = _kernelShape.Length > 0 ? _kernelShape[0] : weightKernel;
            int stride = _strides.Length > 0 ? _strides[0] : 1;
            int dilation = _dilations.Length > 0 ? _dilations[0] : 1;
            int padBegin = _pads.Length > 0 ? _pads[0] : 0;
            int padEnd = _pads.Length > 1 ? _pads[1] : padBegin;

            if (_autoPad is "SAME_UPPER" or "SAME_LOWER" or "VALID")
                (padBegin, padEnd) = AutoPad(_autoPad, length, kernel, stride, dilation, -1);

            return (kernel, stride, dilation, padBegin, padEnd);
        }

        private void Forward1D(Tensor input, Tensor weight, Tensor? bias, Tensor output)
        {
            int batch = input.Shape.Dims[0];
            int inChannels = input.Shape.Dims[1];
            int inLength = input.Shape.Dims[2];
            int outChannels = output.Shape.Dims[1];
            int outLength = output.Shape.Dims[2];

            var (kernel, stride, dilation, padBegin, _) = Resolve1D(inLength, weight.Shape.Dims[2]);
            int channelsPerGroup = inChannels / _group;
            int outPerGroup = outChannels / _group;

            var source = input.AsSpan<float>();
            var weights = weight.AsSpan<float>();
            var target = output.AsSpan<float>();
            var biasSpan = bias is null ? default : bias.AsSpan<float>();

            // stride = 1 时按「输出向量化」做（每个 k 一条连续的 FMA），
            // FSMN / CIF 的 1-D 卷积都是这种形状；stride ≠ 1 才回退到朴素实现。
            if (stride == 1 && Avx.IsSupported)
            {
                unsafe
                {
                    fixed (float* ps = source)
                    fixed (float* pw = weights)
                    fixed (float* pb = biasSpan)
                    fixed (float* pt = target)
                    {
                        for (int n = 0; n < batch; n++)
                        {
                            Conv1DFast(
                                ps + (long)n * inChannels * inLength,
                                pw, bias is null ? null : pb,
                                pt + (long)n * outChannels * outLength,
                                inChannels, inLength, outChannels, outLength,
                                channelsPerGroup, outPerGroup, kernel, dilation, padBegin);
                        }
                    }
                }

                return;
            }

            for (int n = 0; n < batch; n++)
            {
                for (int m = 0; m < outChannels; m++)
                {
                    int inBase = m / outPerGroup * channelsPerGroup;

                    for (int t = 0; t < outLength; t++)
                    {
                        float sum = bias is null ? 0f : biasSpan[m];

                        for (int j = 0; j < channelsPerGroup; j++)
                        {
                            long inRow = ((long)n * inChannels + inBase + j) * inLength;
                            long weightRow = ((long)m * channelsPerGroup + j) * kernel;

                            for (int k = 0; k < kernel; k++)
                            {
                                int position = t * stride - padBegin + k * dilation;
                                if ((uint)position >= (uint)inLength)
                                    continue;
                                sum += source[(int)(inRow + position)] * weights[(int)(weightRow + k)];
                            }
                        }

                        target[(int)(((long)n * outChannels + m) * outLength + t)] = sum;
                    }
                }
            }
        }

        /// <summary>
        /// 一帧 1-D 卷积（stride = 1）：按输出通道并行，每个 (输入通道, k) 做一条 8 路 FMA —— 
        /// 越界位置贡献为 0，所以只在有效区间 <c>[tLo, tHi)</c> 内加减，边界不额外判断。
        /// </summary>
        private static unsafe void Conv1DFast(
            float* source,
            float* weights,
            float* bias,
            float* target,
            int inChannels,
            int inLength,
            int outChannels,
            int outLength,
            int channelsPerGroup,
            int outPerGroup,
            int kernel,
            int dilation,
            int padBegin)
        {
            var options = new ParallelOptions { MaxDegreeOfParallelism = Math.Max(1, ParallelRange.Threads) };

            Parallel.For(0, outChannels, options, m =>
            {
                int inBase = m / outPerGroup * channelsPerGroup;
                float* outRow = target + (long)m * outLength;

                float initial = bias is null ? 0f : bias[m];
                for (int t = 0; t < outLength; t++)
                    outRow[t] = initial;

                for (int j = 0; j < channelsPerGroup; j++)
                {
                    float* inRow = source + (long)(inBase + j) * inLength;
                    float* wRow = weights + (long)(m * channelsPerGroup + j) * kernel;

                    for (int k = 0; k < kernel; k++)
                    {
                        float w = wRow[k];
                        int shift = -padBegin + k * dilation;

                        int tLo = Math.Max(0, -shift);
                        int tHi = Math.Min(outLength, inLength - shift);
                        if (tHi <= tLo)
                            continue;

                        float* src = inRow + shift;
                        var vw = Vector256.Create(w);

                        int t = tLo;
                        for (; t + 8 <= tHi; t += 8)
                        {
                            var value = Vector256.Add(Vector256.Load(outRow + t), Vector256.Multiply(vw, Vector256.Load(src + t)));
                            value.Store(outRow + t);
                        }

                        for (; t < tHi; t++)
                            outRow[t] += w * src[t];
                    }
                }
            });
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

            Tensor? bias = Inputs.Count > 2 && Inputs[2].Idx != 0 && inputs[2].Count > 0 ? inputs[2] : null;

            if (input.Shape.Rank == 3)
            {
                Forward1D(input, weight, bias, output);
                if (FusedActivation is { } fused1D)
                    TensorOps.ElementWiseUnary(fused1D, output, output);
                return;
            }

            if (input.Shape.Rank != 4)
                throw new OnnxSharpException($"Conv '{Name}': only 2D convolution (NCHW) is implemented");

            Resolve(input.Shape, out var strides, out var dilations, out var pads);

            // depthwise（每组 1 个输出通道）先走专用直接卷积：GEMM 类实现在这种形状上会退化成
            // 「K×spatial 的 im2col」，MobileNet 系（YuNet）的 26 个 depthwise 全靠这条路径。
            if (Kernels.DepthwiseConv.TryConv(
                    input, weight, bias, output,
                    _group, strides[0], strides[1], dilations[0], dilations[1], pads[0], pads[1]))
            {
                if (FusedActivation is { } depthwiseActivation)
                    TensorOps.ElementWiseUnary(depthwiseActivation, output, output);
                return;
            }

            // 3 通道输入的首层（yolo / YuNet 的 Conv_0 这类）：K=27 配 M=16，im2col 要为
            // 102400 个像素物化 11 MB 的 col 缓冲，面板式也摊不平这 27 个 K 步 —— 两条通用路径
            // 都只有 20~40 GMAC/s。改沿输出宽方向 16 像素一个 zmm 直接卷：1/4/16 线程
            // 32.8 / 131.9 / 314.2 GMAC/s，是 im2col 的 1.6×（bench.md §39）。
            if (Kernels.FirstConv.TryConv(
                    input, weight, bias, output,
                    _group, strides[0], strides[1], dilations[0], dilations[1], pads[0], pads[1]))
            {
                if (FusedActivation is { } firstActivation)
                    TensorOps.ElementWiseUnary(firstActivation, output, output);
                return;
            }

            // 3×3 / s2 / pad1 走 NCHWc16 直接卷积（planv2 §P7.7 阶段 1）：blocked 布局下
            // 一次 zmm 载入就是「16 个通道」，16 个累加器全压在寄存器里，实测 165~220 GMAC/s，
            // 是同一形状上面板式（88~138）的 1.6~2.4 倍。代价是每层进出一次 reorder。
            if (Kernels.NchwcConv.TryConv(
                    input, weight, bias, output,
                    _group, strides[0], strides[1], dilations[0], dilations[1], pads[0], pads[1],
                    temps.Count > 1 ? temps[0] : null,
                    temps.Count > 1 ? temps[1] : null))
            {
                if (FusedActivation is { } nchwcActivation)
                    TensorOps.ElementWiseUnary(nchwcActivation, output, output);
                return;
            }

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
            {
                long ta = Kernels.ConvProf.On ? System.Diagnostics.Stopwatch.GetTimestamp() : 0;
                TensorOps.ElementWiseUnary(activation, output, output);
                if (Kernels.ConvProf.On)
                    Kernels.ConvProf.AddAct(System.Diagnostics.Stopwatch.GetTimestamp() - ta);
            }
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

        /// <summary>
        /// <c>stride == kernel</c> 专用内核的开关（<c>ONNXSHARP_CONVTRANSPOSE_FAST=0</c> 关闭），
        /// 供 A/B 与回归比对：两条路径应当逐位一致。
        /// </summary>
        private static readonly bool StrideEqualsKernelEnabled =
            !string.Equals(Environment.GetEnvironmentVariable("ONNXSHARP_CONVTRANSPOSE_FAST"), "0",
                StringComparison.Ordinal);

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

                // stride == kernel 且无 padding（DB / FPN 的 2× 上采样全是这种）：输出像素与输入像素
                // 一一对应、相位固定，可以直接按相位展开成「1×1 卷积」，不必每像素做 `% stride` 和边界判断。
                if (StrideEqualsKernelEnabled && _groups == 1 && dH == 1 && dW == 1 && sH == kH && sW == kW
                    && pT == 0 && pL == 0 && outH == inH * sH && outW == inW * sW)
                {
                    ForwardStrideEqualsKernel(px, pp, py, biasArray,
                        batches, inChannels, inH, inW, outChannels, outH, outW,
                        kH, kW, sH, sW, inPerGroup, outPerGroup, inPlane, outPlane);
                    return;
                }

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
        /// <c>stride == kernel</c>、<c>dilation = 1</c>、<c>pad = 0</c>、<c>group = 1</c> 的专用路径。
        /// <para>
        /// 这种参数下每个输出像素 <c>(iy*sH+ky, ix*sW+kx)</c> 只由输入像素 <c>(iy, ix)</c> 贡献（唯一 tap），
        /// 于是整层等价于 <c>kH*kW</c> 个 1×1 卷积。通用路径要为每个输出像素做两次 <c>% stride</c>、
        /// 两次整除和边界判断 —— DB 检出的 2× 上采样（1→1 通道）算力几乎为零，时间全花在这些判断上。
        /// </para>
        /// </summary>
        private static unsafe void ForwardStrideEqualsKernel(
            float* x0, float* w0, float* y0, float[]? biasArray,
            int batches, int inChannels, int inH, int inW,
            int outChannels, int outH, int outW,
            int kH, int kW, int sH, int sW,
            int inPerGroup, int outPerGroup,
            long inPlane, long outPlane)
        {
            int threads = ParallelRange.Threads;
            int blocks = Math.Min(inH, Math.Max(1, threads * 2));
            int rowsPerBlock = Math.Max(1, (inH + blocks - 1) / blocks);
            blocks = (inH + rowsPerBlock - 1) / rowsPerBlock;

            // 单输出通道（DB 检出的最后一层 1→1 上采样）：按 ci 外层 + 行缓冲累加。
            // 「每像素读 16 个跨平面 float」的写法在 258k 像素上要 2 ms，改成行内连续累加、
            // 整行连续写回后降到 0.3 ms 量级。
            if (outPerGroup == 1 && kW == sW && inW <= 8192)
            {
                float bias = biasArray is null ? 0f : biasArray[0];
                int singleTasks = batches * kH * blocks;

                ParallelRange.For(singleTasks, (begin, end) =>
                {
                    float[] lineBuf = ArrayPool<float>.Shared.Rent(kW * inW);
                    try
                    {
                        fixed (float* lb = lineBuf)
                        {
                            for (int task = begin; task < end; task++)
                            {
                                int block = task % blocks;
                                int rest = task / blocks;
                                int ky = rest % kH;
                                int b = rest / kH;

                                int iyBegin = block * rowsPerBlock;
                                int iyEnd = Math.Min(inH, iyBegin + rowsPerBlock);
                                if (iyBegin >= iyEnd)
                                    continue;

                                float* xBatch = x0 + (long)b * inChannels * inPlane;
                                long outBatch = (long)b * outChannels * outPlane;
                                float* wKy = w0 + (long)ky * kW * inPerGroup;

                                for (int iy = iyBegin; iy < iyEnd; iy++)
                                {
                                    float* xRow = xBatch + (long)iy * inW;
                                    float* yRow = y0 + outBatch + (long)(iy * sH + ky) * outW;

                                    for (int kx = 0; kx < kW; kx++)
                                    {
                                        float* dst = lb + (long)kx * inW;
                                        for (int ix = 0; ix < inW; ix++)
                                            dst[ix] = bias;
                                    }

                                    for (int ci = 0; ci < inPerGroup; ci++)
                                    {
                                        float* src = xRow + (long)ci * inPlane;

                                        for (int kx = 0; kx < kW; kx++)
                                        {
                                            float w = wKy[(long)kx * inPerGroup + ci];
                                            if (w == 0f)
                                                continue;

                                            float* dst = lb + (long)kx * inW;
                                            int ix = 0;

                                            if (Avx2.IsSupported)
                                            {
                                                var vw = Vector256.Create(w);
                                                for (; ix + Vector256<float>.Count <= inW; ix += Vector256<float>.Count)
                                                {
                                                    var s = Avx.LoadVector256(src + ix);
                                                    var a = Avx.LoadVector256(dst + ix);
                                                    Avx.Store(dst + ix, Avx.Add(a, Avx.Multiply(vw, s)));
                                                }
                                            }

                                            for (; ix < inW; ix++)
                                                dst[ix] += w * src[ix];
                                        }
                                    }

                                    for (int ix = 0; ix < inW; ix++)
                                    {
                                        int ox = ix * sW;
                                        for (int kx = 0; kx < kW; kx++)
                                            yRow[ox + kx] = lb[(long)kx * inW + ix];
                                    }
                                }
                            }
                        }
                    }
                    finally
                    {
                        ArrayPool<float>.Shared.Return(lineBuf);
                    }
                });
                return;
            }

            // 多通道、kernel 小、行能整行进 L1/L2：同样按 ci 外层 + 行缓冲（每 co 一条输出行），
            // 把「逐像素散写 co 个平面」换成「每行连续写」。FPN 里 16→16 的 2× 上采样走这条。
            if (kW == sW && (long)outPerGroup * kW * inW <= (1 << 16))
            {
                int tiledTasks = batches * kH * blocks;
                int bufStride = kW * inW;

                ParallelRange.For(tiledTasks, (begin, end) =>
                {
                    float[] tileBuf = ArrayPool<float>.Shared.Rent(outPerGroup * bufStride);
                    try
                    {
                        fixed (float* tb = tileBuf)
                        {
                            for (int task = begin; task < end; task++)
                            {
                                int block = task % blocks;
                                int rest = task / blocks;
                                int ky = rest % kH;
                                int b = rest / kH;

                                int iyBegin = block * rowsPerBlock;
                                int iyEnd = Math.Min(inH, iyBegin + rowsPerBlock);
                                if (iyBegin >= iyEnd)
                                    continue;

                                float* xBatch = x0 + (long)b * inChannels * inPlane;
                                long outBatch = (long)b * outChannels * outPlane;

                                for (int iy = iyBegin; iy < iyEnd; iy++)
                                {
                                    float* xRow = xBatch + (long)iy * inW;
                                    float* yRows = y0 + outBatch + (long)(iy * sH + ky) * outW;

                                    for (int co = 0; co < outPerGroup; co++)
                                    {
                                        float bv = biasArray is null ? 0f : biasArray[co];
                                        float* dstBase = tb + (long)co * bufStride;
                                        for (int kx = 0; kx < kW; kx++)
                                        {
                                            float* dst = dstBase + (long)kx * inW;
                                            for (int ix = 0; ix < inW; ix++)
                                                dst[ix] = bv;
                                        }
                                    }

                                    for (int ci = 0; ci < inPerGroup; ci++)
                                    {
                                        float* src = xRow + (long)ci * inPlane;

                                        for (int kx = 0; kx < kW; kx++)
                                        {
                                            float* wTap = w0 + (long)((ky * kW + kx) * inPerGroup + ci) * outPerGroup;

                                            for (int co = 0; co < outPerGroup; co++)
                                            {
                                                float w = wTap[co];
                                                if (w == 0f)
                                                    continue;

                                                float* dst = tb + (long)(co * kW + kx) * inW;
                                                int ix = 0;

                                                if (Avx2.IsSupported)
                                                {
                                                    var vw = Vector256.Create(w);
                                                    for (; ix + Vector256<float>.Count <= inW; ix += Vector256<float>.Count)
                                                    {
                                                        var s = Avx.LoadVector256(src + ix);
                                                        var a = Avx.LoadVector256(dst + ix);
                                                        Avx.Store(dst + ix, Avx.Add(a, Avx.Multiply(vw, s)));
                                                    }
                                                }

                                                for (; ix < inW; ix++)
                                                    dst[ix] += w * src[ix];
                                            }
                                        }
                                    }

                                    for (int co = 0; co < outPerGroup; co++)
                                    {
                                        float* dstBase = tb + (long)co * bufStride;
                                        float* yRow = yRows + (long)co * outPlane;
                                        for (int ix = 0; ix < inW; ix++)
                                        {
                                            int ox = ix * sW;
                                            for (int kx = 0; kx < kW; kx++)
                                                yRow[ox + kx] = dstBase[(long)kx * inW + ix];
                                        }
                                    }
                                }
                            }
                        }
                    }
                    finally
                    {
                        ArrayPool<float>.Shared.Return(tileBuf);
                    }
                });
                return;
            }

            int tasks = batches * kH * blocks;

            ParallelRange.For(tasks, (begin, end) =>
            {
                float[] accBuf = ArrayPool<float>.Shared.Rent(outPerGroup);
                var accSpan = new Span<float>(accBuf, 0, outPerGroup);
                try
                {
                    fixed (float* acc = accBuf)
                    {
                        for (int task = begin; task < end; task++)
                        {
                            int block = task % blocks;
                            int rest = task / blocks;
                            int ky = rest % kH;
                            int b = rest / kH;

                            int iyBegin = block * rowsPerBlock;
                            int iyEnd = Math.Min(inH, iyBegin + rowsPerBlock);
                            if (iyBegin >= iyEnd)
                                continue;

                            float* xBatch = x0 + (long)b * inChannels * inPlane;
                            long outBatch = (long)b * outChannels * outPlane;
                            float* wKy = w0 + (long)ky * kW * inPerGroup * outPerGroup;

                            for (int iy = iyBegin; iy < iyEnd; iy++)
                            {
                                long outRow = outBatch + (long)(iy * sH + ky) * outW;
                                long inRow = (long)iy * inW;

                                for (int ix = 0; ix < inW; ix++)
                                {
                                    for (int kx = 0; kx < kW; kx++)
                                    {
                                        if (biasArray is null)
                                            accSpan.Clear();
                                        else
                                            biasArray.AsSpan(0, outPerGroup).CopyTo(accSpan);

                                        float* wTap = wKy + (long)kx * inPerGroup * outPerGroup;
                                        for (int ci = 0; ci < inPerGroup; ci++)
                                        {
                                            float v = xBatch[(long)ci * inPlane + inRow + ix];
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

                                        float* target = y0 + outRow + (long)(ix * sW + kx);
                                        for (int co = 0; co < outPerGroup; co++)
                                            target[(long)co * outPlane] = acc[co];
                                    }
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

    /// <summary>
    /// 按「平面 × 行块」二维切分并行，回调参数是 <c>(平面, 行起点, 行终点)</c>。
    /// <para>
    /// 池化 / 上采样这类算子只有 16~64 个平面（1×16×H×W），而 <see cref="ParallelRange.For"/>
    /// 在任务数少于线程数时会整段串行 —— 于是在 32 线程的机器上它们其实一直是单线程在跑。
    /// 这里把行也切开，让任务数至少是线程数的两倍。
    /// </para>
    /// </summary>
    private static void ForPlanes(int planes, int rows, Action<int, int, int> body)
    {
        int threads = ParallelRange.Threads;

        int blocks = planes >= threads
            ? 1
            : Math.Min((threads * 2 + planes - 1) / planes, Math.Max(1, rows / 8));

        if (blocks <= 1)
        {
            ParallelRange.For(planes, (begin, end) =>
            {
                for (int t = begin; t < end; t++)
                    body(t, 0, rows);
            });

            return;
        }

        int chunk = (rows + blocks - 1) / blocks;

        ParallelRange.ForEach(planes * blocks, task =>
        {
            int t = task / blocks;
            int oy0 = (task % blocks) * chunk;
            int oy1 = Math.Min(oy0 + chunk, rows);

            if (oy0 < oy1)
                body(t, oy0, oy1);
        });
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

            // 2×2 / stride 2 / 无 padding 的下采样池化（YuNet 的 4 个 MaxPool 全是这形状）：
            // 一次算 8 个输出 —— 两次 8 宽加载 → 去交织出偶/奇列 → 再与相邻行取 max。
            if (input.DType == DType.F32 && kH == 2 && kW == 2 && sH == 2 && sW == 2
                && dH == 1 && dW == 1 && pT == 0 && pL == 0 && Avx.IsSupported)
            {
                ForPlanes(batches * channels, outH, (t, oy0, oy1) =>
                    MaxPool2x2(input.AsSpan<float>(), output.AsSpan<float>(), t, inH, inW, outH, outW, oy0, oy1));

                return;
            }

            // 2×2 / stride 1 的滑动窗口池化（DB 检测的 MaxPool.0：1×16×144×448 → 1×16×144×448）：
            // 竖向取两行 max 到行缓冲 → 横向取相邻两列 max，两级都是 8 宽无分支。
            if (FastPool && input.DType == DType.F32 && kH == 2 && kW == 2 && sH == 1 && sW == 1
                && dH == 1 && dW == 1 && pT >= 0 && pT <= 1 && pL >= 0 && pL <= 1
                && Avx2.IsSupported && inW >= 8)
            {
                ForPlanes(batches * channels, outH, (t, oy0, oy1) =>
                {
                    float[] tmp = ArrayPool<float>.Shared.Rent(inW + 16);
                    try
                    {
                        unsafe
                        {
                            fixed (float* ps = input.AsSpan<float>())
                            fixed (float* pd = output.AsSpan<float>())
                            fixed (float* pt = tmp)
                            {
                                MaxPool2x2Stride1(ps, pd, pt, t, inH, inW, outH, outW, pT, pL, oy0, oy1);
                            }
                        }
                    }
                    finally
                    {
                        ArrayPool<float>.Shared.Return(tmp);
                    }
                });

                return;
            }

            // 3×3 / stride 2 的下采样池化：
            // 走「竖向 3 行取 max 到行缓冲 + 横向去交织」的两段内核。原来的通用路径在这形状上
            // 因为 sW ≠ 1 拿不到向量化分支，只能逐输出点做 9 次带边界的标量比较。
            if (FastPool && input.DType == DType.F32
                && kH == 3 && kW == 3 && sH == 2 && sW == 2
                && dH == 1 && dW == 1 && pT >= 0 && pT <= 1 && pL >= 0 && pL <= 1
                && Avx2.IsSupported && inW >= 16)
            {
                ForPlanes(batches * channels, outH, (t, oy0, oy1) =>
                {
                    // 行缓冲：前置 16 个哨兵 + inW 个数据 + 尾部 24 个哨兵，
                    // 横向 8 宽加载和 -inf 填充都靠它吸收，内层循环不用判越界。
                    int stride = MaxPoolRowPad * 2 + inW + 32;
                    float[] row = ArrayPool<float>.Shared.Rent(stride);
                    try
                    {
                        unsafe
                        {
                            fixed (float* ps = input.AsSpan<float>())
                            fixed (float* pd = output.AsSpan<float>())
                            fixed (float* pr = row)
                            {
                                MaxPool3x3Stride2(ps, pd, pr, t, inH, inW, outH, outW, pT, pL, oy0, oy1);
                            }
                        }
                    }
                    finally
                    {
                        ArrayPool<float>.Shared.Return(row);
                    }
                });

                return;
            }

            // 量化张量的池化：同一 scale/zp 下 max 单调，所以可以直接对整数取 max（ORT 同做法）
            if (input.DType is DType.U8 or DType.I8)
            {
                bool signed = input.DType == DType.I8;

                ParallelRange.For(batches * channels, (begin, end) =>
                {
                    if (signed)
                    {
                        var sbytes = input.AsSpan<sbyte>();
                        var dbytes = output.AsSpan<sbyte>();
                        for (int t = begin; t < end; t++)
                            PoolQuantized(sbytes, dbytes, t, inH, inW, outH, outW, kH, kW, sH, sW, dH, dW, pT, pL);
                    }
                    else
                    {
                        var sbytes = input.AsSpan<byte>();
                        var dbytes = output.AsSpan<byte>();
                        for (int t = begin; t < end; t++)
                            PoolQuantizedBytes(sbytes, dbytes, t, inH, inW, outH, outW, kH, kW, sH, sW, dH, dW, pT, pL);
                    }
                });

                return;
            }


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

        /// <summary>
        /// 2×2 / stride 2 / 无 padding 的池化（一次 8 个输出）：
        /// 两个 8 宽加载覆盖 16 个输入，<c>vshufps</c> 拆成偶列/奇列后取 max，再与下一行合并。
        /// </summary>
        private static void MaxPool2x2(
            ReadOnlySpan<float> source,
            Span<float> target,
            int t,
            int inH,
            int inW,
            int outH,
            int outW,
            int oyBegin,
            int oyEnd)
        {
            long inBase = (long)t * inH * inW;
            long outBase = (long)t * outH * outW;

            var evenIndex = Vector256.Create(0, 1, 4, 5, 2, 3, 6, 7);

            for (int oy = oyBegin; oy < oyEnd; oy++)
            {
                int y0 = oy * 2;
                long row0 = inBase + (long)y0 * inW;
                long row1 = y0 + 1 < inH ? inBase + (long)(y0 + 1) * inW : -1;
                long outRow = outBase + (long)oy * outW;

                int ox = 0;
                for (; ox + 8 <= outW && ox * 2 + 16 <= inW; ox += 8)
                {
                    int x = ox * 2;
                    var a0 = Vector256.LoadUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(source[(int)(row0 + x)..]));
                    var a1 = Vector256.LoadUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(source[(int)(row0 + x + 8)..]));

                    var even = Avx2.PermuteVar8x32(Avx.Shuffle(a0, a1, 0x88), evenIndex);
                    var odd = Avx2.PermuteVar8x32(Avx.Shuffle(a0, a1, 0xDD), evenIndex);
                    var best = Vector256.Max(even, odd);

                    if (row1 >= 0)
                    {
                        var b0 = Vector256.LoadUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(source[(int)(row1 + x)..]));
                        var b1 = Vector256.LoadUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(source[(int)(row1 + x + 8)..]));

                        var evenB = Avx2.PermuteVar8x32(Avx.Shuffle(b0, b1, 0x88), evenIndex);
                        var oddB = Avx2.PermuteVar8x32(Avx.Shuffle(b0, b1, 0xDD), evenIndex);
                        best = Vector256.Max(best, Vector256.Max(evenB, oddB));
                    }

                    best.StoreUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(target[(int)(outRow + ox)..]));
                }

                for (; ox < outW; ox++)
                {
                    int x = ox * 2;
                    float best = source[(int)(row0 + x)];
                    if (x + 1 < inW)
                        best = MathF.Max(best, source[(int)(row0 + x + 1)]);
                    if (row1 >= 0)
                    {
                        best = MathF.Max(best, source[(int)(row1 + x)]);
                        if (x + 1 < inW)
                            best = MathF.Max(best, source[(int)(row1 + x + 1)]);
                    }

                    target[(int)(outRow + ox)] = best;
                }
            }
        }

        /// <summary>3×3/s2 池化的行缓冲前置哨兵长度（见 <see cref="MaxPool3x3Stride2"/>）。</summary>
        private const int MaxPoolRowPad = 16;

        /// <summary>
        /// 2×2 / stride 1 的滑动窗口池化（<c>pad ∈ {0,1}</c>）：竖向把两行 max 写进行缓冲，
        /// 横向再把相邻两列 max 取出来 —— 两级都是 8 宽、循环里没有任何判断。
        /// <para>
        /// 行缓冲末尾多留一个 <c>-inf</c>（<c>tmp[inW]</c>），右边界那一路窗口自然落空。
        /// </para>
        /// </summary>
        private static unsafe void MaxPool2x2Stride1(
            float* src,
            float* dst,
            float* row,
            int t,
            int inH,
            int inW,
            int outH,
            int outW,
            int pT,
            int pL,
            int oyBegin,
            int oyEnd)
        {
            long inBase = (long)t * inH * inW;
            long outBase = (long)t * outH * outW;

            row[inW] = float.NegativeInfinity;

            for (int oy = oyBegin; oy < oyEnd; oy++)
            {
                // pad 行用最靠内的行顶替：那一行本身就在窗口里，取 max 不变。
                int y0 = Math.Max(oy - pT, 0);
                int y1 = Math.Min(oy + 1 - pT, inH - 1);

                float* r0 = src + inBase + (long)y0 * inW;
                float* r1 = src + inBase + (long)y1 * inW;

                int x = 0;
                for (; x <= inW - 8; x += 8)
                    Avx.Store(row + x, Avx.Max(Avx.LoadVector256(r0 + x), Avx.LoadVector256(r1 + x)));

                for (; x < inW; x++)
                    row[x] = MathF.Max(r0[x], r1[x]);

                float* outRow = dst + outBase + (long)oy * outW;

                if (pL == 0)
                {
                    x = 0;
                    for (; x <= outW - 8; x += 8)
                        Avx.Store(outRow + x, Avx.Max(Avx.LoadVector256(row + x), Avx.LoadVector256(row + x + 1)));

                    for (; x < outW; x++)
                        outRow[x] = MathF.Max(row[x], row[x + 1]);
                }
                else
                {
                    // 左 pad：首列只有右半边有值
                    outRow[0] = row[0];

                    x = 1;
                    for (; x <= outW - 8; x += 8)
                        Avx.Store(outRow + x, Avx.Max(Avx.LoadVector256(row + x - 1), Avx.LoadVector256(row + x)));

                    for (; x < outW; x++)
                        outRow[x] = MathF.Max(row[x - 1], row[x]);
                }
            }
        }

        /// <summary>
        /// 3×3 / stride 2 / dilation 1 的池化（<c>pad ∈ {0,1}</c>）。
        /// <para>
        /// 两段式：先竖向把窗口内的 3 行按 8 宽取 max 写进行缓冲，再横向把行缓冲按
        /// 「偶列 / 奇列」去交织 —— 第 k 个输出 = <c>max(even[k], odd[k], even[k+1])</c>，
        /// 一次算 8 个输出。
        /// </para>
        /// <para>
        /// <paramref name="row"/> 的布局：<c>[0, Pad)</c> 与 <c>[Pad + inW, ...)</c> 是 <c>-inf</c> 哨兵，
        /// 数据放在 <c>[Pad, Pad + inW)</c>。所以窗口越界与 pad 区域都自然得到 <c>-inf</c>，内层无分支。
        /// </para>
        /// </summary>
        private static unsafe void MaxPool3x3Stride2(
            float* src,
            float* dst,
            float* row,
            int t,
            int inH,
            int inW,
            int outH,
            int outW,
            int pT,
            int pL,
            int oyBegin,
            int oyEnd)
        {
            const int Pad = MaxPoolRowPad;

            long inBase = (long)t * inH * inW;
            long outBase = (long)t * outH * outW;

            float* data = row + Pad;

            row[Pad - 1] = float.NegativeInfinity;
            for (int i = 0; i < 24; i++)
                data[inW + i] = float.NegativeInfinity;

            var evenIndex = Vector256.Create(0, 1, 4, 5, 2, 3, 6, 7);
            var shiftIndex = Vector256.Create(1, 2, 3, 4, 5, 6, 7, 0);

            for (int oy = oyBegin; oy < oyEnd; oy++)
            {
                // 越界的行用最靠内的行顶替：pad 区域贡献 -inf，被顶替的行本身就在窗口里，取 max 不变。
                int y0 = Math.Max(2 * oy - pT, 0);
                int y1 = 2 * oy;
                int y2 = Math.Min(2 * oy + 1, inH - 1);

                float* r0 = src + inBase + (long)y0 * inW;
                float* r1 = src + inBase + (long)y1 * inW;
                float* r2 = src + inBase + (long)y2 * inW;

                int x = 0;
                for (; x <= inW - 8; x += 8)
                {
                    var v = Avx.Max(Avx.LoadVector256(r0 + x), Avx.LoadVector256(r1 + x));
                    v = Avx.Max(v, Avx.LoadVector256(r2 + x));
                    Avx.Store(data + x, v);
                }

                for (; x < inW; x++)
                    data[x] = MathF.Max(src[inBase + (long)y0 * inW + x],
                        MathF.Max(src[inBase + (long)y1 * inW + x], src[inBase + (long)y2 * inW + x]));

                float* outRow = dst + outBase + (long)oy * outW;

                int j = 0;
                for (; j <= outW - 8; j += 8)
                {
                    // 窗口起点（相对行缓冲）= 2j - pL：pL = 1 时首组取到 data[-1] 的哨兵，正好是 -inf。
                    float* b = data + 2 * j - pL;
                    var l0 = Avx.LoadVector256(b);
                    var l1 = Avx.LoadVector256(b + 8);

                    var even = Avx2.PermuteVar8x32(Avx.Shuffle(l0, l1, 0x88), evenIndex);
                    var odd = Avx2.PermuteVar8x32(Avx.Shuffle(l0, l1, 0xDD), evenIndex);

                    // even[k+1]：组内左移一格，末位补 b[16]（即下一组的 even[0]）
                    var next = Avx2.PermuteVar8x32(even, shiftIndex);
                    next = Avx.Blend(next, Avx.BroadcastScalarToVector256(b + 16), 0x80);

                    var best = Avx.Max(Avx.Max(even, odd), next);
                    Avx.Store(outRow + j, best);
                }

                for (; j < outW; j++)
                {
                    float* b = data + 2 * j - pL;
                    outRow[j] = MathF.Max(b[0], MathF.Max(b[1], b[2]));
                }
            }
        }

        /// <summary>
        /// 量化张量的池化（u8，一次比 32 个输出）：直接比整数，不做反量化 ——
        /// 同一 scale/zp 下 <c>max</c> 与「反量化后取 max」等价。
        /// </summary>
        private static void PoolQuantizedBytes(
            ReadOnlySpan<byte> source,
            Span<byte> target,
            int t,
            int inH,
            int inW,
            int outH,
            int outW,
            int kH,
            int kW,
            int sH,
            int sW,
            int dH,
            int dW,
            int pT,
            int pL)
        {
            long inBase = (long)t * inH * inW;
            long outBase = (long)t * outH * outW;

            // 与 F32 路径同一套判据：sW = 1、dW = 1 且窗口整块落在输入内 → 32 个输出一起算
            bool vectorizable = dW == 1 && sW == 1;
            int vStart = vectorizable ? Math.Max(0, pL) : 0;
            int vLimit = vectorizable
                ? Math.Min(outW, inW - kW - (Vector256<byte>.Count - 1) + pL)
                : 0;

            for (int oy = 0; oy < outH; oy++)
            {
                int y0 = oy * sH - pT;

                for (int ox = 0; ox < outW; ox++)
                {
                    if (vectorizable && ox >= vStart && ox + Vector256<byte>.Count <= vLimit)
                    {
                        int vx = ox - pL;
                        var acc = Vector256<byte>.Zero;

                        for (int ky = 0; ky < kH; ky++)
                        {
                            int y = y0 + ky * dH;
                            if ((uint)y >= (uint)inH)
                                continue;

                            long rowBase = inBase + (long)y * inW + vx;
                            for (int kx = 0; kx < kW; kx++)
                            {
                                var value = Vector256.LoadUnsafe(
                                    ref System.Runtime.InteropServices.MemoryMarshal.GetReference(source[(int)(rowBase + kx)..]));
                                acc = Vector256.Max(acc, value);
                            }
                        }

                        acc.StoreUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(
                            target[(int)(outBase + (long)oy * outW + ox)..]));
                        ox += Vector256<byte>.Count - 1;
                        continue;
                    }

                    int x0 = ox * sW - pL;
                    byte best = 0;
                    bool any = false;

                    for (int ky = 0; ky < kH; ky++)
                    {
                        int y = y0 + ky * dH;
                        if ((uint)y >= (uint)inH)
                            continue;

                        long rowBase = inBase + (long)y * inW;
                        for (int kx = 0; kx < kW; kx++)
                        {
                            int x = x0 + kx * dW;
                            if ((uint)x >= (uint)inW)
                                continue;

                            byte value = source[(int)(rowBase + x)];
                            if (!any || value > best)
                            {
                                best = value;
                                any = true;
                            }
                        }
                    }

                    target[(int)(outBase + (long)oy * outW + ox)] = best;
                }
            }
        }

        /// <summary>
        /// 量化张量的一次池化（<c>u8</c>/<c>i8</c> 通用）：直接比整数，不做反量化 ——
        /// 同一 scale/zp 下 <c>max</c> 与「反量化后取 max」等价。
        /// </summary>
        private static void PoolQuantized<T>(
            ReadOnlySpan<T> source,
            Span<T> target,
            int t,
            int inH,
            int inW,
            int outH,
            int outW,
            int kH,
            int kW,
            int sH,
            int sW,
            int dH,
            int dW,
            int pT,
            int pL)
            where T : IComparable<T>
        {
            long inBase = (long)t * inH * inW;
            long outBase = (long)t * outH * outW;

            for (int oy = 0; oy < outH; oy++)
            {
                int y0 = oy * sH - pT;

                for (int ox = 0; ox < outW; ox++)
                {
                    int x0 = ox * sW - pL;
                    T best = default!;
                    bool any = false;

                    for (int ky = 0; ky < kH; ky++)
                    {
                        int y = y0 + ky * dH;
                        if ((uint)y >= (uint)inH)
                            continue;

                        long rowBase = inBase + (long)y * inW;
                        for (int kx = 0; kx < kW; kx++)
                        {
                            int x = x0 + kx * dW;
                            if ((uint)x >= (uint)inW)
                                continue;

                            T value = source[(int)(rowBase + x)];
                            if (!any || value.CompareTo(best) > 0)
                            {
                                best = value;
                                any = true;
                            }
                        }
                    }

                    target[(int)(outBase + (long)oy * outW + ox)] = best;
                }
            }
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

            // GlobalAveragePool（SE 模块的压缩步）：窗口覆盖整个平面、输出 1×1，
            // 通用路径每个元素都要算四个边界半径，只有 60% 的单线程带宽。
            if (outH == 1 && outW == 1 && kH == inH && kW == inW && sH == inH && sW == inW
                && pT == 0 && pL == 0 && dH == 1 && dW == 1)
            {
                GlobalMeanPerPlane(input.AsSpan<float>(), output.AsSpan<float>(), batches * channels, inH, inW);
                return;
            }

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

        /// <summary>
        /// 整平面求均值（GlobalAveragePool）。每平面 <paramref name="rows"/>×<paramref name="cols"/> 个元素，
        /// 按「平面 × 行块」切分：SE 模块只有 16~64 个平面，平面数少于线程数时只切平面等于串行。
        /// 行内用两个累加器展开，读一遍就到带宽上限。
        /// </summary>
        private static unsafe void GlobalMeanPerPlane(
            ReadOnlySpan<float> source, Span<float> target, int planes, int rows, int cols)
        {
            fixed (float* ps = source)
            fixed (float* pd = target)
            {
                IntPtr src = (IntPtr)ps;
                IntPtr dst = (IntPtr)pd;

                int count = rows * cols;
                int threads = ParallelRange.Threads;

                // 平面数已经够铺满线程时不必再切行（部分和合并也要钱）。
                int blocks = planes >= threads
                    ? 1
                    : Math.Max(1, Math.Min(threads / Math.Max(1, planes), rows / 4));

                if (blocks <= 1)
                {
                    ParallelRange.For(planes, (begin, end) =>
                    {
                        for (int t = begin; t < end; t++)
                            ((float*)dst)[t] = PlaneSum((float*)src + (long)t * count, count) / count;
                    });

                    return;
                }

                int chunk = (rows + blocks - 1) / blocks;
                var partial = new float[planes * blocks];

                ParallelRange.ForEach(planes * blocks, task =>
                {
                    int t = task / blocks;
                    int y0 = (task % blocks) * chunk;
                    int y1 = Math.Min(y0 + chunk, rows);

                    partial[task] = y0 < y1
                        ? PlaneSum((float*)src + (long)t * count + (long)y0 * cols, (y1 - y0) * cols)
                        : 0f;
                });

                for (int t = 0; t < planes; t++)
                {
                    float sum = 0;
                    for (int b = 0; b < blocks; b++)
                        sum += partial[t * blocks + b];

                    ((float*)dst)[t] = sum / count;
                }
            }
        }

        /// <summary>一段连续 float 的和（两路 AVX 累加，尾部标量）。</summary>
        private static unsafe float PlaneSum(float* p, int n)
        {
            float sum = 0;
            int i = 0;

            if (Avx.IsSupported)
            {
                var acc0 = Vector256<float>.Zero;
                var acc1 = Vector256<float>.Zero;

                for (; i <= n - 16; i += 16)
                {
                    acc0 = Avx.Add(acc0, Avx.LoadVector256(p + i));
                    acc1 = Avx.Add(acc1, Avx.LoadVector256(p + i + 8));
                }

                for (; i <= n - 8; i += 8)
                    acc0 = Avx.Add(acc0, Avx.LoadVector256(p + i));

                sum = Vector256.Sum(Avx.Add(acc0, acc1));
            }

            for (; i < n; i++)
                sum += p[i];

            return sum;
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

            // 量化张量（u8/i8）只支持 nearest：整数插值没有意义，量化图（YuNet-int8）也只用 nearest。
            // 元素是 1 字节，拷贝与符号无关，所以 u8/i8 共用一条 byte 通道。
            if (input.DType is DType.U8 or DType.I8)
            {
                if (linear)
                    throw new OnnxSharpException($"Resize '{Name}': linear mode on {input.DType} is not supported");

                ForPlanes(batches * channels, outH, (t, oy0, oy1) =>
                    ResizeNearestTable(input.AsSpan<byte>(), output.AsSpan<byte>(), t, inH, inW, outW,
                        mapY, mapX, oy0, oy1));

                return;
            }


            // nearest 且恰好 2× 上采样（DB 的 FPN 六次 Resize 全是这一形状）：
            // 每 8 个输入元素展开成 16 个输出，不再逐点查 mapX。
            if (FastResize2x && !linear && outH == inH * 2 && outW == inW * 2
                && Avx2.IsSupported && IsDoubleMap(mapY, mapX))
            {
                ForPlanes(batches * channels, outH, (t, oy0, oy1) =>
                    ResizeNearest2x(input.AsSpan<float>(), output.AsSpan<float>(), t, inW, outH, outW, oy0, oy1));

                return;
            }

            ForPlanes(batches * channels, outH, (t, oy0, oy1) =>
            {
                if (linear)
                {
                    ResizeBilinear(input.AsSpan<float>(), output.AsSpan<float>(), t, inH, inW, outW,
                        mapY, mapX, wyTab!, wxTab!, oy0, oy1);
                }
                else
                {
                    ResizeNearestTable(input.AsSpan<float>(), output.AsSpan<float>(), t, inH, inW, outW,
                        mapY, mapX, oy0, oy1);
                }
            });
        }

        /// <summary>按查表取最近邻的一层（float / u8 共用）。</summary>
        private static void ResizeNearestTable<T>(
            ReadOnlySpan<T> source,
            Span<T> target,
            int t,
            int inH,
            int inW,
            int outW,
            int[] mapY,
            int[] mapX,
            int oyBegin,
            int oyEnd)
            where T : unmanaged
        {
            int outH = mapY.Length;
            long inBase = (long)t * inH * inW;
            long outBase = (long)t * outH * outW;

            for (int oy = oyBegin; oy < oyEnd; oy++)
            {
                long rowIn = inBase + (long)mapY[oy] * inW;
                long rowOut = outBase + (long)oy * outW;

                for (int ox = 0; ox < outW; ox++)
                    target[(int)(rowOut + ox)] = source[(int)(rowIn + mapX[ox])];
            }
        }

        /// <summary>双线性插值的一层：行内两个权重来自 <paramref name="wyTab"/> / <paramref name="wxTab"/>。</summary>
        private static void ResizeBilinear(
            ReadOnlySpan<float> source,
            Span<float> target,
            int t,
            int inH,
            int inW,
            int outW,
            int[] mapY,
            int[] mapX,
            double[] wyTab,
            double[] wxTab,
            int oyBegin,
            int oyEnd)
        {
            int outH = mapY.Length;
            long inBase = (long)t * inH * inW;
            long outBase = (long)t * outH * outW;

            for (int oy = oyBegin; oy < oyEnd; oy++)
            {
                int y0 = mapY[oy];
                int y1 = Math.Min(y0 + 1, inH - 1);
                double wy = wyTab[oy];
                double iwy = 1 - wy;

                long topBase = inBase + (long)y0 * inW;
                long bottomBase = inBase + (long)y1 * inW;
                long rowOut = outBase + (long)oy * outW;

                for (int ox = 0; ox < outW; ox++)
                {
                    int x0 = mapX[ox];
                    int x1 = Math.Min(x0 + 1, inW - 1);
                    double wx = wxTab[ox];
                    double iwx = 1 - wx;

                    double top = source[(int)(topBase + x0)] * iwx + source[(int)(topBase + x1)] * wx;
                    double bottom = source[(int)(bottomBase + x0)] * iwx + source[(int)(bottomBase + x1)] * wx;

                    target[(int)(rowOut + ox)] = (float)(top * iwy + bottom * wy);
                }
            }
        }

        /// <summary>映射表是不是纯粹的「2 倍上采样」（<c>map[i] == i / 2</c>）。</summary>
        private static bool IsDoubleMap(int[] mapY, int[] mapX)
        {
            for (int i = 0; i < mapY.Length; i++)
            {
                if (mapY[i] != i / 2)
                    return false;
            }

            for (int i = 0; i < mapX.Length; i++)
            {
                if (mapX[i] != i / 2)
                    return false;
            }

            return true;
        }

        /// <summary>
        /// nearest 2× 上采样的一层：行顺序由 <c>mapY = oy / 2</c> 自动给出，
        /// 行内每 8 个输入元素用两次 <c>vpermd</c> 展开成 16 个输出。
        /// </summary>
        private static unsafe void ResizeNearest2x(
            ReadOnlySpan<float> source,
            Span<float> target,
            int t,
            int inW,
            int outH,
            int outW,
            int oyBegin,
            int oyEnd)
        {
            int inH = outH / 2;
            long inBase = (long)t * inH * inW;
            long outBase = (long)t * outH * outW;

            var lowIndex = Vector256.Create(0, 0, 1, 1, 2, 2, 3, 3);
            var highIndex = Vector256.Create(4, 4, 5, 5, 6, 6, 7, 7);

            fixed (float* ps = source)
            fixed (float* pd = target)
            {
                for (int oy = oyBegin; oy < oyEnd; oy++)
                {
                    float* srcRow = ps + inBase + (long)(oy / 2) * inW;
                    float* outRow = pd + outBase + (long)oy * outW;

                    int x = 0;
                    for (; x <= inW - 8; x += 8)
                    {
                        var v = Avx.LoadVector256(srcRow + x);
                        Avx.Store(outRow + 2 * x, Avx2.PermuteVar8x32(v, lowIndex));
                        Avx.Store(outRow + 2 * x + 8, Avx2.PermuteVar8x32(v, highIndex));
                    }

                    for (; x < inW; x++)
                    {
                        float value = srcRow[x];
                        outRow[2 * x] = value;
                        outRow[2 * x + 1] = value;
                    }
                }
            }
        }
    }

    // ---------------- Softmax ----------------

    /// <summary>图融合用：读取 <c>Softmax</c> 的原始 <c>axis</c>（可能为负）。</summary>
    internal interface ISoftmaxAxis
    {
        long Axis { get; }
    }

    private sealed class SoftmaxLayer : Layer, ISoftmaxAxis
    {
        private readonly long _axis;

        public SoftmaxLayer(LayerParams parameters)
        {
            Type = parameters.Type;
            Name = parameters.Name;
            _axis = parameters.GetInt("axis", -1);
        }

        public long Axis => _axis;

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

    // ---------------- LogSoftmax ----------------

    /// <summary>
    /// <c>LogSoftmax(axis)</c> = <c>x - max - log(Σ exp(x - max))</c>。
    /// Paraformer 的 <c>logits</c> 就是 decoder 输出过一层 LogSoftmax（每个 token 一行）。
    /// </summary>
    private sealed class LogSoftmaxLayer : Layer
    {
        private readonly long _axis;

        public LogSoftmaxLayer(LayerParams parameters)
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
            OpUtil.NormalizeAxis(_axis, inShapes[0].Rank, $"LogSoftmax '{Name}'");
            outShapes.Add(inShapes[0]);
        }

        public override void Forward(
            IReadOnlyList<Tensor> inputs,
            IReadOnlyList<Tensor> outputs,
            IReadOnlyList<Tensor> temps)
        {
            var input = inputs[0];
            var output = outputs[0];

            int axis = OpUtil.NormalizeAxis(_axis, input.Shape.Rank, $"LogSoftmax '{Name}'");
            var dims = input.Shape.ToArray();

            int outer = OpUtil.Product(dims.AsSpan(0, axis));
            int inner = axis + 1 < dims.Length ? OpUtil.Product(dims.AsSpan(axis + 1)) : 1;
            int length = dims[axis];

            if (outer <= 0 || length <= 0)
                return;

            var source = input.AsSpan<float>();
            var target = output.AsSpan<float>();

            // axis 是最后一维（decoder 的 logits 就是 [1,21,8359]）时每行连续：
            // 用 VectorMath 的向量化 exp 一次算完整行，外层的 double Exp（每元素 20ns 级）省掉。
            if (inner == 1 && VectorMath.Available)
            {
                unsafe
                {
                    fixed (float* ps = source)
                    fixed (float* pt = target)
                    {
                        Span<float> lanes = stackalloc float[8];

                        for (int o = 0; o < outer; o++)
                        {
                            float* row = ps + (long)o * length;
                            float* dstRow = pt + (long)o * length;

                            int k = 0;
                            var vMax = Vector256.Create(float.NegativeInfinity);
                            for (; k + 8 <= length; k += 8)
                                vMax = Vector256.Max(vMax, Vector256.Load(row + k));

                            vMax.CopyTo(lanes);
                            float max = float.NegativeInfinity;
                            for (int i = 0; i < lanes.Length; i++)
                                max = MathF.Max(max, lanes[i]);
                            for (; k < length; k++)
                                max = MathF.Max(max, row[k]);

                            // exp(x - max) 写进输出并求和的向量化版本
                            float sum = VectorMath.SumExp(row, dstRow, length, -max);
                            float logSum = MathF.Log(sum) + max;

                            var vLog = Vector256.Create(logSum);
                            k = 0;
                            for (; k + 8 <= length; k += 8)
                                Avx.Subtract(Vector256.Load(row + k), vLog).Store(dstRow + k);
                            for (; k < length; k++)
                                dstRow[k] = row[k] - logSum;
                        }
                    }
                }

                return;
            }

            for (int o = 0; o < outer; o++)
            {
                for (int i = 0; i < inner; i++)
                {
                    long baseIndex = (long)o * length * inner + i;

                    float max = float.NegativeInfinity;
                    for (int k = 0; k < length; k++)
                        max = MathF.Max(max, source[(int)(baseIndex + (long)k * inner)]);

                    double sum = 0;
                    for (int k = 0; k < length; k++)
                        sum += Math.Exp(source[(int)(baseIndex + (long)k * inner)] - max);

                    float logSum = MathF.Log((float)sum) + max;
                    for (int k = 0; k < length; k++)
                        target[(int)(baseIndex + (long)k * inner)] = source[(int)(baseIndex + (long)k * inner)] - logSum;
                }
            }
        }
    }


    // ---------------- MatMul ----------------

    private sealed class MatMulLayer : Layer, IBiasAddable
    {
        /// <summary>
        /// 逐算子把形状和耗时打到 stderr（<c>ONNXSHARP_MM_TRACE=1</c>）。
        /// 用来核对 profile 的 per-op 数字与内核微基准是否自洽。
        /// </summary>
        private static readonly bool TraceEnabled =
            Environment.GetEnvironmentVariable("ONNXSHARP_MM_TRACE") == "1";

        /// <summary>
        /// 转置视图（<c>gemm_view</c>，由 <see cref="GraphFusion"/> 折进来）：12 个整数
        /// <c>[outer, inner, m, k, n, aOuter, aRow, aBatch, bOuter, bRow, bCol, bBatch]</c>。
        /// <para>
        /// attention 的 <c>[B,T,H,D] → Transpose[0,2,1,3] → [B,H,T,D]</c> 折进 GEMM 后，
        /// 层拿到的仍然是<b>未转置</b>的源张量：A 的每行仍是 D 个连续 float，只是行距变成 H*D；
        /// B（K^T）则是「行距 1、列距 H*D」的转置读法。省掉每层 3 次 65 KB 的整块搬运。
        /// </para>
        /// </summary>
        private readonly long[] _view;

        public MatMulLayer(LayerParams parameters)
        {
            Type = parameters.Type;
            Name = parameters.Name;
            _view = parameters.GetInts("gemm_view") ?? Array.Empty<long>();
        }

        /// <summary>由 <see cref="GraphFusion"/> 折进来的 bias add（见 <see cref="IBiasAddable"/>）。</summary>
        public Tensor? FusedBias { get; set; }

        /// <summary>常量 B 的预打包面板（按权重张量实例缓存；非 const 输入不会走这里）。</summary>
        private GemmKernel.PackedB? _packedWeights;
        private Tensor? _packedSource;

        private GemmKernel.PackedB GetPackedWeights(ReadOnlySpan<float> weights, int k, int n, Tensor source)
        {
            if (_packedWeights is not null && ReferenceEquals(_packedSource, source)
                && _packedWeights.K == k && _packedWeights.N == n)
                return _packedWeights;

            _packedWeights = GemmKernel.Pack(weights, k, n);
            _packedSource = source;
            return _packedWeights;
        }

        public override void GetMemoryShapes(
            IReadOnlyList<TensorShape> inShapes,
            int requiredOutputs,
            List<TensorShape> outShapes,
            List<TensorShape> tempShapes)
        {
            var a = inShapes[0];
            var b = inShapes[1];

            // 转置视图：输入是未转置的 [B,T,H,D]，逻辑上是 [B,H,T,D] × [B,H,D,T]。
            if (_view.Length == 12)
            {
                outShapes.Add(new TensorShape(new[]
                {
                    (int)_view[0], (int)_view[1], (int)_view[2], (int)_view[4],
                }));
                return;
            }

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

            bool trace = TraceEnabled;
            var swTrace = trace ? System.Diagnostics.Stopwatch.StartNew() : null;

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

            // 折进来的 bias add：图上 Add(·, [n]) 是**按列**加的，长度必须正好是 n
            var columnBias = FusedBias is { } fusedBias && fusedBias.Count == n && n > 0
                ? fusedBias.AsSpan<float>()
                : default;

            // 转置视图：源张量没被物化，按 stride 直接读（见构造函数注释）。
            // 外层 batch（图像/样本那维）逐次派发，内层 head 一次摊平给批量内核。
            if (_view.Length == 12)
            {
                int outer = (int)_view[0];
                int inner = (int)_view[1];
                int vm = (int)_view[2];
                int vk = (int)_view[3];
                int vn = (int)_view[4];
                long aOuter = _view[5], aRow = _view[6], aBatchStride = _view[7];
                long bOuter = _view[8], bRow = _view[9], bCol = _view[10], bBatchStride = _view[11];
                int cBatch = vm * vn;

                for (int o = 0; o < outer; o++)
                {
                    GemmKernel.MultiplyBatched(
                        srcA.Slice((int)(o * aOuter)),
                        srcB.Slice((int)(o * bOuter)),
                        dst.Slice(o * inner * cBatch, inner * cBatch),
                        inner, vm, vk, vn,
                        columnBias: columnBias,
                        aStride: (int)aRow, aBatchStride: (int)aBatchStride,
                        bColStride: (int)bCol, bBatchStride: (int)bBatchStride, bStride: (int)bRow);
                }

                if (trace)
                    Console.Error.WriteLine(
                        $"MMv m={vm} k={vk} n={vn} bat={outer * inner} aRow={aRow} aBs={aBatchStride} " +
                        $"bRow={bRow} bCol={bCol} bBs={bBatchStride} ms={swTrace!.Elapsed.TotalMilliseconds:F3} {Name}");
                return;
            }

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

                // 常量权重（rf-detr / YOLO / LLM 的投影层都是）：打包一次，每帧零拷贝。
                // 非真实常量时不能缓存（输出缓冲会被复用），所以用 IsConstInput 把关。
                if (GemmKernel.PrepackEnabled && IsConstInput(1))
                {
                    var packed = GetPackedWeights(srcB.Slice(0, k * n), k, n, b);
                    GemmKernel.MultiplyPacked(
                        srcA.Slice(0, rows * k),
                        packed,
                        dst.Slice(0, rows * n),
                        rows,
                        columnBias: columnBias);
                    if (trace)
                        Console.Error.WriteLine(
                            $"MMp m={rows} k={k} n={n} bat={batches} ms={swTrace!.Elapsed.TotalMilliseconds:F3} {Name}");
                    return;
                }

                GemmKernel.Multiply(
                    srcA.Slice(0, rows * k),
                    srcB.Slice(0, k * n),
                    dst.Slice(0, rows * n),
                    rows, k, n,
                    columnBias: columnBias);
                if (trace)
                    Console.Error.WriteLine(
                        $"MMs m={rows} k={k} n={n} bat={batches} ms={swTrace!.Elapsed.TotalMilliseconds:F3} {Name}");
                return;
            }

            // 两个操作数的 batch 维都与输出完全一致时（注意力 QK^T / P·V：[4,6,257,64]×[4,6,64,257]），
            // 每个 batch 的偏移就是连续的 bi*m*k / bi*k*n —— 可以交给批量内核**一次派发**，
            // 而不是 24 次「各自开一个并行域、17 个列块摊给 4 线程」。
            // rf-detr 实测（交替 A/B，min of 3）：238.1 vs 249.6 ms，1.048×。
            bool densePerBatch = true;
            for (int i = 0; i < batchRank; i++)
            {
                if (aBatch[i] != outBatch[i] || bBatch[i] != outBatch[i])
                {
                    densePerBatch = false;
                    break;
                }
            }

            long aTotal = (long)batches * m * k;
            long bTotal = (long)batches * k * n;
            long cTotal = (long)batches * m * n;

            if (densePerBatch && aTotal <= int.MaxValue && bTotal <= int.MaxValue && cTotal <= int.MaxValue)
            {
                GemmKernel.MultiplyBatched(
                    srcA.Slice(0, (int)aTotal),
                    srcB.Slice(0, (int)bTotal),
                    dst.Slice(0, (int)cTotal),
                    (int)batches, m, k, n,
                    columnBias: columnBias);
                if (trace)
                    Console.Error.WriteLine(
                        $"MMb m={m} k={k} n={n} bat={batches} ms={swTrace!.Elapsed.TotalMilliseconds:F3} " +
                        $"sameAb={(ReferenceEquals(a.Buffer, b.Buffer) ? 1 : 0)}sameAc={(ReferenceEquals(a.Buffer, output.Buffer) ? 1 : 0)} " +
                        $"offA={a.ByteOffset} offB={b.ByteOffset} offC={output.ByteOffset} " +
                        $"lenA={a.Buffer.Length} lenC={output.Buffer.Length} {Name}");
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
                    m, k, n,
                    columnBias: columnBias);

                for (int i = batchRank - 1; i >= 0; i--)
                {
                    if (++coords[i] < outBatch[i])
                        break;
                    coords[i] = 0;
                }
            }

            if (trace)
                Console.Error.WriteLine(
                    $"MMl m={m} k={k} n={n} bat={batches} ms={swTrace!.Elapsed.TotalMilliseconds:F3} {Name}");
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
