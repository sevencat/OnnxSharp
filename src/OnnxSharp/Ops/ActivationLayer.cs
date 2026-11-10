using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using OnnxSharp.Core;
using OnnxSharp.Tensors;

namespace OnnxSharp.Ops;

/// <summary>一元逐元素算子（Relu/Sigmoid/Sqrt/Exp/Neg/Abs）。</summary>
public sealed class ActivationLayer : Layer
{
    private readonly UnaryOpKind _op;

    private ActivationLayer(LayerParams parameters, UnaryOpKind op)
    {
        Type = parameters.Type;
        Name = parameters.Name;
        _op = op;
    }

    public static void Register()
    {
        LayerRegistry.Register("Relu", p => new ActivationLayer(p, UnaryOpKind.Relu));
        LayerRegistry.Register("Sigmoid", p => new ActivationLayer(p, UnaryOpKind.Sigmoid));
        LayerRegistry.Register("Sqrt", p => new ActivationLayer(p, UnaryOpKind.Sqrt));
        LayerRegistry.Register("Exp", p => new ActivationLayer(p, UnaryOpKind.Exp));
        LayerRegistry.Register("Neg", p => new ActivationLayer(p, UnaryOpKind.Neg));
        LayerRegistry.Register("Abs", p => new ActivationLayer(p, UnaryOpKind.Abs));
        LayerRegistry.Register("Erf", p => new ActivationLayer(p, UnaryOpKind.Erf));
        LayerRegistry.Register("Sin", p => new ActivationLayer(p, UnaryOpKind.Sin));
        LayerRegistry.Register("Cos", p => new ActivationLayer(p, UnaryOpKind.Cos));
        LayerRegistry.Register("Floor", p => new ActivationLayer(p, UnaryOpKind.Floor));
        LayerRegistry.Register("Ceil", p => new ActivationLayer(p, UnaryOpKind.Ceil));
        LayerRegistry.Register("Log", p => new ActivationLayer(p, UnaryOpKind.Log));
        LayerRegistry.Register("Tanh", p => new ActivationLayer(p, UnaryOpKind.Tanh));
        LayerRegistry.Register("Reciprocal", p => new ActivationLayer(p, UnaryOpKind.Reciprocal));
        LayerRegistry.Register("Sign", p => new ActivationLayer(p, UnaryOpKind.Sign));
        LayerRegistry.Register("Round", p => new ActivationLayer(p, UnaryOpKind.Round));

        LayerRegistry.Register("Clip", p => new ClipLayer(p));
        LayerRegistry.Register("HardSigmoid", p => new HardSigmoidLayer(p));

        // 不是 ONNX 算子：图级融合（Sigmoid → Mul）的产物，见 GraphFusion。
        LayerRegistry.Register("Silu", p => new ActivationLayer(p, UnaryOpKind.Silu));
    }

    // ---------------- Clip ----------------

    /// <summary>
    /// 逐元素夹取。opset ≤ 6 走 min/max 属性，opset ≥ 11 走第 2/3 个输入（可选）。
    /// yolov12 的区域注意力和 RT-DETR 的位置编码都会用。
    /// </summary>
    private sealed class ClipLayer : Layer
    {
        private readonly float _minAttr;
        private readonly float _maxAttr;

        public ClipLayer(LayerParams parameters)
        {
            Type = parameters.Type;
            Name = parameters.Name;
            _minAttr = (float)parameters.GetFloat("min", double.NegativeInfinity);
            _maxAttr = (float)parameters.GetFloat("max", double.PositiveInfinity);
        }

        private static float Scalar(Tensor t, string what, string name) => t.DType switch
        {
            DType.F32 => t.AsSpan<float>()[0],
            DType.I64 => t.AsSpan<long>()[0],
            _ => throw new OnnxSharpException($"Clip '{name}': unsupported {what} dtype {t.DType}"),
        };

        public override void GetMemoryShapes(
            IReadOnlyList<TensorShape> inShapes,
            int requiredOutputs,
            List<TensorShape> outShapes,
            List<TensorShape> tempShapes)
            => outShapes.Add(inShapes[0]);

        public override unsafe void Forward(
            IReadOnlyList<Tensor> inputs,
            IReadOnlyList<Tensor> outputs,
            IReadOnlyList<Tensor> temps)
        {
            // min / max 是**可选**输入：ONNX 用空串占位，importer 会把它变成 Arg(0)。
            // 直接按 inputs.Count 判断会把「缺失」当成「值为 0 的张量」——
            // SAM3 的 Clip(x, min=1) 就是这么被截成 0 的，随后 Div 除零 → ∞ → NaN。
            float min = Inputs.Count > 1 && Inputs[1].Idx != 0 ? Scalar(inputs[1], "min", Name) : _minAttr;
            float max = Inputs.Count > 2 && Inputs[2].Idx != 0 ? Scalar(inputs[2], "max", Name) : _maxAttr;

            var src = inputs[0].AsSpan<float>();
            var dst = outputs[0].AsSpan<float>();

            if (!Avx.IsSupported)
            {
                for (int i = 0; i < dst.Length; i++)
                    dst[i] = MathF.Min(max, MathF.Max(min, src[i]));
                return;
            }

            fixed (float* p = src)
            fixed (float* q = dst)
            {
                var vmin = Vector256.Create(min);
                var vmax = Vector256.Create(max);

                int i = 0;
                for (; i + Vector256<float>.Count <= src.Length; i += Vector256<float>.Count)
                    Avx.Store(q + i, Vector256.Min(vmax, Vector256.Max(vmin, Avx.LoadVector256(p + i))));

                for (; i < src.Length; i++)
                    q[i] = MathF.Min(max, MathF.Max(min, p[i]));
            }
        }
    }

    // ---------------- HardSigmoid ----------------

    /// <summary>
    /// <c>y = max(0, min(1, alpha * x + beta))</c>。PaddleOCR（PP-OCRv6）的 det / rec 骨干里
    /// 用它代替 swish 的一部分，共 18 处；alpha 在 0.1667 与 0.2 两种之间变化，必须读属性。
    /// </summary>
    private sealed class HardSigmoidLayer : Layer
    {
        private readonly float _alpha;
        private readonly float _beta;

        public HardSigmoidLayer(LayerParams parameters)
        {
            Type = parameters.Type;
            Name = parameters.Name;
            _alpha = (float)parameters.GetFloat("alpha", 0.2);
            _beta = (float)parameters.GetFloat("beta", 0.5);
        }

        public override void GetMemoryShapes(
            IReadOnlyList<TensorShape> inShapes,
            int requiredOutputs,
            List<TensorShape> outShapes,
            List<TensorShape> tempShapes)
            => outShapes.Add(inShapes[0]);

        public override void Forward(
            IReadOnlyList<Tensor> inputs,
            IReadOnlyList<Tensor> outputs,
            IReadOnlyList<Tensor> temps)
        {
            var src = inputs[0].AsSpan<float>();
            var dst = outputs[0].AsSpan<float>();

            int i = 0;
            if (Fma.IsSupported)
            {
                var va = Vector256.Create(_alpha);
                var vb = Vector256.Create(_beta);
                var zero = Vector256<float>.Zero;
                var one = Vector256.Create(1f);

                for (; i + Vector256<float>.Count <= src.Length; i += Vector256<float>.Count)
                {
                    var v = Fma.MultiplyAdd(Vector256.LoadUnsafe(ref src[i]), va, vb);
                    v = Vector256.Min(one, Vector256.Max(zero, v));
                    Vector256.StoreUnsafe(v, ref dst[i]);
                }
            }

            for (; i < src.Length; i++)
                dst[i] = MathF.Min(1f, MathF.Max(0f, _alpha * src[i] + _beta));
        }
    }

    public override void GetMemoryShapes(
        IReadOnlyList<TensorShape> inShapes,
        int requiredOutputs,
        List<TensorShape> outShapes,
        List<TensorShape> tempShapes)
    {
        if (inShapes.Count != 1)
            throw new OnnxSharpException($"'{Type}' expects 1 input, got {inShapes.Count}");
        outShapes.Add(inShapes[0]);
    }

    public override void Forward(
        IReadOnlyList<Tensor> inputs,
        IReadOnlyList<Tensor> outputs,
        IReadOnlyList<Tensor> temps)
        => TensorOps.ElementWiseUnary(_op, inputs[0], outputs[0]);
}
