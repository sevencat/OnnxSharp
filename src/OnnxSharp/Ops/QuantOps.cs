using System.Buffers;
using OnnxSharp.Core;
using OnnxSharp.Kernels;
using OnnxSharp.Tensors;

namespace OnnxSharp.Ops;

/// <summary>
/// 量化算子：<c>DynamicQuantizeLinear</c> 与 <c>MatMulInteger</c>（M10-03/06）。
/// <para>
/// 这两个是 ORT 旧式量化（QOperator）最常见的组合：sherpa 的 Paraformer 里有
/// 222 个 <c>MatMulInteger</c> + 211 个 <c>DynamicQuantizeLinear</c>。
/// </para>
/// </summary>
internal static class QuantOps
{
    public static void Register()
    {
        LayerRegistry.Register("DynamicQuantizeLinear", p => new DynamicQuantizeLinearLayer(p));
        LayerRegistry.Register("MatMulInteger", p => new MatMulIntegerLayer(p));
    }

    /// <summary>
    /// <c>MatMulInteger</c> 是否用「F32 GEMM + 权重展开缓存」而不是 <see cref="Int8Gemm"/> 的标量 int32 内核。
    /// <para>
    /// 默认：<b>没有 VNNI 时开启</b>（标量 int32 只有 ~0.9 GFLOPS，同样形状的 F32 GEMM 有 73 GFLOPS）。
    /// 有 VNNI 时关掉 —— 那时 <c>vpdpbusd</c> 既更快也是逐位精确的整数路径。
    /// 环境变量 <c>ONNXSHARP_INT8_F32GEMM=0/1</c> 可强制关闭/开启（对照实验用）。
    /// </para>
    /// </summary>
    internal static readonly bool PreferFloatWeights = ResolvePreferFloatWeights();

    private static bool ResolvePreferFloatWeights()
    {
        string? setting = Environment.GetEnvironmentVariable("ONNXSHARP_INT8_F32GEMM");
        if (setting is "0")
            return false;
        if (setting is "1")
            return true;

        return Int8Kernel.Tier == Int8KernelTier.Scalar;
    }

    // ---------------- DynamicQuantizeLinear ----------------

    /// <summary>
    /// <c>DynamicQuantizeLinear(x)</c> → <c>y(u8), y_scale(f32 标量), y_zero_point(u8 标量)</c>。
    /// 按 ONNX 规范：
    /// <code>
    /// x_min = min(min(x), 0); x_max = max(max(x), 0)
    /// y_scale = (x_max - x_min) / 255
    /// y_zero_point = round(clamp(-x_min / y_scale, 0, 255))
    /// y = saturate(round(x / y_scale) + y_zero_point)
    /// </code>
    /// </summary>
    private sealed class DynamicQuantizeLinearLayer : Layer
    {
        public DynamicQuantizeLinearLayer(LayerParams parameters)
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
            outShapes.Add(inShapes[0]);                    // y
            if (requiredOutputs > 1)
                outShapes.Add(TensorShape.Scalar);         // y_scale
            if (requiredOutputs > 2)
                outShapes.Add(TensorShape.Scalar);         // y_zero_point
        }

        public override void GetTypes(
            IReadOnlyList<DType> inTypes,
            int requiredOutputs,
            int requiredTemps,
            List<DType> outTypes,
            List<DType> tempTypes)
        {
            outTypes.Clear();
            if (requiredOutputs > 0) outTypes.Add(DType.U8);
            if (requiredOutputs > 1) outTypes.Add(DType.F32);
            if (requiredOutputs > 2) outTypes.Add(DType.U8);

            tempTypes.Clear();
            for (int i = 0; i < requiredTemps; i++)
                tempTypes.Add(DType.F32);
        }

        public override void Forward(
            IReadOnlyList<Tensor> inputs,
            IReadOnlyList<Tensor> outputs,
            IReadOnlyList<Tensor> temps)
        {
            var source = inputs[0].AsSpan<float>();

            var (min, max) = Quantizer.MinMax(source);

            float scale = (max - min) / 255f;
            if (!(scale > 0f))
                scale = 1f;                     // 全 0 输入：退化成 scale=1、zp=0

            int zeroPoint = Math.Clamp(
                (int)MathF.Round(-min / scale, MidpointRounding.ToEven), 0, 255);

            Quantizer.QuantizeToUInt8(source, outputs[0].AsSpan<byte>(), scale, zeroPoint);

            if (outputs.Count > 1)
                outputs[1].AsSpan<float>()[0] = scale;
            if (outputs.Count > 2)
                outputs[2].AsSpan<byte>()[0] = (byte)zeroPoint;
        }
    }

    // ---------------- MatMulInteger ----------------

    /// <summary>
    /// <c>MatMulInteger(A, B, a_zero_point?, b_zero_point?)</c> → <c>int32 [m, n]</c>。
    /// <para>支持 2-D 与 batch=1 的 3-D（内部当 2-D）；权重按 <see cref="PackedInt8Weights"/> 只打包一次。</para>
    /// </summary>
    private sealed class MatMulIntegerLayer : Layer
    {
        private PackedInt8Weights? _packed;
        private Tensor? _packedSource;
        private int[]? _packedZeroPoints;

        private float[]? _floatWeights;
        private Tensor? _floatSource;
        private int[]? _floatZeroPoints;

        public MatMulIntegerLayer(LayerParams parameters)
        {
            Type = parameters.Type;
            Name = parameters.Name;
        }

        private static (int M, int K, int N) Resolve(IReadOnlyList<TensorShape> inShapes)
        {
            var a = inShapes[0];
            var b = inShapes[1];

            int aRank = a.Rank == 3 && a[0] == 1 ? 2 : a.Rank;
            int bRank = b.Rank == 3 && b[0] == 1 ? 2 : b.Rank;
            if (aRank != 2 || bRank != 2)
                throw new OnnxSharpException(
                    $"MatMulInteger: only 2-D (or batch=1 3-D) is supported, got {a} x {b}");

            int m = a[a.Rank - 2];
            int k = a[a.Rank - 1];
            int n = b[b.Rank - 1];
            if (b[b.Rank - 2] != k)
                throw new OnnxSharpException($"MatMulInteger: K mismatch ({a} x {b})");

            return (m, k, n);
        }

        public override void GetMemoryShapes(
            IReadOnlyList<TensorShape> inShapes,
            int requiredOutputs,
            List<TensorShape> outShapes,
            List<TensorShape> tempShapes)
        {
            var (m, _, n) = Resolve(inShapes);
            bool batched = inShapes[0].Rank == 3 || inShapes[1].Rank == 3;
            outShapes.Add(batched ? new TensorShape(1, m, n) : new TensorShape(m, n));
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
                outTypes.Add(DType.I32);

            tempTypes.Clear();
            for (int i = 0; i < requiredTemps; i++)
                tempTypes.Add(DType.F32);
        }

        public override void Forward(
            IReadOnlyList<Tensor> inputs,
            IReadOnlyList<Tensor> outputs,
            IReadOnlyList<Tensor> temps)
        {
            var a = inputs[0];
            var b = inputs[1];

            var shapes = new List<TensorShape>(2) { a.Shape, b.Shape };
            var (m, k, n) = Resolve(shapes);

            int aZeroPoint = ReadScalar(inputs, 2);
            int[] bZeroPoints = ReadVector(inputs, 3);

            if (QuantOps.PreferFloatWeights)
            {
                var weights = GetFloatWeights(b, k, n, bZeroPoints);
                float[] buffer = ArrayPool<float>.Shared.Rent(m * k);

                try
                {
                    var activation = buffer.AsSpan(0, m * k);
                    if (a.DType == DType.I8)
                        Int8F32Gemm.WidenActivation(a.AsSpan<sbyte>(), activation, aZeroPoint);
                    else
                        Int8F32Gemm.WidenActivation(a.AsSpan<byte>(), activation, aZeroPoint);

                    Int8F32Gemm.MultiplyInteger(activation, weights, outputs[0].AsSpan<int>(), m, k, n);
                }
                finally
                {
                    ArrayPool<float>.Shared.Return(buffer);
                }

                return;
            }

            PackedInt8Weights packed = GetPacked(b, k, n, bZeroPoints);

            ReadOnlySpan<byte> aBytes = a.DType == DType.I8
                ? System.Runtime.InteropServices.MemoryMarshal.Cast<sbyte, byte>(a.AsSpan<sbyte>())
                : a.AsSpan<byte>();

            Int8Gemm.MultiplyInteger(aBytes, packed, outputs[0].AsSpan<int>(), m, aZeroPoint);
        }

        /// <summary>
        /// 权重的 F32 展开缓存：常量权重只展开一次（体积是 int8 的 4 倍，但换来 80× 的吞吐）。
        /// </summary>
        private float[] GetFloatWeights(Tensor weight, int k, int n, int[] zeroPoints)
        {
            if (_floatWeights is not null && ReferenceEquals(_floatSource, weight) && ZeroPointsEqual(_floatZeroPoints, zeroPoints))
                return _floatWeights;

            var buffer = new float[checked(k * n)];
            if (weight.DType == DType.I8)
                Int8F32Gemm.WidenWeights(weight.AsSpan<sbyte>(), buffer, k, n, zeroPoints);
            else
                Int8F32Gemm.WidenWeights(weight.AsSpan<byte>(), buffer, k, n, zeroPoints);

            _floatWeights = buffer;
            _floatSource = weight;
            _floatZeroPoints = zeroPoints;
            return buffer;
        }

        private static int ReadScalar(IReadOnlyList<Tensor> inputs, int index)
        {
            if (inputs.Count <= index || inputs[index].Count == 0)
                return 0;

            int[] values = OpUtil.ReadInts(inputs[index]);
            return values.Length > 0 ? values[0] : 0;
        }

        private static int[] ReadVector(IReadOnlyList<Tensor> inputs, int index)
            => inputs.Count <= index || inputs[index].Count == 0 ? [0] : OpUtil.ReadInts(inputs[index]);

        private PackedInt8Weights GetPacked(Tensor weight, int k, int n, int[] zeroPoints)
        {
            if (_packed is not null && ReferenceEquals(_packedSource, weight) && ZeroPointsEqual(_packedZeroPoints, zeroPoints))
                return _packed;

            PackedInt8Weights packed = weight.DType == DType.I8
                ? PackedInt8Weights.PackSigned(weight.AsSpan<sbyte>(), k, n, zeroPoints)
                : PackedInt8Weights.Pack(weight.AsSpan<byte>(), k, n, zeroPoints);

            _packed = packed;
            _packedSource = weight;
            _packedZeroPoints = zeroPoints;
            return packed;
        }

        private static bool ZeroPointsEqual(int[]? a, int[] b)
        {
            if (a is null || a.Length != b.Length)
                return false;
            for (int i = 0; i < a.Length; i++)
            {
                if (a[i] != b[i])
                    return false;
            }

            return true;
        }
    }
}
