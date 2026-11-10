using System.Buffers;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using OnnxSharp.Core;
using OnnxSharp.Kernels;
using OnnxSharp.Tensors;

namespace OnnxSharp.Ops;

/// <summary>
/// 图融合产物：<c>DQL → MatMulInteger → Cast → Mul(a_scale·w_scale) → [Add(bias)]</c> 折成一层，
/// 对齐 ONNXRuntime 的 <c>DynamicQuantizeMatMul</c>。
/// <para>
/// 省掉的是：u8 激活的物化与回读、int32 结果的物化与 <c>Cast</c>、以及 <c>Mul/Add</c> 两趟全量读写。
/// 现在是一条链：算 min/max → 边量化边展开成 F32（不落 u8）→ GEMM → epilogue 里乘 scale、加 bias。
/// </para>
/// <para>
/// 数值与原算子链**逐位一致**：整数累加在 F32 里是精确的（已用 k=1280 的用例验证），
/// epilogue 的乘加顺序也照抄（<c>acc * (a_scale * w_scale) + bias</c>）。
/// </para>
/// </summary>
internal static class QuantizedMatMulOps
{
    public static void Register()
        => LayerRegistry.Register("DynamicQuantizeMatMul", p => new DynamicQuantizeMatMulLayer(p));

    private sealed class DynamicQuantizeMatMulLayer : Layer
    {
        /// <summary>分段计时的开关（<c>ONNXSHARP_DQM_BREAKDOWN=1</c>，配合 <c>tools/scripts/dqm_breakdown.py</c>）。</summary>
        private static readonly bool Breakdown =
            Environment.GetEnvironmentVariable("ONNXSHARP_DQM_BREAKDOWN") is { Length: > 0 };

        // (w - w_zp) 的 F32 版本 + 预打包面板（常量权重只算一次）
        private float[]? _floatWeights;
        private GemmKernel.PackedB? _packedWeights;
        private Tensor? _weightSource;
        private int[]? _weightZeroPoints;

        // 每列合成 scale = a_scale * w_scale[col]（复用一个缓冲）
        private float[]? _combinedScale;

        public DynamicQuantizeMatMulLayer(LayerParams parameters)
        {
            Type = parameters.Type;
            Name = parameters.Name;
        }

        private static (int M, int K, int N) Resolve(TensorShape a, TensorShape w)
        {
            if (w.Rank != 2)
                throw new OnnxSharpException($"DynamicQuantizeMatMul '{w}': weight must be 2-D");

            int k = a[a.Rank - 1];
            int n = w[1];
            if (w[0] != k)
                throw new OnnxSharpException($"DynamicQuantizeMatMul: K mismatch ({a} x {w})");

            long m = 1;
            for (int i = 0; i < a.Rank - 1; i++)
                m *= a[i];

            return ((int)m, k, n);
        }

        public override void GetMemoryShapes(
            IReadOnlyList<TensorShape> inShapes,
            int requiredOutputs,
            List<TensorShape> outShapes,
            List<TensorShape> tempShapes)
        {
            var a = inShapes[0];
            var w = inShapes[1];
            var (_, _, n) = Resolve(a, w);

            var dims = a.ToArray();
            dims[^1] = n;
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
            var a = inputs[0];
            var w = inputs[1];
            var output = outputs[0];

            var (m, k, n) = Resolve(a.Shape, w.Shape);

            // 输入顺序：[A, W, w_zero_point, w_scale, bias?]
            var weightZeroPoints = Inputs.Count > 2 && Inputs[2].Idx != 0
                ? OpUtil.ReadInts(inputs[2])
                : [0];

            var weightScale = Inputs.Count > 3 && Inputs[3].Idx != 0
                ? inputs[3].AsSpan<float>()
                : [1f];

            var bias = Inputs.Count > 4 && Inputs[4].Idx != 0
                ? inputs[4].AsSpan<float>()
                : default;

            // 常量权重：默认走「预打包面板」（每帧零拷贝），`ONNXSHARP_NO_PREPACK=1` 退回旧路径
            var packed = GemmKernel.PrepackEnabled ? GetPackedWeights(w, k, n, weightZeroPoints) : null;
            var weights = packed is null ? GetFloatWeights(w, k, n, weightZeroPoints) : null;

            var clock = System.Diagnostics.Stopwatch.StartNew();
            long t0 = 0, t1 = 0, t2 = 0, t3 = 0;

            // ---- 1) 与原 DQL 相同的 per-tensor 动态量化参数
            var source = a.AsSpan<float>();
            var (min, max) = Quantizer.MinMax(source);
            t1 = clock.ElapsedTicks;

            float scale = (max - min) / 255f;
            if (!(scale > 0f))
                scale = 1f;

            int zeroPoint = Math.Clamp((int)MathF.Round(-min / scale, MidpointRounding.ToEven), 0, 255);

            // ---- 2) 每列合成 scale（权重是 per-channel 量化的）
            CombinedScale(scale, weightScale, n, out var combined);

            // ---- 3) 边量化边展开成 F32（不物化 u8）
            float[] activationBuffer = ArrayPool<float>.Shared.Rent(m * k);
            float[]? accumulatorBuffer = packed is null ? ArrayPool<float>.Shared.Rent(m * n) : null;

            try
            {
                var activation = activationBuffer.AsSpan(0, m * k);
                QuantizeAndWiden(source, activation, scale, zeroPoint);
                t2 = clock.ElapsedTicks;

                if (packed is not null)
                {
                    // 反量化 epilogue 折进 GEMM 的写出：out = acc * combined[col] + bias[col]
                    GemmKernel.MultiplyPacked(activation, packed, output.AsSpan<float>(), m,
                        columnBias: bias, columnScale: combined);
                    t3 = clock.ElapsedTicks;
                }
                else
                {
                    var accumulator = accumulatorBuffer!.AsSpan(0, m * n);
                    GemmKernel.Multiply(activation, weights!, accumulator, m, k, n);
                    t3 = clock.ElapsedTicks;
                    Epilogue(accumulator, output.AsSpan<float>(), bias, combined, n);
                }

                // ---- 4) epilogue：乘合成 scale、加 bias
                long t4 = clock.ElapsedTicks;

                if (Breakdown)
                {
                    double tick = 1e3 / System.Diagnostics.Stopwatch.Frequency;
                    Console.Error.WriteLine(
                        $"[dqm] m={m} k={k} n={n}  minmax {(t1) * tick:F3} quant {(t2 - t1) * tick:F3} " +
                        $"gemm {(t3 - t2) * tick:F3} epi {(t4 - t3) * tick:F3} ms");
                }
            }
            finally
            {
                ArrayPool<float>.Shared.Return(activationBuffer);
                if (accumulatorBuffer is not null)
                    ArrayPool<float>.Shared.Return(accumulatorBuffer);
            }
        }

        private void CombinedScale(float activationScale, ReadOnlySpan<float> weightScale, int n, out ReadOnlySpan<float> combined)
        {
            _combinedScale ??= new float[Math.Max(n, 1)];
            if (_combinedScale.Length < n)
                _combinedScale = new float[n];

            if (weightScale.Length == 1)
            {
                float value = activationScale * weightScale[0];
                Array.Fill(_combinedScale, value, 0, n);
            }
            else
            {
                for (int i = 0; i < n; i++)
                    _combinedScale[i] = activationScale * weightScale[i];
            }

            combined = _combinedScale.AsSpan(0, n);
        }

        private float[] GetFloatWeights(Tensor weight, int k, int n, int[] zeroPoints)
        {
            if (_floatWeights is not null && ReferenceEquals(_weightSource, weight) && ZeroPointsEqual(_weightZeroPoints, zeroPoints))
                return _floatWeights;

            var buffer = new float[checked(k * n)];
            if (weight.DType == DType.I8)
                Int8F32Gemm.WidenWeights(weight.AsSpan<sbyte>(), buffer, k, n, zeroPoints);
            else
                Int8F32Gemm.WidenWeights(weight.AsSpan<byte>(), buffer, k, n, zeroPoints);

            _floatWeights = buffer;
            _weightSource = weight;
            _weightZeroPoints = zeroPoints;
            return buffer;
        }

        /// <summary>
        /// <c>(w - w_zp)</c> 展开成 F32 后**打包成列块面板**（常量权重只做一次）。
        /// <para>
        /// 之后每帧的 GEMM 直接读面板，省掉 <see cref="GemmKernel.Multiply"/> 每次都要做的
        /// 「B 重排成 <c>[k, 16]</c> 紧凑面板」——ASR encoder 上那是 160 次 × 3.2 MB。
        /// 面板与 F32 权重同量级，所以不额外占内存（展开的整份 F32 会丢弃）。
        /// </para>
        /// </summary>
        private GemmKernel.PackedB GetPackedWeights(Tensor weight, int k, int n, int[] zeroPoints)
        {
            if (_packedWeights is not null && ReferenceEquals(_weightSource, weight) && ZeroPointsEqual(_weightZeroPoints, zeroPoints))
                return _packedWeights;

            var widened = new float[checked(k * n)];
            if (weight.DType == DType.I8)
                Int8F32Gemm.WidenWeights(weight.AsSpan<sbyte>(), widened, k, n, zeroPoints);
            else
                Int8F32Gemm.WidenWeights(weight.AsSpan<byte>(), widened, k, n, zeroPoints);

            _packedWeights = GemmKernel.Pack(widened, k, n);
            _floatWeights = null;
            _weightSource = weight;
            _weightZeroPoints = zeroPoints;
            return _packedWeights;
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

        /// <summary><c>q = clamp(round_half_even(x / scale) + zp)</c>，写出 <c>q - zp</c>（整数小值）。</summary>
        private static void QuantizeAndWiden(ReadOnlySpan<float> source, Span<float> destination, float scale, int zeroPoint)
        {
            float inverse = 1f / scale;
            var vInverse = Vector256.Create(inverse);
            var vZero = Vector256.Create((float)zeroPoint);
            var vMin = Vector256<float>.Zero;
            var vMax = Vector256.Create(255f);

            int i = 0;
            if (Avx2.IsSupported)
            {
                for (; i + 8 <= source.Length; i += 8)
                {
                    var x = Vector256.LoadUnsafe(ref MemoryMarshal.GetReference(source[i..]));
                    var q = Avx.RoundToNearestInteger(Avx.Multiply(x, vInverse));
                    q = Avx.Add(q, vZero);
                    q = Avx.Min(vMax, Avx.Max(vMin, q));

                    Avx.Subtract(q, vZero).StoreUnsafe(ref destination[i]);
                }
            }

            for (; i < source.Length; i++)
            {
                float q = MathF.Round(source[i] * inverse, MidpointRounding.ToEven) + zeroPoint;
                destination[i] = Math.Clamp(q, 0f, 255f) - zeroPoint;
            }
        }

        /// <summary><c>out = acc * scale[col] + bias[col]</c>（per-column scale / 可选 bias）。</summary>
        private static void Epilogue(
            ReadOnlySpan<float> accumulator,
            Span<float> destination,
            ReadOnlySpan<float> bias,
            ReadOnlySpan<float> scale,
            int n)
        {
            bool hasBias = bias.Length >= n;
            int stride = scale.Length >= n ? n : 1;      // scale 允许是标量

            for (int offset = 0; offset < accumulator.Length; offset += n)
            {
                int column = 0;

                if (Avx.IsSupported)
                {
                    for (; column + 8 <= n; column += 8)
                    {
                        int index = offset + column;
                        var acc = Vector256.LoadUnsafe(ref MemoryMarshal.GetReference(accumulator[index..]));
                        var vs = Vector256.LoadUnsafe(ref MemoryMarshal.GetReference(scale[(stride == 1 ? 0 : column)..]));
                        var value = Avx.Multiply(acc, vs);

                        if (hasBias)
                            value = Avx.Add(value, Vector256.LoadUnsafe(ref MemoryMarshal.GetReference(bias[column..])));

                        value.StoreUnsafe(ref MemoryMarshal.GetReference(destination[index..]));
                    }
                }

                for (; column < n; column++)
                {
                    int index = offset + column;
                    float value = accumulator[index] * scale[stride == 1 ? 0 : column];
                    destination[index] = hasBias ? value + bias[column] : value;
                }
            }
        }
    }
}
