using OnnxSharp.Core;
using OnnxSharp.Tensors;

namespace OnnxSharp.Ops;

/// <summary>
/// <c>DequantizeLinear</c> / <c>QuantizeLinear</c> 的共享实现。
/// <para>
/// ONNX 语义：<c>y = (x - zero_point) * scale</c>；<c>y = saturate(round_half_even(x / scale) + zero_point)</c>。
/// <paramml name="scale"/> 是长度 1 的标量时按 per-tensor，长度 == <c>x.shape[axis]</c> 时按 per-axis
/// （权重最常见）。这两条是 QDQ 图的地基，也覆盖 QOperator 图边界上的反量化/重量化。
/// </para>
/// <para>
/// 未做向量化：它们出现在图的两端（一次前向里个位数调用），向量化收益 <c>&lt; 0.1 ms</c>，
/// 先保证 per-axis / 负 axis / u8-i8 的语义正确。
/// </para>
/// </summary>
internal static class DequantizeOps
{
    public static void Dequantize(
        Tensor source, ReadOnlySpan<float> scale, ReadOnlySpan<int> zeroPoint, int axis, Tensor destination)
    {
        var (outer, channels, inner) = Split(source.Shape, axis, scale.Length);
        var target = destination.AsSpan<float>();
        int count = outer * channels * inner;

        if (source.DType == DType.I8)
        {
            var values = source.AsSpan<sbyte>();
            for (int o = 0; o < outer; o++)
            for (int c = 0; c < channels; c++)
            {
                float s = scale[Math.Min(c, scale.Length - 1)];
                int z = zeroPoint.IsEmpty ? 0 : zeroPoint[Math.Min(c, zeroPoint.Length - 1)];
                int b = (o * channels + c) * inner;
                for (int i = 0; i < inner; i++)
                    target[b + i] = (values[b + i] - z) * s;
            }

            return;
        }

        if (source.DType != DType.U8)
            throw new OnnxSharpException($"DequantizeLinear: input must be u8/i8, got {source.DType}");

        {
            var values = source.AsSpan<byte>();
            for (int o = 0; o < outer; o++)
            for (int c = 0; c < channels; c++)
            {
                float s = scale[Math.Min(c, scale.Length - 1)];
                int z = zeroPoint.IsEmpty ? 0 : zeroPoint[Math.Min(c, zeroPoint.Length - 1)];
                int b = (o * channels + c) * inner;
                for (int i = 0; i < inner; i++)
                    target[b + i] = (values[b + i] - z) * s;
            }
        }
    }

    public static void Quantize(
        Tensor source, ReadOnlySpan<float> scale, ReadOnlySpan<int> zeroPoint, int axis, Tensor destination)
    {
        var (outer, channels, inner) = Split(source.Shape, axis, scale.Length);
        var values = source.AsSpan<float>();

        bool signed = destination.DType == DType.I8;
        if (!signed && destination.DType != DType.U8)
            throw new OnnxSharpException($"QuantizeLinear: output must be u8/i8, got {destination.DType}");

        int min = signed ? sbyte.MinValue : 0;
        int max = signed ? sbyte.MaxValue : 255;
        var quantized = destination.AsSpan<byte>();

        for (int o = 0; o < outer; o++)
        for (int c = 0; c < channels; c++)
        {
            float s = scale[Math.Min(c, scale.Length - 1)];
            int z = zeroPoint.IsEmpty ? 0 : zeroPoint[Math.Min(c, zeroPoint.Length - 1)];
            float inverse = 1f / s;
            int b = (o * channels + c) * inner;

            for (int i = 0; i < inner; i++)
            {
                float q = MathF.Round(values[b + i] * inverse, MidpointRounding.ToEven) + z;
                int clamped = (int)Math.Clamp(q, min, max);
                quantized[b + i] = unchecked((byte)(sbyte)clamped);
            }
        }
    }

    /// <summary>
    /// 把 <paramref name="shape"/> 按 <paramref name="axis"/> 拆成 <c>outer × channels × inner</c>；
    /// per-tensor（scale 只有 1 个）时把 channels 折成 1，让外层循环退化成一条。
    /// </summary>
    private static (int Outer, int Channels, int Inner) Split(TensorShape shape, int axis, int scaleLength)
    {
        int rank = shape.Rank;
        long total = 1;
        for (int i = 0; i < rank; i++)
            total *= shape[i];

        // 标量 scale（per-tensor）：axis 无关，整块一起算（也避开 rank=1 时 axis 越界的误报）
        if (scaleLength <= 1)
            return (1, 1, checked((int)total));

        int normalized = axis < 0 ? axis + rank : axis;
        if (normalized < 0 || normalized >= rank)
            throw new OnnxSharpException($"DequantizeLinear: axis {axis} out of range for rank {rank}");

        int outer = 1;
        for (int i = 0; i < normalized; i++)
            outer *= shape[i];

        int channels = shape[normalized];
        int inner = 1;
        for (int i = normalized + 1; i < rank; i++)
            inner *= shape[i];

        if (scaleLength != channels)
            throw new OnnxSharpException(
                $"DequantizeLinear: scale has {scaleLength} values but axis {axis} has {channels}");

        return (outer, channels, inner);
    }
}
