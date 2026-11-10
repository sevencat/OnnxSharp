using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace OnnxSharp.Tensors;

/// <summary>
/// 量化参数：<c>y = saturate(round(x / scale) + zero_point)</c>。
/// <para>
/// 支持 per-tensor（长度 1）与 per-axis（长度 = 通道数）两种——
/// QDQ 导出与 <c>QLinearConv</c> 的权重通常是 per-axis，激活是 per-tensor。
/// </para>
/// </summary>
public readonly struct QuantizationParams
{
    private readonly float[]? _scales;
    private readonly int[]? _zeroPoints;

    public QuantizationParams(float scale, int zeroPoint = 0)
    {
        if (!(scale > 0))
            throw new ArgumentException("scale must be positive", nameof(scale));

        _scales = [scale];
        _zeroPoints = [zeroPoint];
    }

    public QuantizationParams(ReadOnlySpan<float> scales, ReadOnlySpan<int> zeroPoints)
    {
        if (scales.IsEmpty)
            throw new ArgumentException("scales must not be empty", nameof(scales));
        if (!zeroPoints.IsEmpty && zeroPoints.Length != scales.Length)
            throw new ArgumentException("zero point count must match scale count", nameof(zeroPoints));

        var scaleCopy = new float[scales.Length];
        scales.CopyTo(scaleCopy);

        var zpCopy = new int[scales.Length];
        if (!zeroPoints.IsEmpty)
            zeroPoints.CopyTo(zpCopy);

        for (int i = 0; i < scaleCopy.Length; i++)
        {
            if (!(scaleCopy[i] > 0))
                throw new ArgumentException($"scale[{i}] must be positive", nameof(scales));
        }

        _scales = scaleCopy;
        _zeroPoints = zpCopy;
    }

    public bool IsEmpty => _scales is null || _scales.Length == 0;

    public int Length => _scales?.Length ?? 0;

    /// <summary>per-tensor 时忽略 <paramref name="channel"/>。</summary>
    public float Scale(int channel = 0) => _scales is { Length: > 0 } s ? s[Math.Min(channel, s.Length - 1)] : 1f;

    public int ZeroPoint(int channel = 0) => _zeroPoints is { Length: > 0 } z ? z[Math.Min(channel, z.Length - 1)] : 0;
}

/// <summary>
/// float ↔ int8/uint8 的饱和转换。ONNX <c>QuantizeLinear</c> 的语义：
/// <c>q = clamp(round_half_even(x / scale) + zero_point, qmin, qmax)</c>，
/// <c>x = (q - zero_point) * scale</c>。
/// </summary>
public static class Quantizer
{
    public static void QuantizeToUInt8(ReadOnlySpan<float> source, Span<byte> destination, float scale, int zeroPoint)
    {
        if (destination.Length < source.Length)
            throw new ArgumentException("destination is too short", nameof(destination));

        float inverse = 1f / scale;

        if (Avx.IsSupported)
        {
            var vinv = Vector256.Create(inverse);
            var vzero = Vector256.Create((float)zeroPoint);
            var vmin = Vector256.Create(0f);
            var vmax = Vector256.Create(255f);

            int i = 0;
            for (; i + Vector256<float>.Count <= source.Length; i += Vector256<float>.Count)
            {
                var v = Vector256.LoadUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(source[i..]));
                v = Avx.Multiply(v, vinv);
                // round-to-nearest-even（与 numpy/ORT 的 round 一致）
                v = Avx.RoundToNearestInteger(v);
                v = Avx.Add(v, vzero);
                v = Avx.Min(vmax, Avx.Max(vmin, v));

                for (int j = 0; j < Vector256<float>.Count; j++)
                    destination[i + j] = (byte)v.GetElement(j);
            }

            for (; i < source.Length; i++)
                destination[i] = ClampToByte(MathF.Round(source[i] * inverse, MidpointRounding.ToEven) + zeroPoint);
            return;
        }

        for (int i = 0; i < source.Length; i++)
            destination[i] = ClampToByte(MathF.Round(source[i] * inverse, MidpointRounding.ToEven) + zeroPoint);
    }

    public static void QuantizeToInt8(ReadOnlySpan<float> source, Span<sbyte> destination, float scale, int zeroPoint)
    {
        if (destination.Length < source.Length)
            throw new ArgumentException("destination is too short", nameof(destination));

        float inverse = 1f / scale;
        for (int i = 0; i < source.Length; i++)
        {
            float q = MathF.Round(source[i] * inverse, MidpointRounding.ToEven) + zeroPoint;
            destination[i] = (sbyte)Math.Clamp((int)q, sbyte.MinValue, sbyte.MaxValue);
        }
    }

    public static void DequantizeUInt8(ReadOnlySpan<byte> source, Span<float> destination, float scale, int zeroPoint)
    {
        if (destination.Length < source.Length)
            throw new ArgumentException("destination is too short", nameof(destination));

        for (int i = 0; i < source.Length; i++)
            destination[i] = (source[i] - zeroPoint) * scale;
    }

    public static void DequantizeInt8(ReadOnlySpan<sbyte> source, Span<float> destination, float scale, int zeroPoint)
    {
        if (destination.Length < source.Length)
            throw new ArgumentException("destination is too short", nameof(destination));

        for (int i = 0; i < source.Length; i++)
            destination[i] = (source[i] - zeroPoint) * scale;
    }

    private static byte ClampToByte(float value)
        => value <= 0f ? (byte)0 : value >= 255f ? (byte)255 : (byte)value;
}
