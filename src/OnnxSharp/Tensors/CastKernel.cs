using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace OnnxSharp.Tensors;

/// <summary>
/// <c>Cast</c> 的内核化实现。
/// <para>
/// 原来的通用路径逐元素走 <c>double</c> 中转 + <c>switch</c>，实测只有 ~200 M 元素/秒：
/// Paraformer encoder 里 170 个 <c>Cast(int32 → f32)</c>（每个 97×1280）就吃掉了**整段 33%** 的时间。
/// 这里按 (源类型, 目标类型) 给常见组合开 SIMD 通道（AVX2 8 路 + 标量尾巴）。
/// </para>
/// <para>语义按 ONNX：float → int 向零截断（C# 的 <c>(int)float</c> 就是向零），int → float 精确。</para>
/// </summary>
public static class CastKernel
{
    /// <summary>尝试内置转换；不支持的类型组合返回 <c>false</c>，由调用方走通用路径。</summary>
    public static bool TryCast(Tensor source, Tensor target)
    {
        if (source.Count != target.Count)
            return false;

        if (source.DType == target.DType)
        {
            source.Buffer.AsSpan(source.ByteOffset, source.ByteLength)
                  .CopyTo(target.Buffer.AsSpan(target.ByteOffset, target.ByteLength));
            return true;
        }

        switch (source.DType, target.DType)
        {
            case (DType.I32, DType.F32): Int32ToSingle(source, target); return true;
            case (DType.F32, DType.I32): SingleToInt32(source, target); return true;
            case (DType.U8, DType.F32): UInt8ToSingle(source, target); return true;
            case (DType.I8, DType.F32): Int8ToSingle(source, target); return true;
            case (DType.U8, DType.I32): UInt8ToInt32(source, target); return true;
            case (DType.I8, DType.I32): Int8ToInt32(source, target); return true;
            case (DType.F32, DType.U8): SingleToUInt8(source, target); return true;
            case (DType.I32, DType.U8): Int32ToUInt8(source, target); return true;
            default: return false;
        }
    }

    private static void Int32ToSingle(Tensor source, Tensor target)
    {
        var input = source.AsSpan<int>();
        var output = target.AsSpan<float>();
        int i = 0;

        if (Avx2.IsSupported)
        {
            for (; i + 8 <= input.Length; i += 8)
            {
                var value = Vector256.LoadUnsafe(ref MemoryMarshal.GetReference(input[i..]));
                Avx.ConvertToVector256Single(value).StoreUnsafe(ref output[i]);
            }
        }

        for (; i < input.Length; i++)
            output[i] = input[i];
    }

    private static void SingleToInt32(Tensor source, Tensor target)
    {
        var input = source.AsSpan<float>();
        var output = target.AsSpan<int>();
        int i = 0;

        if (Avx.IsSupported)
        {
            for (; i + 8 <= input.Length; i += 8)
            {
                var value = Vector256.LoadUnsafe(ref MemoryMarshal.GetReference(input[i..]));
                Avx.ConvertToVector256Int32WithTruncation(value).StoreUnsafe(ref output[i]);
            }
        }

        for (; i < input.Length; i++)
            output[i] = (int)input[i];
    }

    private static void UInt8ToSingle(Tensor source, Tensor target)
    {
        var input = source.AsSpan<byte>();
        var output = target.AsSpan<float>();
        int i = 0;

        if (Avx2.IsSupported)
        {
            for (; i + 8 <= input.Length; i += 8)
            {
                var bytes = Vector128.LoadUnsafe(ref MemoryMarshal.GetReference(input[i..]));
                Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(bytes)).StoreUnsafe(ref output[i]);
            }
        }

        for (; i < input.Length; i++)
            output[i] = input[i];
    }

    private static void Int8ToSingle(Tensor source, Tensor target)
    {
        var input = source.AsSpan<sbyte>();
        var output = target.AsSpan<float>();
        int i = 0;

        if (Avx2.IsSupported)
        {
            for (; i + 8 <= input.Length; i += 8)
            {
                var bytes = Vector128.LoadUnsafe(ref MemoryMarshal.GetReference(input[i..]));
                Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(bytes)).StoreUnsafe(ref output[i]);
            }
        }

        for (; i < input.Length; i++)
            output[i] = input[i];
    }

    private static void UInt8ToInt32(Tensor source, Tensor target)
    {
        var input = source.AsSpan<byte>();
        var output = target.AsSpan<int>();
        int i = 0;

        if (Avx2.IsSupported)
        {
            for (; i + 8 <= input.Length; i += 8)
            {
                var bytes = Vector128.LoadUnsafe(ref MemoryMarshal.GetReference(input[i..]));
                Avx2.ConvertToVector256Int32(bytes).StoreUnsafe(ref output[i]);
            }
        }

        for (; i < input.Length; i++)
            output[i] = input[i];
    }

    private static void Int8ToInt32(Tensor source, Tensor target)
    {
        var input = source.AsSpan<sbyte>();
        var output = target.AsSpan<int>();
        int i = 0;

        if (Avx2.IsSupported)
        {
            for (; i + 8 <= input.Length; i += 8)
            {
                var bytes = Vector128.LoadUnsafe(ref MemoryMarshal.GetReference(input[i..]));
                Avx2.ConvertToVector256Int32(bytes).StoreUnsafe(ref output[i]);
            }
        }

        for (; i < input.Length; i++)
            output[i] = input[i];
    }

    /// <summary>float → u8：向零截断后饱和到 [0, 255]（与 ORT 的 <c>Cast</c> 一致）。</summary>
    private static void SingleToUInt8(Tensor source, Tensor target)
    {
        var input = source.AsSpan<float>();
        var output = target.AsSpan<byte>();
        int i = 0;

        if (Avx.IsSupported)
        {
            var vmin = Vector256.Create(0f);
            var vmax = Vector256.Create(255f);

            for (; i + 8 <= input.Length; i += 8)
            {
                var value = Vector256.LoadUnsafe(ref MemoryMarshal.GetReference(input[i..]));
                value = Avx.Min(vmax, Avx.Max(vmin, Avx.RoundToZero(value)));
                var ints = Avx.ConvertToVector256Int32(value);

                for (int lane = 0; lane < 8; lane++)
                    output[i + lane] = (byte)ints.GetElement(lane);
            }
        }

        for (; i < input.Length; i++)
        {
            float value = MathF.Truncate(input[i]);
            output[i] = value <= 0f ? (byte)0 : value >= 255f ? (byte)255 : (byte)value;
        }
    }

    private static void Int32ToUInt8(Tensor source, Tensor target)
    {
        var input = source.AsSpan<int>();
        var output = target.AsSpan<byte>();

        for (int i = 0; i < input.Length; i++)
            output[i] = (byte)Math.Clamp(input[i], 0, 255);
    }
}
