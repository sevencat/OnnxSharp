using System.Buffers.Binary;
using System.Runtime.InteropServices;
using OnnxSharp.Core;

namespace OnnxSharp.Safetensors;

/// <summary>数值转换部分（dtype 表在 <c>SafeTensorFormat.cs</c>）。</summary>
internal static partial class SafeTensorFormat
{
    private static void DecodeFloat(byte[] source, string dtype, float[] destination, string name)
    {
        EnsureLittleEndian(name);

        switch (dtype)
        {
            case "F32":
                MemoryMarshal.Cast<byte, float>(source).CopyTo(destination);
                return;
            case "F16":
                for (int i = 0; i < destination.Length; i++)
                    destination[i] = (float)BitConverter.UInt16BitsToHalf(
                        BinaryPrimitives.ReadUInt16LittleEndian(source.AsSpan(i * 2, 2)));
                return;
            case "BF16":
                for (int i = 0; i < destination.Length; i++)
                    destination[i] = BitConverter.Int32BitsToSingle(
                        BinaryPrimitives.ReadUInt16LittleEndian(source.AsSpan(i * 2, 2)) << 16);
                return;
            case "F64":
                {
                    var values = MemoryMarshal.Cast<byte, double>(source);
                    for (int i = 0; i < destination.Length; i++)
                        destination[i] = (float)values[i];
                    return;
                }
            case "I8":
                for (int i = 0; i < destination.Length; i++)
                    destination[i] = unchecked((sbyte)source[i]);
                return;
            case "U8":
                for (int i = 0; i < destination.Length; i++)
                    destination[i] = source[i];
                return;
            case "BOOL":
                for (int i = 0; i < destination.Length; i++)
                    destination[i] = source[i] != 0 ? 1f : 0f;
                return;
            case "I16":
                for (int i = 0; i < destination.Length; i++)
                    destination[i] = BinaryPrimitives.ReadInt16LittleEndian(source.AsSpan(i * 2, 2));
                return;
            case "U16":
                for (int i = 0; i < destination.Length; i++)
                    destination[i] = BinaryPrimitives.ReadUInt16LittleEndian(source.AsSpan(i * 2, 2));
                return;
            case "I32":
                for (int i = 0; i < destination.Length; i++)
                    destination[i] = BinaryPrimitives.ReadInt32LittleEndian(source.AsSpan(i * 4, 4));
                return;
            case "U32":
                for (int i = 0; i < destination.Length; i++)
                    destination[i] = BinaryPrimitives.ReadUInt32LittleEndian(source.AsSpan(i * 4, 4));
                return;
            case "I64":
                for (int i = 0; i < destination.Length; i++)
                    destination[i] = BinaryPrimitives.ReadInt64LittleEndian(source.AsSpan(i * 8, 8));
                return;
            case "U64":
                for (int i = 0; i < destination.Length; i++)
                    destination[i] = BinaryPrimitives.ReadUInt64LittleEndian(source.AsSpan(i * 8, 8));
                return;
            default:
                throw new OnnxSharpException($"tensor '{name}': cannot convert {dtype} to float");
        }
    }

    private static void DecodeInt(byte[] source, string dtype, long[] destination, string name)
    {
        EnsureLittleEndian(name);

        switch (dtype)
        {
            case "I64":
                MemoryMarshal.Cast<byte, long>(source).CopyTo(destination);
                return;
            case "I32":
                for (int i = 0; i < destination.Length; i++)
                    destination[i] = BinaryPrimitives.ReadInt32LittleEndian(source.AsSpan(i * 4, 4));
                return;
            case "U32":
                for (int i = 0; i < destination.Length; i++)
                    destination[i] = BinaryPrimitives.ReadUInt32LittleEndian(source.AsSpan(i * 4, 4));
                return;
            case "U64":
                for (int i = 0; i < destination.Length; i++)
                    destination[i] = unchecked((long)BinaryPrimitives.ReadUInt64LittleEndian(source.AsSpan(i * 8, 8)));
                return;
            case "I8":
                for (int i = 0; i < destination.Length; i++)
                    destination[i] = unchecked((sbyte)source[i]);
                return;
            case "U8":
            case "BOOL":
                for (int i = 0; i < destination.Length; i++)
                    destination[i] = source[i];
                return;
            case "I16":
                for (int i = 0; i < destination.Length; i++)
                    destination[i] = BinaryPrimitives.ReadInt16LittleEndian(source.AsSpan(i * 2, 2));
                return;
            case "U16":
                for (int i = 0; i < destination.Length; i++)
                    destination[i] = BinaryPrimitives.ReadUInt16LittleEndian(source.AsSpan(i * 2, 2));
                return;
            case "F32":
                {
                    var values = MemoryMarshal.Cast<byte, float>(source);
                    for (int i = 0; i < destination.Length; i++)
                        destination[i] = (long)values[i];
                    return;
                }
            case "F64":
                {
                    var values = MemoryMarshal.Cast<byte, double>(source);
                    for (int i = 0; i < destination.Length; i++)
                        destination[i] = (long)values[i];
                    return;
                }
            case "F16":
                for (int i = 0; i < destination.Length; i++)
                    destination[i] = (long)(float)BitConverter.UInt16BitsToHalf(
                        BinaryPrimitives.ReadUInt16LittleEndian(source.AsSpan(i * 2, 2)));
                return;
            case "BF16":
                for (int i = 0; i < destination.Length; i++)
                    destination[i] = (long)BitConverter.Int32BitsToSingle(
                        BinaryPrimitives.ReadUInt16LittleEndian(source.AsSpan(i * 2, 2)) << 16);
                return;
            default:
                throw new OnnxSharpException($"tensor '{name}': cannot convert {dtype} to integer");
        }
    }

    private static byte[] EncodeFloat(float[] values, int onnxType, string name) => onnxType switch
    {
        1 => MemoryMarshal.AsBytes(values.AsSpan()).ToArray(),
        10 => EncodeHalf(values, name),
        11 => EncodeDouble(values),
        16 => EncodeBFloat16(values),
        _ => throw new OnnxSharpException($"tensor '{name}': cannot encode float as ONNX data_type {onnxType}"),
    };

    private static byte[] EncodeInt(long[] values, int onnxType, string name)
    {
        int size = SizeOfOnnxType(onnxType, name);
        var bytes = new byte[checked(values.Length * size)];

        for (int i = 0; i < values.Length; i++)
        {
            long value = values[i];
            var target = bytes.AsSpan(i * size, size);

            switch (onnxType)
            {
                case 2: // UINT8
                    target[0] = unchecked((byte)value);
                    break;
                case 3: // INT8
                    target[0] = unchecked((byte)(sbyte)value);
                    break;
                case 4: // UINT16
                    BinaryPrimitives.WriteUInt16LittleEndian(target, unchecked((ushort)value));
                    break;
                case 5: // INT16
                    BinaryPrimitives.WriteInt16LittleEndian(target, unchecked((short)value));
                    break;
                case 6: // INT32
                    BinaryPrimitives.WriteInt32LittleEndian(target, unchecked((int)value));
                    break;
                case 7: // INT64
                    BinaryPrimitives.WriteInt64LittleEndian(target, value);
                    break;
                case 9: // BOOL
                    target[0] = value != 0 ? (byte)1 : (byte)0;
                    break;
                case 12: // UINT32
                    BinaryPrimitives.WriteUInt32LittleEndian(target, unchecked((uint)value));
                    break;
                case 13: // UINT64
                    BinaryPrimitives.WriteUInt64LittleEndian(target, unchecked((ulong)value));
                    break;
                default:
                    throw new OnnxSharpException(
                        $"tensor '{name}': cannot encode integer as ONNX data_type {onnxType}");
            }
        }

        return bytes;
    }

    private static byte[] EncodeHalf(float[] values, string name)
    {
        if (!BitConverter.IsLittleEndian)
            throw new OnnxSharpException($"tensor '{name}': big-endian platform is not supported");

        var bytes = new byte[values.Length * 2];
        for (int i = 0; i < values.Length; i++)
            BinaryPrimitives.WriteUInt16LittleEndian(
                bytes.AsSpan(i * 2, 2), BitConverter.HalfToUInt16Bits((Half)values[i]));
        return bytes;
    }

    private static byte[] EncodeDouble(float[] values)
    {
        var doubles = new double[values.Length];
        for (int i = 0; i < values.Length; i++)
            doubles[i] = values[i];
        return MemoryMarshal.AsBytes(doubles.AsSpan()).ToArray();
    }

    /// <summary>float → bfloat16：取高 16 位并按「最近偶数」舍入。</summary>
    private static byte[] EncodeBFloat16(float[] values)
    {
        var bytes = new byte[values.Length * 2];
        for (int i = 0; i < values.Length; i++)
        {
            uint bits = BitConverter.SingleToUInt32Bits(values[i]);
            // NaN / Inf 的指数位已全 1，再加舍入位会溢出成别的值，直接截断
            ushort bf16 = (bits & 0x7F80_0000u) == 0x7F80_0000u
                ? (ushort)(bits >> 16)
                : (ushort)((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16);

            BinaryPrimitives.WriteUInt16LittleEndian(bytes.AsSpan(i * 2, 2), bf16);
        }

        return bytes;
    }

    private static void EnsureLittleEndian(string name)
    {
        if (!BitConverter.IsLittleEndian)
            throw new OnnxSharpException(
                $"tensor '{name}': safetensors/ONNX 都是小端格式，当前平台是大端，需要手工转换");
    }
}
