using System.Buffers.Binary;
using OnnxSharp.Core;
using OnnxSharp.Safetensors;

namespace OnnxSharp.Onnx;

/// <summary>initializer 的「数据字段」解码（raw_data 之外还要兜住 float_data / int*_data）。</summary>
public static partial class OnnxWeightPatcher
{
    /// <summary>读出一个 initializer 的原始字节（外部数据文件、raw_data、typed 数组都支持）。</summary>
    private static byte[] ReadInitializerBytes(InitializerEntry entry, string directory)
    {
        if (TryReadExternalData(entry, directory) is { } external)
            return external;

        int width = SafeTensorFormat.SizeOfOnnxType(entry.OnnxType, entry.Name);

        long count = 1;
        foreach (long dim in entry.Dims)
            count = checked(count * dim);
        long expected = checked(count * width);

        foreach (var field in entry.Fields)
        {
            switch (field.Number)
            {
                case 9: // raw_data
                    if (field.Payload.Length != expected)
                        throw new OnnxSharpException(
                            $"initializer '{entry.Name}': raw_data is {field.Payload.Length} B, expected {expected} B");
                    return field.Payload.ToArray();

                case 4:  // float_data
                case 10: // double_data
                    return DecodeFloatField(entry, field, expected);

                case 6:  // int32_data
                case 7:  // int64_data
                case 11: // uint64_data
                    return DecodeIntField(entry, field, width, expected);
            }
        }

        throw new OnnxSharpException(
            $"initializer '{entry.Name}': no data found (raw_data / float_data / int32_data / "
            + "int64_data / external_data 全缺) —— 这多半是已经剥掉权重的骨架");
    }

    private static byte[] DecodeFloatField(InitializerEntry entry, ProtoRawField field, long expected)
    {
        bool single = field.Number == 4;
        int step = single ? 4 : 8;
        var floats = new List<float>();

        if (field.WireType == 2)
        {
            var span = field.Payload.Span;
            for (int i = 0; i + step <= span.Length; i += step)
                floats.Add(single
                    ? BinaryPrimitives.ReadSingleLittleEndian(span.Slice(i, 4))
                    : (float)BinaryPrimitives.ReadDoubleLittleEndian(span.Slice(i, 8)));
        }
        else if (field.WireType is 1 or 5)
        {
            floats.Add(single
                ? BinaryPrimitives.ReadSingleLittleEndian(field.Payload.Span)
                : (float)BinaryPrimitives.ReadDoubleLittleEndian(field.Payload.Span));
        }
        else
        {
            throw new OnnxSharpException(
                $"initializer '{entry.Name}': unexpected wire type {field.WireType} for float data");
        }

        if (expected != checked(floats.Count * SafeTensorFormat.SizeOfOnnxType(entry.OnnxType, entry.Name)))
            throw new OnnxSharpException(
                $"initializer '{entry.Name}': {floats.Count} float values do not fill "
                + $"[{string.Join(",", entry.Dims)}] of data_type {entry.OnnxType}");

        var bytes = new byte[expected];
        for (int i = 0; i < floats.Count; i++)
        {
            switch (entry.OnnxType)
            {
                case 1: // FLOAT
                    BinaryPrimitives.WriteSingleLittleEndian(bytes.AsSpan(i * 4, 4), floats[i]);
                    break;
                case 10: // FLOAT16
                    BinaryPrimitives.WriteUInt16LittleEndian(
                        bytes.AsSpan(i * 2, 2), BitConverter.HalfToUInt16Bits((Half)floats[i]));
                    break;
                case 11: // DOUBLE
                    BinaryPrimitives.WriteDoubleLittleEndian(bytes.AsSpan(i * 8, 8), floats[i]);
                    break;
                case 16: // BFLOAT16
                    BinaryPrimitives.WriteUInt16LittleEndian(
                        bytes.AsSpan(i * 2, 2), (ushort)(BitConverter.SingleToUInt32Bits(floats[i]) >> 16));
                    break;
                default:
                    throw new OnnxSharpException(
                        $"initializer '{entry.Name}': float data stored for unsupported data_type {entry.OnnxType}");
            }
        }

        return bytes;
    }

    private static byte[] DecodeIntField(
        InitializerEntry entry, ProtoRawField field, int width, long expected)
    {
        var values = new List<long>();
        if (field.WireType == 0)
            values.Add(unchecked((long)field.Value));
        else
            values.AddRange(ProtoRaw.ReadPackedInt64(field.Payload.Span));

        if (expected != checked((long)values.Count * width))
            throw new OnnxSharpException(
                $"initializer '{entry.Name}': {values.Count} integer values do not fill "
                + $"[{string.Join(",", entry.Dims)}] of data_type {entry.OnnxType} ({width} B each)");

        var bytes = new byte[expected];
        for (int i = 0; i < values.Count; i++)
        {
            long value = values[i];
            var target = bytes.AsSpan(i * width, width);
            switch (width)
            {
                case 1:
                    target[0] = unchecked((byte)value);
                    break;
                case 2:
                    BinaryPrimitives.WriteInt16LittleEndian(target, unchecked((short)value));
                    break;
                case 4:
                    BinaryPrimitives.WriteInt32LittleEndian(target, unchecked((int)value));
                    break;
                case 8:
                    BinaryPrimitives.WriteInt64LittleEndian(target, value);
                    break;
                default:
                    throw new OnnxSharpException(
                        $"initializer '{entry.Name}': unexpected element width {width}");
            }
        }

        return bytes;
    }
}
