using System.Text;

namespace OnnxSharp.Onnx;

/// <summary>
/// protobuf 字段的<b>带序</b>原始视图：只解析出 (字段号, wire type, 负载)，不建对象树。
/// <para>
/// 与 <see cref="Proto"/> 的分工：<c>Proto</c> 是「按字段号取值」的读视图，字段间顺序会丢；
/// 本类型用于<b>改写</b>——重放时必须保持原顺序，未知字段（<c>training_info</c>、
/// <c>functions</c>、将来新增的字段）才能原样带走，否则写出来的模型会悄悄丢内容。
/// </para>
/// </summary>
internal readonly record struct ProtoRawField(
    int Number,
    int WireType,
    ulong Value,
    ReadOnlyMemory<byte> Payload)
{
    /// <summary>wire type 2 的负载按 UTF-8 解释（string 字段）。</summary>
    public string Text => Encoding.UTF8.GetString(Payload.Span);
}

/// <summary>protobuf 的裸解析/编码（仅 wire type 0 / 1 / 2 / 5，够 ONNX 用）。</summary>
internal static class ProtoRaw
{
    /// <summary>按出现顺序解析出一条消息的全部字段。</summary>
    public static List<ProtoRawField> Parse(ReadOnlyMemory<byte> message)
    {
        var fields = new List<ProtoRawField>();
        var span = message.Span;
        int i = 0;

        while (i < message.Length)
        {
            ulong tag = ReadVarint(span, ref i);
            int number = checked((int)(tag >> 3));
            int wire = (int)(tag & 7);
            if (number == 0)
                throw new ArgumentException("invalid protobuf tag (field 0)");

            switch (wire)
            {
                case 0:
                    fields.Add(new ProtoRawField(number, wire, ReadVarint(span, ref i), default));
                    break;
                case 1:
                    Ensure(span, i, 8);
                    fields.Add(new ProtoRawField(number, wire, 0, message.Slice(i, 8)));
                    i += 8;
                    break;
                case 2:
                    {
                        int length = checked((int)ReadVarint(span, ref i));
                        Ensure(span, i, length);
                        fields.Add(new ProtoRawField(number, wire, 0, message.Slice(i, length)));
                        i += length;
                        break;
                    }
                case 5:
                    Ensure(span, i, 4);
                    fields.Add(new ProtoRawField(number, wire, 0, message.Slice(i, 4)));
                    i += 4;
                    break;
                default:
                    throw new ArgumentException($"unsupported protobuf wire type {wire}");
            }
        }

        return fields;
    }

    /// <summary>把一段 varint+paced 序列展开成 int64 列表（用于 <c>TensorProto.dims</c> 等）。</summary>
    public static List<long> ReadPackedInt64(ReadOnlySpan<byte> data)
    {
        var values = new List<long>();
        int i = 0;
        while (i < data.Length)
            values.Add(unchecked((long)ReadVarint(data, ref i)));
        return values;
    }

    public static void WriteVarint(List<byte> sink, ulong value)
    {
        while (value >= 0x80)
        {
            sink.Add((byte)(value | 0x80));
            value >>= 7;
        }

        sink.Add((byte)value);
    }

    public static void WriteTag(List<byte> sink, int field, int wireType)
        => WriteVarint(sink, ((ulong)field << 3) | (uint)wireType);

    public static void WriteVarintField(List<byte> sink, int field, ulong value)
    {
        WriteTag(sink, field, 0);
        WriteVarint(sink, value);
    }

    public static void WriteBytesField(List<byte> sink, int field, ReadOnlySpan<byte> payload)
    {
        WriteTag(sink, field, 2);
        WriteVarint(sink, (ulong)payload.Length);
        foreach (byte b in payload)
            sink.Add(b);
    }

    /// <summary>原样重放一个字段（未知字段靠它零损失带过去）。</summary>
    public static void CopyField(List<byte> sink, in ProtoRawField field)
    {
        switch (field.WireType)
        {
            case 0:
                WriteVarintField(sink, field.Number, field.Value);
                break;
            case 2:
                WriteBytesField(sink, field.Number, field.Payload.Span);
                break;
            case 1:
            case 5:
                WriteTag(sink, field.Number, field.WireType);
                sink.AddRange(field.Payload.ToArray());
                break;
            default:
                throw new ArgumentException($"unsupported protobuf wire type {field.WireType}");
        }
    }

    private static ulong ReadVarint(ReadOnlySpan<byte> data, ref int offset)
    {
        ulong value = 0;
        for (int shift = 0; shift < 64; shift += 7)
        {
            Ensure(data, offset, 1);
            byte b = data[offset++];
            if (shift == 63 && b > 1)
                throw new ArgumentException("protobuf varint overflow");
            value |= (ulong)(b & 127) << shift;
            if (b < 128)
                return value;
        }

        throw new ArgumentException("protobuf varint overflow");
    }

    private static void Ensure(ReadOnlySpan<byte> data, int offset, int length)
    {
        if (length < 0 || offset < 0 || length > data.Length - offset)
            throw new ArgumentException("truncated protobuf message");
    }
}
