using System.Text;
using OnnxSharp.Core;

namespace OnnxSharp.Onnx;

/// <summary>
/// 极简 Protocol Buffers 消息视图：只做「按字段号取原始值」，不生成代码、不反射。
/// <para>
/// 移植自 SimdYolo <c>OnnxReader.cs</c> 的 <c>Proto</c>（已验证可读 4 个目标模型）。
/// 与上游的差异：所有长度分隔字段都以 <see cref="ReadOnlyMemory{T}"/> 视图保存，
/// 不复制大数据（权重 <c>raw_data</c> 靠它做到零拷贝）。
/// </para>
/// <para>
/// 支持 wire type 0 / 1 / 2 / 5；未知字段按 wire type 跳过；packed 字段按出现顺序展开。
/// </para>
/// </summary>
internal sealed class Proto
{
    /// <summary>按字段号保存出现顺序（保留重复字段与原始值）。</summary>
    private readonly Dictionary<int, List<object>> _fields = new();

    private Proto()
    {
    }

    public static Proto Parse(ReadOnlyMemory<byte> message) => ParseCore(message);

    private static Proto ParseCore(ReadOnlyMemory<byte> message)
    {
        var proto = new Proto();
        var span = message.Span;
        int i = 0;

        while (i < message.Length)
        {
            ulong tag = Varint(span, ref i);
            int field = checked((int)(tag >> 3));
            if (field == 0)
                throw new OnnxSharpException("invalid protobuf tag (field 0)");

            object value;
            switch (tag & 7)
            {
                case 0: // varint
                    value = Varint(span, ref i);
                    break;
                case 1: // fixed64
                    Ensure(span, i, 8);
                    value = message.Slice(i, 8);
                    i += 8;
                    break;
                case 2: // length-delimited
                    {
                        int length = checked((int)Varint(span, ref i));
                        Ensure(span, i, length);
                        value = message.Slice(i, length);
                        i += length;
                        break;
                    }
                case 5: // fixed32
                    Ensure(span, i, 4);
                    value = message.Slice(i, 4);
                    i += 4;
                    break;
                default:
                    throw new OnnxSharpException($"unsupported protobuf wire type {tag & 7}");
            }

            if (!proto._fields.TryGetValue(field, out var list))
                proto._fields[field] = list = new List<object>(1);
            list.Add(value);
        }

        return proto;
    }

    private static void Ensure(ReadOnlySpan<byte> data, int offset, int length)
    {
        if (length < 0 || offset < 0 || length > data.Length - offset)
            throw new OnnxSharpException("truncated protobuf message");
    }

    /// <summary>读取最多 64 位的无符号 LEB128。</summary>
    internal static ulong Varint(ReadOnlySpan<byte> data, ref int offset)
    {
        ulong value = 0;
        for (int shift = 0; shift < 64; shift += 7)
        {
            Ensure(data, offset, 1);
            byte b = data[offset++];
            if (shift == 63 && b > 1)
                throw new OnnxSharpException("protobuf varint overflow");
            value |= (ulong)(b & 127) << shift;
            if (b < 128)
                return value;
        }
        throw new OnnxSharpException("protobuf varint overflow");
    }

    // ---------------- 取值 ----------------

    public bool Has(int field) => _fields.ContainsKey(field);

    /// <summary>第一个长度分隔字段的视图；缺失返回 null。</summary>
    public ReadOnlyMemory<byte>? Buffer(int field)
    {
        if (!_fields.TryGetValue(field, out var list) || list.Count == 0)
            return null;
        return (ReadOnlyMemory<byte>)list[0];
    }

    /// <summary>按出现顺序枚举长度分隔字段视图。</summary>
    public IEnumerable<ReadOnlyMemory<byte>> Buffers(int field)
    {
        if (!_fields.TryGetValue(field, out var list))
            yield break;
        foreach (var value in list)
            yield return (ReadOnlyMemory<byte>)value;
    }

    public string Text(int field, string fallback = "")
        => Buffer(field) is { } b ? Encoding.UTF8.GetString(b.Span) : fallback;

    /// <summary>第一个 varint 字段（按二进制补码解释为有符号 64 位）。</summary>
    public long Int(int field, long fallback = 0)
        => _fields.TryGetValue(field, out var list) ? unchecked((long)(ulong)list[0]) : fallback;

    /// <summary>重复 varint 或 packed varint 字段，按顺序展开。</summary>
    public long[] Ints(int field)
    {
        if (!_fields.TryGetValue(field, out var list))
            return Array.Empty<long>();

        var result = new List<long>(list.Count);
        foreach (var value in list)
        {
            if (value is ulong single)
            {
                result.Add(unchecked((long)single));
                continue;
            }

            var bytes = (ReadOnlyMemory<byte>)value;
            var span = bytes.Span;
            int i = 0;
            while (i < bytes.Length)
                result.Add(unchecked((long)Varint(span, ref i)));
        }

        return result.ToArray();
    }

    /// <summary>重复或 packed 的小端 FP32 字段。</summary>
    public float[] Floats(int field)
    {
        if (!_fields.TryGetValue(field, out var list))
            return Array.Empty<float>();

        int total = 0;
        foreach (var value in list)
        {
            var bytes = (ReadOnlyMemory<byte>)value;
            if (bytes.Length % 4 != 0)
                throw new OnnxSharpException("invalid packed float field");
            total = checked(total + bytes.Length / 4);
        }

        var floats = GC.AllocateUninitializedArray<float>(total);
        int offset = 0;
        foreach (var value in list)
        {
            var bytes = (ReadOnlyMemory<byte>)value;
            int n = bytes.Length / 4;
            for (int k = 0; k < n; k++)
                floats[offset + k] = BitConverter.ToSingle(bytes.Span.Slice(k * 4, 4));
            offset += n;
        }

        return floats;
    }

    /// <summary>读出全部子消息（长度分隔字段逐个解析）。</summary>
    public IEnumerable<Proto> Messages(int field)
    {
        foreach (var buffer in Buffers(field))
            yield return ParseCore(buffer);
    }
}
