using System.Buffers.Binary;
using System.Text.Json;
using OnnxSharp.Core;

namespace OnnxSharp.Safetensors;

/// <summary>
/// safetensors 写入器（与 <see cref="SafeTensorFile"/> 对称）。
/// <para>
/// 用途：把 ONNX 里的 initializer 抽成独立的权重文件 —— 权重可以单独版本化、量化、拿来微调，
/// 再用 <see cref="Onnx.OnnxWeightPatcher"/> 打回去，图完全不用重导。
/// </para>
/// <para>
/// 头部按 spec 用空格填充到 8 字节对齐（Hugging Face 的 <c>save_file</c> 也这么做），
/// 数据区按加入顺序连续排列。
/// </para>
/// </summary>
public sealed class SafeTensorWriter
{
    private readonly List<Entry> _entries = new();
    private readonly HashSet<string> _names = new(StringComparer.Ordinal);

    private sealed record Entry(string Name, string DType, long[] Shape, byte[] Data);

    /// <summary>加入一个张量（<paramref name="data"/> 必须正好是 dtype × 元素数 的小端字节）。</summary>
    public void Add(string name, string dtype, long[] shape, ReadOnlySpan<byte> data)
    {
        ArgumentException.ThrowIfNullOrEmpty(name);
        ArgumentNullException.ThrowIfNull(shape);

        if (!_names.Add(name))
            throw new OnnxSharpException($"safetensors: duplicate tensor name '{name}'");

        long count = 1;
        foreach (long dim in shape)
            count = checked(count * dim);

        long expected = checked(count * SafeTensorFormat.SizeOf(dtype, "output", name));
        if (data.Length != expected)
            throw new OnnxSharpException(
                $"safetensors: tensor '{name}' is {data.Length} B, but {count} x {dtype} needs {expected} B");

        _entries.Add(new Entry(name, dtype, shape.ToArray(), data.ToArray()));
    }

    /// <summary>已加入的张量数。</summary>
    public int Count => _entries.Count;

    /// <summary>全部张量占用的字节数。</summary>
    public long ByteCount
    {
        get
        {
            long total = 0;
            foreach (var entry in _entries)
                total += entry.Data.Length;
            return total;
        }
    }

    /// <summary>生成完整的 safetensors 字节（8 字节长度 + JSON 头部 + 数据区）。</summary>
    public byte[] ToBytes(IReadOnlyDictionary<string, string>? metadata = null)
    {
        long offset = 0;

        using var headerStream = new MemoryStream();
        using (var json = new Utf8JsonWriter(headerStream))
        {
            json.WriteStartObject();

            foreach (var entry in _entries)
            {
                json.WritePropertyName(entry.Name);
                json.WriteStartObject();
                json.WriteString("dtype", entry.DType);

                json.WriteStartArray("shape");
                foreach (long dim in entry.Shape)
                    json.WriteNumberValue(dim);
                json.WriteEndArray();

                json.WriteStartArray("data_offsets");
                json.WriteNumberValue(offset);
                json.WriteNumberValue(offset + entry.Data.Length);
                json.WriteEndArray();

                json.WriteEndObject();
                offset += entry.Data.Length;
            }

            if (metadata is { Count: > 0 })
            {
                json.WritePropertyName("__metadata__");
                json.WriteStartObject();
                foreach (var (key, value) in metadata)
                    json.WriteString(key, value);
                json.WriteEndObject();
            }

            json.WriteEndObject();
        }

        var header = headerStream.ToArray();

        // spec：头部用空格填充，使数据区从 8 字节对齐处开始
        int padded = header.Length;
        while ((8 + padded) % 8 != 0)
            padded++;

        var result = new byte[8 + padded + offset];
        BinaryPrimitives.WriteUInt64LittleEndian(result.AsSpan(0, 8), (ulong)padded);
        header.CopyTo(result.AsSpan(8));

        for (int i = 8 + header.Length; i < 8 + padded; i++)
            result[i] = (byte)' ';

        int cursor = 8 + padded;
        foreach (var entry in _entries)
        {
            entry.Data.CopyTo(result, cursor);
            cursor += entry.Data.Length;
        }

        return result;
    }

    /// <summary>写到文件（默认在 metadata 里记下来源模型，便于追溯）。</summary>
    public void Save(string path, IReadOnlyDictionary<string, string>? metadata = null)
        => File.WriteAllBytes(path, ToBytes(metadata));
}
