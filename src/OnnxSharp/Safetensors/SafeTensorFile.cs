using System.Buffers.Binary;
using System.Text.Json;
using Microsoft.Win32.SafeHandles;
using OnnxSharp.Core;
using OnnxSharp.Tensors;

namespace OnnxSharp.Safetensors;

/// <summary>safetensors 里的一个张量：头部声明的元信息（数据本身按需从文件读）。</summary>
/// <param name="Name">张量名（等于训练框架 <c>state_dict()</c> 的键）。</param>
/// <param name="DType">safetensors 的 dtype 字符串，如 <c>F32</c> / <c>F16</c> / <c>BF16</c> / <c>I64</c>。</param>
/// <param name="Shape">形状（标量为空数组）。</param>
/// <param name="Begin">相对数据区起始的字节偏移（含）。</param>
/// <param name="End">相对数据区起始的字节偏移（不含）。</param>
public sealed record SafeTensorInfo(string Name, string DType, long[] Shape, long Begin, long End)
{
    /// <summary>该张量占用的字节数。</summary>
    public long ByteLength => End - Begin;

    /// <summary>元素个数（标量为 1）。</summary>
    public long ElementCount
    {
        get
        {
            long count = 1;
            foreach (long dim in Shape)
                count = checked(count * dim);
            return count;
        }
    }

    public override string ToString() => $"{Name}:{DType}[{string.Join(",", Shape)}]";
}

/// <summary>
/// 只读的 safetensors 文件。
/// <para>
/// 格式（safetensors spec）：前 8 字节是小端 <c>u64</c> 的头部长度 N，接着 N 字节 JSON 头部，
/// 再往后是<b>连续数据区</b>；头部里每个张量的 <c>data_offsets</c> 是<b>相对数据区起点</b>
/// 的 <c>[begin, end)</c>。<c>__metadata__</c> 键是字符串字典，不是张量。
/// </para>
/// <para>
/// 读取用 <see cref="RandomAccess"/> 按张量定位，<b>不把整包映射进内存</b> ——
/// 14 GB 的 7B 权重和 18 MB 的 yolo11s 走同一条路径，内存占用只与「当前要的那个张量」有关。
/// </para>
/// </summary>
public sealed class SafeTensorFile : IDisposable
{
    private const int MaxHeaderBytes = 256 * 1024 * 1024;

    private readonly SafeFileHandle _handle;
    private readonly Dictionary<string, SafeTensorInfo> _byName;
    private readonly SafeTensorInfo[] _tensors;
    private readonly long _dataStart;

    private SafeTensorFile(
        string path,
        SafeFileHandle handle,
        long dataStart,
        SafeTensorInfo[] tensors,
        Dictionary<string, string> metadata)
    {
        Path = path;
        _handle = handle;
        _dataStart = dataStart;
        _tensors = tensors;
        _byName = new Dictionary<string, SafeTensorInfo>(tensors.Length, StringComparer.Ordinal);
        foreach (var tensor in tensors)
            _byName[tensor.Name] = tensor;
        Metadata = metadata;
    }

    /// <summary>文件路径。</summary>
    public string Path { get; }

    /// <summary>头部里的 <c>__metadata__</c>（训练框架常放 <c>format</c>；没有则为空字典）。</summary>
    public IReadOnlyDictionary<string, string> Metadata { get; }

    /// <summary>按头部声明顺序列出全部张量。</summary>
    public IReadOnlyList<SafeTensorInfo> Tensors => _tensors;

    /// <summary>全部张量名。</summary>
    public IReadOnlyList<string> Names => Array.ConvertAll(_tensors, t => t.Name);

    public bool Contains(string name) => _byName.ContainsKey(name);

    /// <summary>按名字查元信息。</summary>
    public bool TryGetInfo(string name, out SafeTensorInfo info) => _byName.TryGetValue(name, out info!);

    /// <summary>按名字查元信息，缺失即抛（错误信息里带文件路径）。</summary>
    public SafeTensorInfo Info(string name) => _byName.TryGetValue(name, out var info)
        ? info
        : throw new OnnxSharpException($"safetensors '{Path}': no tensor named '{name}'");

    /// <summary>打开（只读、允许共享）。</summary>
    public static SafeTensorFile Open(string path)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);

        var handle = File.OpenHandle(path, FileMode.Open, FileAccess.Read, FileShare.Read);
        try
        {
            long fileLength = RandomAccess.GetLength(handle);
            if (fileLength < 8)
                throw new OnnxSharpException(
                    $"safetensors '{path}': file is {fileLength} B, header needs at least 8 B");

            Span<byte> prefix = stackalloc byte[8];
            ReadExactly(handle, prefix, 0, path);
            ulong headerLength = BinaryPrimitives.ReadUInt64LittleEndian(prefix);

            if (headerLength == 0 || headerLength > (ulong)MaxHeaderBytes || headerLength > (ulong)(fileLength - 8))
                throw new OnnxSharpException(
                    $"safetensors '{path}': header length {headerLength} is out of range (file {fileLength} B)");

            var header = new byte[(int)headerLength];
            ReadExactly(handle, header, 8, path);

            long dataStart = 8 + (long)headerLength;
            var (tensors, metadata) = ParseHeader(path, header, fileLength - dataStart);

            return new SafeTensorFile(path, handle, dataStart, tensors, metadata);
        }
        catch
        {
            handle.Dispose();
            throw;
        }
    }

    /// <summary>读出一个张量的原始字节。</summary>
    public byte[] ReadRaw(string name) => ReadRaw(Info(name));

    /// <summary>读出一个张量的原始字节。</summary>
    public byte[] ReadRaw(SafeTensorInfo info)
    {
        var buffer = new byte[checked((int)info.ByteLength)];
        ReadRaw(info, buffer);
        return buffer;
    }

    /// <summary>把张量原始字节读进给定缓冲区（长度必须等于 <see cref="SafeTensorInfo.ByteLength"/>）。</summary>
    public void ReadRaw(SafeTensorInfo info, Span<byte> destination)
    {
        if (destination.Length != info.ByteLength)
            throw new OnnxSharpException(
                $"safetensors '{Path}': '{info.Name}' needs {info.ByteLength} B, got {destination.Length} B");

        ReadExactly(_handle, destination, _dataStart + info.Begin, Path);
    }

    /// <summary>
    /// 读成引擎张量：F32 / I64 / Bool 之外的类型会转换（F16 / BF16 / F64 → F32，
    /// 整型 → I64）—— safetensors 的 dtype 契约与 <see cref="DType"/> 不一一对应。
    /// </summary>
    public Tensor GetTensor(string name)
    {
        var info = Info(name);
        int onnxType = SafeTensorFormat.LoadTargetDType(info.DType, $"safetensors '{Path}'", info.Name);
        byte[] raw = SafeTensorFormat.Encode(ReadRaw(info), info.DType, onnxType, info);
        return SafeTensorFormat.ToTensor(raw, onnxType, info);
    }

    /// <summary>把张量编码成指定 ONNX data_type 的 <c>raw_data</c> 字节。</summary>
    internal byte[] EncodeAs(SafeTensorInfo info, int onnxType)
        => SafeTensorFormat.Encode(ReadRaw(info), info.DType, onnxType, info);

    public void Dispose() => _handle.Dispose();

    private static void ReadExactly(SafeFileHandle handle, Span<byte> destination, long offset, string path)
    {
        int total = 0;
        while (total < destination.Length)
        {
            int read = RandomAccess.Read(handle, destination[total..], offset + total);
            if (read <= 0)
                throw new OnnxSharpException($"safetensors '{path}': truncated file at offset {offset + total}");
            total += read;
        }
    }

    private static (SafeTensorInfo[], Dictionary<string, string>) ParseHeader(
        string path, ReadOnlyMemory<byte> header, long dataLength)
    {
        JsonDocument document;
        try
        {
            document = JsonDocument.Parse(header);
        }
        catch (JsonException e)
        {
            throw new OnnxSharpException($"safetensors '{path}': header is not valid JSON ({e.Message})");
        }

        using (document)
        {
            if (document.RootElement.ValueKind != JsonValueKind.Object)
                throw new OnnxSharpException($"safetensors '{path}': header root must be a JSON object");

            var tensors = new List<SafeTensorInfo>();
            var metadata = new Dictionary<string, string>(StringComparer.Ordinal);

            foreach (var property in document.RootElement.EnumerateObject())
            {
                if (property.Name == "__metadata__")
                {
                    foreach (var entry in property.Value.EnumerateObject())
                        metadata[entry.Name] = entry.Value.ValueKind == JsonValueKind.String
                            ? entry.Value.GetString() ?? ""
                            : entry.Value.GetRawText();
                    continue;
                }

                tensors.Add(ParseTensor(path, property.Name, property.Value, dataLength));
            }

            return (tensors.ToArray(), metadata);
        }
    }

    private static SafeTensorInfo ParseTensor(
        string path, string name, JsonElement value, long dataLength)
    {
        if (value.ValueKind != JsonValueKind.Object)
            throw new OnnxSharpException($"safetensors '{path}': tensor '{name}' entry must be an object");

        if (!value.TryGetProperty("dtype", out var dtypeElement) || dtypeElement.ValueKind != JsonValueKind.String)
            throw new OnnxSharpException($"safetensors '{path}': tensor '{name}' has no dtype");
        string dtype = dtypeElement.GetString() ?? "";

        if (!value.TryGetProperty("shape", out var shapeElement) || shapeElement.ValueKind != JsonValueKind.Array)
            throw new OnnxSharpException($"safetensors '{path}': tensor '{name}' has no shape");

        var shape = new long[shapeElement.GetArrayLength()];
        int index = 0;
        foreach (var dim in shapeElement.EnumerateArray())
        {
            if (!dim.TryGetInt64(out long size) || size < 0)
                throw new OnnxSharpException(
                    $"safetensors '{path}': tensor '{name}' has invalid dim '{dim.GetRawText()}'");
            shape[index++] = size;
        }

        if (!value.TryGetProperty("data_offsets", out var offsets)
            || offsets.ValueKind != JsonValueKind.Array
            || offsets.GetArrayLength() != 2
            || !offsets[0].TryGetInt64(out long begin)
            || !offsets[1].TryGetInt64(out long end))
            throw new OnnxSharpException($"safetensors '{path}': tensor '{name}' has no valid data_offsets");

        if (begin < 0 || end < begin || end > dataLength)
            throw new OnnxSharpException(
                $"safetensors '{path}': tensor '{name}' data_offsets [{begin},{end}) "
                + $"out of data section (0..{dataLength})");

        var info = new SafeTensorInfo(name, dtype, shape, begin, end);

        // spec 要求 data_offsets 的大小与 dtype × 元素数严格一致 —— 不一致说明头部被打过补丁
        long expected = checked(info.ElementCount * SafeTensorFormat.SizeOf(dtype, path, name));
        if (end - begin != expected)
            throw new OnnxSharpException(
                $"safetensors '{path}': tensor '{name}' occupies {end - begin} B, "
                + $"but {info.ElementCount} x {dtype} needs {expected} B");

        return info;
    }
}
