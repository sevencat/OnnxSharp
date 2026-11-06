using System.Globalization;
using OnnxSharp.Core;
using OnnxSharp.Tensors;

namespace OnnxSharp.Onnx;

/// <summary>ONNX 张量/输入输出的元信息（维度 -1 表示动态）。</summary>
internal readonly record struct OnnxValueInfo(string Name, DType Type, int[] Dims)
{
    public bool HasDynamicDims => Array.IndexOf(Dims, -1) >= 0;

    public override string ToString() => $"{Name}:{Type}[{string.Join(",", Dims)}]";
}

/// <summary>ONNX 节点（尚未转成 LayerInfo）。</summary>
internal sealed class OnnxNodeProto
{
    internal OnnxFile Owner = null!;

    public string Name = "";
    public string OpType = "";
    public string Domain = "";
    public string[] Inputs = Array.Empty<string>();
    public string[] Outputs = Array.Empty<string>();
    public Dictionary<string, Proto> Attributes = new(StringComparer.Ordinal);

    // ---- 属性便捷读取（对应 AttributeProto） ----

    private Proto? Attr(string name) => Attributes.TryGetValue(name, out var a) ? a : null;

    public long Int(string name, long fallback = 0) => Attr(name)?.Int(3, fallback) ?? fallback;

    public long[] Ints(string name) => Attr(name)?.Ints(8) ?? Array.Empty<long>();

    public float Float(string name, float fallback = 0)
    {
        var buffer = Attr(name)?.Buffer(2);
        return buffer is { Length: >= 4 } b ? BitConverter.ToSingle(b.Span[..4]) : fallback;
    }

    public float[] Floats(string name) => Attr(name)?.Floats(7) ?? Array.Empty<float>();

    public string Text(string name, string fallback = "") => Attr(name)?.Text(4, fallback) ?? fallback;

    /// <summary>读取 tensor 类型的属性（如 Constant 节点的 value）。</summary>
    public Tensor? Tensor(string name)
        => Attr(name)?.Buffer(5) is { } buffer ? Owner.DecodeTensor(Proto.Parse(buffer), name) : null;

    public override string ToString() => $"{OpType}({Name})";
}

/// <summary>
/// 已解析的 ONNX 模型文件：只做 protobuf → 结构体，不建图（建图在 OnnxImporter）。
/// </summary>
internal sealed class OnnxFile
{
    public required string Path { get; init; }

    public required string BaseDirectory { get; init; }

    public required int Opset { get; init; }

    public required string Producer { get; init; }

    public required IReadOnlyList<OnnxNodeProto> Nodes { get; init; }

    public required IReadOnlyDictionary<string, Tensor> Initializers { get; init; }

    public required IReadOnlyList<OnnxValueInfo> Inputs { get; init; }

    public required IReadOnlyList<OnnxValueInfo> Outputs { get; init; }

    public required IReadOnlyList<OnnxValueInfo> ValueInfos { get; init; }

    public static OnnxFile Load(string path)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);

        var fullPath = System.IO.Path.GetFullPath(path);
        var baseDir = System.IO.Path.GetDirectoryName(fullPath) ?? ".";

        var bytes = File.ReadAllBytes(fullPath);
        var model = Proto.Parse(bytes.AsMemory());

        var graphMessage = model.Buffer(7)
            ?? throw new OnnxSharpException($"'{path}' is not an ONNX model (no GraphProto)");
        var graph = Proto.Parse(graphMessage);

        int opset = 0;
        foreach (var import in model.Messages(8))
        {
            var domain = import.Text(1);
            if (domain.Length == 0 || domain == "ai.onnx")
                opset = (int)import.Int(2, opset);
        }

        var file = new OnnxFile
        {
            Path = fullPath,
            BaseDirectory = baseDir,
            Opset = opset,
            Producer = model.Text(2),
            Nodes = Array.Empty<OnnxNodeProto>(),
            Initializers = new Dictionary<string, Tensor>(0, StringComparer.Ordinal),
            Inputs = Array.Empty<OnnxValueInfo>(),
            Outputs = Array.Empty<OnnxValueInfo>(),
            ValueInfos = Array.Empty<OnnxValueInfo>(),
        };

        var initializers = new Dictionary<string, Tensor>(StringComparer.Ordinal);
        foreach (var init in graph.Messages(5))
        {
            var name = init.Text(8);
            if (name.Length == 0)
                throw new OnnxSharpException($"initializer without name in '{path}'");
            initializers[name] = file.DecodeTensor(init, name);
        }

        var nodes = new List<OnnxNodeProto>(512);
        foreach (var nodeMessage in graph.Messages(1))
        {
            var node = new OnnxNodeProto
            {
                Owner = file,
                Name = nodeMessage.Text(3),
                OpType = nodeMessage.Text(4),
                Domain = nodeMessage.Text(7),
                Inputs = Strings(nodeMessage, 1),
                Outputs = Strings(nodeMessage, 2),
            };

            foreach (var attrMessage in nodeMessage.Messages(5))
            {
                var key = attrMessage.Text(1);
                if (key.Length != 0)
                    node.Attributes[key] = attrMessage;
            }

            nodes.Add(node);
        }

        return new OnnxFile
        {
            Path = fullPath,
            BaseDirectory = baseDir,
            Opset = opset,
            Producer = file.Producer,
            Nodes = nodes,
            Initializers = initializers,
            Inputs = ValueInfos(graph, 11, initializers),
            Outputs = ValueInfos(graph, 12, null),
            ValueInfos = ValueInfos(graph, 13, null),
        };
    }

    // ---------------- TensorProto ----------------

    /// <summary>把 TensorProto 解码成 <see cref="Tensor"/>（支持 raw_data / typed_data / 外部数据）。</summary>
    public Tensor DecodeTensor(Proto tensorProto, string debugName)
    {
        var dims = tensorProto.Ints(1);
        var shapeDims = new int[dims.Length];
        for (int i = 0; i < dims.Length; i++)
            shapeDims[i] = checked((int)dims[i]);

        var dtype = MapDType((int)tensorProto.Int(2));
        var shape = new TensorShape(shapeDims);
        var tensor = Tensor.Allocate(shape, dtype);

        long count = shape.Total();
        if (count == 0)
            return tensor;

        int need = checked((int)(count * dtype.SizeOf()));

        // 外部数据（onnx 大模型常用）：data_location == EXTERNAL(1)
        if (tensorProto.Int(14) == 1)
        {
            ReadExternalData(tensorProto, tensor, need, debugName);
            return tensor;
        }

        if (tensorProto.Buffer(9) is { } raw)
        {
            if (raw.Length < need)
                throw new OnnxSharpException(
                    $"tensor '{debugName}' raw_data is {raw.Length} bytes, {need} needed ({shape} {dtype})");
            raw.Span[..need].CopyTo(tensor.Buffer);
            return tensor;
        }

        FillFromTypedData(tensorProto, tensor, need, dtype, debugName);
        return tensor;
    }

    private static void FillFromTypedData(Proto proto, Tensor tensor, int need, DType dtype, string debugName)
    {
        switch (dtype)
        {
            case DType.F32:
                {
                    var data = proto.Floats(4);
                    if (data.Length == 0 && proto.Buffer(10) is not null)
                    {
                        // 少数导出器把 FP32 写进 double_data
                        var doubles = proto.Buffers(10).SelectMany(b => Enumerable.Range(0, b.Length / 8)
                            .Select(k => BitConverter.ToDouble(b.Span.Slice(k * 8, 8)))).ToArray();
                        if (doubles.Length * 4 != need)
                            throw new OnnxSharpException($"tensor '{debugName}' has {doubles.Length} doubles, {need / 4} expected");
                        var span = tensor.AsSpan<float>();
                        for (int i = 0; i < span.Length; i++)
                            span[i] = (float)doubles[i];
                        return;
                    }

                    Expect(data.Length * 4, need, debugName, "float_data");
                    data.CopyTo(tensor.AsSpan<float>());
                    return;
                }

            case DType.I64:
                {
                    var data = proto.Ints(7);
                    Expect(data.Length * 8, need, debugName, "int64_data");
                    data.CopyTo(tensor.AsSpan<long>());
                    return;
                }

            case DType.I32:
                {
                    var data = proto.Ints(5);
                    Expect(data.Length * 4, need, debugName, "int32_data");
                    var span = tensor.AsSpan<int>();
                    for (int i = 0; i < span.Length; i++)
                        span[i] = unchecked((int)data[i]);
                    return;
                }

            case DType.I8:
            case DType.U8:
            case DType.Bool:
                {
                    var data = proto.Ints(5);
                    Expect(data.Length, need, debugName, "int32_data");
                    var bytes = tensor.Buffer.AsSpan(0, need);
                    for (int i = 0; i < bytes.Length; i++)
                    {
                        if (dtype == DType.Bool)
                            bytes[i] = data[i] != 0 ? (byte)1 : (byte)0;
                        else if (dtype == DType.I8)
                            bytes[i] = unchecked((byte)(sbyte)data[i]);
                        else
                            bytes[i] = unchecked((byte)data[i]);
                    }
                    return;
                }

            case DType.F16:
                {
                    // FP16 以 uint16 位模式存放在 int32_data 里
                    var data = proto.Ints(5);
                    Expect(data.Length * 2, need, debugName, "int32_data");
                    var span = tensor.AsSpan<ushort>();
                    for (int i = 0; i < span.Length; i++)
                        span[i] = unchecked((ushort)data[i]);
                    return;
                }

            default:
                throw new OnnxSharpException($"tensor '{debugName}' uses unsupported dtype {dtype}");
        }
    }

    private static void Expect(int actualBytes, int needBytes, string name, string source)
    {
        if (actualBytes != needBytes)
            throw new OnnxSharpException(
                $"tensor '{name}' {source} is {actualBytes} bytes, {needBytes} needed");
    }

    private void ReadExternalData(Proto proto, Tensor tensor, int need, string debugName)
    {
        string? location = null;
        long offset = 0;
        long length = -1;

        foreach (var entry in proto.Messages(13))
        {
            switch (entry.Text(1))
            {
                case "location":
                    location = entry.Text(2);
                    break;
                case "offset":
                    offset = long.Parse(entry.Text(2), CultureInfo.InvariantCulture);
                    break;
                case "length":
                    length = long.Parse(entry.Text(2), CultureInfo.InvariantCulture);
                    break;
            }
        }

        if (string.IsNullOrEmpty(location))
            throw new OnnxSharpException($"tensor '{debugName}' is external but has no location");

        // 只允许模型目录内的文件，防止 ONNX 触发任意路径读取
        var root = System.IO.Path.GetFullPath(BaseDirectory);
        var full = System.IO.Path.GetFullPath(System.IO.Path.Combine(root, location));
        if (!full.StartsWith(root + System.IO.Path.DirectorySeparatorChar, StringComparison.OrdinalIgnoreCase))
            throw new OnnxSharpException($"external data path escapes model directory: '{location}'");

        if (length > 0 && length < need)
            need = checked((int)length);

        using var stream = File.OpenRead(full);
        stream.Seek(offset, SeekOrigin.Begin);

        var destination = tensor.Buffer.AsSpan(0, need);
        int total = 0;
        while (total < destination.Length)
        {
            int read = stream.Read(destination[total..]);
            if (read <= 0)
                throw new OnnxSharpException($"external data '{location}' is truncated");
            total += read;
        }
    }

    private static DType MapDType(int onnxType) => onnxType switch
    {
        2 => DType.F32,     // FLOAT
        3 => DType.U8,      // UINT8
        4 => DType.I8,      // INT8
        7 => DType.I32,     // INT32
        8 => DType.I64,     // INT64
        10 => DType.Bool,   // BOOL
        11 => DType.F16,    // FLOAT16
        _ => throw new OnnxSharpException(
            $"unsupported ONNX data_type {onnxType}; supported: 2(F32) 3(U8) 4(I8) 7(I32) 8(I64) 10(Bool) 11(F16)"),
    };

    // ---------------- graph 元信息 ----------------

    private static string[] Strings(Proto message, int field)
    {
        var list = new List<string>(4);
        foreach (var buffer in message.Buffers(field))
        {
            var text = System.Text.Encoding.UTF8.GetString(buffer.Span);
            if (text.Length != 0)
                list.Add(text);
        }
        return list.ToArray();
    }

    private static List<OnnxValueInfo> ValueInfos(
        Proto graph, int field, IReadOnlyDictionary<string, Tensor>? initializers)
    {
        var result = new List<OnnxValueInfo>();
        foreach (var valueInfo in graph.Messages(field))
        {
            var name = valueInfo.Text(1);
            if (name.Length == 0)
                continue;

            // initializer 同时出现在 graph.input 里是合法的，建图时不算模型输入
            if (initializers is not null && initializers.ContainsKey(name))
                continue;

            if (valueInfo.Buffer(2) is not { } typeMessage)
            {
                result.Add(new OnnxValueInfo(name, DType.F32, Array.Empty<int>()));
                continue;
            }

            var type = Proto.Parse(typeMessage);
            if (type.Buffer(1) is not { } tensorTypeMessage)
            {
                // sequence/map/optional 等非张量类型：首版不支持，保留名字占位
                result.Add(new OnnxValueInfo(name, DType.Invalid, Array.Empty<int>()));
                continue;
            }

            var tensorType = Proto.Parse(tensorTypeMessage);
            var dtype = MapDType((int)tensorType.Int(1, 2));

            var dims = Array.Empty<int>();
            if (tensorType.Buffer(2) is { } shapeMessage)
            {
                var shape = Proto.Parse(shapeMessage);
                var list = new List<int>(4);
                foreach (var dim in shape.Messages(1))
                {
                    // dim_value 缺失 = 动态维（dim_param 或完全未知）
                    list.Add(dim.Has(1) ? checked((int)dim.Int(1)) : -1);
                }
                dims = list.ToArray();
            }

            result.Add(new OnnxValueInfo(name, dtype, dims));
        }
        return result;
    }
}
