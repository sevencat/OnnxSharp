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

    /// <summary>仅用于测试/工具：构造一个空的模型壳子（<see cref="DecodeTensor"/> 只需要 BaseDirectory）。</summary>
    internal static OnnxFile CreateBare(string baseDirectory) => new()
    {
        Path = "",
        BaseDirectory = baseDirectory,
        Opset = 0,
        Producer = "",
        Nodes = Array.Empty<OnnxNodeProto>(),
        Initializers = new Dictionary<string, Tensor>(0, StringComparer.Ordinal),
        Inputs = Array.Empty<OnnxValueInfo>(),
        Outputs = Array.Empty<OnnxValueInfo>(),
        ValueInfos = Array.Empty<OnnxValueInfo>(),
    };

    /// <summary>
    /// 返回一份<b>把动态维固定成具体值</b>的副本（ONNX 里 <c>dim_param</c> / 缺失 <c>dim_value</c> 都解析成 -1）。
    /// <para>
    /// <b>为什么需要</b>：PaddleOCR 的 det / rec 导出是<b>全动态</b>图（<c>x[N,3,H,W]</c>），
    /// 引擎的常量折叠与形状推断都依赖静态维 —— <c>Shape → Slice → Concat → Reshape</c> 链在
    /// 输入维未知时折不出常量，<c>Reshape/Resize</c> 的目标形状就永远是「运行时才知道」，整图卡死。
    /// 调用方给出实际推理尺寸后，这些链在导入期就能折成常量，等价于「为这个尺寸特化一份图」。
    /// </para>
    /// <para>
    /// <paramref name="shapes"/> 的键是图输入名；值是完整形状（长度必须与模型声明一致，每维 &gt; 0）。
    /// 传 null / 空字典时原样返回。返回的副本与原对象共享节点与权重（只换输入元信息）。
    /// </para>
    /// </summary>
    public OnnxFile WithInputShapes(IReadOnlyDictionary<string, int[]>? shapes)
    {
        if (shapes is null || shapes.Count == 0)
            return this;

        var inputs = new List<OnnxValueInfo>(Inputs.Count);
        foreach (var info in Inputs)
            inputs.Add(shapes.TryGetValue(info.Name, out var wanted) ? Specialize(info, wanted) : info);

        // 有输入没给形状且仍是动态：后面推断必然失败，这里提前说清楚是哪一路、当前是什么。
        foreach (var info in inputs)
        {
            if (!info.HasDynamicDims)
                continue;
            throw new OnnxSharpException(
                $"input '{info.Name}' still has dynamic dims [{string.Join(",", info.Dims)}]; " +
                $"pass a concrete shape for it (model inputs: {string.Join(", ", Inputs)})");
        }

        // 少数导出把「图输入」同时写进 value_info；同名条目一起换掉，免得静态形状被旧值覆盖。
        var valueInfos = new List<OnnxValueInfo>(ValueInfos.Count);
        foreach (var info in ValueInfos)
            valueInfos.Add(shapes.TryGetValue(info.Name, out var wanted) ? Specialize(info, wanted) : info);

        return new OnnxFile
        {
            Path = Path,
            BaseDirectory = BaseDirectory,
            Opset = Opset,
            Producer = Producer,
            Nodes = Nodes,
            Initializers = Initializers,
            Inputs = inputs,
            Outputs = Outputs,
            ValueInfos = valueInfos,
        };
    }

    private static OnnxValueInfo Specialize(OnnxValueInfo info, int[] wanted)
    {
        if (wanted.Length != info.Dims.Length)
            throw new OnnxSharpException(
                $"input '{info.Name}': model declares rank {info.Dims.Length} " +
                $"[{string.Join(",", info.Dims)}], shape override has rank {wanted.Length} " +
                $"[{string.Join(",", wanted)}]");

        var dims = new int[wanted.Length];
        for (int i = 0; i < dims.Length; i++)
        {
            if (wanted[i] <= 0)
                throw new OnnxSharpException(
                    $"input '{info.Name}': shape override [{string.Join(",", wanted)}] has non-positive dim #{i}");
            dims[i] = wanted[i];
        }

        return info with { Dims = dims };
    }

    public static OnnxFile Load(string path)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);

        var fullPath = System.IO.Path.GetFullPath(path);
        var baseDir = System.IO.Path.GetDirectoryName(fullPath) ?? ".";

        return Parse(File.ReadAllBytes(fullPath).AsMemory(), fullPath, baseDir);
    }

    /// <summary>
    /// 从<b>内存</b>里的 ONNX 字节加载模型（例如 nuget 模型包的嵌入资源，不必先落盘成临时文件）。
    /// <paramref name="baseDirectory"/> 只在模型带外部数据时才会用到。
    /// </summary>
    public static OnnxFile Load(ReadOnlyMemory<byte> model, string baseDirectory = ".", string displayName = "<memory>")
    {
        if (model.IsEmpty)
            throw new OnnxSharpException("empty ONNX model");

        return Parse(model, displayName, System.IO.Path.GetFullPath(baseDirectory));
    }

    private static OnnxFile Parse(ReadOnlyMemory<byte> content, string path, string baseDir)
    {
        var model = Proto.Parse(content);

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
            Path = path,
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
            Path = path,
            BaseDirectory = baseDir,
            Opset = opset,
            Producer = file.Producer,
            Nodes = nodes,
            Initializers = initializers,
            Inputs = ReadValueInfos(graph, 11, initializers),
            Outputs = ReadValueInfos(graph, 12, null),
            ValueInfos = ReadValueInfos(graph, 13, null),
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

        var dtype = MapDType((int)tensorProto.Int(2), $"tensor '{debugName}'");
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

    /// <summary>
    /// ONNX <c>TensorProto.DataType</c> → <see cref="DType"/>。
    /// 注意该枚举从 <c>FLOAT = 1</c> 起（UNDEFINED 占 0）。
    /// </summary>
    private static DType MapDType(int onnxType, string context) => onnxType switch
    {
        1 => DType.F32,     // FLOAT
        2 => DType.U8,      // UINT8
        3 => DType.I8,      // INT8
        6 => DType.I32,     // INT32
        7 => DType.I64,     // INT64
        9 => DType.Bool,    // BOOL
        10 => DType.F16,    // FLOAT16
        _ => throw new OnnxSharpException(
            $"{context}: unsupported ONNX data_type {onnxType}; supported: 1(F32) 2(U8) 3(I8) 6(I32) 7(I64) 9(Bool) 10(F16)"),
    };

    // ---------------- graph 元信息 ----------------

    /// <summary>
    /// 读取 repeated string 字段。
    /// <para>
    /// <b>必须保留空串</b>：ONNX 用 <c>""</c> 占位「可选输入」（如
    /// <c>Resize(X, "", scales)</c>），丢掉空串会让后面的输入整体错位。
    /// </para>
    /// </summary>
    private static string[] Strings(Proto message, int field)
    {
        var list = new List<string>(4);
        foreach (var buffer in message.Buffers(field))
            list.Add(System.Text.Encoding.UTF8.GetString(buffer.Span));
        return list.ToArray();
    }

    private static List<OnnxValueInfo> ReadValueInfos(
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
            var dtype = MapDType((int)tensorType.Int(1, 2), $"value_info '{name}'");

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
