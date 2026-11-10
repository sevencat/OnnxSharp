using System.Globalization;
using OnnxSharp.Core;
using OnnxSharp.Safetensors;

namespace OnnxSharp.Onnx;

/// <summary>骨架 ONNX 里的一个 initializer（权重/常量）。</summary>
/// <param name="Name">图里的张量名。</param>
/// <param name="OnnxDataType"><c>TensorProto.DataType</c>（1=F32、2=U8、3=I8、6=I32、7=I64、9=Bool、10=F16…）。</param>
/// <param name="Dims">形状。</param>
public sealed record OnnxTensorInfo(string Name, int OnnxDataType, long[] Dims)
{
    public override string ToString() => $"{Name}:{OnnxDataType}[{string.Join(",", Dims)}]";
}

/// <summary>权重补丁的选项。</summary>
public sealed class OnnxWeightPatchOptions
{
    /// <summary>
    /// 显式改名表：键是<b>骨架里的名字</b>，值是 <b>safetensors 里的名字</b>。
    /// 缺省按同名匹配，再按 <see cref="StripPrefixes"/> 去/补前缀。
    /// </summary>
    public Dictionary<string, string> Rename { get; } = new(StringComparer.Ordinal);

    /// <summary>
    /// 尝试去掉（或补上）的公共前缀。训练框架的 <c>state_dict</c> 常带 <c>module.</c>（DDP）、
    /// <c>model.</c>（ultralytics 包了一层）、<c>_orig_mod.</c>（<c>torch.compile</c>）。
    /// </summary>
    public string[] StripPrefixes { get; init; } = { "module.", "model.", "_orig_mod.", "backbone." };

    /// <summary>总字节数超过该值时，权重写进外部数据文件（protobuf 单消息上限 2 GB）。</summary>
    public long ExternalDataThreshold { get; init; } = 1L << 30;

    /// <summary>外部数据文件名（默认 <c>&lt;输出名&gt;.data</c>）。</summary>
    public string? ExternalDataName { get; init; }

    /// <summary>为 true 时，骨架里有、权重里没有的 initializer 视为错误（而不是保留原值）。</summary>
    public bool Strict { get; init; }
}

/// <summary>补丁结果。</summary>
/// <param name="Patched">被权重替换掉的 initializer 数量。</param>
/// <param name="KeptOriginal">保留骨架原值（权重里没有）的数量。</param>
/// <param name="InlineBytes">写进 ONNX 内的权重字节数。</param>
/// <param name="ExternalBytes">写进外部数据文件的权重字节数。</param>
/// <param name="MissingInWeights">骨架有、权重没有的名字。</param>
/// <param name="UnusedWeights">权重有、骨架没用上的名字。</param>
/// <param name="ExternalDataPath">外部数据文件路径（没有则 null）。</param>
public sealed record OnnxWeightPatchReport(
    int Patched,
    int KeptOriginal,
    long InlineBytes,
    long ExternalBytes,
    IReadOnlyList<string> MissingInWeights,
    IReadOnlyList<string> UnusedWeights,
    string? ExternalDataPath);

/// <summary>
/// 把 safetensors 的权重「打补丁」进一份现成的 ONNX 骨架，纯 C#、不依赖 Python。
/// <para>
/// <b>为什么需要骨架</b>：ONNX 文件 = 计算图 + 权重；safetensors 只有权重。
/// 架构（哪层连哪层、BN 是否已折叠、head 怎么排）没法从权重反推，所以图必须由骨架提供。
/// 典型流程：训练侧产出 <c>best.safetensors</c>，部署侧用发布时附带的 <c>skeleton.onnx</c>
/// 生成最终模型 —— 既不需要 Python，也不需要重新导出。
/// </para>
/// <para>
/// <b>实现方式</b>是 protobuf 层的<b>外科手术</b>：只替换 initializer 的数据字段
/// （<c>raw_data</c> / <c>external_data</c> / <c>float_data</c> …），其余字段（节点、
/// value_info、metadata、未知扩展）按原始字节顺序原样带走 ——
/// 因此不会丢 <c>training_info</c>/<c>functions</c>，也不会重排图。
/// </para>
/// </summary>
public static partial class OnnxWeightPatcher
{
    private const int ModelGraphField = 7;
    private const int GraphInitializerField = 5;

    private const int TensorDims = 1;
    private const int TensorDataType = 2;
    private const int TensorName = 8;

    /// <summary>TensorProto 里「承载数据」的字段号，替换时必须整组换掉。</summary>
    private static readonly HashSet<int> TensorDataFields = new()
    {
        4,  // float_data
        5,  // int32_data
        6,  // string_data
        7,  // int64_data
        9,  // raw_data
        10, // double_data
        11, // uint64_data
        13, // external_data
        14, // data_location
    };

    /// <summary>列出骨架里的全部 initializer（不碰权重）。</summary>
    public static IReadOnlyList<OnnxTensorInfo> Inspect(string skeletonPath)
    {
        ArgumentException.ThrowIfNullOrEmpty(skeletonPath);

        var modelFields = ProtoRaw.Parse(File.ReadAllBytes(skeletonPath));
        var entries = ReadInitializers(modelFields, out _, out _);

        var result = new List<OnnxTensorInfo>(entries.Count);
        foreach (var entry in entries)
            result.Add(new OnnxTensorInfo(entry.Name, entry.OnnxType, entry.Dims));
        return result;
    }

    /// <summary>把权重打进骨架并写到 <paramref name="outputPath"/>（必要时另写外部数据文件）。</summary>
    public static OnnxWeightPatchReport Apply(
        string skeletonPath, SafeTensorFile weights, string outputPath, OnnxWeightPatchOptions? options = null)
    {
        ArgumentException.ThrowIfNullOrEmpty(skeletonPath);
        ArgumentNullException.ThrowIfNull(weights);
        ArgumentException.ThrowIfNullOrEmpty(outputPath);

        options ??= new OnnxWeightPatchOptions();

        var (report, model, external) = RewriteCore(
            File.ReadAllBytes(skeletonPath), skeletonPath, weights, outputPath, options);

        string? directory = Path.GetDirectoryName(Path.GetFullPath(outputPath));
        if (directory is { Length: > 0 })
            Directory.CreateDirectory(directory);

        File.WriteAllBytes(outputPath, model);
        if (external is not null && report.ExternalDataPath is { } externalPath)
            File.WriteAllBytes(externalPath, external);

        return report;
    }

    private static (OnnxWeightPatchReport Report, byte[] Model, byte[]? External) RewriteCore(
        byte[] skeleton,
        string skeletonPath,
        SafeTensorFile weights,
        string outputPath,
        OnnxWeightPatchOptions options)
    {
        var modelFields = ProtoRaw.Parse(skeleton);
        var entries = ReadInitializers(modelFields, out int graphIndex, out var graphFields);

        string skeletonDirectory = Path.GetDirectoryName(Path.GetFullPath(skeletonPath)) ?? ".";

        // ---- 1) 逐个 initializer 决定「换成什么数据」----
        var replacements = new Dictionary<int, byte[]>(entries.Count); // graph 字段下标 → 新数据
        var missing = new List<string>();
        var used = new HashSet<string>(StringComparer.Ordinal);
        long totalBytes = 0;
        int patched = 0;

        foreach (var entry in entries)
        {
            if (TryResolve(weights, options, entry.Name, out var info))
            {
                ValidateShape(entry, info);
                replacements[entry.GraphIndex] = weights.EncodeAs(info, entry.OnnxType);
                used.Add(info.Name);
                totalBytes += replacements[entry.GraphIndex].LongLength;
                patched++;
                continue;
            }

            missing.Add(entry.Name);

            // 骨架自己用外部数据文件的话，必须把字节搬进来 —— 否则输出的模型换个目录就废了
            if (TryReadExternalData(entry, skeletonDirectory) is { } external)
            {
                replacements[entry.GraphIndex] = external;
                totalBytes += external.LongLength;
            }
        }

        if (options.Strict && missing.Count > 0)
            throw new OnnxSharpException(
                $"skeleton has {missing.Count} initializer(s) missing from the weights: "
                + string.Join(", ", missing.Take(10)) + (missing.Count > 10 ? " …" : ""));

        // ---- 2) 决定内联还是外部数据 ----
        bool useExternal = totalBytes > options.ExternalDataThreshold;
        string? externalPath = useExternal
            ? Path.Combine(
                Path.GetDirectoryName(Path.GetFullPath(outputPath)) ?? ".",
                options.ExternalDataName ?? (Path.GetFileName(outputPath) + ".data"))
            : null;

        string externalLocation = externalPath is null ? "" : Path.GetFileName(externalPath);
        using var externalStream = useExternal ? new MemoryStream() : null;

        // ---- 3) 重建 graph（只有 initializer 变了，其余原样重放）----
        var graphBytes = new List<byte>((int)Math.Min(int.MaxValue, skeleton.LongLength / 4));
        for (int i = 0; i < graphFields.Count; i++)
        {
            var field = graphFields[i];
            if (field.Number == GraphInitializerField && replacements.TryGetValue(i, out var data))
            {
                RebuildTensor(graphBytes, field, data, externalStream, externalLocation);
                continue;
            }

            ProtoRaw.CopyField(graphBytes, field);
        }

        // ---- 4) 重建 model（只换 graph 字段）----
        var modelBytes = new List<byte>(skeleton.Length);
        for (int i = 0; i < modelFields.Count; i++)
        {
            if (i == graphIndex)
            {
                ProtoRaw.WriteBytesField(modelBytes, ModelGraphField, graphBytes.ToArray());
                continue;
            }

            ProtoRaw.CopyField(modelBytes, modelFields[i]);
        }

        var unused = new List<string>();
        foreach (string name in weights.Names)
        {
            if (!used.Contains(name))
                unused.Add(name);
        }

        var report = new OnnxWeightPatchReport(
            patched,
            entries.Count - patched,
            useExternal ? 0 : totalBytes,
            useExternal ? totalBytes : 0,
            missing,
            unused,
            externalPath);

        return (report, modelBytes.ToArray(), externalStream?.ToArray());
    }

    /// <summary>一个 initializer 在图里的原始视图。</summary>
    private sealed record InitializerEntry(
        int GraphIndex, string Name, int OnnxType, long[] Dims, List<ProtoRawField> Fields);

    /// <summary>从 ModelProto 里取出 graph，再列出 graph 的全部 initializer 及其字段。</summary>
    private static List<InitializerEntry> ReadInitializers(
        List<ProtoRawField> modelFields, out int graphIndex, out List<ProtoRawField> graphFields)
    {
        graphIndex = -1;
        for (int i = 0; i < modelFields.Count; i++)
        {
            if (modelFields[i].Number != ModelGraphField)
                continue;
            graphIndex = i;
            break;
        }

        if (graphIndex < 0)
            throw new OnnxSharpException("ONNX model has no graph (ModelProto field 7) — not a valid ONNX file");

        graphFields = ProtoRaw.Parse(modelFields[graphIndex].Payload);

        var entries = new List<InitializerEntry>();
        for (int i = 0; i < graphFields.Count; i++)
        {
            var field = graphFields[i];
            if (field.Number != GraphInitializerField)
                continue;

            var tensorFields = ProtoRaw.Parse(field.Payload);
            string name = "";
            int onnxType = 0;
            var dims = new List<long>();

            foreach (var tensorField in tensorFields)
            {
                switch (tensorField.Number)
                {
                    case TensorName:
                        name = tensorField.Text;
                        break;
                    case TensorDataType:
                        onnxType = (int)tensorField.Value;
                        break;
                    case TensorDims:
                        if (tensorField.WireType == 2)
                            dims.AddRange(ProtoRaw.ReadPackedInt64(tensorField.Payload.Span));
                        else
                            dims.Add(unchecked((long)tensorField.Value));
                        break;
                }
            }

            // 无名 initializer 无法与权重对应，原样保留（极少见，但别因为一个就整包失败）
            if (name.Length == 0)
                continue;

            entries.Add(new InitializerEntry(i, name, onnxType, dims.ToArray(), tensorFields));
        }

        return entries;
    }

    /// <summary>骨架名 → 权重张量：先看显式改名表，再同名，最后去/补公共前缀。</summary>
    private static bool TryResolve(
        SafeTensorFile weights, OnnxWeightPatchOptions options, string name, out SafeTensorInfo info)
    {
        if (options.Rename.TryGetValue(name, out string? renamed))
            return weights.TryGetInfo(renamed, out info!);

        if (weights.TryGetInfo(name, out info!))
            return true;

        foreach (string prefix in options.StripPrefixes)
        {
            if (name.StartsWith(prefix, StringComparison.Ordinal)
                && weights.TryGetInfo(name[prefix.Length..], out info!))
                return true;

            if (weights.TryGetInfo(prefix + name, out info!))
                return true;
        }

        info = null!;
        return false;
    }

    private static void ValidateShape(InitializerEntry entry, SafeTensorInfo info)
    {
        if (DimsEquivalent(entry.Dims, info.Shape))
            return;

        throw new OnnxSharpException(
            $"initializer '{entry.Name}': skeleton expects [{string.Join(",", entry.Dims)}], "
            + $"weights '{info.Name}' has [{string.Join(",", info.Shape)}]. "
            + "形状不一致通常意味着骨架与训练的架构/类别数不同，或者骨架是用 fuse=True 导出的"
            + "（BatchNorm 已折进 Conv，bn.* 权重没有对应项）；后者请用 fuse=False 重新导出骨架");
    }

    /// <summary>
    /// 形状是否等价：<b>去掉前导 1 后必须逐维相等</b>。
    /// <para>
    /// 为什么放宽前导 1：ONNX 的标量 initializer 写成 <c>dims=[]</c>（rank 0），
    /// 而 numpy/PyTorch 存 safetensors 时常是 <c>(1,)</c> —— 元素数与字节数完全相同，
    /// 只有 rank 写法不同（真实案例：yolo11n 的 <c>/model.10/.../Constant_1_output_0</c>）。
    /// </para>
    /// <para>
    /// 但 <c>[2,3]</c> 与 <c>[3,2]</c> 这种「元素数相同、语义不同」必须报错 —— 否则转置错误会被静默吞掉。
    /// </para>
    /// </summary>
    private static bool DimsEquivalent(long[] skeleton, long[] weights)
    {
        int i = 0;
        while (i < skeleton.Length && skeleton[i] == 1)
            i++;

        int j = 0;
        while (j < weights.Length && weights[j] == 1)
            j++;

        if (skeleton.Length - i != weights.Length - j)
            return false;

        for (int k = 0; k < skeleton.Length - i; k++)
        {
            if (skeleton[i + k] != weights[j + k])
                return false;
        }

        return true;
    }

    /// <summary>用新数据重建一个 TensorProto：只换数据字段，其余（name/dims/data_type…）原样。</summary>
    private static void RebuildTensor(
        List<byte> sink, ProtoRawField original, byte[] data, MemoryStream? external, string externalLocation)
    {
        var rebuilt = new List<byte>(data.Length + 64);

        foreach (var field in ProtoRaw.Parse(original.Payload))
        {
            if (TensorDataFields.Contains(field.Number))
                continue;
            ProtoRaw.CopyField(rebuilt, field);
        }

        if (external is null)
        {
            ProtoRaw.WriteBytesField(rebuilt, 9, data); // raw_data
        }
        else
        {
            // 4096 对齐：外部数据常被 mmap 读取，避免张量跨页
            long aligned = Align(external.Length, 4096);
            while (external.Length < aligned)
                external.WriteByte(0);

            long offset = external.Length;
            external.Write(data, 0, data.Length);

            ProtoRaw.WriteBytesField(rebuilt, 13, StringEntry("location", externalLocation));
            ProtoRaw.WriteBytesField(rebuilt, 13, StringEntry("offset", offset.ToString(CultureInfo.InvariantCulture)));
            ProtoRaw.WriteBytesField(rebuilt, 13, StringEntry("length", data.Length.ToString(CultureInfo.InvariantCulture)));
            ProtoRaw.WriteVarintField(rebuilt, 14, 1); // data_location = EXTERNAL
        }

        ProtoRaw.WriteBytesField(sink, GraphInitializerField, rebuilt.ToArray());
    }

    /// <summary><c>StringStringEntryProto</c>（key=1, value=2）。</summary>
    private static byte[] StringEntry(string key, string value)
    {
        var entry = new List<byte>(32);
        ProtoRaw.WriteBytesField(entry, 1, System.Text.Encoding.UTF8.GetBytes(key));
        ProtoRaw.WriteBytesField(entry, 2, System.Text.Encoding.UTF8.GetBytes(value));
        return entry.ToArray();
    }

    private static long Align(long value, long alignment) => (value + alignment - 1) / alignment * alignment;

    /// <summary>
    /// 读出骨架自带的外部数据（<c>external_data</c> + <c>data_location=EXTERNAL</c>）。
    /// 不搬进来的话，输出模型一换目录就会去错误的位置找权重。
    /// </summary>
    private static byte[]? TryReadExternalData(InitializerEntry entry, string skeletonDirectory)
    {
        string? location = null;
        long offset = 0;
        long length = -1;
        bool isExternal = false;

        foreach (var field in entry.Fields)
        {
            if (field.Number == 14 && field.Value == 1)
                isExternal = true;

            if (field.Number != 13)
                continue;

            string key = "";
            string value = "";
            foreach (var item in ProtoRaw.Parse(field.Payload))
            {
                if (item.Number == 1)
                    key = item.Text;
                else if (item.Number == 2)
                    value = item.Text;
            }

            switch (key)
            {
                case "location":
                    location = value;
                    break;
                case "offset":
                    offset = long.Parse(value, CultureInfo.InvariantCulture);
                    break;
                case "length":
                    length = long.Parse(value, CultureInfo.InvariantCulture);
                    break;
            }
        }

        if (!isExternal)
            return null;
        if (string.IsNullOrEmpty(location))
            throw new OnnxSharpException($"initializer '{entry.Name}' is external but has no location");

        long needed = 1;
        foreach (long dim in entry.Dims)
            needed = checked(needed * dim);
        needed *= SafeTensorFormat.SizeOfOnnxType(entry.OnnxType, entry.Name);
        if (length <= 0 || length < needed)
            length = needed;

        var root = Path.GetFullPath(skeletonDirectory);
        var full = Path.GetFullPath(Path.Combine(root, location));
        if (!full.StartsWith(root + Path.DirectorySeparatorChar, StringComparison.OrdinalIgnoreCase))
            throw new OnnxSharpException($"external data path escapes model directory: '{location}'");

        var data = new byte[checked((int)length)];
        using var stream = File.OpenRead(full);
        stream.Seek(offset, SeekOrigin.Begin);
        int total = 0;
        while (total < data.Length)
        {
            int read = stream.Read(data, total, data.Length - total);
            if (read <= 0)
                throw new OnnxSharpException($"external data '{location}' is truncated at {offset + total}");
            total += read;
        }

        return data;
    }
}

