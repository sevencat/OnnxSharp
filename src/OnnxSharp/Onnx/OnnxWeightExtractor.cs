using System.Buffers.Binary;
using OnnxSharp.Core;
using OnnxSharp.Safetensors;

namespace OnnxSharp.Onnx;

/// <summary>
/// <see cref="OnnxWeightPatcher"/> 的<b>反向</b>：把 ONNX 里的 initializer 抽成 safetensors，
/// 顺便（可选）写出一份「剥掉权重数据」的骨架。
/// <para>
/// 为什么这个方向便宜：ONNX = 图 + 权重，两者都显式存在，拆开只是搬运；
/// 反过来（safetensors + config → ONNX）才需要架构代码，因为图不存在。
/// </para>
/// <para>
/// 注意：剥掉数据的骨架<b>不能直接推理</b>（引擎/ORT 都会说 initializer 长度不对），
/// 它只能配合 <see cref="OnnxWeightPatcher.Apply"/> 使用 —— 这也正是骨架的定义。
/// </para>
/// </summary>
public static partial class OnnxWeightPatcher
{
    /// <summary>抽取结果。</summary>
    /// <param name="Tensors">抽出的张量数。</param>
    /// <param name="WeightBytes">权重总字节数。</param>
    /// <param name="WeightsPath">写出的 safetensors 路径。</param>
    /// <param name="SkeletonPath">写出的骨架路径（没要求则为 null）。</param>
    /// <param name="SkeletonBytes">骨架文件大小。</param>
    public sealed record OnnxExtractReport(
        int Tensors,
        long WeightBytes,
        string WeightsPath,
        string? SkeletonPath,
        long SkeletonBytes);

    /// <summary>
    /// 把 <paramref name="modelPath"/> 的 initializer 抽成 safetensors；
    /// 给 <paramref name="skeletonPath"/> 时再写一份不带权重数据的骨架。
    /// </summary>
    public static OnnxExtractReport Extract(
        string modelPath,
        string weightsPath,
        string? skeletonPath = null,
        IReadOnlyDictionary<string, string>? metadata = null)
    {
        ArgumentException.ThrowIfNullOrEmpty(modelPath);
        ArgumentException.ThrowIfNullOrEmpty(weightsPath);

        var modelFields = ProtoRaw.Parse(File.ReadAllBytes(modelPath));
        var entries = ReadInitializers(modelFields, out int graphIndex, out var graphFields);
        string directory = Path.GetDirectoryName(Path.GetFullPath(modelPath)) ?? ".";

        var writer = new SafeTensorWriter();
        long totalBytes = 0;

        foreach (var entry in entries)
        {
            byte[] data = ReadInitializerBytes(entry, directory);
            string dtype = SafeTensorFormat.ToSafeTensorDType(entry.OnnxType, entry.Name);
            writer.Add(entry.Name, dtype, entry.Dims, data);
            totalBytes += data.Length;
        }

        var meta = new Dictionary<string, string>(StringComparer.Ordinal) { ["format"] = "pt" };
        meta["source"] = Path.GetFileName(modelPath);
        if (metadata is not null)
        {
            foreach (var (key, value) in metadata)
                meta[key] = value;
        }

        string? weightsDirectory = Path.GetDirectoryName(Path.GetFullPath(weightsPath));
        if (weightsDirectory is { Length: > 0 })
            Directory.CreateDirectory(weightsDirectory);

        writer.Save(weightsPath, meta);

        long skeletonBytes = 0;
        if (skeletonPath is not null)
        {
            byte[] skeleton = StripInitializerData(modelFields, graphIndex, graphFields);
            File.WriteAllBytes(skeletonPath, skeleton);
            skeletonBytes = skeleton.LongLength;
        }

        return new OnnxExtractReport(entries.Count, totalBytes, weightsPath, skeletonPath, skeletonBytes);
    }

    /// <summary>重放模型，但把每个 initializer 的数据字段（raw_data / *_data / external_data）丢掉。</summary>
    private static byte[] StripInitializerData(
        List<ProtoRawField> modelFields, int graphIndex, List<ProtoRawField> graphFields)
    {
        var graphBytes = new List<byte>(graphFields.Count * 64);

        foreach (var field in graphFields)
        {
            if (field.Number != GraphInitializerField)
            {
                ProtoRaw.CopyField(graphBytes, field);
                continue;
            }

            var stripped = new List<byte>(64);
            foreach (var tensorField in ProtoRaw.Parse(field.Payload))
            {
                if (TensorDataFields.Contains(tensorField.Number))
                    continue;
                ProtoRaw.CopyField(stripped, tensorField);
            }

            ProtoRaw.WriteBytesField(graphBytes, GraphInitializerField, stripped.ToArray());
        }

        var modelBytes = new List<byte>(modelFields.Count * 8);
        for (int i = 0; i < modelFields.Count; i++)
        {
            if (i == graphIndex)
            {
                ProtoRaw.WriteBytesField(modelBytes, ModelGraphField, graphBytes.ToArray());
                continue;
            }

            ProtoRaw.CopyField(modelBytes, modelFields[i]);
        }

        return modelBytes.ToArray();
    }
}
