using OnnxSharp.Core;

namespace OnnxSharp.Ocr.PaddleOCR;

/// <summary>
/// 一套 PaddleOCR 模型（det / rec / 可选 cls + 字典）。
/// <para>
/// 模型只以<b>字节</b>形式持有：既可以从 ONNX 文件读，也可以直接吃 nuget 模型包
/// （<c>Sdcb.SimdPaddleOCR.Models.ChineseV6Tiny</c> / <c>TextLineOrientation</c>）里的嵌入资源 ——
/// 那些包只提供 ONNX 流与字典，不含推理代码，所以和本引擎不冲突。
/// </para>
/// </summary>
public sealed class PaddleOcrModel
{
    private readonly byte[] _detection;
    private readonly byte[] _recognition;
    private readonly byte[]? _classification;
    private readonly string[] _classes;

    private PaddleOcrModel(byte[] detection, byte[] recognition, byte[]? classification, string[] classes, string name)
    {
        _detection = detection;
        _recognition = recognition;
        _classification = classification;
        _classes = classes;
        Name = name;
    }

    /// <summary>模型名（仅用于日志）。</summary>
    public string Name { get; }

    /// <summary>CTC 类别表：索引 0 = blank，末位 = 空格（PaddleOCR 约定）。</summary>
    public IReadOnlyList<string> Classes => _classes;

    /// <summary>是否带文本行方向分类模型。</summary>
    public bool HasClassification => _classification is not null;

    /// <summary>从 ONNX 文件 + 字典文件加载。</summary>
    public static PaddleOcrModel FromFiles(
        string detectionPath,
        string recognitionPath,
        string dictionaryPath,
        string? classificationPath = null)
    {
        ArgumentException.ThrowIfNullOrEmpty(detectionPath);
        ArgumentException.ThrowIfNullOrEmpty(recognitionPath);
        ArgumentException.ThrowIfNullOrEmpty(dictionaryPath);

        return FromBytes(
            File.ReadAllBytes(detectionPath),
            File.ReadAllBytes(recognitionPath),
            File.ReadAllText(dictionaryPath),
            classificationPath is null ? null : File.ReadAllBytes(classificationPath),
            Path.GetFileNameWithoutExtension(detectionPath));
    }

    /// <summary>从流加载（模型包的 <c>OpenRead()</c> 可以直接传进来）。</summary>
    public static PaddleOcrModel FromStreams(
        Stream detection,
        Stream recognition,
        TextReader dictionary,
        Stream? classification = null,
        string? name = null)
    {
        ArgumentNullException.ThrowIfNull(detection);
        ArgumentNullException.ThrowIfNull(recognition);
        ArgumentNullException.ThrowIfNull(dictionary);

        return FromBytes(
            ReadAll(detection),
            ReadAll(recognition),
            dictionary.ReadToEnd(),
            classification is null ? null : ReadAll(classification),
            name ?? "paddleocr");
    }

    /// <summary>从内存字节加载（字典是文本内容，不是路径）。</summary>
    public static PaddleOcrModel FromBytes(
        byte[] detection,
        byte[] recognition,
        string dictionary,
        byte[]? classification = null,
        string name = "paddleocr")
    {
        ArgumentNullException.ThrowIfNull(detection);
        ArgumentNullException.ThrowIfNull(recognition);
        ArgumentNullException.ThrowIfNull(dictionary);

        string[] classes = ParseDictionary(dictionary);
        if (classes.Length < 2)
            throw new ArgumentException("dictionary is empty", nameof(dictionary));

        return new PaddleOcrModel(detection, recognition, classification, classes, name);
    }

    /// <summary>
    /// PaddleOCR 字典解析：每行一个字符（空行也要保留 —— 字典里确实有空白字符），
    /// 索引 0 表示 blank，末尾补一个空格（<c>use_space_char</c>）。
    /// </summary>
    private static string[] ParseDictionary(string text)
    {
        var raw = new List<string>();
        int start = 0;
        while (start <= text.Length)
        {
            int end = text.IndexOf('\n', start);
            if (end < 0)
                end = text.Length;

            raw.Add(text[start..end].TrimEnd('\r'));
            start = end + 1;
            if (end == text.Length)
                break;
        }

        // 文件末尾的换行会多解析出一条空项，去掉它（中间的空行保留）。
        if (raw.Count > 0 && raw[^1].Length == 0)
            raw.RemoveAt(raw.Count - 1);

        var classes = new string[raw.Count + 2];
        classes[0] = string.Empty;
        for (int i = 0; i < raw.Count; i++)
            classes[i + 1] = raw[i];
        classes[^1] = " ";
        return classes;
    }

    private static byte[] ReadAll(Stream stream)
    {
        if (stream is MemoryStream memory)
            return memory.ToArray();

        using var buffer = new MemoryStream();
        stream.CopyTo(buffer);
        return buffer.ToArray();
    }

    internal Net CreateDetectionNet(int width, int height)
        => Net.ReadModel(_detection, new[] { 1, 3, height, width });

    internal Net CreateRecognitionNet(int width, int height)
        => Net.ReadModel(_recognition, new[] { 1, 3, height, width });

    internal Net CreateClassificationNet(int width, int height)
        => _classification is null
            ? throw new InvalidOperationException("this PaddleOCR model has no classification model")
            : Net.ReadModel(_classification, new[] { 1, 3, height, width });

    public override string ToString()
        => $"{Name} classes={_classes.Length} cls={(HasClassification ? "on" : "off")}";
}
