using OnnxSharp.Core;

namespace OnnxSharp.Ocr.PaddleOCR;

/// <summary>
/// 涓€濂?PaddleOCR 妯″瀷锛坉et / rec / 鍙€?cls + 瀛楀吀锛夈€?
/// <para>
/// 妯″瀷鍙互<b>瀛楄妭</b>褰㈠紡鎸佹湁锛氭棦鍙互浠?ONNX 鏂囦欢璇伙紝涔熷彲浠ョ洿鎺ュ悆 nuget 妯″瀷鍖?
/// 锛?c>Sdcb.SimdPaddleOCR.Models.ChineseV6Tiny</c> / <c>TextLineOrientation</c>锛夐噷鐨勫祵鍏ヨ祫婧?鈥斺€?
/// 閭ｄ簺鍖呭彧鎻愪緵 ONNX 娴佷笌瀛楀吀锛屼笉鍚帹鐞嗕唬鐮侊紝鎵€浠ュ拰鏈紩鎿庝笉鍐茬獊銆?
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

    /// <summary>妯″瀷鍚嶏紙浠呯敤浜庢棩蹇楋級銆?/summary>
    public string Name { get; }

    /// <summary>CTC 绫诲埆琛細绱㈠紩 0 = blank锛屾湯浣?= 绌烘牸锛圥addleOCR 绾﹀畾锛夈€?/summary>
    public IReadOnlyList<string> Classes => _classes;

    /// <summary>鏄惁甯︽枃鏈鏂瑰悜鍒嗙被妯″瀷銆?/summary>
    public bool HasClassification => _classification is not null;

    /// <summary>浠?ONNX 鏂囦欢 + 瀛楀吀鏂囦欢鍔犺浇銆?/summary>
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

    /// <summary>浠庢祦鍔犺浇锛堟ā鍨嬪寘鐨?<c>OpenRead()</c> 鍙互鐩存帴浼犺繘鏉ワ級銆?/summary>
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

    /// <summary>浠庡唴瀛樺瓧鑺傚姞杞斤紙瀛楀吀鏄枃鏈唴瀹癸紝涓嶆槸璺緞锛夈€?/summary>
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
    /// PaddleOCR 瀛楀吀瑙ｆ瀽锛氭瘡琛屼竴涓瓧绗︼紙绌鸿涔熻淇濈暀 鈥斺€?瀛楀吀閲岀‘瀹炴湁绌虹櫧瀛楃锛夛紝
    /// 绱㈠紩 0 琛?blank锛屾湯灏捐ˉ涓€涓┖鏍硷紙<c>use_space_char</c>锛夈€?
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

        // 鏂囦欢鏈熬鐨勬崲琛屼細澶氳В鏋愬嚭涓€鏉＄┖涓诧紝鍘绘帀瀹冿紙涓棿鐨勭┖琛屼繚鐣欙級銆?
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
