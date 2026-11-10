using System.Globalization;
using OnnxSharp.Core;
using OnnxSharp.Onnx;

namespace OnnxSharp.Speech;

/// <summary>一条语音的特征矩阵（行优先）。</summary>
public readonly struct FeatureMatrix
{
    public required float[] Values { get; init; }

    public required int Frames { get; init; }

    public required int Dim { get; init; }
}

/// <summary>前端参数：fbank + LFR + CMVN。</summary>
public sealed class SpeechFeatureOptions
{
    public FbankOptions Fbank { get; init; } = new();

    public int LfrWindowSize { get; init; } = 7;

    public int LfrWindowShift { get; init; } = 6;

    /// <summary>CMVN 加性项（模型 metadata 的 <c>neg_mean</c>）。</summary>
    public float[]? NegMean { get; init; }

    /// <summary>CMVN 乘性项（模型 metadata 的 <c>inv_stddev</c>）。</summary>
    public float[]? InvStdDev { get; init; }
}

/// <summary>
/// 语音特征前端：<c>wav → fbank(80) → LFR(7,6) → CMVN → [T, 560]</c>，
/// 产出的矩阵直接就是 Paraformer 的 <c>speech</c> 输入。
/// <para>
/// 参数优先从模型 metadata 读（<see cref="FromModel"/>）：本项目的 Paraformer 模型带着
/// <c>lfr_window_size</c> / <c>lfr_window_shift</c> / <c>neg_mean</c> / <c>inv_stddev</c>，
/// 不用额外传 CMVN 文件。
/// </para>
/// </summary>
public sealed class SpeechFeatureExtractor
{
    private readonly FbankExtractor _fbank;
    private readonly LfrCmvn _lfr;

    public SpeechFeatureExtractor(SpeechFeatureOptions? options = null)
    {
        options ??= new SpeechFeatureOptions();

        _fbank = new FbankExtractor(options.Fbank);
        _lfr = new LfrCmvn
        {
            WindowSize = options.LfrWindowSize,
            WindowShift = options.LfrWindowShift,
            Additive = options.NegMean,
            Scale = options.InvStdDev,
        };
    }

    /// <summary>fbank 之后再 LFR：输出帧数 = ceil(fbank 帧数 / lfr_n)。</summary>
    public FeatureMatrix Compute(ReadOnlySpan<float> samples)
    {
        var fbank = _fbank.Compute(samples, out int frames);
        var values = _lfr.Apply(fbank, frames, _fbank.Dimension, out int outFrames);

        return new FeatureMatrix
        {
            Values = values,
            Frames = outFrames,
            Dim = _fbank.Dimension * _lfr.WindowSize,
        };
    }

    /// <summary>直接从 wav 文件算（16 kHz，否则报错）。</summary>
    public FeatureMatrix ComputeFromWav(string path)
    {
        var (samples, sampleRate) = WavReader.Read(path);
        if (sampleRate != _fbank.Options.SampleRate)
            throw new OnnxSharpException(
                $"{path}: sample rate {sampleRate} Hz, expected {_fbank.Options.SampleRate} Hz " +
                "(resample before calling this)");

        return Compute(samples);
    }

    /// <summary>从 ONNX 模型的 metadata 建前端（LFR / CMVN 参数都在里面）。</summary>
    public static SpeechFeatureExtractor FromModel(string modelPath, FbankOptions? fbank = null)
    {
        var file = OnnxFile.Load(modelPath);
        return FromMetadata(file.Metadata, fbank);
    }

    /// <summary>用已经读出来的 metadata 建前端（<c>Net</c> 侧已经有 <c>OnnxFile</c> 时用这个）。</summary>
    public static SpeechFeatureExtractor FromMetadata(
        IReadOnlyDictionary<string, string> metadata,
        FbankOptions? fbank = null)
    {
        int window = metadata.TryGetValue("lfr_window_size", out string? w)
            ? int.Parse(w, CultureInfo.InvariantCulture)
            : 1;
        int shift = metadata.TryGetValue("lfr_window_shift", out string? s)
            ? int.Parse(s, CultureInfo.InvariantCulture)
            : 1;

        float[]? negMean = metadata.TryGetValue("neg_mean", out string? nm) ? ParseFloats(nm) : null;
        float[]? invStdDev = metadata.TryGetValue("inv_stddev", out string? iv) ? ParseFloats(iv) : null;

        return new SpeechFeatureExtractor(new SpeechFeatureOptions
        {
            Fbank = fbank ?? new FbankOptions(),
            LfrWindowSize = window,
            LfrWindowShift = shift,
            NegMean = negMean,
            InvStdDev = invStdDev,
        });
    }

    private static float[] ParseFloats(string text)
    {
        string[] parts = text.Split(',', StringSplitOptions.RemoveEmptyEntries);
        var values = new float[parts.Length];
        for (int i = 0; i < parts.Length; i++)
            values[i] = float.Parse(parts[i], CultureInfo.InvariantCulture);

        return values;
    }
}
