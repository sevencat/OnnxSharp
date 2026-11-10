namespace OnnxSharp.Speech;

/// <summary>
/// fbank 参数，默认值对齐 sherpa-onnx / kaldi-native-fbank 用在 Paraformer 上的配置。
/// </summary>
public sealed class FbankOptions
{
    public int SampleRate { get; init; } = 16000;

    public double FrameLengthMilliseconds { get; init; } = 25;

    public double FrameShiftMilliseconds { get; init; } = 10;

    /// <summary><c>povey</c> / <c>hann</c> / <c>hamming</c> / <c>rectangular</c>。</summary>
    public string WindowType { get; init; } = "povey";

    /// <summary>抖动（kaldi 默认 1.0，sherpa 的推理配置是 0 —— 推理必须确定性）。</summary>
    public double Dither { get; init; }

    public double PreemphasisCoefficient { get; init; } = 0.97;

    public bool RemoveDcOffset { get; init; } = true;

    public bool UseEnergy { get; init; }

    public int MelBins { get; init; } = 80;

    /// <summary>低频截止（Hz）。</summary>
    public double LowFrequency { get; init; } = 20;

    /// <summary>高频截止：<c>&lt;= 0</c> 表示 <c>nyquist + HighFrequency</c>（kaldi 约定）。</summary>
    public double HighFrequency { get; init; } = -400;

    public bool SnipEdges { get; init; } = true;

    public bool UseLogFbank { get; init; } = true;

    /// <summary>帧长（采样点）＝ round(sr × ms / 1000)。</summary>
    public int FrameLength => (int)Math.Round(SampleRate * FrameLengthMilliseconds / 1000.0);

    /// <summary>帧移（采样点）。</summary>
    public int FrameShift => (int)Math.Round(SampleRate * FrameShiftMilliseconds / 1000.0);

    /// <summary>FFT 点数 = 帧长向上取到 2 的幂（kaldi 的 <c>RoundUpToNearestPowerOfTwo</c>）。</summary>
    public int FftSize
    {
        get
        {
            int size = 1;
            while (size < FrameLength)
                size <<= 1;
            return size;
        }
    }

    /// <summary>kaldi 只算到 <c>FftSize / 2</c>（不含 Nyquist）。</summary>
    public int FftBins => FftSize / 2;

    /// <summary>实际的梅尔高频截止（Hz）。</summary>
    public double ResolvedHighFrequency =>
        HighFrequency > 0 ? HighFrequency : SampleRate / 2.0 + HighFrequency;
}
