using OnnxSharp.Core;

namespace OnnxSharp.Speech;

/// <summary>
/// kaldi 风格的 fbank（对数梅尔滤波器组能量），逐项对齐 kaldi-native-fbank：
/// 分帧 → 去直流 → 预加重 → 加窗（povey）→ 补零到 2 的幂 → 实 FFT → 功率谱 → 三角梅尔滤波 → 取对数。
/// <para>
/// 与 kaldi 几处「必须照抄」的细节：梅尔刻度是 <c>1127·ln(1+f/700)</c>；三角滤波器的权重在
/// <b>梅尔域</b>按「每个 FFT bin 的中心频率」算；只算到 <c>FftSize/2</c>（不含 Nyquist）；
/// 取对数前的下限是 <c>1.1920929e-7</c>（<c>float epsilon</c>，不是 <c>float.Epsilon</c>）。
/// </para>
/// </summary>
public sealed class FbankExtractor
{
    private const float LogFloor = 1.1920929E-07f;

    private readonly FbankOptions _options;
    private readonly float[] _window;
    private readonly int[][] _filterBins;
    private readonly float[][] _filterWeights;
    private readonly float[] _cosine;
    private readonly float[] _sine;

    public FbankExtractor(FbankOptions? options = null)
    {
        _options = options ?? new FbankOptions();

        if (_options.SampleRate <= 0 || _options.FrameLength <= 0 || _options.FrameShift <= 0)
            throw new OnnxSharpException("Fbank: invalid frame options");
        if (_options.MelBins <= 0)
            throw new OnnxSharpException("Fbank: MelBins must be positive");

        _window = BuildWindow(_options);

        int fftSize = _options.FftSize;
        int half = fftSize / 2;
        _cosine = new float[half];
        _sine = new float[half];
        for (int i = 0; i < half; i++)
        {
            double angle = -2.0 * Math.PI * i / fftSize;
            _cosine[i] = (float)Math.Cos(angle);
            _sine[i] = (float)Math.Sin(angle);
        }

        BuildMelFilters(_options, out _filterBins, out _filterWeights);
    }

    public FbankOptions Options => _options;

    /// <summary>输出维度（梅尔 bin 数）。</summary>
    public int Dimension => _options.MelBins;

    /// <summary>帧数（与 kaldi 的 <c>NumFrames</c> 一致）。</summary>
    public int FrameCount(int sampleCount)
    {
        long frameLength = _options.FrameLength;
        long frameShift = _options.FrameShift;

        if (_options.SnipEdges)
            return sampleCount < frameLength ? 0 : (int)(1 + (sampleCount - frameLength) / frameShift);

        long frames = (sampleCount + frameShift / 2) / frameShift;
        long end = FirstSampleOfFrame(frames - 1) + frameLength;
        while (frames > 0 && end > sampleCount)
        {
            frames--;
            end -= frameShift;
        }

        return (int)frames;
    }

    private long FirstSampleOfFrame(long frame)
        => _options.SnipEdges ? frame * _options.FrameShift : frame * _options.FrameShift - _options.FrameShift / 2;

    /// <summary>算 fbank，返回 <c>[frames, Dimension]</c>（行优先）。</summary>
    public float[] Compute(ReadOnlySpan<float> samples, out int frames)
    {
        frames = FrameCount(samples.Length);
        var features = new float[(long)frames * _options.MelBins];

        int frameLength = _options.FrameLength;
        int fftSize = _options.FftSize;
        int fftBins = _options.FftBins;

        var re = new float[fftSize];
        var im = new float[fftSize];
        var power = new float[fftBins];
        var time = new float[frameLength];

        for (int frame = 0; frame < frames; frame++)
        {
            long start = FirstSampleOfFrame(frame);

            // 1) 取帧（超出范围补 0）
            for (int i = 0; i < frameLength; i++)
            {
                long index = start + i;
                time[i] = (uint)index < (uint)samples.Length ? samples[(int)index] : 0f;
            }

            // 2) 抖动（推理默认关闭）
            if (_options.Dither != 0)
                AddDither(time, _options.Dither, frame);

            // 3) 去直流
            if (_options.RemoveDcOffset)
            {
                double sum = 0;
                for (int i = 0; i < frameLength; i++)
                    sum += time[i];

                float mean = (float)(sum / frameLength);
                for (int i = 0; i < frameLength; i++)
                    time[i] -= mean;
            }

            // 4) 预加重
            if (_options.PreemphasisCoefficient != 0)
                Preemphasize(time, (float)_options.PreemphasisCoefficient);

            // 5) 加窗
            for (int i = 0; i < frameLength; i++)
                time[i] *= _window[i];

            // 6) FFT（补零到 fftSize）
            Array.Clear(re);
            Array.Clear(im);
            time.CopyTo(re, 0);
            Forward(re, im);

            // 7) 功率谱（kaldi 只到 FftSize/2，不含 Nyquist）
            for (int i = 0; i < fftBins; i++)
                power[i] = re[i] * re[i] + im[i] * im[i];

            // 8) 梅尔三角滤波 + 取对数（累加用 float，与 kaldi 的 MelBanks::Compute 一致）
            long offset = (long)frame * _options.MelBins;
            for (int bin = 0; bin < _options.MelBins; bin++)
            {
                var indices = _filterBins[bin];
                var weights = _filterWeights[bin];

                float energy = 0f;
                for (int t = 0; t < indices.Length; t++)
                    energy += power[indices[t]] * weights[t];

                features[offset + bin] = _options.UseLogFbank
                    ? MathF.Log(MathF.Max(energy, LogFloor))
                    : energy;
            }
        }

        return features;
    }

    /// <summary>原地基 2 FFT（实数输入补零到 <paramref name="re"/>.Length）。</summary>
    private void Forward(float[] re, float[] im)
    {
        int n = re.Length;

        for (int i = 1, j = 0; i < n; i++)
        {
            int bit = n >> 1;
            for (; (j & bit) != 0; bit >>= 1)
                j ^= bit;
            j ^= bit;

            if (i < j)
            {
                (re[i], re[j]) = (re[j], re[i]);
                (im[i], im[j]) = (im[j], im[i]);
            }
        }

        int half = n / 2;
        for (int length = 2; length <= n; length <<= 1)
        {
            int halfLength = length / 2;
            int step = n / length;

            for (int start = 0; start < n; start += length)
            {
                for (int k = 0; k < halfLength; k++)
                {
                    int twiddle = k * step % half;
                    float wr = _cosine[twiddle];
                    float wi = _sine[twiddle];

                    int a = start + k;
                    int b = a + halfLength;

                    float tr = re[b] * wr - im[b] * wi;
                    float ti = re[b] * wi + im[b] * wr;

                    re[b] = re[a] - tr;
                    im[b] = im[a] - ti;
                    re[a] += tr;
                    im[a] += ti;
                }
            }
        }
    }

    private static void Preemphasize(Span<float> data, float coefficient)
    {
        for (int i = data.Length - 1; i > 0; i--)
            data[i] -= coefficient * data[i - 1];
        data[0] -= coefficient * data[0];
    }

    /// <summary>抖动：用帧号驱动的 LCG，保证同一输入可复现（kaldi 默认用全局 RNG，不可复现）。</summary>
    private static void AddDither(Span<float> data, double dither, int frame)
    {
        uint state = (uint)(0x9E3779B9u * (uint)(frame + 1)) | 1u;
        for (int i = 0; i < data.Length; i++)
        {
            state = state * 1664525u + 1013904223u;
            double uniform = (state >> 8) / (double)(1 << 24);
            data[i] += (float)(dither * (2.0 * uniform - 1.0));
        }
    }

    private static float[] BuildWindow(FbankOptions options)
    {
        int length = options.FrameLength;
        var window = new float[length];
        double cycle = 2.0 * Math.PI / (length - 1);

        for (int i = 0; i < length; i++)
        {
            window[i] = options.WindowType switch
            {
                "povey" => (float)Math.Pow(0.5 - 0.5 * Math.Cos(cycle * i), 0.85),
                "hann" or "hanning" => (float)(0.5 - 0.5 * Math.Cos(cycle * i)),
                "hamming" => (float)(0.54 - 0.46 * Math.Cos(cycle * i)),
                "rectangular" or "rect" => 1f,
                _ => throw new OnnxSharpException($"Fbank: unsupported window type '{options.WindowType}'"),
            };
        }

        return window;
    }

    /// <summary>
    /// 按 kaldi 的 <c>MelBanks</c> 构造：权重在梅尔域、按每个 FFT bin 的中心频率算。
    /// <para>
    /// <b>全程用 float</b>（连 <c>MelScale</c> 也是）：用 double 算会让「恰好落在三角边上的 FFT bin」
    /// 的<b>包含关系</b>翻转，实测在响亮 bin 上造成 ~3e-4 的相对能量差。照抄 float 运算后降到 1e-6 量级。
    /// </para>
    /// </summary>
    private static void BuildMelFilters(FbankOptions options, out int[][] bins, out float[][] weights)
    {
        float sampleFrequency = options.SampleRate;
        float windowLengthPadded = options.FftSize;
        int numFftBins = options.FftBins;                 // window_length_padded / 2
        float nyquist = 0.5f * sampleFrequency;

        float lowFrequency = (float)options.LowFrequency;
        float highFrequency = options.HighFrequency > 0
            ? (float)options.HighFrequency
            : nyquist + (float)options.HighFrequency;

        float fftBinWidth = sampleFrequency / windowLengthPadded;
        float melLow = MelScale(lowFrequency);
        float melHigh = MelScale(highFrequency);

        int numBins = options.MelBins;
        float delta = (melHigh - melLow) / (numBins + 1);

        bins = new int[numBins][];
        weights = new float[numBins][];

        for (int bin = 0; bin < numBins; bin++)
        {
            float left = melLow + bin * delta;
            float center = melLow + (bin + 1) * delta;
            float right = melLow + (bin + 2) * delta;

            var indices = new List<int>(16);
            var values = new List<float>(16);

            for (int i = 0; i < numFftBins; i++)
            {
                float mel = MelScale(fftBinWidth * i);
                if (!(mel > left && mel < right))
                    continue;

                float weight = mel <= center
                    ? (mel - left) / (center - left)
                    : (right - mel) / (right - center);

                indices.Add(i);
                values.Add(weight);
            }

            bins[bin] = indices.ToArray();
            weights[bin] = values.ToArray();
        }
    }

    private static float MelScale(float frequency) => 1127.0f * MathF.Log(1.0f + frequency / 700.0f);
}
