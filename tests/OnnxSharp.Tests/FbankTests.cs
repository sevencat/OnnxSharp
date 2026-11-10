using OnnxSharp.Speech;

namespace OnnxSharp.Tests;

/// <summary>
/// M9-02：纯 C# 的语音前端（fbank → LFR → CMVN）。
/// fbank 的金标准由 <c>tools/scripts/fbank_probe.py</c> 用 **kaldi-native-fbank**（sherpa 内部同一个库）生成；
/// LFR+CMVN 用 funasr <c>WavFrontend</c> 的语义生成，参数取自模型 metadata。
/// </summary>
internal static class FbankTests
{
    public static IEnumerable<TestCase> Cases()
    {
        yield return new("fbank: matches kaldi-native-fbank frames", MatchesKaldi);
        yield return new("fbank: LFR + CMVN matches funasr semantics", MatchesFunasrLfr);
    }

    private static void MatchesKaldi()
    {
        string? directory = FindFixture();
        if (directory is null)
            throw new TestSkipException("fbank fixture not found (run tools/scripts/fbank_probe.py)");

        var (samples, sampleRate) = WavReader.Read(Path.Combine(directory, "fbank.wav"));
        Assert.Equal(16000, sampleRate, "sample rate");

        var extractor = new FbankExtractor();
        var features = extractor.Compute(samples, out int frames);

        var golden = Read(Path.Combine(directory, "fbank_golden.f32"));
        int goldenFrames = golden.Length / 80;

        Assert.Equal(goldenFrames, frames, "frame count");
        Assert.Equal(80, extractor.Dimension, "dimension");

        double max = 0;
        int worst = -1;
        for (int i = 0; i < golden.Length; i++)
        {
            double diff = Math.Abs(golden[i] - features[i]);
            if (diff > max)
            {
                max = diff;
                worst = i / 80;
            }
        }

        Console.WriteLine($"        fbank {frames}x80: max|diff| = {max:E2} (worst frame {worst})");

        // 诊断用：把我们的 fbank 落盘，方便在 Python 侧按能量域看相对误差
        File.WriteAllBytes(Path.Combine(directory, "ours_fbank.f32"), ToBytes(features));

        // 判据分三层：
        //  ① 中位数要非常小（浮点级一致）；② 有声 bin 的绝对差要小；
        //  ③ 静音 bin（能量贴着 log 下限 1.19e-7）允许较大对数值差 ——
        //     那里 |Δlog| ≈ Δenergy/energy，1e-8 的绝对能量差就能放大成 1e-2。
        double median = Median(golden, features);
        double audibleMax = 0;
        int audible = 0;
        for (int i = 0; i < golden.Length; i++)
        {
            if (MathF.Exp(golden[i]) <= 1e-3f)
                continue;

            audible++;
            audibleMax = Math.Max(audibleMax, Math.Abs((double)golden[i] - features[i]));
        }

        Console.WriteLine($"        median|diff| = {median:E2}, audible bins {audible}, max|diff| there = {audibleMax:E2}");

        Assert.True(median < 1e-5, $"median|diff| {median:E2} >= 1e-5");
        Assert.True(audibleMax < 1e-4, $"audible max|diff| {audibleMax:E2} >= 1e-4");
    }

    private static double Median(float[] expected, float[] actual)
    {
        var diffs = new double[expected.Length];
        for (int i = 0; i < expected.Length; i++)
            diffs[i] = Math.Abs((double)expected[i] - actual[i]);

        Array.Sort(diffs);
        return diffs[diffs.Length / 2];
    }

    private static void MatchesFunasrLfr()
    {
        string? directory = FindFixture();
        if (directory is null)
            throw new TestSkipException("fbank fixture not found (run tools/scripts/fbank_probe.py)");

        var (samples, _) = WavReader.Read(Path.Combine(directory, "fbank.wav"));

        var negMean = Read(Path.Combine(directory, "cmvn_neg_mean.f32"));
        var invStdDev = Read(Path.Combine(directory, "cmvn_inv_stddev.f32"));

        var extractor = new SpeechFeatureExtractor(new SpeechFeatureOptions
        {
            LfrWindowSize = 7,
            LfrWindowShift = 6,
            NegMean = negMean,
            InvStdDev = invStdDev,
        });

        var features = extractor.Compute(samples);
        Assert.Equal(560, features.Dim, "LFR dimension");

        var golden = Read(Path.Combine(directory, "lfr_golden.f32"));
        Assert.Equal(golden.Length / 560, features.Frames, "LFR frame count");

        double max = 0;
        for (int i = 0; i < golden.Length; i++)
            max = Math.Max(max, Math.Abs(golden[i] - features.Values[i]));

        double median = Median(golden, features.Values);
        Console.WriteLine($"        lfr {features.Frames}x560: median|diff| = {median:E2}, max|diff| = {max:E2}");

        // 与 fbank 同样的口径：中位数必须浮点级一致；静音 bin 经对数+CMVN 放大后允许到 1e-2
        Assert.True(median < 1e-5, $"LFR median|diff| {median:E2} >= 1e-5");
        Assert.True(max < 1e-2, $"LFR max|diff| {max:E2} >= 1e-2");

        // funasr 的左补 3 帧：第 0 个输出帧 = [f0,f0,f0,f0,f1,f2,f3]
        var fbank = new FbankExtractor().Compute(samples, out int fbankFrames);
        for (int window = 0; window < 7; window++)
        {
            int expectedFrame = Math.Clamp(window - 3, 0, fbankFrames - 1);
            for (int d = 0; d < 80; d++)
            {
                float expected = (fbank[expectedFrame * 80 + d] + negMean[window * 80 + d]) * invStdDev[window * 80 + d];
                float actual = features.Values[window * 80 + d];
                Assert.Near(expected, actual, 1e-5f, $"row0 window {window} dim {d}");
            }
        }
    }

    private static float[] Read(string path)
    {
        byte[] bytes = File.ReadAllBytes(path);
        var values = new float[bytes.Length / sizeof(float)];
        Buffer.BlockCopy(bytes, 0, values, 0, bytes.Length);
        return values;
    }

    private static byte[] ToBytes(float[] values)
    {
        var bytes = new byte[values.Length * sizeof(float)];
        Buffer.BlockCopy(values, 0, bytes, 0, bytes.Length);
        return bytes;
    }

    private static string? FindFixture()
    {
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); dir is not null; dir = dir.Parent)
        {
            if (!File.Exists(Path.Combine(dir.FullName, "OnnxSharp.slnx")))
                continue;

            string candidate = Path.Combine(dir.FullName, ".tmp", "fbank");
            return File.Exists(Path.Combine(candidate, "fbank_golden.f32")) ? candidate : null;
        }

        return null;
    }
}
