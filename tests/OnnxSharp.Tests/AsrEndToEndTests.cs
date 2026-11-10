using OnnxSharp.Speech;
using OnnxSharp.Speech.Paraformer;

namespace OnnxSharp.Tests;

/// <summary>
/// M9-08/09：token 解码 + 端到端识别。
/// fixture：`tools/scripts/tts_huihui.ps1` 生成中文 wav、`sherpa_e2e_reference.py` 用 ORT 跑完整模型得到参考文本，
/// `split_*.py` 切出 encoder/decoder。缺任何一个就 SKIP。
/// </summary>
internal static class AsrEndToEndTests
{
    public static IEnumerable<TestCase> Cases()
    {
        yield return new("asr: token decode drops blanks, folds repeats and stops at EOS", TokenDecodeRule);
        yield return new("asr: huihui wav transcribes to the ORT reference text", HuihuiEndToEnd);
    }

    private static void TokenDecodeRule()
    {
        var table = TokenTable.Load(FindTokens() ?? throw new TestSkipException("tokens.txt not found"));

        // 0 = blank、2 = EOS；8301 连续重复要折叠
        long[] raw = [5308, 0, 8301, 8301, 4474, 2, 554, 2986];
        var ids = table.DecodeIds(raw);

        Assert.Equal(3, ids.Count, "id count");
        Assert.Equal(5308, ids[0], "[0]");
        Assert.Equal(8301, ids[1], "[1] folded repeat");
        Assert.Equal(4474, ids[2], "[2]");

        // EOS 之后的一律丢弃
        Assert.True(ids[^1] == 4474, "stop at EOS");
    }

    private static void HuihuiEndToEnd()
    {
        string? models = FindModels();
        string? wav = FindWav();
        string? expectedPath = FindReferenceText();
        if (models is null || wav is null || expectedPath is null)
            throw new TestSkipException("asr fixture not found (see AsrEndToEndTests header)");

        string expected = File.ReadAllText(expectedPath).Trim();

        var recognizer = new ParaformerRecognizer(new ParaformerOptions
        {
            EncoderPath = Path.Combine(models, "encoder.onnx"),
            DecoderPath = Path.Combine(models, "decoder.onnx"),
            TokensPath = Path.Combine(models, "tokens.txt"),
        });

        var result = recognizer.TranscribeWav(wav);

        Console.WriteLine(
            $"        asr: {result.FeatureFrames} frames → {result.TokenCount} tokens → '{result.Text}'");
        Console.WriteLine(
            $"        time: total {result.TotalMs:F0} ms (feat {result.FeatureMs:F0} / enc {result.EncoderMs:F0} " +
            $"/ dec {result.DecoderMs:F0}; load {result.EncoderLoadMs:F0}+{result.DecoderLoadMs:F0})");

        Assert.Equal(expected, result.Text, "text");
        Assert.True(result.TokenCount > 0, "token count");
        Assert.True(result.FeatureFrames > 0, "feature frames");
    }

    private static string? FindTokens() => FindModels() is { } models
        ? Path.Combine(models, "tokens.txt")
        : null;

    private static string? FindModels()
    {
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); dir is not null; dir = dir.Parent)
        {
            if (!File.Exists(Path.Combine(dir.FullName, "OnnxSharp.slnx")))
                continue;

            string candidate = Path.Combine(dir.FullName, ".tmp", "sherpa");
            return File.Exists(Path.Combine(candidate, "decoder.onnx")) &&
                   File.Exists(Path.Combine(candidate, "tokens.txt"))
                ? candidate
                : null;
        }

        return null;
    }

    private static string? FindWav()
    {
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); dir is not null; dir = dir.Parent)
        {
            if (!File.Exists(Path.Combine(dir.FullName, "OnnxSharp.slnx")))
                continue;

            string candidate = Path.Combine(dir.FullName, ".tmp", "asr", "huihui.wav");
            return File.Exists(candidate) ? candidate : null;
        }

        return null;
    }

    private static string? FindReferenceText()
    {
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); dir is not null; dir = dir.Parent)
        {
            if (!File.Exists(Path.Combine(dir.FullName, "OnnxSharp.slnx")))
                continue;

            string candidate = Path.Combine(dir.FullName, ".tmp", "asr", "e2e_ref_text.txt");
            return File.Exists(candidate) ? candidate : null;
        }

        return null;
    }
}
