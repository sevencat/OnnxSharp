using OnnxSharp.Speech.Paraformer;

namespace OnnxSharp.Tests;

/// <summary>
/// M9-06：用 C# 重写的 CIF（Paraformer 的 integrate-and-excite）与模型里的 ONNX <c>Loop</c> body 逐值对齐。
/// 金标准由 <c>tools/scripts/sherpa_cif_probe.py</c> 录（把 Loop 的中间张量加成 graph output）。
/// </summary>
internal static class CifTests
{
    public static IEnumerable<TestCase> Cases()
    {
        yield return new("cif: candidates/fires/leftovers match the ONNX loop body", MatchesLoopBody);
        yield return new("cif: token count matches the model output", TokenCountMatches);
    }

    private static void MatchesLoopBody()
    {
        string? directory = FindFixture();
        if (directory is null)
            throw new TestSkipException("cif fixture not found (run tools/scripts/sherpa_cif_probe.py)");

        const int frames = 97, dim = 320;

        var encoder = Read(Path.Combine(directory, "ort__predictor_Concat_5_output_0.f32"));
        var alphas = Read(Path.Combine(directory, "ort__predictor_Add_output_0.f32"));

        var cif = new ContinuousIntegrateAndFire();
        var result = cif.Process(encoder, frames, dim, alphas);

        // frames：每个 step 的候选帧（模型里的 `frames`）
        var expectedFrames = Read(Path.Combine(directory, "ort_frames.f32"));
        Assert.Equal(expectedFrames.Length, result.Candidates.Length, "frames length");
        double frameMax = MaxDiff(expectedFrames, result.Candidates);

        // fires：每个 step 累加后的 alpha
        var expectedFires = Read(Path.Combine(directory, "ort_fires.f32"));
        Assert.Equal(expectedFires.Length, result.FireAlphas.Length, "fires length");
        double fireMax = MaxDiff(expectedFires, result.FireAlphas);

        // frame.7 / integrate.7：循环结束时的残留
        var expectedLeftoverFrame = Read(Path.Combine(directory, "ort_frame_7.f32"));
        double leftoverMax = MaxDiff(expectedLeftoverFrame, result.LeftoverFrame);

        var expectedLeftoverAlpha = Read(Path.Combine(directory, "ort_integrate_7.f32"));
        double alphaDiff = MathF.Abs(expectedLeftoverAlpha[0] - result.LeftoverAlpha);

        Console.WriteLine(
            $"        cif: frames {frameMax:E2}, fires {fireMax:E2}, frame.7 {leftoverMax:E2}, " +
            $"integrate.7 {alphaDiff:E2}, tokens {result.TokenCount}");

        Assert.True(frameMax < 1e-5, $"frames max|diff| {frameMax:E2} >= 1e-5");
        Assert.True(fireMax < 1e-6, $"fires max|diff| {fireMax:E2} >= 1e-6");
        Assert.True(leftoverMax < 1e-5, $"frame.7 max|diff| {leftoverMax:E2} >= 1e-5");
        Assert.True(alphaDiff < 1e-6, $"integrate.7 diff {alphaDiff:E2} >= 1e-6");
    }

    private static void TokenCountMatches()
    {
        string? directory = FindFixture();
        if (directory is null)
            throw new TestSkipException("cif fixture not found (run tools/scripts/sherpa_cif_probe.py)");

        const int frames = 97, dim = 320;

        var encoder = Read(Path.Combine(directory, "ort__predictor_Concat_5_output_0.f32"));
        var alphas = Read(Path.Combine(directory, "ort__predictor_Add_output_0.f32"));
        var expectedTokens = (int)Read(Path.Combine(directory, "ort_token_num.f32"))[0];

        var result = new ContinuousIntegrateAndFire().Process(encoder, frames, dim, alphas);

        Console.WriteLine($"        cif: token count {result.TokenCount} (ORT {expectedTokens})");
        Assert.Equal(expectedTokens, result.TokenCount, "token count");

        // 选中帧与「候选帧按触发下标取子集」必须一致
        Assert.Equal(result.TokenCount * dim, result.Tokens.Length, "tokens length");
        for (int i = 0; i < result.FireIndices.Length; i++)
        {
            int step = result.FireIndices[i];
            for (int d = 0; d < dim; d++)
            {
                if (result.Tokens[i * dim + d] != result.Candidates[step * dim + d])
                    throw new TestFailureException($"tokens[{i},{d}] 不等于 candidates[{step},{d}]");
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

    private static double MaxDiff(float[] expected, float[] actual)
    {
        Assert.Equal(expected.Length, actual.Length, "length");
        double max = 0;
        for (int i = 0; i < expected.Length; i++)
            max = Math.Max(max, Math.Abs((double)expected[i] - actual[i]));
        return max;
    }

    private static string? FindFixture()
    {
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); dir is not null; dir = dir.Parent)
        {
            if (!File.Exists(Path.Combine(dir.FullName, "OnnxSharp.slnx")))
                continue;

            string candidate = Path.Combine(dir.FullName, ".tmp", "sherpa");
            return File.Exists(Path.Combine(candidate, "ort_frames.f32")) ? candidate : null;
        }

        return null;
    }
}
