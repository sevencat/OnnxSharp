using OnnxSharp.Vision;

namespace OnnxSharp.Tests;

/// <summary>
/// YuNet 人脸检测：与 OpenCV <c>FaceDetectorYN</c> 逐值对拍。
/// <para>
/// 夹具由 <c>tools/scripts/yunet_probe.py &lt;image&gt; .tmp/yunet</c> 生成：把图 resize 到模型的
/// 静态输入 640×640，**同一个输入**同时喂给 cv2 和我们的引擎，所以比的是解码公式而不是预处理。
/// 夹具缺失则 SKIP。
/// </para>
/// </summary>
internal static class YuNetTests
{
    public static IEnumerable<TestCase> Cases()
    {
        yield return new("yunet: box and landmarks match opencv FaceDetectorYN", MatchesOpenCv);
        yield return new("yunet: pixel preprocessing matches the reference blob", PreprocessingMatches);
        yield return new("yunet: score threshold gates the detections", ThresholdGate);
    }

    private static void MatchesOpenCv()
    {
        string? directory = FindFixture();
        if (directory is null)
            throw new TestSkipException(
                "yunet fixture not found (run tools/scripts/yunet_probe.py .tmp/lena.jpg .tmp/yunet)");

        var detector = new YuNetDetector(Path.Combine(directory, "yunet.onnx"));
        byte[] pixels = File.ReadAllBytes(Path.Combine(directory, "yunet_input.bgr"));

        var faces = detector.Detect(pixels, detector.InputWidth, detector.InputHeight);
        var expected = ReadReference(Path.Combine(directory, "yunet_faces.txt"));

        Assert.Equal(expected.Length, faces.Count, "face count");

        double maxBox = 0;
        double maxKeyPoint = 0;
        double maxScore = 0;

        for (int i = 0; i < expected.Length; i++)
        {
            float[] want = expected[i];
            Face got = faces[i];

            maxBox = Math.Max(maxBox, Math.Abs(want[0] - got.X));
            maxBox = Math.Max(maxBox, Math.Abs(want[1] - got.Y));
            maxBox = Math.Max(maxBox, Math.Abs(want[2] - got.Width));
            maxBox = Math.Max(maxBox, Math.Abs(want[3] - got.Height));

            FaceLandmarks marks = got.Landmarks;
            (float X, float Y)[] ours =
            [
                (marks.RightEye.X, marks.RightEye.Y),
                (marks.LeftEye.X, marks.LeftEye.Y),
                (marks.Nose.X, marks.Nose.Y),
                (marks.RightMouth.X, marks.RightMouth.Y),
                (marks.LeftMouth.X, marks.LeftMouth.Y),
            ];

            for (int k = 0; k < 5; k++)
            {
                maxKeyPoint = Math.Max(maxKeyPoint, Math.Abs(want[4 + k * 2] - ours[k].X));
                maxKeyPoint = Math.Max(maxKeyPoint, Math.Abs(want[4 + k * 2 + 1] - ours[k].Y));
            }

            maxScore = Math.Max(maxScore, Math.Abs(want[14] - got.Score));
        }

        Console.WriteLine(
            $"        yunet vs cv2: faces={faces.Count} max|dBox|={maxBox:E2}px " +
            $"max|dKps|={maxKeyPoint:E2}px max|dScore|={maxScore:E2}");

        // 我们的 Conv/GEMM 与 ORT（cv2 的内核）不是逐位一致，留半像素余量；
        // 解码公式错了的话这里会差几十像素。
        Assert.True(maxBox < 0.5, $"max|dBox| {maxBox:E2}px >= 0.5");
        Assert.True(maxKeyPoint < 0.5, $"max|dKps| {maxKeyPoint:E2}px >= 0.5");
        Assert.True(maxScore < 1e-3, $"max|dScore| {maxScore:E2} >= 1e-3");
    }

    /// <summary>
    /// 门面吃的是 0..255 原始像素（不 /255、不减均值），这条用例把「像素 → [1,3,H,W] 张量」
    /// 这一步单独钉住：与 python 侧 <c>cv2.resize + transpose</c> 的结果逐值比。
    /// </summary>
    private static void PreprocessingMatches()
    {
        string? directory = FindFixture();
        if (directory is null)
            throw new TestSkipException(
                "yunet fixture not found (run tools/scripts/yunet_probe.py .tmp/lena.jpg .tmp/yunet)");

        var detector = new YuNetDetector(Path.Combine(directory, "yunet.onnx"));
        byte[] pixels = File.ReadAllBytes(Path.Combine(directory, "yunet_input.bgr"));
        byte[] raw = File.ReadAllBytes(Path.Combine(directory, "yunet_input.f32"));
        var expected = new float[raw.Length / sizeof(float)];
        Buffer.BlockCopy(raw, 0, expected, 0, raw.Length);

        var tensor = detector.CreateInput(pixels, 640, 640, bgr: true);

        ReadOnlySpan<float> got = tensor.AsSpan<float>();
        Assert.Equal(expected.Length, got.Length, "blob length");

        double sum = 0;
        float max = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            float diff = MathF.Abs(expected[i] - got[i]);
            sum += diff;
            if (diff > max)
                max = diff;
        }

        Console.WriteLine($"        preprocessing vs python blob: mean|d|={sum / expected.Length:E2} max|d|={max:E2}");
        Assert.True(max < 1e-3, $"预处理与参考 blob 不一致: max|d| {max:E2}");
    }

    private static void ThresholdGate()
    {
        string? directory = FindFixture();
        if (directory is null)
            throw new TestSkipException(
                "yunet fixture not found (run tools/scripts/yunet_probe.py .tmp/lena.jpg .tmp/yunet)");

        byte[] pixels = File.ReadAllBytes(Path.Combine(directory, "yunet_input.bgr"));
        string modelPath = Path.Combine(directory, "yunet.onnx");

        var strict = new YuNetDetector(modelPath, new FaceDetectorOptions { ScoreThreshold = 0.99f });
        Assert.Equal(0, strict.Detect(pixels, strict.InputWidth, strict.InputHeight).Count,
            "threshold 0.99 should drop the face");

        var loose = new YuNetDetector(modelPath, new FaceDetectorOptions { ScoreThreshold = 0.5f });
        int looseCount = loose.Detect(pixels, loose.InputWidth, loose.InputHeight).Count;
        Assert.True(looseCount >= 1, "threshold 0.5 should keep at least one face");
        Console.WriteLine($"        threshold 0.5 -> {looseCount} face(s), 0.99 -> 0");
    }

    /// <summary>每行 15 个数：x, y, w, h, 5×2 关键点, score（cv2 的原样输出）。</summary>
    private static float[][] ReadReference(string path)
    {
        var rows = new List<float[]>();
        foreach (string line in File.ReadAllLines(path))
        {
            if (string.IsNullOrWhiteSpace(line))
                continue;

            string[] parts = line.Split(' ', StringSplitOptions.RemoveEmptyEntries);
            var values = new float[parts.Length];
            for (int i = 0; i < parts.Length; i++)
                values[i] = float.Parse(parts[i], System.Globalization.CultureInfo.InvariantCulture);

            rows.Add(values);
        }

        return rows.ToArray();
    }

    private static string? FindFixture()
    {
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); dir is not null; dir = dir.Parent)
        {
            if (!File.Exists(Path.Combine(dir.FullName, "OnnxSharp.slnx")))
                continue;

            string candidate = Path.Combine(dir.FullName, ".tmp", "yunet");
            return File.Exists(Path.Combine(candidate, "yunet.onnx")) ? candidate : null;
        }

        return null;
    }
}
