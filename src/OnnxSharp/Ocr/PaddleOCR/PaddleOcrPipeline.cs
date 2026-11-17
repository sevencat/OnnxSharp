using System.Diagnostics;
using OnnxSharp.Core;
using OnnxSharp.Tensors;

namespace OnnxSharp.Ocr.PaddleOCR;

/// <summary>
/// PaddleOCR 三段式流水线：DB 检测 →（可选）文本行方向分类 → CTC 识别。
/// <para>
/// <b>用法</b>：
/// <code>
/// var model = PaddleOcrModel.FromFiles("det.onnx", "rec.onnx", "ppocr_keys.txt", "cls.onnx");
/// var ocr = new PaddleOcrPipeline(model);
/// foreach (var line in ocr.Run(bgra, width, height, OcrPixelFormat.Bgra32, stride))
///     Console.WriteLine(line);
/// </code>
/// </para>
/// <para>
/// <b>为什么要「按尺寸份图」</b>：PP-OCRv6 的 det / rec 都是全动态导出（<c>x[N,3,H,W]</c>），
/// 引擎靠 <see cref="Net.ReadModel(string, int[])"/> 把输入维固定成具体值才能折叠形状链。
/// 于是每个 <c>(H,W)</c> 组合对应一份特化后的图：检测按 32 对齐、识别按 16 分桶；
/// 这里按形状缓存（默认 det 2 份 / rec 8 份），同一尺寸的后续调用不需要重新导入。
/// </para>
/// <para>不做线程安全：一个实例串行使用；多线程请各自持有一个实例（模型字节可以共享）。</para>
/// </summary>
public sealed class PaddleOcrPipeline
{
    private const int DetSizeStep = 32;
    private const int RecWidthStep = 16;
    private const int ClsWidth = 160;
    private const int ClsHeight = 80;

    private readonly PaddleOcrModel _model;
    private readonly Dictionary<long, Net> _detNets = new();
    private readonly Dictionary<int, Net> _recNets = new();
    private Net? _clsNet;

    private float[] _detBuffer = Array.Empty<float>();
    private float[] _recBuffer = Array.Empty<float>();
    private readonly float[] _clsBuffer = new float[3 * ClsHeight * ClsWidth];

    // 喂给引擎的输入张量按形状缓存：引擎的 SetInput 只引用不拷贝，所以可以原地复用。
    // 键与对应 Net 缓存的键一致，清 Net 缓存时一起清。
    private readonly Dictionary<int, Tensor> _detInputs = new();
    private readonly Dictionary<int, Tensor> _recInputs = new();
    private readonly Dictionary<int, Tensor> _clsInputs = new();

    public PaddleOcrPipeline(PaddleOcrModel model, PaddleOcrOptions? options = null)
    {
        _model = model ?? throw new ArgumentNullException(nameof(model));
        Options = options ?? new PaddleOcrOptions();
    }

    /// <summary>识别流程参数（可在构造后继续改，下一帧生效）。</summary>
    public PaddleOcrOptions Options { get; }

    public PaddleOcrModel Model => _model;

    /// <summary>
    /// 上一帧的分段耗时；仅当 <see cref="PaddleOcrOptions.CollectTimings"/> 为 <c>true</c> 时才更新，
    /// 否则一直是 <c>null</c>（关掉计时就不付 <see cref="Stopwatch"/> 的代价）。
    /// </summary>
    public OcrTimings? LastTimings { get; private set; }

    /// <summary>
    /// 整页识别：返回按阅读顺序排列的文本行（坐标是<b>源图像素</b>）。
    /// </summary>
    /// <param name="pixels">交错像素内存；每行 <paramref name="rowStride"/> 字节（0 = 按宽紧凑排列）。</param>
    /// <param name="width">图像宽（像素）。</param>
    /// <param name="height">图像高（像素）。</param>
    /// <param name="format">像素排列，默认 BGR24（OpenCV / 多数相机 SDK）。</param>
    /// <param name="rowStride">源图每行字节数（含行尾对齐填充）；0 表示 <c>width * 每像素字节数</c>。</param>
    public IReadOnlyList<OcrLine> Run(
        ReadOnlySpan<byte> pixels,
        int width,
        int height,
        OcrPixelFormat format = OcrPixelFormat.Bgr24,
        int rowStride = 0)
    {
        if (width <= 0 || height <= 0)
            throw new ArgumentException($"invalid image size {width}x{height}");

        Stopwatch? stage = Options.CollectTimings ? Stopwatch.StartNew() : null;
        double preMs = 0, detMs = 0, postMs = 0, clsMs = 0, recMs = 0;

        // ---- 1) DB 检测 ----
        List<DbPostprocess.Box> boxes = DetectBoxes(
            pixels, width, height, format, rowStride, stage,
            out preMs, out detMs, out postMs);

        // ---- 2) 逐框裁剪 → 方向分类 → 识别 ----
        bool useClassification = Options.UseClassification && _model.HasClassification;

        var lines = new List<OcrLine>(boxes.Count);
        foreach (DbPostprocess.Box scaled in boxes)
        {
            int x = Math.Clamp((int)MathF.Floor(scaled.X), 0, width - 1);
            int y = Math.Clamp((int)MathF.Floor(scaled.Y), 0, height - 1);
            int right = Math.Clamp((int)MathF.Ceiling(scaled.X + scaled.Width), x + 1, width);
            int bottom = Math.Clamp((int)MathF.Ceiling(scaled.Y + scaled.Height), y + 1, height);

            int cropWidth = right - x;
            int cropHeight = bottom - y;
            if (cropWidth < 4 || cropHeight < 4)
                continue;

            bool rotate = useClassification &&
                IsUpsideDown(pixels, width, height, format, rowStride, x, y, cropWidth, cropHeight);

            if (stage is not null)
            {
                clsMs += stage.Elapsed.TotalMilliseconds;
                stage.Restart();
            }

            (string text, float score) = Recognize(
                pixels, width, height, format, rowStride, x, y, cropWidth, cropHeight, rotate);

            if (stage is not null)
            {
                recMs += stage.Elapsed.TotalMilliseconds;
                stage.Restart();
            }

            if (text.Length != 0)
                lines.Add(new OcrLine(text, score, x, y, cropWidth, cropHeight));
        }

        if (stage is not null)
            LastTimings = new OcrTimings(preMs, detMs, postMs, clsMs, recMs);

        return lines;
    }

    // ---------------- 检测 ----------------

    /// <summary>
    /// 只跑 DB 检测，返回<b>源图像素坐标</b>里的文本框（按阅读顺序）。
    /// 车牌模块用它拿到候选框后自己做受限解码，不必先走一遍通用 CTC。
    /// </summary>
    internal List<DbPostprocess.Box> DetectBoxes(
        ReadOnlySpan<byte> pixels,
        int width,
        int height,
        OcrPixelFormat format,
        int rowStride)
        => DetectBoxes(pixels, width, height, format, rowStride, null, out _, out _, out _);

    private List<DbPostprocess.Box> DetectBoxes(
        ReadOnlySpan<byte> pixels,
        int width,
        int height,
        OcrPixelFormat format,
        int rowStride,
        Stopwatch? stage,
        out double preMs,
        out double detMs,
        out double postMs)
    {
        preMs = detMs = postMs = 0;

        (int detWidth, int detHeight) = DetectSize(width, height);
        Net detNet = GetDetectionNet(detWidth, detHeight);
        EnsureDetBuffer(3 * detWidth * detHeight);

        OcrImage.ToChw(pixels, width, height, format, rowStride,
            0, 0, width, height, detWidth, detHeight,
            OcrImage.DetMean, OcrImage.DetStd, _detBuffer, detWidth);

        // 喂给引擎的张量按形状复用：每次新建会分配 3 MB（>85 KB 直接进 LOH），
        // 几十帧就攒够一次 Gen2，端到端延迟出现数百毫秒的尖刺（实测单帧 991 ms）。
        Tensor detInput = InputFor(_detInputs, (detHeight << 16) | detWidth,
            new TensorShape(1, 3, detHeight, detWidth));
        _detBuffer.AsSpan(0, 3 * detHeight * detWidth).CopyTo(detInput.AsSpan<float>());

        if (stage is not null)
        {
            preMs = stage.Elapsed.TotalMilliseconds;
            stage.Restart();
        }

        detNet.SetInput(detNet.InputNames[0], detInput);
        Tensor probability = detNet.Forward();

        if (stage is not null)
        {
            detMs = stage.Elapsed.TotalMilliseconds;
            stage.Restart();
        }

        List<DbPostprocess.Box> boxes = DbPostprocess.Detect(
            probability.AsSpan<float>(),
            detWidth, detHeight,
            Options.DetThreshold, Options.BoxThreshold, Options.UnclipRatio);

        if (stage is not null)
        {
            postMs = stage.Elapsed.TotalMilliseconds;
            stage.Restart();
        }

        // 概率图坐标 → 源图像素坐标
        float scaleX = (float)width / detWidth;
        float scaleY = (float)height / detHeight;

        for (int i = 0; i < boxes.Count; i++)
            boxes[i] = DbPostprocess.Scale(boxes[i], scaleX, scaleY, width, height);

        return boxes;
    }

    /// <summary>
    /// 检测输入尺寸：最长边压到 <see cref="PaddleOcrOptions.DetLimitSide"/>，
    /// 宽高都取 32 的整数倍（PaddleOCR 的 <c>limit_type=min</c> 规则）。
    /// </summary>
    internal (int Width, int Height) DetectSize(int width, int height)
    {
        int limit = Math.Max(DetSizeStep, Options.DetLimitSide);
        float ratio = MathF.Min(1f, limit / (float)Math.Max(width, height));

        int detWidth = Math.Max(DetSizeStep, (int)(MathF.Round(width * ratio / DetSizeStep) * DetSizeStep));
        int detHeight = Math.Max(DetSizeStep, (int)(MathF.Round(height * ratio / DetSizeStep) * DetSizeStep));
        return (detWidth, detHeight);
    }

    private Net GetDetectionNet(int width, int height)
    {
        long key = ((long)width << 32) | (uint)height;
        if (_detNets.TryGetValue(key, out Net? net))
            return net;

        if (_detNets.Count >= Math.Max(1, Options.DetShapeCacheSize))
        {
            _detNets.Clear();
            _detInputs.Clear();
        }

        net = _model.CreateDetectionNet(width, height);
        _detNets[key] = net;
        return net;
    }

    /// <summary>取一个按 <paramref name="key"/> 复用的输入张量（形状不符时重建）。</summary>
    private static Tensor InputFor(Dictionary<int, Tensor> cache, int key, TensorShape shape)
    {
        if (cache.TryGetValue(key, out Tensor? tensor) && tensor.Shape.Equals(shape))
            return tensor;

        tensor = Tensor.Allocate(shape, DType.F32);
        cache[key] = tensor;
        return tensor;
    }

    private void EnsureDetBuffer(int size)
    {
        if (_detBuffer.Length < size)
            _detBuffer = new float[size];
    }

    // ---------------- 方向分类 ----------------

    private bool IsUpsideDown(
        ReadOnlySpan<byte> pixels,
        int width,
        int height,
        OcrPixelFormat format,
        int rowStride,
        int x,
        int y,
        int cropWidth,
        int cropHeight)
    {
        _clsNet ??= _model.CreateClassificationNet(ClsWidth, ClsHeight);

        OcrImage.ToChw(pixels, width, height, format, rowStride,
            x, y, cropWidth, cropHeight, ClsWidth, ClsHeight,
            OcrImage.Half, OcrImage.Half, _clsBuffer, ClsWidth);

        Tensor clsInput = InputFor(_clsInputs, (ClsHeight << 16) | ClsWidth,
            new TensorShape(1, 3, ClsHeight, ClsWidth));
        _clsBuffer.CopyTo(clsInput.AsSpan<float>());

        _clsNet.SetInput(_clsNet.InputNames[0], clsInput);
        ReadOnlySpan<float> logits = _clsNet.Forward().AsSpan<float>();
        if (logits.Length < 2)
            return false;

        // cls 图末尾是 2 类 softmax：p1 就是「倒置」的概率
        float p1 = logits[1];
        if (logits[0] > 1.0001f || logits[1] > 1.0001f)
        {
            float max = MathF.Max(logits[0], logits[1]);
            float e0 = MathF.Exp(logits[0] - max);
            float e1 = MathF.Exp(logits[1] - max);
            p1 = e1 / (e0 + e1);
        }

        return p1 > Options.UpsideDownThreshold;
    }

    // ---------------- 识别 ----------------

    /// <summary>
    /// 识别一块<b>已经裁好</b>的文本行区域（流程与 <see cref="Run"/> 内部相同）。
    /// 暴露为 internal 是给回归测试直接覆盖 rec 模型用的。
    /// </summary>
    internal (string Text, float Score) Recognize(
        ReadOnlySpan<byte> pixels,
        int width,
        int height,
        OcrPixelFormat format,
        int rowStride,
        int x,
        int y,
        int cropWidth,
        int cropHeight,
        bool rotate)
    {
        (Tensor logits, int classCount) = RecognizeLogits(
            pixels, width, height, format, rowStride, x, y, cropWidth, cropHeight, rotate);

        return CtcDecoder.Decode(logits.AsSpan<float>(), classCount, _model.Classes);
    }

    /// <summary>
    /// 只跑到 rec 网络输出（logits / 概率，行主序 <c>steps × classCount</c>），解码交给调用方。
    /// 车牌模块用它在同一份 rec 图上做受限解码，避免重复 import 图。
    /// <para>返回的 <see cref="Tensor"/> 是引擎的输出缓冲，下一次 <c>Forward</c> 会被覆盖。</para>
    /// </summary>
    internal (Tensor Logits, int ClassCount) RecognizeLogits(
        ReadOnlySpan<byte> pixels,
        int width,
        int height,
        OcrPixelFormat format,
        int rowStride,
        int x,
        int y,
        int cropWidth,
        int cropHeight,
        bool rotate)
    {
        int recHeight = Math.Max(8, Options.RecHeight);
        int maxWidth = Math.Max(RecWidthStep, Options.RecMaxWidth);

        // 等比缩放到高 recHeight；宽度按 16 分桶（桶宽以外补 0，与 PaddleOCR 的 padding 一致）
        int resizedWidth = Math.Clamp(
            (int)MathF.Round(recHeight * (float)cropWidth / cropHeight),
            RecWidthStep, maxWidth);
        int bucket = Math.Min(maxWidth, ((resizedWidth + RecWidthStep - 1) / RecWidthStep) * RecWidthStep);

        Net net = GetRecognitionNet(recHeight, bucket);

        int size = 3 * recHeight * bucket;
        if (_recBuffer.Length < size)
            _recBuffer = new float[size];
        Array.Clear(_recBuffer, 0, size);

        OcrImage.ToChw(pixels, width, height, format, rowStride,
            x, y, cropWidth, cropHeight, resizedWidth, recHeight,
            OcrImage.Half, OcrImage.Half, _recBuffer, bucket, rotate);

        Tensor recInput = InputFor(_recInputs, (recHeight << 16) | bucket,
            new TensorShape(1, 3, recHeight, bucket));
        _recBuffer.AsSpan(0, size).CopyTo(recInput.AsSpan<float>());

        net.SetInput(net.InputNames[0], recInput);

        return (net.Forward(), _model.Classes.Count);
    }

    private Net GetRecognitionNet(int height, int width)
    {
        int key = (height << 16) | width;
        if (_recNets.TryGetValue(key, out Net? net))
            return net;

        if (_recNets.Count >= Math.Max(1, Options.RecShapeCacheSize))
        {
            _recNets.Clear();
            _recInputs.Clear();
        }

        net = _model.CreateRecognitionNet(width, height);
        _recNets[key] = net;
        return net;
    }
}
