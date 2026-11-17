using OnnxSharp.Ocr.PaddleOCR;
using OnnxSharp.Tensors;

namespace OnnxSharp.Ocr.Plate;

/// <summary>车牌读取的参数。</summary>
public sealed class PlateReaderOptions
{
    /// <summary>解码参数（beam 宽度、位数范围、是否用结构先验）。</summary>
    public PlateDecoderOptions Decoder { get; init; } = new();

    /// <summary>逐字几何平均分的下限；低于它按「不是车牌」处理。</summary>
    public float MinScore { get; init; } = 0.5f;

    /// <summary>候选框的宽高比下限（号牌约 440×140 ≈ 3.1）。</summary>
    public float MinAspect { get; init; } = 1.6f;

    /// <summary>候选框的宽高比上限。</summary>
    public float MaxAspect { get; init; } = 6.5f;

    /// <summary>是否先用宽高比筛掉明显不是车牌的文本行（省掉一次 rec 前向）。</summary>
    public bool FilterByAspect { get; init; } = true;

    /// <summary>一帧最多返回几个车牌。</summary>
    public int MaxPlates { get; init; } = 4;
}

/// <summary>一张车牌：解码结果 + 它在源图里的位置。</summary>
public sealed class PlateResult
{
    internal PlateResult(PlateDecoding decoding, int x, int y, int width, int height)
    {
        Decoding = decoding;
        X = x;
        Y = y;
        Width = width;
        Height = height;
    }

    /// <summary>完整解码结果（逐字概率、对齐时间步都在这里）。</summary>
    public PlateDecoding Decoding { get; }

    /// <summary>归一化后的号牌文本。</summary>
    public string Text => Decoding.Text;

    /// <summary>逐字几何平均分。</summary>
    public float Score => Decoding.Score;

    /// <summary>结构化号牌（省 / 发牌机关 / 序号 / 类型）。</summary>
    public PlateNumber? Plate => Decoding.Plate;

    public int X { get; }
    public int Y { get; }
    public int Width { get; }
    public int Height { get; }

    public override string ToString() => $"{Text} ({Score:P1}) [{X},{Y} {Width}x{Height}]";
}

/// <summary>
/// 车牌读取器：复用 <see cref="PaddleOcrPipeline"/> 的 rec 图，把解码换成
/// <see cref="PlateDecoder"/> 的受限 beam。
/// <para>
/// <b>用法</b>：
/// <code>
/// var model = PaddleOcrModel.FromFiles("det.onnx", "rec.onnx", "ppocr_keys.txt");
/// var pipeline = new PaddleOcrPipeline(model);
/// var reader = new PlateReader(pipeline);
///
/// // 0) 整张图就是车牌（上游已抠出 / 手工裁好）——不跑检测，直接识别
/// var whole = reader.ReadWhole(bgra, width, height, OcrPixelFormat.Bgra32, stride);
///
/// // 1) 自己知道框（车牌检测器的输出）——只跑一次 rec
/// var plate = reader.Read(bgra, width, height, OcrPixelFormat.Bgra32, stride, x, y, w, h);
///
/// // 2) 不知道框——用 DB 检测先找文本行，再逐行做车牌解码
/// foreach (var found in reader.DetectAndRead(bgra, width, height, OcrPixelFormat.Bgra32, stride))
///     Console.WriteLine($"{found.Text} {found.Score:P1}");
/// </code>
/// </para>
/// <para>
/// 不做方向分类：车牌框基本是水平的，多跑一次 cls 网络不值。真遇到倒置号牌请调用方自己转 180°。
/// 与 <see cref="PaddleOcrPipeline"/> 一样不是线程安全的，一个实例串行使用。
/// </para>
/// </summary>
public sealed class PlateReader
{
    private readonly PlateCharset _charset;

    public PlateReader(PaddleOcrPipeline pipeline, PlateReaderOptions? options = null, PlateCharset? charset = null)
    {
        Pipeline = pipeline ?? throw new ArgumentNullException(nameof(pipeline));
        Options = options ?? new PlateReaderOptions();
        _charset = charset ?? PlateCharset.Default;
    }

    /// <summary>底下的 OCR 流水线（det / rec 图与缓存都在它身上）。</summary>
    public PaddleOcrPipeline Pipeline { get; }

    /// <summary>参数。</summary>
    public PlateReaderOptions Options { get; }

    /// <summary>字符集。</summary>
    public PlateCharset Charset => _charset;

    /// <summary>
    /// 识别一块<b>已知是车牌</b>的区域（源图像素坐标）。分数不够或形态不合法返回 <c>null</c>。
    /// </summary>
    public PlateResult? Read(
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
        int left = Math.Clamp(x, 0, width - 1);
        int top = Math.Clamp(y, 0, height - 1);
        int right = Math.Clamp(x + cropWidth, left + 1, width);
        int bottom = Math.Clamp(y + cropHeight, top + 1, height);
        int cropW = right - left;
        int cropH = bottom - top;

        if (cropW < 8 || cropH < 8)
            return null;

        (Tensor logits, int classCount) = Pipeline.RecognizeLogits(
            pixels, width, height, format, rowStride, left, top, cropW, cropH, rotate: false);

        PlateDecoding decoding = PlateDecoder.Decode(
            logits.AsSpan<float>(), classCount, Pipeline.Model.Classes, _charset, Options.Decoder);

        if (decoding.Text.Length == 0 || decoding.Score < Options.MinScore)
            return null;

        return new PlateResult(decoding, left, top, cropW, cropH);
    }

    /// <summary>
    /// 整张图<b>就是一张车牌</b>（上游检测器已抠出、或用户手工裁好的裁剪图）：跳过 DB 检测
    /// 与宽高比筛选，直接对整幅图跑一次 rec + 受限解码。
    /// <para>
    /// 这是「先用车牌检测器（YOLO / 颜色分割 / 自己的算法）抠出车牌，再把裁剪图丢进来识别」的入口：
    /// 比 <see cref="DetectAndRead"/> 少一次 DB 前向，也不受 DB 对小图 / 贴边图的召回影响。
    /// 分数不够或形态不合法返回 <c>null</c>。
    /// </para>
    /// </summary>
    public PlateResult? ReadWhole(
        ReadOnlySpan<byte> pixels,
        int width,
        int height,
        OcrPixelFormat format = OcrPixelFormat.Bgr24,
        int rowStride = 0)
    {
        if (width < 8 || height < 8)
            return null;

        return Read(pixels, width, height, format, rowStride, 0, 0, width, height);
    }

    /// <summary>
    /// 整图：DB 检测 → 宽高比筛选 → 逐框车牌解码，按分数从高到低返回（最多
    /// <see cref="PlateReaderOptions.MaxPlates"/> 个）。
    /// </summary>
    public IReadOnlyList<PlateResult> DetectAndRead(
        ReadOnlySpan<byte> pixels,
        int width,
        int height,
        OcrPixelFormat format = OcrPixelFormat.Bgr24,
        int rowStride = 0)
    {
        List<DbPostprocess.Box> boxes = Pipeline.DetectBoxes(pixels, width, height, format, rowStride);
        var results = new List<PlateResult>();

        foreach (DbPostprocess.Box box in boxes)
        {
            int x = Math.Clamp((int)MathF.Floor(box.X), 0, width - 1);
            int y = Math.Clamp((int)MathF.Floor(box.Y), 0, height - 1);
            int right = Math.Clamp((int)MathF.Ceiling(box.X + box.Width), x + 1, width);
            int bottom = Math.Clamp((int)MathF.Ceiling(box.Y + box.Height), y + 1, height);

            int cropW = right - x;
            int cropH = bottom - y;

            if (cropW < 16 || cropH < 8)
                continue;

            if (Options.FilterByAspect)
            {
                float aspect = (float)cropW / cropH;
                if (aspect < Options.MinAspect || aspect > Options.MaxAspect)
                    continue;
            }

            PlateResult? result = Read(pixels, width, height, format, rowStride, x, y, cropW, cropH);

            if (result is not null)
                results.Add(result);
        }

        results.Sort(static (a, b) => b.Score.CompareTo(a.Score));

        if (results.Count > Options.MaxPlates)
            results.RemoveRange(Options.MaxPlates, results.Count - Options.MaxPlates);

        return results;
    }
}
