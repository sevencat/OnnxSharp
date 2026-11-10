using OnnxSharp.Core;
using OnnxSharp.Tensors;

namespace OnnxSharp.Vision;

/// <summary>YuNet 人脸检测的参数（默认值与 OpenCV <c>FaceDetectorYN</c> 一致）。</summary>
public sealed class FaceDetectorOptions
{
    /// <summary>置信度阈值，<c>sqrt(cls × obj)</c> 低于它直接丢。默认 0.6。</summary>
    public float ScoreThreshold { get; set; } = 0.6f;

    /// <summary>NMS 的 IoU 阈值。默认 0.3（人脸互相重叠很正常，别调太高）。</summary>
    public float IouThreshold { get; set; } = 0.3f;

    /// <summary>送进 NMS 之前的候选上限，默认 5000。</summary>
    public int TopK { get; set; } = 5000;

    /// <summary>最多返回几张脸，默认不限。</summary>
    public int MaxFaces { get; set; } = int.MaxValue;

    /// <summary>
    /// YuNet 吃的是 <b>0..255 的原始像素</b>（不 <c>/255</c>、不减均值）。
    /// <see cref="ChannelNormalization"/> 的定义是 <c>(v/255 - mean) / std</c>，
    /// 所以这里取 <c>std = 1/255</c> 把除法抵消掉。
    /// </summary>
    internal static readonly ChannelNormalization RawPixel = new(0, 0, 0, 1f / 255f, 1f / 255f, 1f / 255f);
}

/// <summary>
/// 「一张图进、人脸出」的 YuNet 门面：内部完成拉伸预处理（到模型的静态输入尺寸）、推理、
/// 解码（cls/obj/bbox/kps × 3 个 stride）和 NMS，坐标换回原图空间。
/// <para>
/// 用法：
/// <code>
/// var detector = new YuNetDetector("yunet.onnx");
/// foreach (var face in detector.Detect(bgr, width, height))
///     Console.WriteLine(face);          // 框 + 5 点关键点 + 分数，坐标是原图像素
/// </code>
/// 检测器只持有托管内存，不实现 <see cref="IDisposable"/>；它不是线程安全的（共享同一份输入缓冲）。
/// </para>
/// </summary>
public sealed class YuNetDetector
{
    /// <summary>输出头名字里的 stride 部分（与 OpenCV <c>FaceDetectorYN</c> 里写死的顺序一致）。</summary>
    private static readonly string[] OutputStrides = ["8", "16", "32"];

    private readonly Net _net;
    private readonly string _inputName;

    /// <summary>加载模型并准备检测器。</summary>
    /// <param name="modelPath">YuNet 的 ONNX 路径（官方 <c>face_detection_yunet_*.onnx</c> 或其 int8 版）。</param>
    /// <param name="options">后处理参数；null 用默认值（阈值 0.6、IoU 0.3、TopK 5000）。</param>
    public YuNetDetector(string modelPath, FaceDetectorOptions? options = null)
        : this(Net.ReadModel(modelPath), options)
    {
    }

    /// <summary>复用已经加载好的模型（多个检测器共享同一份权重时用这个重载）。</summary>
    public YuNetDetector(Net net, FaceDetectorOptions? options = null)
    {
        _net = net ?? throw new ArgumentNullException(nameof(net));
        Options = options ?? new FaceDetectorOptions();

        var inputNames = _net.InputNames;
        if (inputNames.Count == 0)
            throw new InvalidOperationException("yunet: model has no inputs");

        _inputName = inputNames[0];

        var inputData = _net.Impl.GetArgData(_net.Impl.Require(_inputName));
        if (inputData.Shape is not { Rank: 4 } shape)
            throw new NotSupportedException($"yunet: input '{_inputName}' must be rank-4 NCHW, got {inputData.Shape}");

        if (shape[1] != 3)
            throw new NotSupportedException($"yunet: input has {shape[1]} channels, expected 3");

        if (shape[2] <= 0 || shape[3] <= 0)
            throw new NotSupportedException(
                $"yunet: input '{_inputName}' must have a static H/W (got {shape}); " +
                "this facade resizes the image itself, so use the fixed-size export");

        InputHeight = shape[2];
        InputWidth = shape[3];

        // 12 个输出必须齐全，缺了就明确报出来（比后面 NullReference 好查）
        var outputs = new HashSet<string>(_net.OutputNames);
        foreach (string head in new[] { "cls_", "obj_", "bbox_", "kps_" })
        {
            foreach (string stride in OutputStrides)
            {
                if (!outputs.Contains(head + stride))
                    throw new NotSupportedException(
                        $"yunet: model is missing output '{head + stride}' (outputs: {string.Join(", ", _net.OutputNames)})");
            }
        }
    }

    /// <summary>底层模型，用于 profile 或自定义前后处理。</summary>
    public Net Net => _net;

    /// <summary>网络输入宽（模型静态形状）。</summary>
    public int InputWidth { get; }

    /// <summary>网络输入高（模型静态形状）。</summary>
    public int InputHeight { get; }

    /// <summary>后处理参数。</summary>
    public FaceDetectorOptions Options { get; }

    /// <summary>
    /// 把像素转成 YuNet 的输入张量：<c>[1,3,H,W]</c>、<b>BGR 顺序</b>、<b>0..255 原始值</b>
    /// （与 OpenCV <c>blobFromImage(pad_image)</c> 完全一致，不减均值、不 /255）。
    /// <para>
    /// ⚠️ <see cref="ImagePreprocessor"/> 产出的张量**恒为 RGB**（它的 <c>bgr</c> 参数描述的是
    /// *源像素* 的顺序），而 YuNet 要 BGR —— 所以这里把标志反过来传，让张量的通道 0..2 = B,G,R。
    /// </para>
    /// </summary>
    /// <param name="pixels">24 位像素（每行 <paramref name="rowStride"/> 字节）。</param>
    /// <param name="width">源图宽。</param>
    /// <param name="height">源图高。</param>
    /// <param name="bgr">true 表示源像素是 BGR（OpenCV <c>imread</c> 的默认顺序）。</param>
    /// <param name="rowStride">源图每行字节数；0 表示 <c>width * 3</c>。</param>
    public Tensor CreateInput(
        ReadOnlySpan<byte> pixels,
        int width,
        int height,
        bool bgr = true,
        int rowStride = 0)
        => ImagePreprocessor.Stretch(
            pixels,
            width,
            height,
            InputWidth,
            InputHeight,
            bgr: !bgr,          // 见方法注释：把「源是 BGR」反转，得到的张量才是 BGR
            rowStride,
            FaceDetectorOptions.RawPixel);

    /// <summary>
    /// 检测一张图。<paramref name="pixels"/> 是 <paramref name="width"/> × <paramref name="height"/> 的
    /// 24 位像素（默认 BGR，与 OpenCV <c>imread</c> 一致）。
    /// </summary>
    /// <returns>按分数降序的人脸列表，坐标已换回原图空间。</returns>
    public IReadOnlyList<Face> Detect(
        ReadOnlySpan<byte> pixels,
        int width,
        int height,
        bool bgr = true,
        int rowStride = 0)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(width);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(height);

        return DetectFromTensor(CreateInput(pixels, width, height, bgr, rowStride), width, height);
    }

    /// <summary>
    /// 已经有一份 <c>[1,3,H,W]</c> 的输入张量时直接跑（测试与自定义预处理走这条）。
    /// <paramref name="imageWidth"/> / <paramref name="imageHeight"/> 是这张张量对应原图的尺寸，
    /// 用来把坐标换回去。
    /// </summary>
    public IReadOnlyList<Face> DetectFromTensor(Tensor input, int imageWidth, int imageHeight)
    {
        var shape = input.Shape;
        if (shape.Rank != 4 || shape[0] != 1 || shape[1] != 3 || shape[2] != InputHeight || shape[3] != InputWidth)
            throw new ArgumentException(
                $"yunet: input tensor must be [1,3,{InputHeight},{InputWidth}], got {shape}", nameof(input));

        _net.SetInput(_inputName, input);
        var outputs = _net.ForwardAll();

        var byName = new Dictionary<string, Tensor>(StringComparer.Ordinal);
        var names = _net.OutputNames;
        for (int i = 0; i < names.Count && i < outputs.Count; i++)
            byName[names[i]] = outputs[i];

        var candidates = new List<Face>(capacity: 512);
        foreach (int stride in YuNetDecoder.Strides)
        {
            string suffix = stride.ToString();
            YuNetDecoder.DecodeStride(
                byName["cls_" + suffix].AsSpan<float>(),
                byName["obj_" + suffix].AsSpan<float>(),
                byName["bbox_" + suffix].AsSpan<float>(),
                byName["kps_" + suffix].AsSpan<float>(),
                InputWidth,
                InputHeight,
                stride,
                Options.ScoreThreshold,
                candidates);
        }

        if (candidates.Count > Options.TopK)
        {
            candidates.Sort((a, b) => b.Score.CompareTo(a.Score));
            candidates.RemoveRange(Options.TopK, candidates.Count - Options.TopK);
        }

        var kept = YuNetDecoder.Suppress(candidates, Options.IouThreshold, Options.MaxFaces);
        return YuNetDecoder.ScaleToImage(kept, InputWidth, InputHeight, imageWidth, imageHeight);
    }
}
