using OnnxSharp.Core;
using OnnxSharp.Tensors;

namespace OnnxSharp.Vision;

/// <summary>
/// 「一张图进、检测框出」的 YOLO 封装：内部完成 letterbox 预处理、推理、解码和 NMS，
/// 调用方不需要关心 NCHW 排布、坐标缩放和填边。
/// <para>
/// 用法：
/// <code>
/// var detector = new YoloDetector("yolo11n.onnx");
/// foreach (var d in detector.Detect(rgb, width, height))
///     Console.WriteLine(d);
/// </code>
/// （检测器只持有托管内存，不实现 <see cref="IDisposable"/>；模型用完交给 GC 即可。）
/// </para>
/// </summary>
public sealed class YoloDetector
{
    private readonly Net _net;
    private readonly string _inputName;

    /// <summary>加载模型并准备检测器。</summary>
    /// <param name="modelPath">ONNX 文件路径。</param>
    /// <param name="options">后处理参数；null 使用默认值（conf 0.25、IoU 0.45、最多 300 个框）。</param>
    public YoloDetector(string modelPath, DetectorOptions? options = null)
        : this(Net.ReadModel(modelPath), options)
    {
    }

    /// <summary>复用已经加载好的模型（多个检测器共享同一份权重时用这个重载）。</summary>
    public YoloDetector(Net net, DetectorOptions? options = null)
    {
        _net = net ?? throw new ArgumentNullException(nameof(net));
        Options = options ?? new DetectorOptions();

        var inputNames = _net.InputNames;
        if (inputNames.Count == 0)
            throw new InvalidOperationException("detect: model has no inputs");

        _inputName = inputNames[0];

        var inputData = _net.Impl.GetArgData(_net.Impl.Require(_inputName));
        if (inputData.Shape is not { Rank: 4 } shape)
            throw new NotSupportedException(
                $"detect: input '{_inputName}' must be rank-4 NCHW, got {inputData.Shape}");

        if (shape[1] != 3)
            throw new NotSupportedException(
                $"detect: input '{_inputName}' has {shape[1]} channels, expected 3 (RGB/BGR image)");

        // 静态形状直接用模型的；动态形状（-1）退回 options，再不行按 640。
        TargetHeight = shape[2] > 0 ? shape[2] : Options.TargetHeight > 0 ? Options.TargetHeight : 640;
        TargetWidth = shape[3] > 0 ? shape[3] : Options.TargetWidth > 0 ? Options.TargetWidth : 640;
    }

    /// <summary>底层模型，用于自定义前后处理或取 profile。</summary>
    public Net Net => _net;

    /// <summary>后处理参数。</summary>
    public DetectorOptions Options { get; }

    /// <summary>模型输入张量的名字。</summary>
    public string InputName => _inputName;

    /// <summary>模型输入宽（letterbox 的目标宽）。</summary>
    public int TargetWidth { get; }

    /// <summary>模型输入高（letterbox 的目标高）。</summary>
    public int TargetHeight { get; }

    /// <summary>完整的输入形状 <c>[1, 3, H, W]</c>。</summary>
    public TensorShape InputShape => new(1, 3, TargetHeight, TargetWidth);

    /// <summary>只做预处理，返回送进模型的张量与还原参数（需要自己推理时用）。</summary>
    /// <param name="pixels">8 位三通道像素，每行 <paramref name="rowStride"/> 字节。</param>
    /// <param name="width">图像宽，像素。</param>
    /// <param name="height">图像高，像素。</param>
    /// <param name="bgr">true 表示像素是 BGR 顺序（OpenCV 默认），false 为 RGB。</param>
    /// <param name="rowStride">源图每行字节数（含行尾填充）；0 表示 <c>width * 3</c>。</param>
    public (Tensor Input, Letterbox Transform) Preprocess(
        ReadOnlySpan<byte> pixels,
        int width,
        int height,
        bool bgr = false,
        int rowStride = 0)
        => ImagePreprocessor.Letterbox(
            pixels, width, height, TargetWidth, TargetHeight, bgr, rowStride);

    /// <summary>
    /// 对一张已解码的图像做检测，返回<b>原图坐标系</b>下的检测框（按分数降序，已裁剪到图内）。
    /// </summary>
    /// <param name="pixels">8 位三通道像素，每行 <paramref name="rowStride"/> 字节。</param>
    /// <param name="width">图像宽，像素。</param>
    /// <param name="height">图像高，像素。</param>
    /// <param name="bgr">true 表示像素是 BGR 顺序（OpenCV 默认），false 为 RGB。</param>
    /// <param name="rowStride">源图每行字节数（含行尾填充）；0 表示 <c>width * 3</c>。</param>
    public IReadOnlyList<Detection> Detect(
        ReadOnlySpan<byte> pixels,
        int width,
        int height,
        bool bgr = false,
        int rowStride = 0)
    {
        var (input, transform) = Preprocess(pixels, width, height, bgr, rowStride);
        return Detect(input, transform);
    }

    /// <summary>对已经预处理好的输入做推理 + 解码（批量复用同一输入时用）。</summary>
    public IReadOnlyList<Detection> Detect(Tensor input, in Letterbox transform)
    {
        _net.SetInput(_inputName, input);
        var output = _net.Forward();
        return Decode(output, transform);
    }

    /// <summary>只做后处理：把检测头张量解成原图坐标的框（需要自己计时或自己前向时用）。</summary>
    public IReadOnlyList<Detection> Decode(Tensor output, in Letterbox transform)
        => YoloDecoder.Decode(output, transform, Options);
}
