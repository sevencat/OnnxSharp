using OnnxSharp.Core;
using OnnxSharp.Tensors;

namespace OnnxSharp.Vision;

/// <summary>
/// 「一张图进、检测框出」的 RF-DETR 封装：内部完成拉伸 + ImageNet 归一化、推理、解码（sigmoid / 背景剔除 /
/// top-k / 坐标还原），调用方不需要关心 <c>dets</c>+<c>labels</c> 双输出和归一化坐标。
/// <para>
/// 用法：
/// <code>
/// var detector = new RfDetrDetector("rfdetr-small.onnx");
/// foreach (var d in detector.Detect(rgb, width, height))
///     Console.WriteLine(d);
/// </code>
/// 与 <see cref="YoloDetector"/> 的差别：没有 NMS（DETR 的查询之间已经互相抑制），也不需要 letterbox
/// —— RF-DETR 是把整幅图直接拉伸成正方形，所以还原只需按宽高比例缩放。
/// </para>
/// </summary>
public sealed class RfDetrDetector
{
    private readonly Net _net;
    private readonly string _inputName;
    private int _boxesIndex;
    private int _logitsIndex;

    /// <summary>加载模型并准备检测器。</summary>
    /// <param name="modelPath">ONNX 文件路径。</param>
    /// <param name="options">后处理参数；null 使用默认值（conf 0.3、最多 300 个框、末列是背景）。</param>
    public RfDetrDetector(string modelPath, RfDetrOptions? options = null)
        : this(Net.ReadModel(modelPath), options)
    {
    }

    /// <summary>复用已经加载好的模型（多个检测器共享同一份权重时用这个重载）。</summary>
    public RfDetrDetector(Net net, RfDetrOptions? options = null)
    {
        _net = net ?? throw new ArgumentNullException(nameof(net));
        Options = options ?? new RfDetrOptions();

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

        // 静态形状直接用模型的；动态形状（-1）退回 options，再不行按 512。
        TargetHeight = shape[2] > 0 ? shape[2] : Options.TargetHeight > 0 ? Options.TargetHeight : 512;
        TargetWidth = shape[3] > 0 ? shape[3] : Options.TargetWidth > 0 ? Options.TargetWidth : 512;

        _boxesIndex = -1;
        _logitsIndex = -1;
        ResolveOutputs();
    }

    /// <summary>底层模型，用于自定义前后处理或取 profile。</summary>
    public Net Net => _net;

    /// <summary>后处理参数。</summary>
    public RfDetrOptions Options { get; }

    /// <summary>模型输入张量的名字。</summary>
    public string InputName => _inputName;

    /// <summary>模型输入宽（拉伸的目标宽）。</summary>
    public int TargetWidth { get; }

    /// <summary>模型输入高（拉伸的目标高）。</summary>
    public int TargetHeight { get; }

    /// <summary>框输出（<c>dets</c>）的名字。</summary>
    public string BoxesOutputName => _net.OutputNames[_boxesIndex];

    /// <summary>类别输出（<c>labels</c>）的名字。</summary>
    public string LogitsOutputName => _net.OutputNames[_logitsIndex];

    /// <summary>
    /// 只做预处理：拉伸到模型尺寸 + 归一化，返回 <c>[1, 3, H, W]</c> 的输入张量（需要自己推理时用）。
    /// </summary>
    public Tensor Preprocess(ReadOnlySpan<byte> pixels, int width, int height, bool bgr = false)
        => ImagePreprocessor.Stretch(
            pixels, width, height, TargetWidth, TargetHeight, bgr, 0, Options.Normalization);

    /// <summary>
    /// 对一张已解码的图像做检测，返回<b>原图坐标系</b>下的检测框（按分数降序，已裁剪到图内）。
    /// </summary>
    /// <param name="pixels">紧凑排布的 8 位三通道像素。</param>
    /// <param name="width">图像宽，像素。</param>
    /// <param name="height">图像高，像素。</param>
    /// <param name="bgr">true 表示像素是 BGR 顺序（OpenCV 默认），false 为 RGB。</param>
    public IReadOnlyList<Detection> Detect(ReadOnlySpan<byte> pixels, int width, int height, bool bgr = false)
    {
        var input = Preprocess(pixels, width, height, bgr);
        return Detect(input, width, height);
    }

    /// <summary>对已经预处理好的输入做推理 + 解码（同一输入多次推理时用）。</summary>
    public IReadOnlyList<Detection> Detect(Tensor input, int sourceWidth, int sourceHeight)
    {
        _net.SetInput(_inputName, input);
        return Decode(_net.ForwardAll(), sourceWidth, sourceHeight);
    }

    /// <summary>
    /// 只做后处理：从 <see cref="Net.ForwardAll"/> 的结果里按名字取两路输出再解码
    /// （需要分别计时前向与后处理时用）。
    /// </summary>
    /// <param name="outputs">模型全部输出，顺序同 <see cref="Net.OutputNames"/>。</param>
    /// <param name="sourceWidth">原图宽。</param>
    /// <param name="sourceHeight">原图高。</param>
    public IReadOnlyList<Detection> Decode(IReadOnlyList<Tensor> outputs, int sourceWidth, int sourceHeight)
    {
        ArgumentNullException.ThrowIfNull(outputs);

        int required = Math.Max(_boxesIndex, _logitsIndex);
        if (outputs.Count <= required)
            throw new ArgumentException(
                $"detect: expected at least {required + 1} outputs, got {outputs.Count}", nameof(outputs));

        return Decode(outputs[_boxesIndex], outputs[_logitsIndex], sourceWidth, sourceHeight);
    }

    /// <summary>
    /// 只做后处理：把 <c>dets</c> / <c>labels</c> 解成原图坐标的框（需要自己计时或自己前向时用）。
    /// </summary>
    public IReadOnlyList<Detection> Decode(Tensor boxes, Tensor logits, int sourceWidth, int sourceHeight)
        => RfDetrDecoder.Decode(boxes, logits, sourceWidth, sourceHeight, Options);

    /// <summary>
    /// 定位双输出：先按名字（<c>dets</c>/<c>boxes</c> 与 <c>labels</c>/<c>logits</c>），
    /// 再按形状（唯一一个末维 4 的 rank-3 输出是框），都不行就按导出顺序（0 框、1 类别）。
    /// </summary>
    private void ResolveOutputs()
    {
        var outputNames = _net.OutputNames;
        if (outputNames.Count < 2)
            throw new NotSupportedException(
                $"detect: RF-DETR needs two outputs (dets, labels), got {outputNames.Count}: " +
                string.Join(", ", outputNames));

        int boxes = FindByKeyword(outputNames, "dets", "boxes", "pred_boxes");
        int logits = FindByKeyword(outputNames, "labels", "logits", "pred_logits", "scores");
        if (boxes >= 0 && logits >= 0 && boxes != logits)
        {
            _boxesIndex = boxes;
            _logitsIndex = logits;
            return;
        }

        // 形状兜底：末维为 4 的 rank-3 输出只可能是框。
        int boxLike = -1;
        int boxLikeCount = 0;
        for (int index = 0; index < outputNames.Count; index++)
        {
            var shape = _net.Impl.GetArgData(_net.Impl.Require(outputNames[index])).Shape;
            if (shape is { Rank: 3 } realShape && realShape[2] == 4)
            {
                boxLike = index;
                boxLikeCount++;
            }
        }

        if (boxLikeCount == 1)
        {
            _boxesIndex = boxLike;
            _logitsIndex = boxLike == 0 ? 1 : 0;
            return;
        }

        // 形状也分辨不出来（例如 3 类时两路末维都是 4）：按导出顺序兜底。
        _boxesIndex = 0;
        _logitsIndex = 1;
    }

    private static int FindByKeyword(IReadOnlyList<string> names, params string[] keywords)
    {
        for (int index = 0; index < names.Count; index++)
        {
            string lowered = names[index].ToLowerInvariant();
            foreach (string keyword in keywords)
            {
                if (lowered.Contains(keyword, StringComparison.Ordinal))
                    return index;
            }
        }

        return -1;
    }
}
