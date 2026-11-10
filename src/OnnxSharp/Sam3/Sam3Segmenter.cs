using OnnxSharp.Core;
using OnnxSharp.Imaging;
using OnnxSharp.Tensors;
using OnnxSharp.Vision;

namespace OnnxSharp.Sam3;

/// <summary>
/// SAM3（Segment Anything Model 3）的<b>框提示</b>分段流水线：图像 + 若干框 → 掩码 + 框 + 分数。
/// <para>
/// 三个子模型串起来跑（模型需先离线转成 FP32，见 <c>tools/scripts/onnx_fp16_to_f32.py</c>）：
/// <code>
/// vision-encoder  : images[1,3,1008,1008] → fpn_feat_0/1/2 + fpn_pos_2
/// geometry-encoder: input_boxes[1,N,4] + labels[1,N] + fpn_feat_2 + fpn_pos_2 → geometry_features[1,N+1,256] + mask
/// decoder         : 四个 FPN 特征 + prompt_features/prompt_mask → pred_masks/pred_boxes/pred_logits/presence
/// </code>
/// </para>
/// <para>
/// <b>性能</b>：vision-encoder 在 1008² 上单帧约 26 s（16 线程、FP32），整条链路 30~40 s；
/// 内存约 3~5 GB（vision 权重 1.84 GB 常驻）。纯 CPU 下只适合离线批处理。
/// </para>
/// <para>不做线程安全：一个实例串行使用。</para>
/// </summary>
public sealed class Sam3Segmenter
{
    private readonly Sam3Options _options;
    private readonly string _visionPath;
    private readonly string _geometryPath;
    private readonly string _decoderPath;

    private Net? _visionNet;
    private readonly Dictionary<int, Net> _geometryNets = new();
    private readonly Dictionary<int, Net> _decoderNets = new();

    public Sam3Segmenter(Sam3Options options)
    {
        _options = options ?? throw new ArgumentNullException(nameof(options));
        if (string.IsNullOrEmpty(options.ModelDirectory))
            throw new ArgumentException("Sam3Options.ModelDirectory is required", nameof(options));

        string directory = Path.GetFullPath(options.ModelDirectory);
        _visionPath = Path.Combine(directory, "vision-encoder.onnx");
        _geometryPath = Path.Combine(directory, "geometry-encoder.onnx");
        _decoderPath = Path.Combine(directory, "decoder.onnx");

        foreach (string path in new[] { _visionPath, _geometryPath, _decoderPath })
        {
            if (!File.Exists(path))
                throw new FileNotFoundException(
                    $"SAM3 模型缺失：{path}（先用 tools/scripts/onnx_fp16_to_f32.py 把官方 fp16 导出转成 fp32）", path);
        }

        Options = options;
    }

    public Sam3Options Options { get; }

    /// <summary>
    /// 对一张图做框提示分段。掩码坐标 → 源图坐标用 <see cref="Sam3Result.MaskScaleX"/> /
    /// <see cref="Sam3Result.MaskScaleY"/>（框已经是源图像素）。
    /// </summary>
    /// <param name="pixels">交错像素（RGB24 / BGR24 由 <paramref name="bgr"/> 决定）。</param>
    /// <param name="width">源图宽。</param>
    /// <param name="height">源图高。</param>
    /// <param name="boxes">提示框（<b>源图像素</b>坐标）。至少一个。</param>
    /// <param name="bgr">true 表示像素是 BGR 顺序。</param>
    /// <param name="rowStride">源图每行字节数；0 = <c>width * 3</c>。</param>
    public Sam3Result Segment(
        ReadOnlySpan<byte> pixels,
        int width,
        int height,
        IReadOnlyList<Sam3Box> boxes,
        bool bgr = false,
        int rowStride = 0)
    {
        if (boxes is null || boxes.Count == 0)
            throw new ArgumentException("SAM3 需要至少一个提示框", nameof(boxes));

        int imageSize = _options.ImageSize;

        // ---- 1) 预处理：等比缩放到 1008（左上对齐、白边填充）、/255、NCHW ----
        // 共享的 ImageTensor 负责「裁剪 + 双线性 + 归一化 + 填边」，SAM3 只是参数组合之一。
        ImagePlacement placement = ImageGeometry.Contain(width, height, imageSize, imageSize, centered: false);
        ImagePixels pixelsView = ImagePixels.FromBgr(pixels, width, height, bgr, rowStride);

        float[] input = new float[3 * imageSize * imageSize];
        ImageTensor.ToChw(
            pixelsView,
            cropX: 0,
            cropY: 0,
            cropWidth: width,
            cropHeight: height,
            outWidth: placement.ResizedWidth,
            outHeight: placement.ResizedHeight,
            offsetX: 0,
            offsetY: 0,
            destinationHeight: imageSize,
            destinationStride: imageSize,
            padValue: 1f,                                  // 白边，归一化之后正好是 1.0
            destination: input,
            normalization: ChannelNormalization.Divide255);

        int embeddedWidth = placement.ResizedWidth;
        int embeddedHeight = placement.ResizedHeight;

        // ---- 2) vision-encoder ----
        Net vision = _visionNet ??= Net.ReadModel(_visionPath,
            new Dictionary<string, int[]>(StringComparer.Ordinal)
            {
                ["images"] = [1, 3, imageSize, imageSize],
            });

        vision.SetInput("images", Tensor.FromArray(input, new TensorShape(1, 3, imageSize, imageSize)));

        IReadOnlyList<Tensor> features = vision.ForwardAll();
        IReadOnlyList<string> names = vision.OutputNames;
        Tensor feat0 = default!, feat1 = default!, feat2 = default!, pos2 = default!;
        for (int i = 0; i < features.Count; i++)
        {
            switch (names[i])
            {
                case "fpn_feat_0": feat0 = features[i]; break;
                case "fpn_feat_1": feat1 = features[i]; break;
                case "fpn_feat_2": feat2 = features[i]; break;
                case "fpn_pos_2": pos2 = features[i]; break;
            }
        }

        // 输出张量指向引擎的缓冲槽，下一次 Forward 会被覆盖：先拷成独立副本再往下传。
        return SegmentWithFeatures(
            feat0.Clone(), feat1.Clone(), feat2.Clone(), pos2.Clone(),
            boxes, width, height, embeddedWidth, embeddedHeight);
    }

    /// <summary>等比缩放到 <paramref name="size"/>² 时实际占用的宽高（左上对齐，与预处理一致）。</summary>
    internal static (int Width, int Height) FitSize(int width, int height, int size)
    {
        ImagePlacement placement = ImageGeometry.Contain(width, height, size, size, centered: false);
        return (placement.ResizedWidth, placement.ResizedHeight);
    }
    // ---------------- 三个模型串起来 ----------------

    private Sam3Result SegmentWithFeatures(
        Tensor feat0,
        Tensor feat1,
        Tensor feat2,
        Tensor pos2,
        IReadOnlyList<Sam3Box> boxes,
        int width,
        int height,
        int embeddedWidth,
        int embeddedHeight)
    {
        int boxCount = boxes.Count;
        int promptLength = boxCount + 1;       // geometry-encoder 会多出一个「全局/无提示」token
        int imageSize = _options.ImageSize;
        int maskSize = _options.MaskSize;

        // ---- 3) geometry-encoder：把框编码成 prompt ----
        Net geometry = GetGeometryNet(boxCount);
        geometry.SetInput("input_boxes",
            Tensor.FromArray(BuildBoxTensor(boxes, width, height, embeddedWidth, embeddedHeight),
                new TensorShape(1, boxCount, 4)));
        geometry.SetInput("input_boxes_labels",
            Tensor.FromInt64(BuildLabels(boxCount), new TensorShape(1, boxCount)));
        geometry.SetInput("fpn_feat_2", feat2);
        geometry.SetInput("fpn_pos_2", pos2);

        IReadOnlyList<Tensor> geometryOutputs = geometry.ForwardAll();
        IReadOnlyList<string> geometryNames = geometry.OutputNames;
        Tensor? promptFeatures = null;
        Tensor? promptMask = null;
        for (int i = 0; i < geometryOutputs.Count; i++)
        {
            if (geometryNames[i] == "geometry_features") promptFeatures = geometryOutputs[i];
            else if (geometryNames[i] == "geometry_mask") promptMask = geometryOutputs[i];
        }

        if (promptFeatures is null || promptMask is null)
            throw new OnnxSharpException("SAM3 geometry-encoder 缺少 geometry_features / geometry_mask 输出");

        Diagnose("geometry_features", promptFeatures);
        Diagnose("geometry_mask", promptMask);

        // ---- 4) decoder ----
        Net decoder = GetDecoderNet(promptLength);
        decoder.SetInput("fpn_feat_0", feat0);
        decoder.SetInput("fpn_feat_1", feat1);
        decoder.SetInput("fpn_feat_2", feat2);
        decoder.SetInput("fpn_pos_2", pos2);
        decoder.SetInput("prompt_features", promptFeatures);
        decoder.SetInput("prompt_mask", promptMask);

        IReadOnlyList<Tensor> decoderOutputs = decoder.ForwardAll();
        IReadOnlyList<string> decoderNames = decoder.OutputNames;
        Tensor? masks = null, predBoxes = null, logits = null, presence = null;
        for (int i = 0; i < decoderOutputs.Count; i++)
        {
            switch (decoderNames[i])
            {
                case "pred_masks": masks = decoderOutputs[i]; break;
                case "pred_boxes": predBoxes = decoderOutputs[i]; break;
                case "pred_logits": logits = decoderOutputs[i]; break;
                case "presence_logits": presence = decoderOutputs[i]; break;
            }
        }

        if (masks is null || predBoxes is null || logits is null || presence is null)
            throw new OnnxSharpException("SAM3 decoder 输出不全（需要 pred_masks / pred_boxes / pred_logits / presence_logits）");

        // ---- 5) 后处理（必须在任何下一次 Forward 之前完成，输出指向引擎缓冲槽）----
        Diagnose("pred_logits", logits);
        Diagnose("pred_boxes", predBoxes);
        Diagnose("pred_masks", masks);
        Diagnose("presence_logits", presence);

        float presenceScore = Sigmoid(presence.AsSpan<float>()[0]);
        var logitsSpan = logits.AsSpan<float>();
        var boxesSpan = predBoxes.AsSpan<float>();
        var masksSpan = masks.AsSpan<float>();

        float scaleToSourceX = (float)width / embeddedWidth;
        float scaleToSourceY = (float)height / embeddedHeight;

        var items = new List<Sam3Item>();
        for (int i = 0; i < logitsSpan.Length; i++)
        {
            float score = Sigmoid(logitsSpan[i]) * presenceScore;

            // NaN 比较永远为 false，用「!(score > 阈值)」把 NaN 一起筛掉。
            if (!(score > _options.ScoreThreshold))
                continue;

            float x1 = Math.Clamp(boxesSpan[i * 4 + 0] * imageSize * scaleToSourceX, 0, width);
            float y1 = Math.Clamp(boxesSpan[i * 4 + 1] * imageSize * scaleToSourceY, 0, height);
            float x2 = Math.Clamp(boxesSpan[i * 4 + 2] * imageSize * scaleToSourceX, 0, width);
            float y2 = Math.Clamp(boxesSpan[i * 4 + 3] * imageSize * scaleToSourceY, 0, height);
            if (x2 <= x1 || y2 <= y1)
                continue;

            // pred_masks 是 logits（实测范围 -6.7~0.28），转成概率再交给调用方阈值化。
            float[] mask = new float[maskSize * maskSize];
            var maskSource = masksSpan.Slice(i * maskSize * maskSize, mask.Length);
            for (int m = 0; m < mask.Length; m++)
                mask[m] = 1f / (1f + MathF.Exp(-maskSource[m]));

            items.Add(new Sam3Item(score, new Sam3Box(x1, y1, x2, y2), mask, maskSize, maskSize));
        }

        items.Sort((a, b) => b.Score.CompareTo(a.Score));
        items = Sam3Nms.Suppress(items, _options.IouThreshold, _options.MaxItems);

        return new Sam3Result(items, presenceScore, width, height, embeddedWidth, embeddedHeight, imageSize, maskSize);
    }

    // ---------------- 输入构造 / 模型缓存 ----------------

    /// <summary>源图像素框 → geometry-encoder 期望的 <c>[1,N,4]</c>（见 <see cref="Sam3PromptBoxFormat"/>）。</summary>
    private float[] BuildBoxTensor(
        IReadOnlyList<Sam3Box> boxes,
        int width,
        int height,
        int embeddedWidth,
        int embeddedHeight)
    {
        float imageSize = _options.ImageSize;
        float scaleX = (float)embeddedWidth / width;
        float scaleY = (float)embeddedHeight / height;

        var data = new float[boxes.Count * 4];
        for (int i = 0; i < boxes.Count; i++)
        {
            Sam3Box box = boxes[i];
            float x1 = box.X1 * scaleX;
            float y1 = box.Y1 * scaleY;
            float x2 = box.X2 * scaleX;
            float y2 = box.Y2 * scaleY;

            switch (_options.BoxFormat)
            {
                case Sam3PromptBoxFormat.XyxyModelPixels:
                    data[i * 4 + 0] = x1;
                    data[i * 4 + 1] = y1;
                    data[i * 4 + 2] = x2;
                    data[i * 4 + 3] = y2;
                    break;

                case Sam3PromptBoxFormat.XyxyNormalized:
                    data[i * 4 + 0] = x1 / imageSize;
                    data[i * 4 + 1] = y1 / imageSize;
                    data[i * 4 + 2] = x2 / imageSize;
                    data[i * 4 + 3] = y2 / imageSize;
                    break;

                default: // CxcywhNormalized
                    data[i * 4 + 0] = (x1 + x2) * 0.5f / imageSize;
                    data[i * 4 + 1] = (y1 + y2) * 0.5f / imageSize;
                    data[i * 4 + 2] = (x2 - x1) / imageSize;
                    data[i * 4 + 3] = (y2 - y1) / imageSize;
                    break;
            }
        }

        return data;
    }

    private long[] BuildLabels(int count)
    {
        var labels = new long[count];
        Array.Fill(labels, _options.BoxLabel);
        return labels;
    }

    private Net GetGeometryNet(int boxCount)
    {
        if (_geometryNets.TryGetValue(boxCount, out Net? net))
            return net;

        if (_geometryNets.Count >= Math.Max(1, _options.ShapeCacheSize))
            _geometryNets.Clear();

        int size = _options.ImageSize;
        int feature = size / 14;      // 72×72
        net = Net.ReadModel(_geometryPath, new Dictionary<string, int[]>(StringComparer.Ordinal)
        {
            ["input_boxes"] = [1, boxCount, 4],
            ["input_boxes_labels"] = [1, boxCount],
            ["fpn_feat_2"] = [1, 256, feature, feature],
            ["fpn_pos_2"] = [1, 256, feature, feature],
        });
        _geometryNets[boxCount] = net;
        return net;
    }

    private Net GetDecoderNet(int promptLength)
    {
        if (_decoderNets.TryGetValue(promptLength, out Net? net))
            return net;

        if (_decoderNets.Count >= Math.Max(1, _options.ShapeCacheSize))
            _decoderNets.Clear();

        int size = _options.ImageSize;
        int feat0 = size / 3 - 48;    // 288
        int feat1 = feat0 / 2;        // 144
        int feat2 = feat1 / 2;        // 72

        net = Net.ReadModel(_decoderPath, new Dictionary<string, int[]>(StringComparer.Ordinal)
        {
            ["fpn_feat_0"] = [1, 256, feat0, feat0],
            ["fpn_feat_1"] = [1, 256, feat1, feat1],
            ["fpn_feat_2"] = [1, 256, feat2, feat2],
            ["fpn_pos_2"] = [1, 256, feat2, feat2],
            ["prompt_features"] = [1, promptLength, 256],
            ["prompt_mask"] = [1, promptLength],
        });
        _decoderNets[promptLength] = net;
        return net;
    }

    private static float Sigmoid(float value) => 1f / (1f + MathF.Exp(-value));

    /// <summary>打印一个中间张量的统计（min / max / NaN 数 / 非零数），只在 <see cref="Sam3Options.Diagnostic"/> 非空时执行。</summary>
    private void Diagnose(string name, Tensor tensor)
    {
        if (_options.Diagnostic is not { } write)
            return;

        if (tensor.DType == DType.Bool)
        {
            var bytes = tensor.AsSpan<byte>();
            int trueCount = 0;
            foreach (byte value in bytes)
            {
                if (value != 0)
                    trueCount++;
            }

            write($"{name,-18} Bool {tensor.Shape}  true {trueCount}/{bytes.Length}");
            return;
        }

        var span = tensor.AsSpan<float>();
        float min = float.MaxValue, max = float.MinValue;
        int nan = 0, nonZero = 0;
        foreach (float value in span)
        {
            if (float.IsNaN(value))
            {
                nan++;
                continue;
            }

            if (value < min) min = value;
            if (value > max) max = value;
            if (value != 0f) nonZero++;
        }

        write($"{name,-18} F32 {tensor.Shape}  min {min:F4} max {max:F4} nan {nan} nonzero {nonZero}/{span.Length}");
    }
}
