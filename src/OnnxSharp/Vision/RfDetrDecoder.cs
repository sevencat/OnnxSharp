using OnnxSharp.Tensors;

namespace OnnxSharp.Vision;

/// <summary>RF-DETR 的后处理参数。</summary>
public sealed class RfDetrOptions
{
    /// <summary>置信度阈值（严格大于才保留）。RF-DETR 官方 ONNX helper 的默认值是 0.3。</summary>
    public float Confidence { get; init; } = 0.3f;

    /// <summary>
    /// 在阈值<b>之前</b>最多取多少个「查询×类别」对（对应官方实现的 <c>num_select</c>，
    /// 默认等于查询数 300）。0 表示不限制。
    /// </summary>
    public int MaxDetections { get; init; } = 300;

    /// <summary>
    /// 需要排除的背景槽位（负数按 Python 习惯从末尾数，默认 -1 即最后一列）。
    /// COCO 91 类导出里最后一列是背景；自定义类别数导出时改成 <c>ClassCount - 1</c>，
    /// 没有背景列时设为 <c>null</c>。
    /// </summary>
    public int? BackgroundClassId { get; init; } = -1;

    /// <summary>模型输入为动态尺寸时使用的宽（0 表示 512）。</summary>
    public int TargetWidth { get; init; }

    /// <summary>模型输入为动态尺寸时使用的高（0 表示 512）。</summary>
    public int TargetHeight { get; init; }

    /// <summary>预处理归一化，默认 ImageNet 均值方差（RF-DETR 的官方预处理）。</summary>
    public ChannelNormalization Normalization { get; init; } = ChannelNormalization.ImageNet;
}

/// <summary>
/// 把 RF-DETR 的双输出解成原图坐标的检测框。语义对齐官方
/// <c>rfdetr.export._runtime.decode.decode_detections</c>（等价于 <c>PostProcess.forward</c>）：
/// <list type="number">
/// <item>逐类 sigmoid（<b>不是 softmax</b>：一个查询可以同时在多个类别上超过阈值）；</item>
/// <item>丢掉背景槽位（默认最后一列）；</item>
/// <item>把 <c>(Q, C)</c> 摊平成 <c>Q×C</c> 个「查询×类别」对，按分数降序取前 <c>num_select</c> 个，再按阈值过滤；</item>
/// <item>归一化 <c>cxcywh</c> → 原图像素 <c>xyxy</c>，裁剪到图内。</item>
/// </list>
/// 第 3 步的顺序看似多余，但<b>不能</b>简化成「每个查询取最大类别」：那会把一个查询在其它类别上的
/// 高分全部丢掉。反过来「先按阈值过滤再取前 N」与本实现等价（阈值单调），所以这里用前者，省一次全量排序。
/// <para>
/// RF-DETR 的框是相对<b>整幅拉伸后的图</b>归一化的（没有 letterbox 填边），所以还原只需要乘原图宽高。
/// </para>
/// </summary>
public static class RfDetrDecoder
{
    /// <summary>一条候选：分数、排除背景后的摊平序号（仅用于稳定排序）、查询下标、原始类别槽位。</summary>
    private readonly record struct Candidate(float Score, long Order, int Query, int ClassId);

    /// <summary>
    /// 解码一张图的输出，返回原图坐标系下按分数降序的检测框（已裁剪到图内）。
    /// </summary>
    /// <param name="boxes">框输出 <c>[1, Q, 4]</c>，归一化的 <c>cxcywh</c>。</param>
    /// <param name="logits">类别输出 <c>[1, Q, C]</c>，<b>未经 sigmoid 的 logits</b>。</param>
    /// <param name="sourceWidth">原图宽，用于把归一化坐标还原成像素。</param>
    /// <param name="sourceHeight">原图高。</param>
    /// <param name="options">阈值与背景槽位；null 使用默认值。</param>
    /// <exception cref="NotSupportedException">输出不是 F32、不是三维、或批次不为 1。</exception>
    /// <exception cref="ArgumentException">形状不匹配、尺寸非正或背景槽位越界。</exception>
    public static IReadOnlyList<Detection> Decode(
        Tensor boxes,
        Tensor logits,
        int sourceWidth,
        int sourceHeight,
        RfDetrOptions? options = null)
    {
        options ??= new RfDetrOptions();

        if (sourceWidth <= 0 || sourceHeight <= 0)
            throw new ArgumentException("Source image dimensions must be positive.");

        if (boxes.DType != DType.F32 || logits.DType != DType.F32)
            throw new NotSupportedException(
                $"detect: RF-DETR outputs must be F32, got {boxes.DType} / {logits.DType}");

        var boxShape = boxes.Shape;
        var logitShape = logits.Shape;

        if (boxShape.Rank != 3 || boxShape[0] != 1 || boxShape[2] != 4)
            throw new NotSupportedException($"detect: expected boxes [1, Q, 4], got {boxShape}");
        if (logitShape.Rank != 3 || logitShape[0] != 1)
            throw new NotSupportedException($"detect: expected logits [1, Q, C], got {logitShape}");
        if (logitShape[1] != boxShape[1])
            throw new ArgumentException(
                $"detect: boxes {boxShape} and logits {logitShape} disagree on the query count");

        int queries = boxShape[1];
        int classes = logitShape[2];

        int background = -1;
        if (options.BackgroundClassId is { } declared)
        {
            if (declared < -classes || declared >= classes)
                throw new ArgumentException(
                    $"detect: BackgroundClassId {declared} is outside {classes} class slots", nameof(options));

            background = ((declared % classes) + classes) % classes;
        }

        int foreground = background >= 0 ? classes - 1 : classes;
        if (foreground <= 0)
            throw new ArgumentException($"detect: logits {logitShape} has no foreground class");

        var candidates = CollectCandidates(logits, queries, classes, background, foreground, options.Confidence);
        if (candidates.Count == 0)
            return Array.Empty<Detection>();

        // 分数降序；分数相同时按摊平序号升序，与官方实现的可复现排序一致。
        candidates.Sort(static (left, right) => left.Score != right.Score
            ? right.Score.CompareTo(left.Score)
            : left.Order.CompareTo(right.Order));

        int cap = options.MaxDetections <= 0
            ? candidates.Count
            : Math.Min(options.MaxDetections, candidates.Count);
        if (cap < candidates.Count)
            candidates.RemoveRange(cap, candidates.Count - cap);

        return Materialize(boxes, candidates, sourceWidth, sourceHeight);
    }

    private static List<Candidate> CollectCandidates(
        Tensor logits,
        int queries,
        int classes,
        int background,
        int foreground,
        float threshold)
    {
        var scores = logits.AsReadOnlySpan<float>();
        var candidates = new List<Candidate>();

        for (int query = 0; query < queries; query++)
        {
            int row = query * classes;

            // foregroundIndex 是排除背景后的列号，用它拼摊平序号才能和官方排序键一致。
            int foregroundIndex = 0;
            for (int slot = 0; slot < classes; slot++)
            {
                if (slot == background)
                    continue;

                float score = Sigmoid(scores[row + slot]);
                if (float.IsFinite(score) && score > threshold)
                    candidates.Add(new Candidate(score, (long)query * foreground + foregroundIndex, query, slot));

                foregroundIndex++;
            }
        }

        return candidates;
    }

    private static List<Detection> Materialize(
        Tensor boxes,
        List<Candidate> candidates,
        int sourceWidth,
        int sourceHeight)
    {
        var data = boxes.AsReadOnlySpan<float>();
        var result = new List<Detection>(candidates.Count);

        foreach (var candidate in candidates)
        {
            int offset = candidate.Query * 4;
            float cx = data[offset];
            float cy = data[offset + 1];
            float width = data[offset + 2];
            float height = data[offset + 3];

            if (!float.IsFinite(cx) || !float.IsFinite(cy) || !float.IsFinite(width) || !float.IsFinite(height))
                continue;

            // 与官方一致：先在归一化空间算 xyxy，再乘原图宽高，最后 clip。
            float x1 = Math.Clamp((cx - width * 0.5f) * sourceWidth, 0, sourceWidth);
            float y1 = Math.Clamp((cy - height * 0.5f) * sourceHeight, 0, sourceHeight);
            float x2 = Math.Clamp((cx + width * 0.5f) * sourceWidth, 0, sourceWidth);
            float y2 = Math.Clamp((cy + height * 0.5f) * sourceHeight, 0, sourceHeight);

            // 官方实现保留裁剪后的零面积框（框完全落在图外），这里丢掉 —— 零面积的框对调用方没有意义。
            if (x2 <= x1 || y2 <= y1)
                continue;

            result.Add(new Detection(candidate.ClassId, candidate.Score, x1, y1, x2 - x1, y2 - y1));
        }

        return result;
    }

    /// <summary>逐类 sigmoid，入参先夹到 ±88（官方实现同样这么夹，避免 <c>exp</c> 溢出）。</summary>
    private static float Sigmoid(float value)
    {
        if (value > 88f)
            value = 88f;
        else if (value < -88f)
            value = -88f;

        return 1f / (1f + MathF.Exp(-value));
    }
}
