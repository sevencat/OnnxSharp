namespace OnnxSharp.Vision;

/// <summary>
/// COCO 80 类的顺序表（yolov8 / yolo11 / yolo26 的公开权重都是这套，检测头维度 <c>4 + 80 = 84</c>）。
/// <para>
/// 只做「类别编号 → 名字」的查表，不参与任何推理。自定义训练的模型类别顺序不同，请自己传名字表。
/// </para>
/// </summary>
public static class CocoLabels
{
    /// <summary>类别数。</summary>
    public const int Count = 80;

    /// <summary>检测头字段数（4 个坐标 + 80 个类别分）；输出这一维就说明是 COCO 权重。</summary>
    public const int DetectionFields = 4 + Count;

    private static readonly string[] NamesInternal =
    [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
        "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
        "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
        "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
        "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
        "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
        "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
        "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
    ];

    /// <summary>按类别编号索引的名字表（只读）。</summary>
    public static IReadOnlyList<string> Names => NamesInternal;

    /// <summary>取类别名；编号越界返回 <c>null</c>。</summary>
    public static string? NameOf(int classId)
        => (uint)classId < (uint)NamesInternal.Length ? NamesInternal[classId] : null;

    /// <summary>取类别名，越界时退回 <c>class{id}</c>（日志/CLI 用，永不返回 null）。</summary>
    public static string LabelOf(int classId)
        => NameOf(classId) ?? $"class{classId}";
}
