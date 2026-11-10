namespace OnnxSharp.Vision;

/// <summary>一个人脸关键点，坐标原点在<b>原图</b>左上角，单位是原图像素。</summary>
public readonly record struct FacePoint(float X, float Y)
{
    public override string ToString() => $"({X:F1}, {Y:F1})";
}

/// <summary>
/// YuNet 的 5 点人脸关键点，顺序与 OpenCV <c>FaceDetectorYN</c> 一致：
/// <b>右眼、左眼、鼻尖、右嘴角、左嘴角</b>（“左右”是图中的左右，不是人的左右）。
/// </summary>
public readonly record struct FaceLandmarks(
    FacePoint RightEye,
    FacePoint LeftEye,
    FacePoint Nose,
    FacePoint RightMouth,
    FacePoint LeftMouth);

/// <summary>
/// 一张人脸：检测框 + 5 个关键点 + 置信度。坐标都在<b>原图</b>空间
/// （<see cref="YuNetDetector"/> 已把网络输入尺寸换算回去）。
/// </summary>
/// <param name="Score">置信度，<c>sqrt(cls × obj)</c>，与 OpenCV 同口径。</param>
/// <param name="X">框左上角横坐标。</param>
/// <param name="Y">框左上角纵坐标。</param>
/// <param name="Width">框宽。</param>
/// <param name="Height">框高。</param>
public readonly record struct Face(
    float Score,
    float X,
    float Y,
    float Width,
    float Height,
    FaceLandmarks Landmarks)
{
    public float Right => X + Width;

    public float Bottom => Y + Height;

    public float Area => Width * Height;

    /// <summary>框的 IoU（不含关键点）。不相交时返回 0。</summary>
    public static float IoU(in Face a, in Face b)
    {
        float x1 = MathF.Max(a.X, b.X);
        float y1 = MathF.Max(a.Y, b.Y);
        float x2 = MathF.Min(a.Right, b.Right);
        float y2 = MathF.Min(a.Bottom, b.Bottom);

        float w = x2 - x1;
        float h = y2 - y1;
        if (w <= 0 || h <= 0)
            return 0;

        float intersection = w * h;
        float union = a.Area + b.Area - intersection;
        return union <= 0 ? 0 : intersection / union;
    }

    public override string ToString()
        => $"face score {Score:F4} box [{X:F1}, {Y:F1}, {Width:F1} x {Height:F1}] " +
           $"eye({Landmarks.RightEye.X:F0},{Landmarks.RightEye.Y:F0})";
}
