namespace OnnxSharp.Vision;

/// <summary>
/// YuNet 的后处理：把 12 个输出头（cls/obj/bbox/kps × stride 8/16/32）解成 <see cref="Face"/>。
/// <para>
/// 公式直接对齐 OpenCV <c>modules/objdetect/src/face_detect.cpp</c> 的 <c>postProcess</c>
/// （<b>不是</b> SSD 的 prior/variances 那一套，别按 SSD 猜）：
/// </para>
/// <code>
/// score = sqrt(max(cls,0) · max(obj,0))
/// cx = (col + bbox[0]) · stride      cy = (row + bbox[1]) · stride
/// w  = exp(bbox[2]) · stride         h  = exp(bbox[3]) · stride
/// kps[n] = ((kps[2n] + col) · stride, (kps[2n+1] + row) · stride)
/// </code>
/// <para>
/// 注意 <c>col</c>/<c>row</c> 是<b>格子的下标</b>（不是中心 +0.5）—— 这一点用 lena 上的
/// <c>cv2.FaceDetectorYN</c> 逐值验证过（见 <c>tools/scripts/yunet_probe.py</c>）。
/// </para>
/// </summary>
public static class YuNetDecoder
{
    /// <summary>YuNet 的三个输出 stride，顺序与输出名 <c>*_8 / *_16 / *_32</c> 一致。</summary>
    public static ReadOnlySpan<int> Strides => [8, 16, 32];

    /// <summary>
    /// 解码一层（一个 stride）。<paramref name="bbox"/> 是 <c>[N,4]</c>，
    /// <paramref name="kps"/> 是 <c>[N,10]</c>，坐标以网络输入尺寸为准。
    /// </summary>
    public static void DecodeStride(
        ReadOnlySpan<float> cls,
        ReadOnlySpan<float> obj,
        ReadOnlySpan<float> bbox,
        ReadOnlySpan<float> kps,
        int inputWidth,
        int inputHeight,
        int stride,
        float scoreThreshold,
        List<Face> destination)
    {
        int columns = inputWidth / stride;
        int rows = inputHeight / stride;
        int cells = columns * rows;

        for (int index = 0; index < cells; index++)
        {
            // OpenCV 先 clamp 到 0（网络偶尔吐负值），再开方
            float clsScore = MathF.Max(cls[index], 0f);
            float objScore = MathF.Max(obj[index], 0f);
            float score = MathF.Sqrt(clsScore * objScore);

            if (score < scoreThreshold)
                continue;

            int row = index / columns;
            int column = index % columns;

            int boxOffset = index * 4;
            float centerX = (column + bbox[boxOffset]) * stride;
            float centerY = (row + bbox[boxOffset + 1]) * stride;
            float width = MathF.Exp(bbox[boxOffset + 2]) * stride;
            float height = MathF.Exp(bbox[boxOffset + 3]) * stride;

            int keyOffset = index * 10;
            destination.Add(new Face(
                score,
                centerX - width / 2f,
                centerY - height / 2f,
                width,
                height,
                new FaceLandmarks(
                    KeyPoint(kps, keyOffset, 0, column, row, stride),
                    KeyPoint(kps, keyOffset, 1, column, row, stride),
                    KeyPoint(kps, keyOffset, 2, column, row, stride),
                    KeyPoint(kps, keyOffset, 3, column, row, stride),
                    KeyPoint(kps, keyOffset, 4, column, row, stride))));
        }
    }

    private static FacePoint KeyPoint(
        ReadOnlySpan<float> kps,
        int offset,
        int index,
        int column,
        int row,
        int stride)
        => new(
            (kps[offset + index * 2] + column) * stride,
            (kps[offset + index * 2 + 1] + row) * stride);

    /// <summary>
    /// 贪心 NMS（与 OpenCV <c>NMSBoxes</c> 同语义：按分数降序，和已保留框的 IoU 严格大于阈值就压掉）。
    /// </summary>
    public static List<Face> Suppress(List<Face> candidates, float iouThreshold, int maxFaces)
    {
        var kept = new List<Face>();
        if (candidates.Count == 0 || maxFaces <= 0)
            return kept;

        candidates.Sort((a, b) => b.Score.CompareTo(a.Score));

        foreach (var candidate in candidates)
        {
            bool suppressed = false;
            foreach (var accepted in kept)
            {
                if (Face.IoU(candidate, accepted) > iouThreshold)
                {
                    suppressed = true;
                    break;
                }
            }

            if (suppressed)
                continue;

            kept.Add(candidate);
            if (kept.Count >= maxFaces)
                break;
        }

        return kept;
    }

    /// <summary>
    /// 把 <paramref name="faces"/> 从网络输入空间线性缩放到原图空间。
    /// 模型输入是静态 <c>[1,3,640,640]</c>，而原图宽高比一般不相等，所以 x/y 用各自的缩放系数。
    /// </summary>
    public static List<Face> ScaleToImage(List<Face> faces, int inputWidth, int inputHeight, int imageWidth, int imageHeight)
    {
        if (inputWidth == imageWidth && inputHeight == imageHeight)
            return faces;

        float scaleX = (float)imageWidth / inputWidth;
        float scaleY = (float)imageHeight / inputHeight;

        for (int i = 0; i < faces.Count; i++)
        {
            var face = faces[i];
            var marks = face.Landmarks;
            faces[i] = new Face(
                face.Score,
                face.X * scaleX,
                face.Y * scaleY,
                face.Width * scaleX,
                face.Height * scaleY,
                new FaceLandmarks(
                    new FacePoint(marks.RightEye.X * scaleX, marks.RightEye.Y * scaleY),
                    new FacePoint(marks.LeftEye.X * scaleX, marks.LeftEye.Y * scaleY),
                    new FacePoint(marks.Nose.X * scaleX, marks.Nose.Y * scaleY),
                    new FacePoint(marks.RightMouth.X * scaleX, marks.RightMouth.Y * scaleY),
                    new FacePoint(marks.LeftMouth.X * scaleX, marks.LeftMouth.Y * scaleY)));
        }

        return faces;
    }
}
