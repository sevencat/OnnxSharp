using OnnxSharp.Core;
using OnnxSharp.Tensors;

namespace OnnxSharp.Ocr.PaddleOCR;

/// <summary>
/// PaddleOCR 涓夋寮忔祦姘寸嚎锛欴B 妫€娴?鈫掞紙鍙€夛級鏂囨湰琛屾柟鍚戝垎绫?鈫?CTC 璇嗗埆銆?
/// <para>
/// <b>鐢ㄦ硶</b>锛?
/// <code>
/// var model = PaddleOcrModel.FromFiles("det.onnx", "rec.onnx", "ppocr_keys.txt", "cls.onnx");
/// var ocr = new PaddleOcrPipeline(model);
/// foreach (var line in ocr.Run(bgra, width, height, OcrPixelFormat.Bgra32, stride))
///     Console.WriteLine(line);
/// </code>
/// </para>
/// <para>
/// <b>涓轰粈涔堣銆屾寜灏哄浠藉浘銆?/b>锛歅P-OCRv6 鐨?det / rec 閮芥槸鍏ㄥ姩鎬佸鍑猴紙<c>x[N,3,H,W]</c>锛夛紝
/// 寮曟搸闈?<see cref="Net.ReadModel(string, int[])"/> 鎶婅緭鍏ョ淮鍥哄畾鎴愬叿浣撳€兼墠鑳芥姌鍙犲舰鐘堕摼銆?
/// 浜庢槸姣忎釜 <c>(H,W)</c> 缁勫悎瀵瑰簲涓€浠界壒鍖栧悗鐨勫浘锛氭娴嬫寜 32 瀵归綈銆佽瘑鍒寜 16 鍒嗘《锛?
/// 杩欓噷鎸夊舰鐘剁紦瀛橈紙榛樿 det 2 浠?/ rec 8 浠斤級锛屽悓涓€灏哄鐨勫悗缁皟鐢ㄤ笉闇€瑕侀噸鏂板鍏ャ€?
/// </para>
/// <para>涓嶅仛绾跨▼瀹夊叏锛氫竴涓疄渚嬩覆琛屼娇鐢紱澶氱嚎绋嬭鍚勮嚜鎸佹湁涓€涓疄渚嬶紙妯″瀷瀛楄妭鍙互鍏变韩锛夈€?/para>
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

    public PaddleOcrPipeline(PaddleOcrModel model, PaddleOcrOptions? options = null)
    {
        _model = model ?? throw new ArgumentNullException(nameof(model));
        Options = options ?? new PaddleOcrOptions();
    }

    /// <summary>璇嗗埆娴佺▼鍙傛暟锛堝彲鍦ㄦ瀯閫犲悗缁х画鏀癸紝涓嬩竴甯х敓鏁堬級銆?/summary>
    public PaddleOcrOptions Options { get; }

    public PaddleOcrModel Model => _model;

    /// <summary>
    /// 鏁撮〉璇嗗埆锛氳繑鍥炴寜闃呰椤哄簭鎺掑垪鐨勬枃鏈锛堝潗鏍囨槸<b>婧愬浘鍍忕礌</b>锛夈€?
    /// </summary>
    /// <param name="pixels">浜ら敊鍍忕礌鍐呭瓨锛涙瘡琛?<paramref name="rowStride"/> 瀛楄妭锛? = 鎸夊绱у噾鎺掑垪锛夈€?/param>
    /// <param name="width">鍥惧儚瀹斤紙鍍忕礌锛夈€?/param>
    /// <param name="height">鍥惧儚楂橈紙鍍忕礌锛夈€?/param>
    /// <param name="format">鍍忕礌鎺掑垪锛岄粯璁?BGR24锛圤penCV / 澶氭暟鐩告満 SDK锛夈€?/param>
    /// <param name="rowStride">婧愬浘姣忚瀛楄妭鏁帮紙鍚灏惧榻愬～鍏咃級锛? 琛ㄧず <c>width * 姣忓儚绱犲瓧鑺傛暟</c>銆?/param>
    public IReadOnlyList<OcrLine> Run(
        ReadOnlySpan<byte> pixels,
        int width,
        int height,
        OcrPixelFormat format = OcrPixelFormat.Bgr24,
        int rowStride = 0)
    {
        if (width <= 0 || height <= 0)
            throw new ArgumentException($"invalid image size {width}x{height}");

        // ---- 1) DB 妫€娴?----
        (int detWidth, int detHeight) = DetectSize(width, height);
        Net detNet = GetDetectionNet(detWidth, detHeight);
        EnsureDetBuffer(3 * detWidth * detHeight);

        OcrImage.ToChw(pixels, width, height, format, rowStride,
            0, 0, width, height, detWidth, detHeight,
            OcrImage.DetMean, OcrImage.DetStd, _detBuffer, detWidth);

        detNet.SetInput(detNet.InputNames[0],
            Tensor.FromArray(_detBuffer.AsSpan(0, 3 * detHeight * detWidth), new TensorShape(1, 3, detHeight, detWidth)));
        Tensor probability = detNet.Forward();

        List<DbPostprocess.Box> boxes = DbPostprocess.Detect(
            probability.AsSpan<float>(),
            detWidth, detHeight,
            Options.DetThreshold, Options.BoxThreshold, Options.UnclipRatio);

        // ---- 2) 閫愭瑁佸壀 鈫?鏂瑰悜鍒嗙被 鈫?璇嗗埆 ----
        float scaleX = (float)width / detWidth;
        float scaleY = (float)height / detHeight;
        bool useClassification = Options.UseClassification && _model.HasClassification;

        var lines = new List<OcrLine>(boxes.Count);
        foreach (DbPostprocess.Box box in boxes)
        {
            DbPostprocess.Box scaled = DbPostprocess.Scale(box, scaleX, scaleY, width, height);

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

            (string text, float score) = Recognize(
                pixels, width, height, format, rowStride, x, y, cropWidth, cropHeight, rotate);

            if (text.Length != 0)
                lines.Add(new OcrLine(text, score, x, y, cropWidth, cropHeight));
        }

        return lines;
    }

    // ---------------- 妫€娴?----------------

    /// <summary>
    /// 妫€娴嬭緭鍏ュ昂瀵革細鏈€闀胯竟鍘嬪埌 <see cref="PaddleOcrOptions.DetLimitSide"/>锛?
    /// 瀹介珮閮藉彇 32 鐨勬暣鏁板€嶏紙PaddleOCR 鐨?<c>limit_type=min</c> 瑙勫垯锛夈€?
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
            _detNets.Clear();

        net = _model.CreateDetectionNet(width, height);
        _detNets[key] = net;
        return net;
    }

    private void EnsureDetBuffer(int size)
    {
        if (_detBuffer.Length < size)
            _detBuffer = new float[size];
    }

    // ---------------- 鏂瑰悜鍒嗙被 ----------------

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

        _clsNet.SetInput(_clsNet.InputNames[0], Tensor.FromArray(_clsBuffer, new TensorShape(1, 3, ClsHeight, ClsWidth)));
        ReadOnlySpan<float> logits = _clsNet.Forward().AsSpan<float>();
        if (logits.Length < 2)
            return false;

        // cls 鍥炬湯灏炬槸 2 绫?softmax锛歱1 灏辨槸銆屽€掔疆銆嶇殑姒傜巼
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

    // ---------------- 璇嗗埆 ----------------

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
        int recHeight = Math.Max(8, Options.RecHeight);
        int maxWidth = Math.Max(RecWidthStep, Options.RecMaxWidth);

        // 绛夋瘮缂╂斁鍒伴珮 recHeight锛涘搴︽寜 16 鍒嗘《锛堟《瀹戒互澶栬ˉ 0锛屼笌 PaddleOCR 鐨?padding 涓€鑷达級
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

        net.SetInput(net.InputNames[0],
            Tensor.FromArray(_recBuffer.AsSpan(0, size), new TensorShape(1, 3, recHeight, bucket)));
        ReadOnlySpan<float> logits = net.Forward().AsSpan<float>();

        return CtcDecoder.Decode(logits, _model.Classes.Count, _model.Classes);
    }

    private Net GetRecognitionNet(int height, int width)
    {
        int key = (height << 16) | width;
        if (_recNets.TryGetValue(key, out Net? net))
            return net;

        if (_recNets.Count >= Math.Max(1, Options.RecShapeCacheSize))
            _recNets.Clear();

        net = _model.CreateRecognitionNet(width, height);
        _recNets[key] = net;
        return net;
    }
}
