using System.Diagnostics;
using OnnxSharp.Core;
using OnnxSharp.Tensors;

namespace OnnxSharp.Speech.Paraformer;

/// <summary>Paraformer 的模型路径与张量名约定。</summary>
public sealed class ParaformerOptions
{
    /// <summary>切出来的 encoder 段（输入 <c>speech/speech_lengths</c>，输出 encoder 特征与 CIF 的 alphas）。</summary>
    public required string EncoderPath { get; init; }

    /// <summary>切出来的 decoder 段（输入 <c>speech/speech_lengths/memory/max_label_len</c>，输出 <c>logits</c>）。</summary>
    public required string DecoderPath { get; init; }

    public required string TokensPath { get; init; }

    public string EncoderOutput { get; init; } = "/predictor/Concat_5_output_0";

    public string AlphaOutput { get; init; } = "/predictor/Add_output_0";

    public string MemoryInput { get; init; } = "onnx::Slice_6785";

    public string MaxLabelLengthInput { get; init; } = "max_label_len.28";

    public string LogitsOutput { get; init; } = "logits";
}

/// <summary>识别结果。</summary>
public sealed class RecognitionResult
{
    public required string Text { get; init; }

    public required IReadOnlyList<int> TokenIds { get; init; }

    /// <summary>LFR 之后的特征帧数（= encoder 的时间步）。</summary>
    public required int FeatureFrames { get; init; }

    /// <summary>CIF 触发出来的 token 数。</summary>
    public required int TokenCount { get; init; }

    public required double FeatureMs { get; init; }

    public required double EncoderMs { get; init; }

    public required double DecoderMs { get; init; }

    /// <summary>encoder 图加载 + 形状特化的耗时（每次识别都要重来）。</summary>
    public double EncoderLoadMs { get; init; }

    /// <summary>decoder 图加载 + 形状特化的耗时。</summary>
    public double DecoderLoadMs { get; init; }

    public double TotalMs => FeatureMs + EncoderMs + DecoderMs + EncoderLoadMs + DecoderLoadMs;

    public override string ToString() => Text;
}

/// <summary>
/// 纯 C# 的 Paraformer 离线识别：<c>wav → fbank/LFR/CMVN → encoder → CIF → decoder → argmax → 文本</c>。
/// <para>
/// 两段模型由 <c>tools/scripts/split_*.py</c> 从 sherpa 的 <c>model.int8.onnx</c> 切出来。
/// 每次识别都会按实际帧数**特化**并重新加载两张图 —— 引擎的形状特化发生在 <c>ReadModel</c>，
/// 不同长度的音频没法复用同一份图（要复用就得按长度做缓存）。
/// </para>
/// </summary>
public sealed class ParaformerRecognizer
{
    private readonly ParaformerOptions _options;
    private readonly SpeechFeatureExtractor _features;
    private readonly TokenTable _tokens;
    private readonly ContinuousIntegrateAndFire _cif = new();

    public ParaformerRecognizer(ParaformerOptions options)
    {
        _options = options;
        _features = SpeechFeatureExtractor.FromModel(options.EncoderPath);
        _tokens = TokenTable.Load(options.TokensPath);
    }

    /// <summary>识别一个 16 kHz 单声道 wav。</summary>
    public RecognitionResult TranscribeWav(string wavPath)
    {
        var (samples, sampleRate) = WavReader.Read(wavPath);
        if (sampleRate != 16000)
            throw new OnnxSharpException(
                $"{wavPath}: sample rate {sampleRate} Hz, Paraformer needs 16 kHz (resample first)");

        return Transcribe(samples);
    }

    /// <summary>识别 16 kHz 单声道采样（取值范围 [-1, 1]）。</summary>
    public RecognitionResult Transcribe(ReadOnlySpan<float> samples)
    {
        // ---- 1) 特征：fbank(80) → LFR(7,6) → CMVN → [T, 560]
        var watch = Stopwatch.StartNew();
        var features = _features.Compute(samples);
        double featureMs = watch.Elapsed.TotalMilliseconds;

        int frames = features.Frames;
        int dim = features.Dim;

        // ---- 2) encoder：一次拿到 encoder 特征与 CIF 的 alphas
        watch.Restart();
        var encoder = Net.ReadModel(_options.EncoderPath, new Dictionary<string, int[]>
        {
            ["speech"] = [1, frames, dim],
            ["speech_lengths"] = [1],
        });

        double encoderLoadMs = watch.Elapsed.TotalMilliseconds;
        watch.Restart();

        var speech = Tensor.Allocate(new TensorShape(1, frames, dim), DType.F32);
        Buffer.BlockCopy(features.Values, 0, speech.Buffer, 0, features.Values.Length * sizeof(float));

        var lengths = Tensor.Allocate(new TensorShape(1), DType.I32);
        lengths.AsSpan<int>()[0] = frames;

        encoder.SetInput("speech", speech);
        encoder.SetInput("speech_lengths", lengths);

        var encoderOutputs = encoder.ForwardAll();
        var hidden = FindOutput(encoder, encoderOutputs, _options.EncoderOutput);
        var alphas = FindOutput(encoder, encoderOutputs, _options.AlphaOutput);
        double encoderMs = watch.Elapsed.TotalMilliseconds;

        // ---- 3) CIF（C# 重写，替代图里的 Loop）
        // 注意：encoder 的输出比输入特征**多一帧**（模型在末尾补了一帧），
        // CIF 必须按 encoder 的帧数跑，否则会少触发一个 token。
        int hiddenDim = hidden.Shape.Dims[2];
        int encoderFrames = hidden.Shape.Dims[1];
        var cif = _cif.Process(hidden.AsSpan<float>(), encoderFrames, hiddenDim, alphas.AsSpan<float>());
        int tokenCount = cif.TokenCount;

        if (tokenCount <= 0)
        {
            return new RecognitionResult
            {
                Text = string.Empty,
                TokenIds = Array.Empty<int>(),
                FeatureFrames = frames,
                TokenCount = 0,
                FeatureMs = featureMs,
                EncoderMs = encoderMs,
                DecoderMs = 0,
            };
        }

        // ---- 4) decoder：memory = CIF 的 [1, tokenNum, D]
        watch.Restart();
        var decoder = Net.ReadModel(_options.DecoderPath, new Dictionary<string, int[]>
        {
            ["speech"] = [1, frames, dim],
            ["speech_lengths"] = [1],
            [_options.MemoryInput] = [1, tokenCount, hiddenDim],
        });

        double decoderLoadMs = watch.Elapsed.TotalMilliseconds;
        watch.Restart();

        var memory = Tensor.Allocate(new TensorShape(1, tokenCount, hiddenDim), DType.F32);
        Buffer.BlockCopy(cif.Tokens, 0, memory.Buffer, 0, cif.Tokens.Length * sizeof(float));

        var maxLength = Tensor.Allocate(TensorShape.Scalar, DType.I64);
        maxLength.AsSpan<long>()[0] = tokenCount;

        decoder.SetInput("speech", speech);
        decoder.SetInput("speech_lengths", lengths);
        decoder.SetInput(_options.MemoryInput, memory);
        decoder.SetInput(_options.MaxLabelLengthInput, maxLength);

        var decoderOutputs = decoder.ForwardAll();
        var logits = FindOutput(decoder, decoderOutputs, _options.LogitsOutput);
        double decoderMs = watch.Elapsed.TotalMilliseconds;

        // ---- 5) argmax + token 解码（去 blank、折叠重复、遇 EOS 停）
        var argMax = ArgMaxRows(logits, tokenCount, logits.Shape.Dims[^1]);
        var ids = _tokens.DecodeIds(argMax);

        return new RecognitionResult
        {
            Text = _tokens.ToText(ids),
            TokenIds = ids,
            FeatureFrames = frames,
            TokenCount = tokenCount,
            FeatureMs = featureMs,
            EncoderMs = encoderMs,
            DecoderMs = decoderMs,
            EncoderLoadMs = encoderLoadMs,
            DecoderLoadMs = decoderLoadMs,
        };
    }

    private static Tensor FindOutput(Net net, List<Tensor> outputs, string name)
    {
        var names = net.OutputNames;
        for (int i = 0; i < names.Count && i < outputs.Count; i++)
        {
            if (names[i] == name)
                return outputs[i];
        }

        throw new OnnxSharpException(
            $"model has no output '{name}' (outputs: {string.Join(", ", names)})");
    }

    /// <summary>逐行 argmax（词表维在最后一维）。</summary>
    private static long[] ArgMaxRows(Tensor logits, int rows, int columns)
    {
        var source = logits.AsSpan<float>();
        var result = new long[rows];

        for (int row = 0; row < rows; row++)
        {
            long offset = (long)row * columns;
            int best = 0;
            float bestValue = source[(int)offset];

            for (int column = 1; column < columns; column++)
            {
                float value = source[(int)(offset + column)];
                if (value > bestValue)
                {
                    bestValue = value;
                    best = column;
                }
            }

            result[row] = best;
        }

        return result;
    }
}

