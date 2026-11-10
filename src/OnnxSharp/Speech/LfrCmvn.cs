namespace OnnxSharp.Speech;

/// <summary>
/// funasr 的 LFR（low frame rate）+ CMVN，语义照搬 <c>WavFrontend</c>：
/// <code>
/// T_lfr = ceil(T / lfr_n)
/// 左侧补 (lfr_m-1)/2 帧（复制第 0 帧），右侧不足时重复最后一帧
/// 每个输出帧 = 连续 lfr_m 个 fbank 帧拼接 → (x + neg_mean) * inv_stddev
/// </code>
/// <para>
/// 参数来自模型 metadata（本模型的 <c>lfr_m=7</c>、<c>lfr_n=6</c>、80 维 fbank → 560 维）。
/// 注意 metadata 里 <c>neg_mean</c> / <c>inv_stddev</c> 是 <b>560 维</b>（80 维按 7 次平铺），
/// 所以「先 CMVN 再 LFR」与「先 LFR 再 CMVN」结果相同。
/// </para>
/// </summary>
public sealed class LfrCmvn
{
    public int WindowSize { get; init; } = 7;

    public int WindowShift { get; init; } = 6;

    /// <summary>CMVN 的加性项（<c>neg_mean</c>），长度可为 <c>dim</c> 或 <c>window*dim</c>。</summary>
    public float[]? Additive { get; init; }

    /// <summary>CMVN 的乘性项（<c>inv_stddev</c>）。</summary>
    public float[]? Scale { get; init; }

    /// <summary>输出帧数（funasr 的 <c>ceil(T / shift)</c>）。</summary>
    public int FrameCount(int frames)
        => frames <= 0 ? 0 : (frames + WindowShift - 1) / WindowShift;

    /// <summary>LFR + CMVN，输出 <c>[FrameCount(frames), window*dim]</c>。</summary>
    public float[] Apply(ReadOnlySpan<float> fbank, int frames, int dim, out int outFrames)
    {
        if (frames <= 0 || dim <= 0)
        {
            outFrames = 0;
            return Array.Empty<float>();
        }

        outFrames = FrameCount(frames);
        int window = WindowSize;
        int outDim = window * dim;
        var result = new float[(long)outFrames * outDim];

        int left = (window - 1) / 2;

        bool hasCmvn = Additive is not null && Scale is not null;
        bool cmvnIsFull = hasCmvn && Additive!.Length >= outDim && Scale!.Length >= outDim;

        for (int frame = 0; frame < outFrames; frame++)
        {
            long offset = (long)frame * outDim;

            for (int j = 0; j < window; j++)
            {
                int index = frame * WindowShift + j - left;
                index = Math.Clamp(index, 0, frames - 1);

                var source = fbank.Slice(index * dim, dim);
                for (int d = 0; d < dim; d++)
                {
                    int target = j * dim + d;
                    float value = source[d];

                    if (hasCmvn)
                    {
                        int c = cmvnIsFull ? target : d;
                        value = (value + Additive![c]) * Scale![c];
                    }

                    result[offset + target] = value;
                }
            }
        }

        return result;
    }
}
