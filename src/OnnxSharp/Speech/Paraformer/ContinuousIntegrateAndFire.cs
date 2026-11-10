using OnnxSharp.Core;

namespace OnnxSharp.Speech.Paraformer;

/// <summary>
/// Paraformer 的 CIF（Continuous Integrate-and-Fire），**逐节点移植自模型里的 ONNX <c>Loop</c> body**
/// （`tools/scripts/dump_loop_body.py` 可以打印那份 body）。
/// <para>
/// 为什么要自己写：CIF 在导出图里是一个 80 节点的 <c>Loop</c> + 两个 <c>Sequence*</c> 输出，
/// 引擎不支持序列/子图；而它的语义其实只有十几行：
/// </para>
/// <code>
/// integrate = 0, frame = 0
/// for t in 0..T-1:
///     alpha = alphas[t]
///     completion  = 1 - integrate          # 距离下一次触发还差多少
///     integrate  += alpha
///     fire        = integrate >= 1
///     cur         = fire ? completion : alpha
///     remainds    = alpha - cur            # 触发后溢出到下一个 token 的部分
///     frame      += cur * enc[t]
///     frames[t]   = frame                  # 每个 step 的候选帧（真的被选中要看 fire）
///     if fire: integrate -= 1; frame = remainds * enc[t]
/// </code>
/// <para>
/// 图里 <c>frames</c> 是「每步的候选帧」，之后用 <c>frames[fires >= 1]</c>（<c>NonZero</c> + <c>GatherND</c>）
/// 选出真正的 token 级 acoustic embeddings —— 等价于本类返回的 <see cref="CifResult.Tokens"/>。
/// </para>
/// </summary>
public sealed class ContinuousIntegrateAndFire
{
    /// <summary>触发阈值（模型里是常数 1.0）。</summary>
    public float Threshold { get; init; } = 1f;

    /// <summary>
    /// 处理单条语音。
    /// </summary>
    /// <param name="encoder">encoder 输出 <c>[T, dim]</c>（行优先）。</param>
    /// <param name="frames">时间步数 <c>T</c>。</param>
    /// <param name="dim">每帧维度。</param>
    /// <param name="alphas">CIF 权重 <c>[T]</c>（模型的 <c>/predictor/Add_output_0</c>）。</param>
    public CifResult Process(ReadOnlySpan<float> encoder, int frames, int dim, ReadOnlySpan<float> alphas)
    {
        if (frames < 0 || dim <= 0)
            throw new OnnxSharpException($"CIF: invalid shape [{frames},{dim}]");
        if (encoder.Length < (long)frames * dim)
            throw new OnnxSharpException($"CIF: encoder has {encoder.Length} values, [{frames},{dim}] needed");
        if (alphas.Length < frames)
            throw new OnnxSharpException($"CIF: alphas has {alphas.Length} values, {frames} needed");

        var candidates = new float[(long)frames * dim];
        var fireAlphas = new float[frames];
        var fireIndices = new List<int>(frames);
        var current = new float[dim];

        float integrate = 0f;

        for (int t = 0; t < frames; t++)
        {
            float alpha = alphas[t];
            float completion = Threshold - integrate;
            float accumulate = integrate + alpha;
            bool fire = accumulate >= Threshold;

            // cur：触发时只取「补满」的那一部分，否则整段 alpha
            float cur = fire ? completion : alpha;
            float remainds = alpha - cur;

            var frame = encoder.Slice((int)((long)t * dim), dim);
            for (int d = 0; d < dim; d++)
                current[d] += cur * frame[d];

            Array.Copy(current, 0, candidates, (long)t * dim, dim);
            fireAlphas[t] = accumulate;

            if (fire)
            {
                integrate = accumulate - Threshold;
                fireIndices.Add(t);

                // 重置成「溢出部分 × 当前帧」，作为下一个 token 的起点
                for (int d = 0; d < dim; d++)
                    current[d] = remainds * frame[d];
            }
            else
            {
                integrate = accumulate;
            }
        }

        var tokens = new float[(long)fireIndices.Count * dim];
        for (int i = 0; i < fireIndices.Count; i++)
            Array.Copy(candidates, (long)fireIndices[i] * dim, tokens, (long)i * dim, dim);

        return new CifResult
        {
            Candidates = candidates,
            FireAlphas = fireAlphas,
            FireIndices = fireIndices.ToArray(),
            Tokens = tokens,
            TokenCount = fireIndices.Count,
            Dim = dim,
            LeftoverFrame = current,
            LeftoverAlpha = integrate,
        };
    }
}

/// <summary>CIF 的结果。形状全部是「单条语音」的。</summary>
public sealed class CifResult
{
    /// <summary>每个时间步的候选帧 <c>[T, dim]</c>（模型里的 <c>frames</c>）。</summary>
    public required float[] Candidates { get; init; }

    /// <summary>每个时间步累加后的 alpha <c>[T]</c>（模型里的 <c>fires</c>，触发判据是 <c>&gt;= 1</c>）。</summary>
    public required float[] FireAlphas { get; init; }

    /// <summary>触发的时间步下标。</summary>
    public required int[] FireIndices { get; init; }

    /// <summary>token 级 acoustic embeddings <c>[TokenCount, dim]</c>（decoder 的输入）。</summary>
    public required float[] Tokens { get; init; }

    /// <summary>token 数。</summary>
    public required int TokenCount { get; init; }

    /// <summary>每帧维度。</summary>
    public required int Dim { get; init; }

    /// <summary>循环结束时残留的帧 <c>[dim]</c>（模型里的 <c>frame.7</c>）。</summary>
    public required float[] LeftoverFrame { get; init; }

    /// <summary>循环结束时残留的 alpha（模型里的 <c>integrate.7</c>）。</summary>
    public required float LeftoverAlpha { get; init; }
}
