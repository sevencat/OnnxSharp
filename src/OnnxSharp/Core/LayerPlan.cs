using OnnxSharp.Tensors;

namespace OnnxSharp.Core;

/// <summary>算子计划的完成度。单调上升，只有 <see cref="LayerPlan.Invalidate"/> 才归零。</summary>
internal enum PlanState
{
    /// <summary>从未推断过：签名与输出形状都不可信。</summary>
    None = 0,

    /// <summary>输出形状/类型已推断，可以按 <see cref="LayerPlan.InShapes"/> 复用。</summary>
    Shaped = 1,

    /// <summary>已对当前输入签名调用过 <c>PrepackWeights + Finalize</c>。</summary>
    Finalized = 2,
}

/// <summary>
/// 一个算子的形状/类型推断缓存与 <c>Finalize</c> 状态，**只有一份输入签名**。
/// <para>
/// 之前这里散着 10 个字段、<b>两套签名</b>（<c>CachedIn*</c> 给形状缓存用、
/// <c>LastIn*</c> 给 Finalize 失效判定用），而形状缓存那套由一个 <c>ShapeCacheValid</c> 标志
/// 把关 —— 动态输出形状的算子从不写这个标志，于是它的 Finalize 失效判定永远为「签名变了」，
/// 每帧重跑 <c>PrepackWeights + Finalize</c>（rfdetr-small 上每帧白分配 52 MB）。
/// </para>
/// <para>
/// 现在 <see cref="PlanState.Shaped"/> 与 <see cref="PlanState.Finalized"/> 共用同一份
/// <see cref="InShapes"/>/<see cref="InTypes"/>，两个判定读同一个事实来源，那类不同步
/// 从「靠注释提醒」变成「结构上写不出来」。
/// </para>
/// </summary>
internal sealed class LayerPlan
{
    public PlanState State;

    /// <summary>输入形状签名（推断时写入，逐帧比对）。</summary>
    public readonly List<TensorShape> InShapes = new();

    /// <summary>输入类型签名（OpenCV 同时比较 shapes 与 types）。</summary>
    public readonly List<DType> InTypes = new();

    /// <summary>推断出的输出形状（<see cref="PlanState.Shaped"/> 之后可直接复用）。</summary>
    public readonly List<TensorShape> OutShapes = new();

    /// <summary>推断出的临时张量形状。</summary>
    public readonly List<TensorShape> TempShapes = new();

    /// <summary>推断出的输出类型。</summary>
    public readonly List<DType> OutTypes = new();

    /// <summary>推断出的临时张量类型。</summary>
    public readonly List<DType> TempTypes = new();

    /// <summary>
    /// 当前输入签名是否与计划里记录的一致。
    /// <para>
    /// 纯签名比较，<b>不掺状态</b>：<see cref="PlanState.None"/> 时两边都是空列表的零输入算子
    /// （<c>Constant</c>）也会返回 true，所以缓存复用必须先判 <c>State &gt;= Shaped</c>。
    /// </para>
    /// </summary>
    public bool Matches(IReadOnlyList<TensorShape> shapes, IReadOnlyList<DType> types)
    {
        if (InShapes.Count != shapes.Count || InTypes.Count != types.Count)
            return false;

        for (int i = 0; i < InShapes.Count; i++)
        {
            if (!InShapes[i].Equals(shapes[i]))
                return false;
        }

        for (int i = 0; i < InTypes.Count; i++)
        {
            if (InTypes[i] != types[i])
                return false;
        }

        return true;
    }

    /// <summary>覆盖写入输入签名（不清容量，稳态下无分配）。</summary>
    public void WriteInputs(IReadOnlyList<TensorShape> shapes, IReadOnlyList<DType> types)
    {
        InShapes.Clear();
        InShapes.AddRange(shapes);
        InTypes.Clear();
        InTypes.AddRange(types);
    }

    /// <summary>
    /// 让计划作废。只改状态、不清列表 —— 列表容量留着，重新推断时不重新分配。
    /// </summary>
    public void Invalidate() => State = PlanState.None;
}
