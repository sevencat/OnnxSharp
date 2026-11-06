using OnnxSharp.Tensors;

namespace OnnxSharp.Core;

/// <summary>
/// 后端无关的算子描述：只负责 shape / type 推断，不执行计算。
/// 对应 OpenCV 新引擎的 <c>LayerInfo</c>。
/// </summary>
public class LayerInfo
{
    public string Name = string.Empty;

    public string Type = string.Empty;

    /// <summary>权重（ONNX initializer）。</summary>
    public List<Tensor> Blobs = new();

    public List<Arg> Inputs = new();

    public List<Arg> Outputs = new();

    /// <summary>
    /// 所属网络的 Arg 表。等价 OpenCV 的 <c>getNetImpl(this)</c>。
    /// <para>
    /// <b>为什么必须有</b>：<see cref="GetMemoryShapes"/> 只拿到形状，
    /// 而 <c>Reshape/Slice/Split/Expand/TopK/Reduce*/ConstantOfShape</c> 必须读常量输入的<b>值</b>才能推断。
    /// OpenCV 的做法就在形状推断里直接
    /// <c>netimpl_->argTensor(this-&gt;inputs[1])</c>（见 <c>slice2_layer.cpp:363-376</c>）。
    /// </para>
    /// <para>由 <see cref="NetImpl"/> 在首次推理前回填，算子实现不要自己赋值。</para>
    /// </summary>
    public NetImpl? Net { get; internal set; }

    /// <summary>输入个数（便于算子实现写断言）。</summary>
    public int InputCount => Inputs.Count;

    /// <summary>
    /// 取第 <paramref name="index"/> 个输入张量（含常量值）。
    /// 形状推断阶段用它读常量，例如 <c>Reshape</c> 的目标 shape、<c>Slice</c> 的 starts/ends。
    /// </summary>
    public Tensor In(int index)
    {
        if (Net is null)
            throw new OnnxSharpException($"op '{Name}' ({Type}) is not bound to a network");
        if ((uint)index >= (uint)Inputs.Count)
            throw new OnnxSharpException($"op '{Name}' ({Type}) has no input #{index}");
        return Net.GetArgTensor(Inputs[index]);
    }

    /// <summary>第 <paramref name="index"/> 个输入是否为常量（常量才能在形状推断时读值）。</summary>
    public bool IsConstInput(int index)
    {
        if (Net is null || (uint)index >= (uint)Inputs.Count)
            return false;
        return Net.GetArgKind(Inputs[index]) == ArgKind.Const;
    }

    /// <summary>推断输出/临时张量形状。必须由具体算子实现。</summary>
    public virtual void GetMemoryShapes(
        IReadOnlyList<TensorShape> inShapes,
        int requiredOutputs,
        List<TensorShape> outShapes,
        List<TensorShape> tempShapes)
        => throw new NotSupportedException($"op '{Type}' does not implement GetMemoryShapes()");

    /// <summary>推断输出/临时张量类型。默认沿用第 0 个输入的类型。</summary>
    public virtual void GetTypes(
        IReadOnlyList<DType> inTypes,
        int requiredOutputs,
        int requiredTemps,
        List<DType> outTypes,
        List<DType> tempTypes)
    {
        var t = inTypes.Count > 0 ? inTypes[0] : DType.F32;
        outTypes.Clear();
        for (int i = 0; i < requiredOutputs; i++)
            outTypes.Add(t);
        tempTypes.Clear();
        for (int i = 0; i < requiredTemps; i++)
            tempTypes.Add(DType.F32);
    }

    /// <summary>
    /// 存储布局推断（对应 OpenCV <c>LayerInfo::getLayouts</c>）。
    /// <para>
    /// 首版所有算子恒返回 NCHW，<b>没有任何 pass 调用它</b>。
    /// 保留纯属接口预留：将来做图级 blocked layout 时，只需实现这个方法和一个
    /// <c>TransformLayout</c> 算子，不必改 42 个算子的签名。
    /// </para>
    /// </summary>
    public virtual void GetLayouts(IReadOnlyList<Layout> inputLayouts, List<Layout> outputLayouts)
    {
        outputLayouts.Clear();
        var l = inputLayouts.Count > 0 ? inputLayouts[0] : Layout.NCHW;
        for (int i = 0; i < Outputs.Count; i++)
            outputLayouts.Add(l);
    }

    /// <summary>输出形状依赖运行时数值（如 Where/Expand/ConstantOfShape）。</summary>
    public virtual bool DynamicOutputShapes() => false;

    /// <summary>控制流子图（If/Loop/Scan）。首版不做。</summary>
    public virtual IReadOnlyList<Graph>? Subgraphs() => null;

    public override string ToString() => $"{Type}({Name})";
}

/// <summary>
/// 可执行算子。CPU 路径下与 <see cref="LayerInfo"/> 为同一对象
/// （对齐 OpenCV 中 <c>Layer : LayerInfo</c> 的设计）。
/// </summary>
public class Layer : LayerInfo
{
    public bool FinalizedOnce;

    /// <summary>上次 Finalize 时的输入形状签名（形状变化时重新 Finalize）。</summary>
    public readonly List<TensorShape> LastInShapes = new();

    /// <summary>上次 Finalize 时的输入类型签名（OpenCV 同时比较 shapes 与 types）。</summary>
    public readonly List<DType> LastInTypes = new();

    /// <summary>权重预处理（打包/重排）。权重变化时会被再次调用。</summary>
    public virtual void PrepackWeights()
    {
    }

    /// <summary>根据实际输入形状做一次性准备（选择 kernel、预计算索引等）。</summary>
    public virtual void Finalize(IReadOnlyList<Tensor> inputs, IReadOnlyList<Tensor> outputs)
    {
    }

    /// <summary>
    /// 执行。
    /// <para>
    /// <b>动态形状约定</b>（对齐 OpenCV <c>net_impl2.cpp:1891-1898</c>）：
    /// 当 <see cref="LayerInfo.DynamicOutputShapes"/> 为 true 时，<paramref name="outputs"/>
    /// 里的 Tensor <b>没有被预先 Fit</b>，实现必须自己
    /// <c>outputs[i].Fit(shape, dtype)</c> 之后再写数据。
    /// </para>
    /// </summary>
    public virtual void Forward(
        IReadOnlyList<Tensor> inputs,
        IReadOnlyList<Tensor> outputs,
        IReadOnlyList<Tensor> temps)
        => throw new NotSupportedException($"op '{Type}' has no Forward() implementation");
}
