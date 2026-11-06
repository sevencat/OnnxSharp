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

    /// <summary>权重预处理（打包/重排）。权重变化时会被再次调用。</summary>
    public virtual void PrepackWeights()
    {
    }

    /// <summary>根据实际输入形状做一次性准备（选择 kernel、预计算索引等）。</summary>
    public virtual void Finalize(IReadOnlyList<Tensor> inputs, IReadOnlyList<Tensor> outputs)
    {
    }

    public virtual void Forward(
        IReadOnlyList<Tensor> inputs,
        IReadOnlyList<Tensor> outputs,
        IReadOnlyList<Tensor> temps)
        => throw new NotSupportedException($"op '{Type}' has no Forward() implementation");
}
