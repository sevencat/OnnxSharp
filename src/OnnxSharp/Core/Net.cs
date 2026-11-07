using OnnxSharp.Tensors;

namespace OnnxSharp.Core;

/// <summary>对外门面（对应 OpenCV <c>cv::dnn::Net</c> 的最小子集）。</summary>
public sealed class Net
{
    private readonly NetImpl _impl;

    public Net() => _impl = new NetImpl();

    private Net(NetImpl impl) => _impl = impl;

    public NetImpl Impl => _impl;

    public Graph? MainGraph => _impl.MainGraph;

    public IReadOnlyList<string> InputNames => _impl.GetInputNames();

    public IReadOnlyList<string> OutputNames => _impl.GetOutputNames();

    /// <summary>
    /// 从 ONNX 文件加载模型。导入后立刻做<b>常量折叠</b>：
    /// RF-DETR 的 <c>Shape → Gather → Concat → Reshape</c> 链必须折叠掉，否则形状推断无从下手。
    /// </summary>
    public static Net ReadModel(string path)
    {
        var impl = Onnx.OnnxImporter.Build(Onnx.OnnxFile.Load(path));

        if (impl is { MainGraph: { } graph })
        {
            // 先折叠（会删节点、把上游改成常量），再融合（合并剩下的运行时节点）。
            // 顺序不能反：折叠后可能又暴露出新的融合机会，而且融合不处理常量输入。
            ConstFolder.Run(impl, graph);
            GraphFusion.Run(impl, graph);
        }

        return new Net(impl);
    }

    public void SetInput(string name, Tensor input) => _impl.SetInput(name, input);

    public Tensor Forward() => _impl.Forward();

    public Tensor Forward(string outputName) => _impl.Forward(outputName);

    public List<Tensor> ForwardAll() => _impl.ForwardAll();

    /// <summary>
    /// 逐算子耗时统计（对应 OpenCV <c>Net::getPerfProfile</c>），按耗时降序返回。
    /// <para>
    /// 第 1 次迭代包含 <c>Finalize</c> / 权重打包，找热点请用 <c>iters &gt;= 2</c> 并只看平均值。
    /// </para>
    /// </summary>
    public IReadOnlyList<OpTiming> ProfileAll(int iters = 2)
    {
        if (_impl.MainGraph is not { } graph)
            throw new OnnxSharpException("model is not loaded");

        iters = Math.Max(1, iters);
        var acc = new Dictionary<LayerInfo, (double Milliseconds, int Calls)>();

        for (int i = 0; i < iters; i++)
            Interpreter.Run(_impl, graph, executeLayers: true, acc);

        var result = new List<OpTiming>(acc.Count);
        foreach (var (layer, value) in acc)
            result.Add(new OpTiming(layer.Name, layer.Type, value.Milliseconds, value.Calls));

        result.Sort((a, b) => b.Milliseconds.CompareTo(a.Milliseconds));
        return result;
    }
}

/// <summary>单个算子的累计耗时。</summary>
public sealed record OpTiming(string Name, string Type, double Milliseconds, int Calls)
{
    public double Average => Calls > 0 ? Milliseconds / Calls : 0;
}
