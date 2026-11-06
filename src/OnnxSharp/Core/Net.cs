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

        if (impl.MainGraph is { } graph)
            ConstFolder.Run(impl, graph);

        return new Net(impl);
    }

    public void SetInput(string name, Tensor input) => _impl.SetInput(name, input);

    public Tensor Forward() => _impl.Forward();

    public Tensor Forward(string outputName) => _impl.Forward(outputName);

    public List<Tensor> ForwardAll() => _impl.ForwardAll();
}
