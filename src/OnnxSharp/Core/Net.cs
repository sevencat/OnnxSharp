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
    public static Net ReadModel(string path) => ReadModel(path, (IReadOnlyDictionary<string, int[]>?)null);

    /// <summary>
    /// 加载模型并把<b>动态输入维固定</b>成给定形状后再折叠（见 <see cref="Onnx.OnnxFile.WithInputShapes"/>）。
    /// <para>
    /// 用于全动态导出的模型（PaddleOCR 的 <c>det[N,3,H,W]</c> / <c>rec[N,3,48,W]</c>）：
    /// 折叠依赖静态维，不给尺寸就只能得到「Reshape 目标形状未知」的失败。
    /// </para>
    /// </summary>
    /// <param name="path">ONNX 文件路径。</param>
    /// <param name="inputShapes">图输入名 → 完整形状（每维 &gt; 0）；null 表示按模型声明（可含动态维）。</param>
    public static Net ReadModel(string path, IReadOnlyDictionary<string, int[]>? inputShapes)
    {
        Onnx.OnnxFile file = Onnx.OnnxFile.Load(path);
        if (inputShapes is { Count: > 0 })
            file = file.WithInputShapes(inputShapes);

        return Build(file);
    }

    /// <summary>
    /// 单输入模型的便捷重载：只给一个形状，输入名自动取（模型多于一个输入时会抛错，请用字典重载）。
    /// </summary>
    public static Net ReadModel(string path, int[] inputShape)
    {
        ArgumentNullException.ThrowIfNull(inputShape);

        Onnx.OnnxFile probe = Onnx.OnnxFile.Load(path);
        return ReadModel(path, inputShape, probe);
    }

    /// <summary>
    /// 从<b>内存</b>里的 ONNX 加载（模型包的嵌入资源可以直接喂进来），并把动态输入维固定成给定形状。
    /// </summary>
    public static Net ReadModel(ReadOnlyMemory<byte> model, IReadOnlyDictionary<string, int[]>? inputShapes,
        string baseDirectory = ".")
    {
        Onnx.OnnxFile file = Onnx.OnnxFile.Load(model, baseDirectory);
        if (inputShapes is { Count: > 0 })
            file = file.WithInputShapes(inputShapes);

        return Build(file);
    }

    /// <summary>单输入内存模型的便捷重载。</summary>
    public static Net ReadModel(ReadOnlyMemory<byte> model, int[] inputShape, string baseDirectory = ".")
    {
        ArgumentNullException.ThrowIfNull(inputShape);

        Onnx.OnnxFile probe = Onnx.OnnxFile.Load(model, baseDirectory);
        return Build(Specialize(probe, inputShape));
    }

    private static Net ReadModel(string path, int[] inputShape, Onnx.OnnxFile probe)
    {
        if (inputShape.Length == 0)
            return ReadModel(path);

        // 单输入模型：名字直接取；多动态输入的模型必须用字典重载。
        int dynamicInputs = 0;
        string? name = null;
        foreach (var info in probe.Inputs)
        {
            if (!info.HasDynamicDims)
                continue;
            dynamicInputs++;
            name = info.Name;
        }

        if (dynamicInputs > 1)
            throw new OnnxSharpException(
                $"model has {dynamicInputs} dynamic inputs (must use Net.ReadModel(path, Dictionary<string,int[]>)); " +
                $"got shape [{string.Join(",", inputShape)}]");

        // 全部输入都静态时也允许显式覆盖（例如换 batch / 换分辨率），此时取第一个输入。
        if (name is null)
        {
            if (probe.Inputs.Count != 1)
                throw new OnnxSharpException(
                    $"model has {probe.Inputs.Count} static inputs; use Net.ReadModel(path, Dictionary<string,int[]>)");
            name = probe.Inputs[0].Name;
        }

        return ReadModel(path, new Dictionary<string, int[]>(StringComparer.Ordinal) { [name] = inputShape });
    }

    /// <summary>单输入模型：把 <paramref name="inputShape"/> 绑到唯一的（或第一个）图输入上。</summary>
    private static Onnx.OnnxFile Specialize(Onnx.OnnxFile file, int[] inputShape)
    {
        if (inputShape.Length == 0)
            return file;

        int dynamicInputs = 0;
        string? name = null;
        foreach (var info in file.Inputs)
        {
            if (!info.HasDynamicDims)
                continue;
            dynamicInputs++;
            name = info.Name;
        }

        if (dynamicInputs > 1)
            throw new OnnxSharpException(
                $"model has {dynamicInputs} dynamic inputs; pass a dictionary of input shapes");

        if (name is null)
        {
            if (file.Inputs.Count != 1)
                throw new OnnxSharpException(
                    $"model has {file.Inputs.Count} static inputs; pass a dictionary of input shapes");
            name = file.Inputs[0].Name;
        }

        var shapes = new Dictionary<string, int[]>(StringComparer.Ordinal) { [name] = inputShape };
        return file.WithInputShapes(shapes);
    }

    private static Net Build(Onnx.OnnxFile file)
    {
        var impl = Onnx.OnnxImporter.Build(file);

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

    /// <summary>调试版前向：逐算子检查 NaN，命中即抛（见 <see cref="NetImpl.ForwardAll(bool)"/>）。</summary>
    public List<Tensor> ForwardAll(bool detectNaN) => _impl.ForwardAll(detectNaN);

    /// <summary>
    /// 逐算子耗时统计（对应 OpenCV <c>Net::getPerfProfile</c>），按耗时降序返回。
    /// <para>
    /// <paramref name="warmup"/> 次迭代只跑不计：分层编译下前几次是 tier-0 代码，
    /// 而 tier-0 的 MatMul 比 tier-1 慢约 4 倍（逐元素只慢 2 倍左右），
    /// 不预热会让**计算密集算子的份额被系统性高估**——不加预热量的 rfdetr-small 得到
    /// 「MatMul 占 66%」，预热后是 60%，而定点微模型显示内核本身已接近峰值。
    /// </para>
    /// <para>
    /// 怎么和 <c>bench</c> 对账：**各算子 <see cref="OpTiming.Milliseconds"/> 之和（均值口径）**
    /// 才等于端到端中位数（rfdetr-small：251 vs 250~275 ms）。
    /// 各算子 <see cref="OpTiming.MinMilliseconds"/> 之和只是**下界**（215 ms），不能拿去和中位数比 ——
    /// 每个算子的最小值基本不会同时出现。
    /// </para>
    /// </summary>
    public IReadOnlyList<OpTiming> ProfileAll(int iters = 2, int warmup = 0)
    {
        if (_impl.MainGraph is not { } graph)
            throw new OnnxSharpException("model is not loaded");

        iters = Math.Max(1, iters);

        for (int i = 0; i < Math.Max(0, warmup); i++)
            Interpreter.Run(_impl, graph, executeLayers: true);

        var sum = new Dictionary<LayerInfo, (double Milliseconds, int Calls)>();
        var min = new Dictionary<LayerInfo, double>();

        // 每轮单独统计再合并：均值会被后台负载的重尾放大（测量机上均值比中位数高 25%），
        // 而上百个算子合起来看时只有「每算子的最小耗时」能和 bench 的 min 对上。
        for (int i = 0; i < iters; i++)
        {
            var once = new Dictionary<LayerInfo, (double Milliseconds, int Calls)>();
            Interpreter.Run(_impl, graph, executeLayers: true, once);

            foreach (var (layer, value) in once)
            {
                var acc = sum.GetValueOrDefault(layer);
                sum[layer] = (acc.Milliseconds + value.Milliseconds, acc.Calls + value.Calls);

                if (!min.TryGetValue(layer, out double best) || value.Milliseconds < best)
                    min[layer] = value.Milliseconds;
            }
        }

        var result = new List<OpTiming>(sum.Count);
        foreach (var (layer, value) in sum)
            result.Add(new OpTiming(layer.Name, layer.Type, value.Milliseconds, value.Calls, min[layer]));

        // 按最小耗时排序：均值的排序会被偶发的慢迭代打乱
        result.Sort((a, b) => b.MinMilliseconds.CompareTo(a.MinMilliseconds));
        return result;
    }
}

/// <summary>单个算子的累计耗时。<paramref name="MinMilliseconds"/> 是单次调用的最小值。</summary>
public sealed record OpTiming(
    string Name,
    string Type,
    double Milliseconds,
    int Calls,
    double MinMilliseconds)
{
    public double Average => Calls > 0 ? Milliseconds / Calls : 0;
}
