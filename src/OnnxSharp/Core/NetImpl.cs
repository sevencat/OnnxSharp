using OnnxSharp.Tensors;

namespace OnnxSharp.Core;

/// <summary>
/// 推理引擎内核：SSA 值表 + 图管理 + 输入输出。
/// 对应 OpenCV 新引擎的 <c>Net::Impl</c>。
/// </summary>
public sealed class NetImpl
{
    private readonly List<ArgData> _args = new() { new ArgData() }; // 0 号保留为“空参数”
    private readonly Dictionary<string, int> _names = new(StringComparer.Ordinal);
    private readonly List<Graph> _graphs = new();

    public NetImpl()
    {
        LayerRegistry.EnsureBuiltins();
    }

    public Graph? MainGraph { get; private set; }

    public IReadOnlyList<Graph> Graphs => _graphs;

    public int ArgCount => _args.Count;

    public Graph NewGraph(string name, IEnumerable<Arg> inputs, bool isMain)
    {
        var graph = new Graph(name, inputs.ToList());
        _graphs.Add(graph);
        if (isMain)
            MainGraph = graph;
        return graph;
    }

    // ---------------- Arg 表 ----------------

    public Arg NewArg(string name, ArgKind kind)
    {
        if (!string.IsNullOrEmpty(name) && _names.ContainsKey(name))
            throw new OnnxSharpException($"arg '{name}' already exists");

        int idx = _args.Count;
        _args.Add(new ArgData { Name = name, Kind = kind });
        if (!string.IsNullOrEmpty(name))
            _names[name] = idx;
        return new Arg(idx);
    }

    /// <summary>按名字取 Arg；不存在则创建一个空名的临时 Arg。</summary>
    public Arg GetArg(string name) =>
        _names.TryGetValue(name, out var idx) ? new Arg(idx) : NewArg(name, ArgKind.Temp);

    public bool HaveArg(string name) => _names.ContainsKey(name);

    public Arg NewConstArg(string name, Tensor value)
    {
        var arg = NewArg(name, ArgKind.Const);
        var data = GetArgData(arg);
        data.Const = value;
        data.Shape = value.Shape;
        data.DType = value.DType;
        return arg;
    }

    public ArgData GetArgData(Arg arg)
    {
        CheckArg(arg);
        return _args[arg.Idx];
    }

    public ArgKind GetArgKind(Arg arg) => GetArgData(arg).Kind;

    public string GetArgName(Arg arg) => GetArgData(arg).Name;

    private void CheckArg(Arg arg)
    {
        if ((uint)arg.Idx >= (uint)_args.Count)
            throw new OnnxSharpException($"arg index {arg.Idx} is out of range");
    }

    // ---------------- 张量访问 ----------------

    public Tensor GetArgTensor(Arg arg)
    {
        var data = GetArgData(arg);
        if (data.Kind == ArgKind.Const)
            return data.Const ?? throw new OnnxSharpException($"const arg '{data.Name}' has no tensor");

        data.Runtime ??= Tensor.Allocate(data.Shape, data.DType);
        return data.Runtime;
    }

    /// <summary>按推断出的形状/类型绑定（并复用）运行期缓冲。</summary>
    public Tensor BindArgTensor(Arg arg, TensorShape shape, DType dtype)
    {
        var data = GetArgData(arg);
        data.Shape = shape;
        data.DType = dtype;
        data.Runtime ??= Tensor.Allocate(shape, dtype);
        data.Runtime.Fit(shape, dtype);
        return data.Runtime;
    }

    // ---------------- 输入 ----------------

    /// <summary>设置模型输入。张量由调用方持有，引擎直接引用（不拷贝）。</summary>
    public void SetInput(string name, Tensor input)
    {
        if (!_names.TryGetValue(name, out var idx))
            throw new OnnxSharpException($"input '{name}' is not found");

        var data = _args[idx];
        if (data.Kind == ArgKind.Const)
            throw new OnnxSharpException($"arg '{name}' is a constant, cannot be used as input");

        data.Kind = ArgKind.Input;
        data.Shape = input.Shape;
        data.DType = input.DType;
        data.Runtime = input;
    }

    public IReadOnlyList<string> GetInputNames() =>
        _args.Where(a => a.Kind == ArgKind.Input).Select(a => a.Name).ToList();

    public IReadOnlyList<string> GetOutputNames() =>
        MainGraph?.Outputs.Select(GetArgName).ToList() ?? new List<string>();

    // ---------------- 推理 ----------------

    public Tensor Forward()
    {
        if (MainGraph is null)
            throw new OnnxSharpException("model is not loaded");

        Interpreter.Run(this, MainGraph);

        if (MainGraph.Outputs.Count != 1)
            throw new OnnxSharpException($"graph has {MainGraph.Outputs.Count} outputs; use Forward(outputName)");

        return GetArgTensor(MainGraph.Outputs[0]);
    }

    public Tensor Forward(string outputName)
    {
        if (MainGraph is null)
            throw new OnnxSharpException("model is not loaded");
        if (!_names.TryGetValue(outputName, out var idx))
            throw new OnnxSharpException($"output '{outputName}' is not found");

        Interpreter.Run(this, MainGraph);
        return GetArgTensor(new Arg(idx));
    }

    public List<Tensor> ForwardAll()
    {
        if (MainGraph is null)
            throw new OnnxSharpException("model is not loaded");

        Interpreter.Run(this, MainGraph);
        return MainGraph.Outputs.Select(GetArgTensor).ToList();
    }
}
