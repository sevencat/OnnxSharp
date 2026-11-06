using OnnxSharp.Engine;
using OnnxSharp.Tensors;

namespace OnnxSharp.Core;

/// <summary>
/// 推理引擎内核：SSA 值表 + 图管理 + 运行期缓冲池。
/// 对应 OpenCV 新引擎的 <c>Net::Impl</c>。
/// </summary>
public sealed class NetImpl
{
    private readonly List<ArgData> _args = new() { new ArgData() }; // 0 号保留为“空参数”
    private readonly Dictionary<string, int> _names = new(StringComparer.Ordinal);
    private readonly List<Graph> _graphs = new();

    /// <summary>共享缓冲槽（由 <see cref="BufferAllocator"/> 分配，只增不减）。</summary>
    private readonly List<Tensor> _slots = new();

    /// <summary>全局临时张量池（对应 OpenCV 的 <c>globalTemps</c>）。</summary>
    private readonly List<Tensor> _temps = new();

    /// <summary>当前 op 的 temp 视图，复用同一个 List 避免每 op 分配。</summary>
    private readonly List<Tensor> _tempView = new();

    private int _buffersVersion = int.MinValue;

    public NetImpl()
    {
        LayerRegistry.EnsureBuiltins();
    }

    public Graph? MainGraph { get; private set; }

    public IReadOnlyList<Graph> Graphs => _graphs;

    public int ArgCount => _args.Count;

    /// <summary>已分配的共享缓冲槽数量（诊断/测试用）。</summary>
    public int SlotCount => _slots.Count;

    /// <summary>全局临时张量对象数量（诊断/测试用）。</summary>
    public int TempCount => _temps.Count;

    public Graph NewGraph(string name, IEnumerable<Arg> inputs, bool isMain)
    {
        var graph = new Graph(name, inputs.ToList());
        _graphs.Add(graph);
        if (isMain)
            MainGraph = graph;
        return graph;
    }

    // ---------------- Arg 表 ----------------

    /// <summary>
    /// 新建一个 Arg。同名已存在时抛异常（供 importer 使用，拼错名字会立刻暴露）。
    /// </summary>
    public Arg Declare(string name, ArgKind kind)
    {
        if (!string.IsNullOrEmpty(name) && _names.ContainsKey(name))
            throw new OnnxSharpException($"arg '{name}' already exists");

        int idx = _args.Count;
        _args.Add(new ArgData { Name = name, Kind = kind });
        if (!string.IsNullOrEmpty(name))
            _names[name] = idx;
        return new Arg(idx);
    }

    /// <summary>按名字取已存在的 Arg；不存在则抛异常。</summary>
    public Arg Require(string name)
    {
        ArgumentException.ThrowIfNullOrEmpty(name);
        if (!_names.TryGetValue(name, out var idx))
            throw new OnnxSharpException($"arg '{name}' is not found");
        return new Arg(idx);
    }

    /// <summary>
    /// 按名字取 Arg；不存在则创建一个 <see cref="ArgKind.Temp"/>。
    /// <remarks>手工建图方便用；importer 请用 <see cref="Require"/> 以免静默吞掉拼写错误。</remarks>
    /// </summary>
    public Arg GetArg(string name) =>
        _names.TryGetValue(name, out var idx) ? new Arg(idx) : Declare(name, ArgKind.Temp);

    public bool HaveArg(string name) => _names.ContainsKey(name);

    public bool TryGetArg(string name, out Arg arg)
    {
        if (_names.TryGetValue(name, out var idx))
        {
            arg = new Arg(idx);
            return true;
        }
        arg = default;
        return false;
    }

    public Arg NewConstArg(string name, Tensor value)
    {
        var arg = Declare(name, ArgKind.Const);
        var data = GetArgData(arg);
        data.Const = value;
        data.Shape = value.Shape;
        data.DType = value.DType;
        data.Layout = value.Layout;
        return arg;
    }

    public ArgData GetArgData(Arg arg)
    {
        CheckArg(arg);
        return _args[arg.Idx];
    }

    public ArgKind GetArgKind(Arg arg) => GetArgData(arg).Kind;

    public bool IsConstArg(Arg arg) => GetArgData(arg).Kind == ArgKind.Const;

    public string GetArgName(Arg arg) => GetArgData(arg).Name;

    private void CheckArg(Arg arg)
    {
        if ((uint)arg.Idx >= (uint)_args.Count)
            throw new OnnxSharpException($"arg index {arg.Idx} is out of range");
    }

    // ---------------- 缓冲分配 ----------------

    /// <summary>
    /// 绑定 <see cref="LayerInfo.Net"/> 并按当前 <c>prog</c> 分配共享缓冲槽。
    /// 图结构变化（<see cref="Graph.Version"/> 递增）后自动重跑。
    /// </summary>
    internal void PrepareForGraph(Graph graph)
    {
        if (_buffersVersion == graph.Version)
            return;
        _buffersVersion = graph.Version;

        foreach (var op in graph.Prog)
            op.Net = this;

        var slotOf = BufferAllocator.Assign(graph.Prog, graph.Outputs, _args.Count, out int slotCount);

        while (_slots.Count < slotCount)
            _slots.Add(new Tensor());
        // 槽只增不减：多出来的保留，避免下次又分配。

        for (int i = 0; i < _args.Count; i++)
        {
            var data = _args[i];
            if (data.Kind is ArgKind.Input or ArgKind.Const)
            {
                // 图输入持有调用方的张量，常量持有自己的张量，都不进槽。
                data.BufferIndex = BufferAllocator.NoSlot;
                continue;
            }

            int slot = slotOf[i];
            data.BufferIndex = slot;
            data.Runtime = slot >= 0 ? _slots[slot] : null;
        }
    }

    // ---------------- 张量访问 ----------------

    public Tensor GetArgTensor(Arg arg)
    {
        var data = GetArgData(arg);

        if (data.Kind == ArgKind.Const)
            return data.Const ?? throw new OnnxSharpException($"const arg '{data.Name}' has no tensor");

        if ((uint)data.BufferIndex < (uint)_slots.Count)
            return _slots[data.BufferIndex];   // 共享槽：形状由生产者 Fit 时确定

        data.Runtime ??= Tensor.Allocate(data.Shape, data.DType);
        return data.Runtime;
    }

    /// <summary>按推断出的形状/类型绑定（并复用）运行期缓冲。</summary>
    public Tensor BindArgTensor(Arg arg, TensorShape shape, DType dtype)
    {
        var data = GetArgData(arg);
        data.Shape = shape;
        data.DType = dtype;
        data.Layout = Layout.NCHW;

        var tensor = GetArgTensor(arg);
        tensor.Fit(shape, dtype);
        return tensor;
    }

    /// <summary>
    /// 按形状需求从全局临时池取出 temp 张量（对象跨 op、跨 forward 复用）。
    /// 返回的列表本身也被复用，调用方不得持有。
    /// </summary>
    public IReadOnlyList<Tensor> FitTemps(IReadOnlyList<TensorShape> shapes, IReadOnlyList<DType> types)
    {
        while (_temps.Count < shapes.Count)
            _temps.Add(new Tensor());

        _tempView.Clear();
        for (int i = 0; i < shapes.Count; i++)
        {
            var t = _temps[i];
            t.Fit(shapes[i], types[i]);
            _tempView.Add(t);
        }

        return _tempView;
    }

    /// <summary>释放所有池化缓冲（<c>Reset()</c> 语义，与 OpenCV 一致：只增不减，需显式释放）。</summary>
    public void ResetBuffers()
    {
        _slots.Clear();
        _temps.Clear();
        _tempView.Clear();
        _buffersVersion = int.MinValue;
    }

    // ---------------- 输入 ----------------

    /// <summary>设置模型输入。张量由调用方持有，引擎直接引用（不拷贝）。</summary>
    public void SetInput(string name, Tensor input)
    {
        ArgumentNullException.ThrowIfNull(input);
        var arg = Require(name);
        var data = GetArgData(arg);

        if (data.Kind == ArgKind.Const)
            throw new OnnxSharpException($"arg '{name}' is a constant, cannot be used as input");

        data.Kind = ArgKind.Input;
        data.Shape = input.Shape;
        data.DType = input.DType;
        data.Layout = input.Layout;
        data.Runtime = input;
        data.BufferIndex = BufferAllocator.NoSlot;
        _buffersVersion = int.MinValue; // 输入变化后重算一次（输入不参与着色，但保持一致性）
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

        var arg = Require(outputName);
        Interpreter.Run(this, MainGraph);
        return GetArgTensor(arg);
    }

    public List<Tensor> ForwardAll()
    {
        if (MainGraph is null)
            throw new OnnxSharpException("model is not loaded");

        Interpreter.Run(this, MainGraph);
        return MainGraph.Outputs.Select(GetArgTensor).ToList();
    }
}
