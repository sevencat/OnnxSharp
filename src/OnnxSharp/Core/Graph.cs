namespace OnnxSharp.Core;

/// <summary>
/// 一张计算图：输入 Arg、按拓扑序排列的算子表、输出 Arg。
/// 对应 OpenCV 新引擎的 <c>Graph</c>（<c>prog</c> 线性表）。
/// <para>
/// <see cref="Version"/> 在每次结构变更时递增，<see cref="NetImpl"/> 用它判断
/// 缓冲槽分配是否失效（常量折叠删节点、追加算子都会触发重算）。
/// </para>
/// </summary>
public sealed class Graph
{
    private readonly List<LayerInfo> _prog = new();

    public Graph(string name, IReadOnlyList<Arg> inputs)
    {
        Name = name;
        Inputs = inputs.ToList();
    }

    public string Name { get; }

    public List<Arg> Inputs { get; }

    public List<Arg> Outputs { get; private set; } = new();

    public IReadOnlyList<LayerInfo> Prog => _prog;

    /// <summary>结构版本号，任何增删改都会递增。</summary>
    public int Version { get; private set; }

    public void Append(LayerInfo op)
    {
        ArgumentNullException.ThrowIfNull(op);
        if (op.Outputs.Count == 0)
            throw new OnnxSharpException($"op '{op.Name}' ({op.Type}) has no outputs");
        _prog.Add(op);
        Version++;
    }

    /// <summary>
    /// 用新序列整体替换 <c>prog</c>。常量折叠后调用（折叠会删掉节点、把上游改成常量 Arg）。
    /// 对应 OpenCV <c>prepareForInference()</c> 之后的 <c>updateGraphOfs</c>。
    /// </summary>
    public void Reset(IEnumerable<LayerInfo> ops)
    {
        ArgumentNullException.ThrowIfNull(ops);
        _prog.Clear();
        foreach (var op in ops)
        {
            if (op.Outputs.Count == 0)
                throw new OnnxSharpException($"op '{op.Name}' ({op.Type}) has no outputs");
            _prog.Add(op);
        }
        Version++;
    }

    /// <summary>删除满足条件的算子（返回删除个数）。用于死节点消除。</summary>
    public int RemoveAll(Predicate<LayerInfo> match)
    {
        ArgumentNullException.ThrowIfNull(match);
        int removed = _prog.RemoveAll(match);
        if (removed > 0)
            Version++;
        return removed;
    }

    public void SetOutputs(IEnumerable<Arg> outputs)
    {
        Outputs = outputs.ToList();
        if (Outputs.Count == 0)
            throw new OnnxSharpException($"graph '{Name}' has no outputs");
        Version++;
    }

    public override string ToString() => $"Graph({Name}, {_prog.Count} ops)";
}
