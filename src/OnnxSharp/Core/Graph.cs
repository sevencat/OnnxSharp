namespace OnnxSharp.Core;

/// <summary>
/// 一张计算图：输入 Arg、按拓扑序排列的算子表、输出 Arg。
/// 对应 OpenCV 新引擎的 <c>Graph</c>（<c>prog</c> 线性表）。
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

    public void Append(LayerInfo op)
    {
        if (op.Outputs.Count == 0)
            throw new OnnxSharpException($"op '{op.Name}' ({op.Type}) has no outputs");
        _prog.Add(op);
    }

    public void SetOutputs(IEnumerable<Arg> outputs)
    {
        Outputs = outputs.ToList();
        if (Outputs.Count == 0)
            throw new OnnxSharpException($"graph '{Name}' has no outputs");
    }

    public override string ToString() => $"Graph({Name}, {_prog.Count} ops)";
}
