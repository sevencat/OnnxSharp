// 段层（P1-8）：把「一段连续算子」物化成一个普通 `Layer`。
//
// 与 P1-7 的 `VkPartitionLayer` 的分工：
//   * `VkPartitionLayer` 是**手工**建段（槽位、dispatch、push constant 都要调用方写）；
//   * `VkGraphLayer` 是**按图**建段：吃 `VkSegmentPlan`，自己做段内形状推断，
//     `Finalize` 时按推断结果开槽位、叫各个 `IVkOpEmitter` 追加 dispatch。
//
// 为什么形状推断必须推迟到这里：`Interpreter` 的顺序是「先 GetMemoryShapes → 再 Finalize」，
// 而图改写发生在加载期——那时候中间张量形状还未知。所以段层在自己的 `GetMemoryShapes`
// 里把子算子按拓扑序推一遍，`Finalize` 才拿得到完整的槽位表。
//
// 这也是「`Interpreter` 一行都不用改」的代价：形状推断要重做一遍，
// 但只做一次（静态形状），且段内中间结果永不回 host。
using OnnxSharp.Core;
using OnnxSharp.Tensors;

namespace OnnxSharp.Vulkan;

internal sealed class VkGraphLayer : Layer, IDisposable
{
    private readonly VkDevice _dev;
    private readonly IReadOnlyList<LayerInfo> _ops;
    private readonly VkWeightArena _weights;
    private readonly Dictionary<int, VkArgInfo> _args = [];

    private VkPartitionLayer? _lower;
    private bool _disposed;

    public VkGraphLayer(VkDevice dev, string name, VkSegmentPlan plan, VkWeightArena weights)
    {
        ArgumentNullException.ThrowIfNull(dev);
        ArgumentNullException.ThrowIfNull(plan);
        ArgumentNullException.ThrowIfNull(weights);

        _dev = dev;
        _ops = plan.Ops;
        _weights = weights;
        Name = name;
        Type = "VkSegment";
        Inputs = [.. plan.LiveIn];
        Outputs = [.. plan.LiveOut];
    }

    /// <summary>段内算子（诊断用：拼接后原来的算子都被这一个层替掉了）。</summary>
    public IReadOnlyList<LayerInfo> Ops => _ops;

    /// <summary>真正执行的下层（`Finalize` 之后才存在）。</summary>
    public VkPartitionLayer? Lower => _lower;

    /// <summary>段内 arg 的推断结果（诊断/测试用）。</summary>
    public IReadOnlyDictionary<int, VkArgInfo> ArgInfo => _args;

    // ---------------------------------------------------------------- 形状推断

    /// <summary>
    /// 段内按拓扑序逐算子推断，返回 live-out 的形状。
    /// 输入的 DType 这里还拿不到（`GetMemoryShapes` 只给形状），先按 F32 记，`GetTypes` 再修正。
    /// </summary>
    public override void GetMemoryShapes(
        IReadOnlyList<TensorShape> inShapes,
        int requiredOutputs,
        List<TensorShape> outShapes,
        List<TensorShape> tempShapes)
    {
        if (inShapes.Count != Inputs.Count)
            throw new OnnxSharpException(
                $"段 \"{Name}\" 声明了 {Inputs.Count} 个输入，实际 {inShapes.Count} 个");
        if (requiredOutputs != Outputs.Count)
            throw new OnnxSharpException(
                $"段 \"{Name}\" 声明了 {Outputs.Count} 个输出，图要求 {requiredOutputs} 个");

        _args.Clear();
        for (int i = 0; i < Inputs.Count; i++)
            _args[Inputs[i].Idx] = new VkArgInfo(inShapes[i], DType.F32);

        // 常量输入不在 live-in 里（由发射器绑权重），但段内形状推断照样要读它的形状。
        // 初值/权重在加载期就带静态形状，这里直接从网络的 arg 表拿。
        foreach (LayerInfo op in _ops)
        {
            foreach (Arg a in op.Inputs)
            {
                if (a.IsEmpty || _args.ContainsKey(a.Idx) || op.Net is null) continue;
                ArgData data = op.Net.GetArgData(a);
                if (data.HasShape) _args[a.Idx] = new VkArgInfo(data.Shape, data.DType);
            }
        }

        var subIn = new List<TensorShape>();
        var subOut = new List<TensorShape>();
        var subTemps = new List<TensorShape>();

        foreach (LayerInfo op in _ops)
        {
            subIn.Clear();
            foreach (Arg a in op.Inputs)
            {
                if (a.IsEmpty) { subIn.Add(default); continue; }
                if (!_args.TryGetValue(a.Idx, out VkArgInfo info))
                    throw new OnnxSharpException(
                        $"段内算子 '{op.Name}' 的输入 arg#{a.Idx} 既不在 live-in 也不是有静态形状的常量");
                subIn.Add(info.Shape);
            }

            subOut.Clear();
            subTemps.Clear();
            op.GetMemoryShapes(subIn, op.Outputs.Count, subOut, subTemps);

            if (subOut.Count != op.Outputs.Count)
                throw new OnnxSharpException(
                    $"段内算子 '{op.Name}' 推断出 {subOut.Count} 个输出，实际有 {op.Outputs.Count} 个");

            for (int i = 0; i < op.Outputs.Count; i++)
            {
                Arg a = op.Outputs[i];
                if (a.IsEmpty) continue;
                _args[a.Idx] = new VkArgInfo(subOut[i], DType.F32);
            }
        }

        outShapes.Clear();
        foreach (Arg a in Outputs) outShapes.Add(_args[a.Idx].Shape);
        tempShapes.Clear(); // 段内中间结果全在显存
    }

    /// <summary>段内按拓扑序补上类型，返回 live-out 的类型。</summary>
    public override void GetTypes(
        IReadOnlyList<DType> inTypes,
        int requiredOutputs,
        int requiredTemps,
        List<DType> outTypes,
        List<DType> tempTypes)
    {
        if (_args.Count > 0)
        {
            for (int i = 0; i < Inputs.Count && i < inTypes.Count; i++)
            {
                Arg a = Inputs[i];
                if (_args.TryGetValue(a.Idx, out VkArgInfo info))
                    _args[a.Idx] = info with { Type = inTypes[i] };
            }

            var subIn = new List<DType>();
            var subOut = new List<DType>();
            var subTemps = new List<DType>();

            foreach (LayerInfo op in _ops)
            {
                subIn.Clear();
                foreach (Arg a in op.Inputs)
                    subIn.Add(!a.IsEmpty && _args.TryGetValue(a.Idx, out VkArgInfo info) ? info.Type : DType.F32);

                subOut.Clear();
                subTemps.Clear();
                op.GetTypes(subIn, op.Outputs.Count, 0, subOut, subTemps);

                for (int i = 0; i < op.Outputs.Count && i < subOut.Count; i++)
                {
                    Arg a = op.Outputs[i];
                    if (a.IsEmpty) continue;
                    if (_args.TryGetValue(a.Idx, out VkArgInfo info))
                        _args[a.Idx] = info with { Type = subOut[i] };
                }
            }
        }

        outTypes.Clear();
        foreach (Arg a in Outputs)
            outTypes.Add(_args.TryGetValue(a.Idx, out VkArgInfo info) ? info.Type : DType.F32);
        tempTypes.Clear();
    }

    public override bool DynamicOutputShapes() => false;

    // ---------------------------------------------------------------- 建段

    /// <summary>
    /// 形状就绪后一次性建段：按推断结果开槽位 → 叫发射器追加 dispatch → Seal。
    /// 权重是模型级的（已 Commit），这里只取切片，不重传。
    /// </summary>
    public override void Finalize(IReadOnlyList<Tensor> inputs, IReadOnlyList<Tensor> outputs)
    {
        ThrowIfDisposed();
        if (_lower is not null) return;

        var lower = new VkPartitionLayer(_dev, Name, _weights);
        try
        {
            var slotOf = new Dictionary<int, int>();

            foreach (Arg a in Inputs)
                slotOf[a.Idx] = lower.AddSlot(_args[a.Idx].Shape, _args[a.Idx].Type);

            foreach (LayerInfo op in _ops)
            {
                foreach (Arg a in op.Outputs)
                {
                    if (a.IsEmpty || slotOf.ContainsKey(a.Idx)) continue;
                    if (!_args.TryGetValue(a.Idx, out VkArgInfo info))
                        throw new OnnxSharpException($"段内 arg#{a.Idx} 没有推断出形状");
                    slotOf[a.Idx] = lower.AddSlot(info.Shape, info.Type);
                }
            }

            foreach (LayerInfo op in _ops)
                VkOpRegistry.Get(op).Emit(new VkEmitContext(lower, op, slotOf, _args));

            var inSlots = new int[Inputs.Count];
            for (int i = 0; i < Inputs.Count; i++) inSlots[i] = slotOf[Inputs[i].Idx];
            var outSlots = new int[Outputs.Count];
            for (int i = 0; i < Outputs.Count; i++) outSlots[i] = slotOf[Outputs[i].Idx];

            lower.SetInputs(inSlots);
            lower.SetOutputs(outSlots);
            lower.Seal();
            _lower = lower;
        }
        catch
        {
            lower.Dispose();
            throw;
        }

        if (inputs.Count != Inputs.Count || outputs.Count != Outputs.Count)
            throw new OnnxSharpException(
                $"段 \"{Name}\" 的 I/O 数量与声明不一致（{inputs.Count}/{Inputs.Count}, {outputs.Count}/{Outputs.Count}）");
    }

    public override void Forward(
        IReadOnlyList<Tensor> inputs,
        IReadOnlyList<Tensor> outputs,
        IReadOnlyList<Tensor> temps)
    {
        ThrowIfDisposed();
        if (_lower is null)
            throw new OnnxSharpException($"段 \"{Name}\" 还没 Finalize 就被执行");
        _lower.Forward(inputs, outputs, temps);
    }

    public override string ToString() =>
        $"{Type}({Name}: ops={_ops.Count}, in={Inputs.Count}, out={Outputs.Count})";

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(nameof(VkGraphLayer));
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _lower?.Dispose();
        _lower = null;
    }
}
