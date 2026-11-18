// 分区集合（P1-8）：一次规划 = 一个模型级上下文。
//
// 职责有三件，顺序不能换：
//   1. 用 <see cref="VkPartitionPlanner"/> 选出连续可发射的段；
//   2. 把段内所有 blob / 常量输入登记进**模型级**共享权重区，统一上传一次
//      （权重属于模型，不属于段：换 shape 重建段时不该重传 10 MB）；
//   3. 建 <see cref="VkGraphLayer"/> 并用 `Graph.Reset` 把原 prog 里的那一段替换成一个层。
//
// `Interpreter` 一行都不用改：段对上层就是一个普通 `Layer`。
using OnnxSharp.Core;
using OnnxSharp.Tensors;

namespace OnnxSharp.Vulkan;

internal sealed class VkPartitionSet : IDisposable
{
    private readonly VkDevice _dev;
    private readonly VkWeightArena _weights;
    private readonly List<VkGraphLayer> _layers = [];

    private bool _disposed;

    /// <summary>
    /// 段级成本判据的默认开关（<c>ONNXSHARP_VK_COST=0</c> 关闭）。
    /// 默认**开**：实测 1×1 卷积 @32×32 端到端 415 us，而 CPU 只要 250–380 us，
    /// 不拦就是稳稳的亏。关掉的口子留给「怀疑模型估错」时的 A/B。
    /// </summary>
    public static bool CostGate { get; set; } = ReadCostGateDefault();

    /// <summary>最近一次被成本判据丢掉的理由（探针与诊断用；没丢过则为空）。</summary>
    public string? LastReject { get; private set; }

    /// <summary>
    /// 最近一次 <see cref="Apply"/> 的段表：每段一行，含预测的 gpu/cpu 耗时与字节数。
    /// 判据关闭时为空。真模型调优要靠它对账——「模型说该发」和「实测确实更快」是两件事，
    /// 没有这张表就无法区分判据估错与 kernel 本身慢。
    /// </summary>
    public IReadOnlyList<string> CostReport => _costReport;

    private readonly List<string> _costReport = [];

    private static bool ReadCostGateDefault()
        => Environment.GetEnvironmentVariable("ONNXSHARP_VK_COST") is not "0";

    public VkPartitionSet(VkDevice dev, string name = "model")
    {
        ArgumentNullException.ThrowIfNull(dev);
        ArgumentException.ThrowIfNullOrEmpty(name);

        _dev = dev;
        _weights = new VkWeightArena(dev, name + ".weights");
    }

    /// <summary>模型级权重常驻区（跨段共享，Commit 由本类负责）。</summary>
    public VkWeightArena Weights => _weights;

    /// <summary>已物化的段层。</summary>
    public IReadOnlyList<VkGraphLayer> Layers => _layers;

    /// <summary>
    /// 规划并改写 <paramref name="graph"/>。返回替换出的段数（0 = 没有可发射的连续段，图未改动）。
    /// <para>
    /// 幂等：段层自己的 Type 是 <c>VkSegment</c>，没有注册发射器，重复调用不会再切一次。
    /// </para>
    /// </summary>
    /// <param name="costGate">
    /// 是否用 <see cref="VkCostModel"/> 丢掉「预测比 CPU 还慢」的段。
    /// <c>null</c> = 用 <see cref="CostGate"/> 的全局默认。微基准要测 kernel 本身，
    /// 必须显式传 false——否则小段会被判据挡掉，量到的是「什么都没跑」。
    /// </param>
    public int Apply(NetImpl net, Graph graph, int minOpsPerSegment = -1, bool? costGate = null)
    {
        ArgumentNullException.ThrowIfNull(net);
        ArgumentNullException.ThrowIfNull(graph);
        ThrowIfDisposed();

        _costReport.Clear();
        LastReject = null;

        // 规划前先绑网络：`CanEmit` 要查 dtype（int64/FP16 的逐元素算子不能上 GPU），
        // 而 dtype 在加载期就已知，不必等形状推断。
        foreach (LayerInfo op in graph.Prog) net.Attach(op);

        // 两道门串联：CanEmit 管「能不能」，VkDevicePolicy 管「值不值得」（同一份代码要能跑独显/核显）。
        // 被设备门拒掉的算子等价于「不可发射」，会把段切断——这正是想要的：核显上整段回落 CPU。
        bool CanEmitOn(LayerInfo op) => VkOpRegistry.CanEmit(op) && VkDevicePolicy.Allows(_dev, op, out _);

        // 常量输入由发射器绑权重，不进 stage in
        List<VkSegmentPlan> plans = VkPartitionPlanner.Plan(
            graph, CanEmitOn, minOpsPerSegment, a => net.GetArgKind(a) == ArgKind.Const);

        // 段内生产、段外没人读的段没有任何意义：`Graph.Reset` 也会拒绝「没有输出」的层。
        plans.RemoveAll(static p => p.LiveOut.Count == 0);

        // 第三道门：值不值得（按实测成本模型预测）。被丢掉的段不发射，算子留在 CPU 上。
        // 这一门只在「已知形状」时生效——信息不足时宁可发射，也不做拍脑袋的决定。
        if (costGate ?? CostGate)
        {
            // 倒序遍历，便于就地丢弃并保持段表顺序与拓扑一致
            for (int i = plans.Count - 1; i >= 0; i--)
            {
                VkSegmentPlan plan = plans[i];
                VkCostModel.Estimate(plan, net, out double gpuUs, out double cpuUs,
                                     out double bytes, out double flops);
                // 形状未知（NaN）时放行：判据是为了止损，不是为了在信息不足时下注
                bool keep = double.IsNaN(gpuUs) || gpuUs <= cpuUs;

                _costReport.Insert(0,
                    $"{(keep ? "发" : "弃")} {plan.Count} op @{plan.Start,-3} " +
                    $"gpu={gpuUs,6:F0} cpu={cpuUs,6:F0} " +
                    $"{bytes / 1048576.0,5:F2} MB {flops / 1e6,6:F1} MF");

                if (!keep)
                {
                    LastReject = $"预测 gpu={gpuUs:F0} us > cpu={cpuUs:F0} us" +
                                 $"（{plan.Count} dispatch, {bytes / 1048576.0:F2} MB 往返, {flops / 1e6:F1} MFLOP）";
                    plans.RemoveAt(i);
                }
            }
        }

        if (plans.Count == 0) return 0;

        RegisterWeights(net, plans);

        // 按原拓扑顺序拼接：段外的算子原样保留，段内的整段换成一个层。
        IReadOnlyList<LayerInfo> prog = graph.Prog;
        var next = new List<LayerInfo>(prog.Count);
        int p = 0;

        foreach (VkSegmentPlan plan in plans)
        {
            while (p < plan.Start) next.Add(prog[p++]);

            var layer = new VkGraphLayer(_dev, $"vkgpu{_layers.Count}", plan, _weights);
            _layers.Add(layer);
            next.Add(layer);

            p = plan.Start + plan.Count;
        }

        while (p < prog.Count) next.Add(prog[p++]);

        graph.Reset(next);
        return plans.Count;
    }

    /// <summary>
    /// 登记段内全部权重并统一上传。形状此刻还未知，但权重不需要形状——
    /// 这正是把权重提到模型级的意义：建段可以晚到形状就绪，权重不必跟着重来。
    /// </summary>
    private void RegisterWeights(NetImpl net, List<VkSegmentPlan> plans)
    {
        foreach (VkSegmentPlan plan in plans)
        {
            foreach (LayerInfo op in plan.Ops)
            {
                // 段内算子不是 prog 的成员，PrepareForGraph 不会给它绑 Net；
                // 但 Reshape/Slice 这类算子的形状推断要读常量值，必须现在就绑。
                net.Attach(op);

                foreach (Tensor blob in op.Blobs) _weights.Add(blob);

                foreach (Arg a in op.Inputs)
                {
                    if (a.IsEmpty || net.GetArgKind(a) != ArgKind.Const) continue;
                    _weights.Add(net.GetArgTensor(a));
                }
            }
        }

        _weights.Commit();
    }

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(nameof(VkPartitionSet));
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        foreach (VkGraphLayer layer in _layers) layer.Dispose();
        _layers.Clear();
        _weights.Dispose();
    }
}
