// 算子发射器注册表（P1-8 的接缝）。
//
// 分区规划只回答「哪段连续算子该上 GPU」；把一段真正编成 dispatch 列表需要每个算子
// 自己的形状规则、binding 约定、push constant 布局——那些属于 P2 的算子实现。
//
// 所以这里只定接口：P2 每实现一个 GPU 算子就 `Register("Conv", ...)`，
// 规划器立刻能把含它的连续段挑出来，不需要改规划逻辑一行。
// 注册表为空时 <see cref="CanEmit"/> 恒 false，规划结果为空——这正是 P1-8 阶段的预期状态，
// 探针会临时注册假算子来验证边界划分与整条拼接链。
using OnnxSharp.Core;
using OnnxSharp.Tensors;

namespace OnnxSharp.Vulkan;

/// <summary>段内一个 arg 的静态形状/类型（段内推断出来的，运行期不再重算）。</summary>
internal readonly record struct VkArgInfo(TensorShape Shape, DType Type);

/// <summary>把一个 <see cref="LayerInfo"/> 编进段的钩子（P2 的算子实现填这里）。</summary>
internal interface IVkOpEmitter
{
    /// <summary>能否把该算子编进段（形状静态、数据类型受支持等）。</summary>
    bool CanEmit(LayerInfo op);

    /// <summary>追加该算子的 dispatch。槽位与权重切片已由宿主准备好。</summary>
    void Emit(VkEmitContext ctx);

    /// <summary>
    /// 发射器是否自己负责补上 <see cref="LayerInfo.FusedActivationKind"/>。
    /// <para>
    /// 默认 false：规划期会拒绝这类算子。图融合后算子的输出 Arg 已经是激活后的那个，
    /// 少做一次激活既不报错、形状也全对，只会静默算出错的数（yolo11n 上逐层放大到 err=25）。
    /// 「不说自己能处理，就别让它上 GPU」是这里唯一能防住这类错误的地方。
    /// </para>
    /// </summary>
    bool HandlesFusedActivation => false;
}

/// <summary>发射一个算子时的只读上下文：算子本身，加上「arg → 槽位 / 权重切片 / 形状」的映射。</summary>
internal sealed class VkEmitContext
{
    private readonly Dictionary<int, int> _slotOf;
    private readonly IReadOnlyDictionary<int, VkArgInfo> _args;

    public VkEmitContext(VkPartitionLayer layer, LayerInfo op,
        Dictionary<int, int> slotOf, IReadOnlyDictionary<int, VkArgInfo> args)
    {
        Layer = layer;
        Op = op;
        _slotOf = slotOf;
        _args = args;
    }

    public VkPartitionLayer Layer { get; }

    public LayerInfo Op { get; }

    /// <summary>该 arg 在段 arena 里的槽位 id。未登记槽位会抛——漏一个槽位就是整段算错。</summary>
    public int SlotOf(Arg arg)
    {
        if (!_slotOf.TryGetValue(arg.Idx, out int slot))
            throw new OnnxSharpException($"op '{Op.Name}' 的 arg#{arg.Idx} 没有对应的段内槽位");
        return slot;
    }

    /// <summary>
    /// 输出槽位的绑定：与 <see cref="SlotOf"/> 相同，但标成「会被写」。
    /// <para>
    /// 屏障判定靠这个标记区分读写方向（读-读不插屏障）。漏标不会算错，只会退回保守判定；
    /// 标错（把只读标成写）同样只是多插屏障。真正危险的是反过来——所以输出一律走这里。
    /// </para>
    /// </summary>
    public VkBind Out(Arg arg) => VkBind.Local(SlotOf(arg)).AsWrite();

    public VkArgInfo InfoOf(Arg arg)
    {
        if (!_args.TryGetValue(arg.Idx, out VkArgInfo info))
            throw new OnnxSharpException($"op '{Op.Name}' 的 arg#{arg.Idx} 没有推断出形状");
        return info;
    }

    public TensorShape ShapeOf(Arg arg) => InfoOf(arg).Shape;

    public DType TypeOf(Arg arg) => InfoOf(arg).Type;

    /// <summary>元素数（绝大多数 dispatch 的 workgroup 规模都由它决定）。</summary>
    public ulong ElementsOf(Arg arg) => (ulong)ShapeOf(arg).Total();

    /// <summary>把算子第 <paramref name="index"/> 个 blob 绑成只读权重切片。</summary>
    public VkBind WeightOf(int index)
    {
        if ((uint)index >= (uint)Op.Blobs.Count)
            throw new OnnxSharpException($"op '{Op.Name}' 没有第 {index} 个 blob（共 {Op.Blobs.Count} 个）");
        return Layer.Weights.BindOf(Op.Blobs[index]);
    }

    /// <summary>把常量输入（initializer 作为 arg 传进来）绑成只读权重切片。</summary>
    public VkBind WeightOf(Arg constArg)
    {
        if (Op.Net is null || Op.Net.GetArgKind(constArg) != ArgKind.Const)
            throw new OnnxSharpException($"op '{Op.Name}' 的 arg#{constArg.Idx} 不是常量");
        return Layer.Weights.BindOf(Op.Net.GetArgTensor(constArg));
    }
}

internal static class VkOpRegistry
{
    private static readonly Dictionary<string, IVkOpEmitter> ByType = new(StringComparer.Ordinal);

    /// <summary>注册（或替换）某类算子的发射器。线程安全由调用方保证——只在建段阶段用。</summary>
    public static void Register(string type, IVkOpEmitter emitter)
    {
        ArgumentException.ThrowIfNullOrEmpty(type);
        ArgumentNullException.ThrowIfNull(emitter);
        ByType[type] = emitter;
    }

    public static bool Unregister(string type) => ByType.Remove(type);

    public static void Clear() => ByType.Clear();

    public static int Count => ByType.Count;

    /// <summary>
    /// 注册后端内置的全部发射器。P2 起每个新算子在这里加一行，规划器立刻能选它。
    /// 不调用 = 注册表为空 = 一个段都不发射（P1 阶段的默认状态）。
    /// <para>
    /// 「注册」不等于「一定发射」：设备相关的取舍在 <see cref="VkDevicePolicy"/> 里，
    /// 例如核显上的 Conv 会被它拦下。所以连核显一起跑也不必再靠「不注册」来兜底。
    /// </para>
    /// </summary>
    public static void RegisterBuiltins()
    {
        VkElementwiseEmitter.Register();
        VkConvEmitter.Register();
        // Concat/Split 本身几乎不耗算力，但它们是把段连起来的关键：yolo11n 的 162 个算子里
        // 有 34 个是这两类，它们一旦不能发射，整图就被切成 21 段、每段都要往返一次 host。
        VkConcatEmitter.Register();
        VkSplitEmitter.Register();
        // Reshape 一族本身零算力，但每漏一个就是一次段切割（yolo11n 里有 6 个）。
        VkReshapeEmitter.Register();
        // SPPF 的三个 5×5 MaxPool 夹在大段中间：算力可忽略，切断段的代价却不小。
        VkMaxPoolEmitter.Register();
        // neck 的两次 2× 上采样同理，卡在两路分支之间。
        VkResizeEmitter.Register();
        // 下面三个都在检测头的输出分支上，彼此紧挨着：Sigmoid → DflSoftmax → Slice ×2。
        // 它们单个都只值零点几毫秒，但少发一个就多一次段切割（每段约 3 ms 的 host 往返地板），
        // 所以这里的目标不是算力，是**别把段切断**。
        VkUnaryEmitter.Register();
        VkDflEmitter.Register();
        VkSliceEmitter.Register();
        // C2PSA 的 attention：这三个一起发才能把 vkgpu0/vkgpu1/vkgpu2 连成一段。
        // 只做一个没有意义——链条上任何一环留在 CPU，段照样被切断。
        //
        // ONNXSHARP_VK_NOEMIT=transpose,matmul,softmax 可以单独摘掉某一个：
        // 新 kernel 算错时，端到端 err 只会告诉你「错了」，不会告诉你「谁错了」，
        // 而逐个禁用 + 重跑 err 是最快的二分。
        string[] disabled = (Environment.GetEnvironmentVariable("ONNXSHARP_VK_NOEMIT") ?? "")
            .Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        if (!disabled.Contains("transpose")) VkTransposeEmitter.Register();
        if (!disabled.Contains("softmax")) VkSoftmaxEmitter.Register();
        if (!disabled.Contains("matmul")) VkMatMulEmitter.Register();
    }

    /// <summary>
    /// 能否发射。注册过还不够——必须由发射器自己判断（dtype、形状规则、属性边界）。
    /// 这里若只查注册表，int64 的 Add 也会被规划进段，然后在建段时炸掉整个模型。
    /// </summary>
    public static bool CanEmit(LayerInfo op)
    {
        ArgumentNullException.ThrowIfNull(op);
        if (string.IsNullOrEmpty(op.Type) || !ByType.TryGetValue(op.Type, out IVkOpEmitter? emitter))
            return false;

        // 统一防线：带融合激活、而发射器又没声明自己会补的，一律不发射。
        // 这一条不写在各个发射器里，是因为漏写的后果（静默算错）比漏发（慢）严重得多。
        if (op.FusedActivationKind is not null && !emitter.HandlesFusedActivation)
            return false;

        return emitter.CanEmit(op);
    }

    public static IVkOpEmitter Get(LayerInfo op)
    {
        if (!CanEmit(op))
            throw new OnnxSharpException($"算子类型 '{op.Type}' 没有注册 GPU 发射器");
        return ByType[op.Type];
    }
}
