// 分区层（P1-7）：主包 `Layer` 与 Vulkan 段之间的唯一接缝。
//
// 为什么要一个 `Layer` 而不是直接暴露 `VkSegment`：计划里的集成形态是
// `VkPartitionLayer : Layer` + `Graph.Reset`，`Interpreter` 一行都不用改。
// 于是段内所有算子对上层就是一个普通算子——入参出参都是 `Tensor`，
// 中间结果（段内 slot）既不进 `BufferAllocator`，也不回 host。
//
// 两类数据在这里分开：
//   * 权重：`VkWeightArena`，整个推理生命周期只上传一次，绑定成外部切片（只读）；
//   * 激活：段 arena 的槽位，每帧从头 stage in / stage out。
//
// `Tensor` 是 `byte[] + ByteOffset`（POH 上的 pinned 数组，见 Tensor.Alignment），
// 所以 stage in 就是一次 memcpy，没有装箱也没有中间缓冲。
using System.Runtime.InteropServices;
using OnnxSharp.Core;
using OnnxSharp.Tensors;

namespace OnnxSharp.Vulkan;

internal sealed unsafe class VkPartitionLayer : Layer, IDisposable
{
    /// <summary>一个段内激活槽位（形状/类型只用于校验，数据在 GPU 上）。</summary>
    private readonly record struct Slot(TensorShape Shape, DType DType, ulong Bytes);

    private readonly VkDevice _dev;
    private readonly VkSegment _seg;
    private readonly VkWeightArena _weights;
    private readonly bool _ownsWeights;
    private readonly List<Slot> _slots = [];

    private int[] _inSlots = [];
    private int[] _outSlots = [];
    private VkSegment.StageIn[] _stageIn = [];
    private VkSegment.StageOut[] _stageOut = [];

    /// <summary>Forward 期间把 Tensor.Buffer 钉住，避免 GC 在提交与回读之间搬动它。</summary>
    private GCHandle[] _pins = new GCHandle[8];
    private int _pinCount;

    private bool _sealed, _disposed;

    public VkPartitionLayer(VkDevice dev, string name)
        : this(dev, name, new VkWeightArena(dev, name + ".weights"), ownsWeights: true)
    {
    }

    /// <summary>
    /// 挂到模型级共享权重区（P1-8）。权重属于模型、不属于段：换 shape 重建段时
    /// 只重建 arena 与 dispatch 列表，10 MB 权重不必再传一遍——
    /// 实测每个分区的「重建」里 619/691 us 有 619 us 是权重重传。
    /// <para>
    /// 共享时本层<b>不</b>负责 Commit：调用方把所有分区的权重登记完之后统一 Commit 一次。
    /// </para>
    /// </summary>
    public VkPartitionLayer(VkDevice dev, string name, VkWeightArena sharedWeights)
        : this(dev, name, sharedWeights, ownsWeights: false)
    {
    }

    private VkPartitionLayer(VkDevice dev, string name, VkWeightArena weights, bool ownsWeights)
    {
        ArgumentNullException.ThrowIfNull(dev);
        ArgumentNullException.ThrowIfNull(weights);
        if (string.IsNullOrEmpty(name)) throw new ArgumentException("分区名不能为空", nameof(name));

        _dev = dev;
        Name = name;
        Type = "VkPartition";
        _seg = new VkSegment(dev, name);
        _weights = weights;
        _ownsWeights = ownsWeights;
    }

    /// <summary>权重区是不是本层自己的（false = 模型级共享，Commit/Dispose 由调用方负责）。</summary>
    public bool OwnsWeights => _ownsWeights;

    /// <summary>段本身（P1-8 的分区 pass 从这里读槽位/arena 规模做成本估计）。</summary>
    public VkSegment Segment => _seg;

    /// <summary>权重常驻区。用法：`AddWeight(...)` 全部登记 → `Commit()` → `BindOf(...)` → `AddDispatch`。</summary>
    public VkWeightArena Weights => _weights;

    /// <summary>可选的时间戳计时器（P2 做 A/B 时打开）。</summary>
    public VkTimer? Timer { get; set; }

    public int SlotCount => _slots.Count;
    public ulong ArenaBytes => _seg.ArenaBytes;

    /// <summary>登记一个激活槽位，返回槽位 id。按 4 B 对齐：`vkCmdCopyBuffer` 的 region 要求。</summary>
    public int AddSlot(TensorShape shape, DType dtype)
    {
        ThrowIfSealed();
        long total = shape.Total();
        if (total <= 0)
            throw new ArgumentException($"槽位形状 {shape} 的元素数为 {total}", nameof(shape));
        if (total * dtype.SizeOf() > int.MaxValue)
            throw new NotSupportedException($"槽位 {shape} {dtype} 超过单块 buffer 上限");

        ulong bytes = (ulong)total * (ulong)dtype.SizeOf();
        if ((bytes & 3) != 0)
            throw new NotSupportedException(
                $"槽位 {shape} {dtype} = {bytes} B 不是 4 的倍数；P1 的 stage 边界要求 4 B 对齐（P2 段内 relayout 后再放宽）");

        int id = _seg.NewSlot(Align4(bytes));
        _slots.Add(new Slot(shape, dtype, bytes));
        return id;
    }

    /// <summary>登记一个权重张量（返回 id，稍后用 <see cref="VkWeightArena.BindOf(int)"/> 取绑定）。</summary>
    public int AddWeight(Tensor tensor) => _weights.Add(tensor);

    /// <summary>
    /// 一次性上传全部权重。之后再 `BindOf` 才拿得到有效切片。
    /// <para>共享权重区时这里只是转发（幂等），但**不应**在这里第一个 Commit：那样后面分区就登记不进权重了。</para>
    /// </summary>
    public void CommitWeights() => _weights.Commit();

    /// <summary>
    /// 追加一个 dispatch。<paramref name="binds"/> 里既可以是段内槽位（裸写 slot id），
    /// 也可以是 <see cref="VkWeightArena.BindOf(int)"/> 给出的外部权重切片。
    /// </summary>
    public void AddDispatch(string shader, int bindings, int pushConstBytes, ReadOnlySpan<VkBind> binds,
        ReadOnlySpan<byte> pushConstants, ulong elements)
    {
        ThrowIfSealed();
        ArgumentException.ThrowIfNullOrEmpty(shader);

        // 归因开关 ONNXSHARP_VK_ZERO="conv3x3,conv2d"：命中的核只发 1 个工作组。
        // 为什么用这个而不是 GPU 时间戳：这台机器（WDDM）上 vkCmdWriteTimestamp 给单
        // dispatch 报 0.8 us、给 28 个 3×3 卷积报 49 us（=88 TFLOP/s），物理上不可能，
        // 只有「段级一次提交」的时间是可信的。缩成 1 个工作组后段结构、绑定、屏障全不变，
        // 端到端掉多少就是该核真实吃掉的多少——不需要计时器自证。
        if (VkZero.Hit(shader) && elements > 64) elements = 64;

        VkPipelineTemplate tpl = VkPipelineTemplate.Get(_dev, shader, bindings, pushConstBytes);
        _seg.AddDispatch(tpl, binds, pushConstants, elements);
    }

    /// <summary>声明本层的输入按顺序来自哪些槽位（`Forward` 的 inputs 顺序）。</summary>
    public void SetInputs(ReadOnlySpan<int> slots)
    {
        ThrowIfSealed();
        _inSlots = CheckSlots(slots, nameof(slots));
    }

    /// <summary>声明本层的输出按顺序写进哪些槽位（`Forward` 的 outputs 顺序）。</summary>
    public void SetOutputs(ReadOnlySpan<int> slots)
    {
        ThrowIfSealed();
        _outSlots = CheckSlots(slots, nameof(slots));
    }

    /// <summary>分配 arena。幂等；之后不能再加槽位/dispatch。</summary>
    public void Seal()
    {
        ThrowIfDisposed();
        if (_sealed) return;
        if (_inSlots.Length == 0 || _outSlots.Length == 0)
            throw new InvalidOperationException($"分区 \"{Name}\" 还没声明输入/输出槽位");

        _seg.Seal();
        _sealed = true;

        // 输出张量必须由本层 Fit（静态形状），否则 GetMemoryShapes 与实际会不一致
        for (int i = 0; i < _outSlots.Length; i++)
        {
            Slot s = _slots[_outSlots[i]];
            if (s.Bytes == 0) throw new InvalidOperationException($"分区 \"{Name}\" 的输出槽位 {i} 是空的");
        }
    }

    // ---------------------------------------------------------------- Layer 接口

    /// <summary>
    /// 静态形状：槽位形状在建段时就定了（P1 口径是静态形状），这里直接回填声明值。
    /// 段内算子的形状推断发生在建段阶段，运行期不再重复。
    /// </summary>
    public override void GetMemoryShapes(
        IReadOnlyList<TensorShape> inShapes,
        int requiredOutputs,
        List<TensorShape> outShapes,
        List<TensorShape> tempShapes)
    {
        if (inShapes.Count != _inSlots.Length)
            throw new InvalidOperationException(
                $"分区 \"{Name}\" 声明了 {_inSlots.Length} 个输入，实际 {inShapes.Count} 个");
        if (requiredOutputs != _outSlots.Length)
            throw new InvalidOperationException(
                $"分区 \"{Name}\" 声明了 {_outSlots.Length} 个输出，图要求 {requiredOutputs} 个");

        outShapes.Clear();
        for (int i = 0; i < _outSlots.Length; i++)
            outShapes.Add(_slots[_outSlots[i]].Shape);
        // 中间结果全在显存里的段内槽位，不占 host 临时张量
        tempShapes.Clear();
    }

    public override void GetTypes(
        IReadOnlyList<DType> inTypes,
        int requiredOutputs,
        int requiredTemps,
        List<DType> outTypes,
        List<DType> tempTypes)
    {
        outTypes.Clear();
        for (int i = 0; i < _outSlots.Length; i++)
            outTypes.Add(_slots[_outSlots[i]].DType);
        tempTypes.Clear();
    }

    public override bool DynamicOutputShapes() => false;

    /// <summary>形状确定后做一次性准备：分配 arena 并核对实际张量与槽位一致。</summary>
    public override void Finalize(IReadOnlyList<Tensor> inputs, IReadOnlyList<Tensor> outputs)
    {
        Seal();
        CheckIo(inputs, outputs);
    }

    /// <summary>stage in → 一次提交跑完整段 → stage out。中间结果不离开显存。</summary>
    public override void Forward(
        IReadOnlyList<Tensor> inputs,
        IReadOnlyList<Tensor> outputs,
        IReadOnlyList<Tensor> temps)
    {
        Seal();
        CheckIo(inputs, outputs);

        try
        {
            if (_stageIn.Length < inputs.Count)
            {
                _stageIn = new VkSegment.StageIn[inputs.Count];
                _stageOut = new VkSegment.StageOut[outputs.Count];
            }

            for (int i = 0; i < inputs.Count; i++)
            {
                Tensor t = inputs[i];
                byte* p = Pin(t.Buffer) + t.ByteOffset;
                _stageIn[i] = new VkSegment.StageIn(_inSlots[i], p, _slots[_inSlots[i]].Bytes);
            }

            for (int i = 0; i < outputs.Count; i++)
            {
                Tensor t = outputs[i];
                byte* p = Pin(t.Buffer) + t.ByteOffset;
                _stageOut[i] = new VkSegment.StageOut(_outSlots[i], p, _slots[_outSlots[i]].Bytes);
            }

            _seg.Execute(_stageIn.AsSpan(0, inputs.Count), _stageOut.AsSpan(0, outputs.Count), Timer);
        }
        finally
        {
            UnpinAll();
        }

        // 调试：ONNXSHARP_VK_DUMP=first 时把第一个执行段的每个槽位读回 host 打统计。
        if (!_dumped && Environment.GetEnvironmentVariable("ONNXSHARP_VK_DUMP") == "first")
        {
            _dumped = true;
            DumpSlots(inputs, outputs);
        }
    }

    private bool _dumped;

    /// <summary>
    /// 把 host 输入与段内**每个**槽位的内容打统计。两件事只能靠它分辨：
    /// stage in 没把 host 数据送对（host-in 与 gpu-slot0 不同），还是段内算错（相同但结果错）。
    /// </summary>
    private void DumpSlots(IReadOnlyList<Tensor> inputs, IReadOnlyList<Tensor> outputs)
    {
        Console.WriteLine($"    [dump] {Name}: slots={_slots.Count} dispatch={_seg.DispatchCount}" +
                          $" in={inputs.Count} out={outputs.Count}");
        for (int i = 0; i < inputs.Count; i++)
        {
            Tensor t = inputs[i];
            int n = t.ByteLength / 4;
            fixed (byte* p = t.Buffer)
                Console.WriteLine($"    [dump] host-in{i} {t.Shape} {Stat((float*)(p + t.ByteOffset), n)}");
        }

        for (int s = 0; s < _slots.Count; s++)
        {
            int n = (int)(_slots[s].Bytes / 4);
            if (n <= 0) continue;
            float[] tmp = new float[n];
            fixed (float* p = tmp) _seg.DownloadSlot(s, p);
            Console.WriteLine($"    [dump] gpu-slot{s} {_slots[s].Shape} {Stat(tmp, n)}");
        }
    }

    private static string Stat(float* p, int n)
    {
        double sum = 0;
        float lo = float.MaxValue, hi = float.MinValue;
        for (int i = 0; i < n; i++)
        {
            float v = p[i];
            sum += v;
            if (v < lo) lo = v;
            if (v > hi) hi = v;
        }
        return $"min={lo:F4} max={hi:F4} mean={sum / Math.Max(1, n):F4}" +
               $" avg|x|={AvgAbs(p, n):F4} head=[{p[0]:F4},{p[1]:F4},{p[2]:F4},{p[3]:F4}]";
    }

    private static string Stat(float[] a, int n) { fixed (float* p = a) return Stat(p, n); }

    private static double AvgAbs(float* p, int n)
    {
        double s = 0;
        for (int i = 0; i < n; i++) s += Math.Abs(p[i]);
        return s / Math.Max(1, n);
    }

    public override string ToString() =>
        $"{Type}({Name}: slots={_slots.Count}, dispatches={_seg.DispatchCount}, arena={_seg.ArenaBytes / 1024} KB)";

    // ---------------------------------------------------------------- 内部

    private int[] CheckSlots(ReadOnlySpan<int> slots, string what)
    {
        if (slots.Length == 0) throw new ArgumentException("至少需要一个槽位", what);
        var copy = new int[slots.Length];
        for (int i = 0; i < slots.Length; i++)
        {
            if ((uint)slots[i] >= (uint)_slots.Count)
                throw new ArgumentOutOfRangeException(what, $"槽位 {slots[i]} 不存在（共 {_slots.Count} 个）");
            copy[i] = slots[i];
        }
        return copy;
    }

    /// <summary>
    /// 校验 host 侧张量与槽位声明一致。必须是紧凑布局：非紧凑视图直接拷字节是错的，
    /// 且这类错误不会报错只会算出垃圾——所以在这里挡住，等 P2 段内 relayout 再支持。
    /// </summary>
    private void CheckIo(IReadOnlyList<Tensor> inputs, IReadOnlyList<Tensor> outputs)
    {
        if (inputs.Count != _inSlots.Length)
            throw new OnnxSharpException(
                $"分区 \"{Name}\" 声明了 {_inSlots.Length} 个输入，实际给了 {inputs.Count} 个");
        if (outputs.Count != _outSlots.Length)
            throw new OnnxSharpException(
                $"分区 \"{Name}\" 声明了 {_outSlots.Length} 个输出，实际给了 {outputs.Count} 个");

        for (int i = 0; i < inputs.Count; i++) CheckTensor(inputs[i], _inSlots[i], "输入", i);
        for (int i = 0; i < outputs.Count; i++) CheckTensor(outputs[i], _outSlots[i], "输出", i);
    }

    private void CheckTensor(Tensor t, int slot, string what, int index)
    {
        Slot s = _slots[slot];
        if (t.DType != s.DType || !t.Shape.Equals(s.Shape))
            throw new OnnxSharpException(
                $"分区 \"{Name}\" 的{what} {index} 是 {t.Shape} {t.DType}，槽位 {slot} 是 {s.Shape} {s.DType}");
        if ((ulong)t.ByteLength != s.Bytes)
            throw new OnnxSharpException(
                $"分区 \"{Name}\" 的{what} {index} 字节数 {t.ByteLength} 与槽位 {slot} 的 {s.Bytes} 不一致");
        if (!t.IsContiguous)
            throw new OnnxSharpException(
                $"分区 \"{Name}\" 的{what} {index} 是非紧凑视图（Strides != null），stage 不支持");
    }

    private byte* Pin(byte[] buffer)
    {
        if (_pinCount == _pins.Length) Array.Resize(ref _pins, _pins.Length * 2);
        GCHandle h = GCHandle.Alloc(buffer, GCHandleType.Pinned);
        _pins[_pinCount++] = h;
        return (byte*)h.AddrOfPinnedObject();
    }

    private void UnpinAll()
    {
        for (int i = 0; i < _pinCount; i++) _pins[i].Free();
        _pinCount = 0;
    }

    private void ThrowIfSealed()
    {
        if (_sealed) throw new InvalidOperationException($"分区 \"{Name}\" 已 Seal，不能再改结构");
    }

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(nameof(VkPartitionLayer));
    }

    private static ulong Align4(ulong v) => (v + 3) & ~3UL;

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        UnpinAll();
        if (_ownsWeights) _weights.Dispose(); // 共享权重区由模型级持有者释放
        _seg.Dispose();
    }
}

/// <summary>
/// 归因用的「按核缩量」开关：<c>ONNXSHARP_VK_ZERO="conv3x3,conv2d_1x1"</c> 时命中的核只发
/// 1 个工作组。纯测量工具——数值当然会错，换来的是「这个核到底吃掉多少端到端时间」的直接答案。
/// <para>
/// 为什么需要它：这台机器上 GPU 时间戳不可信（单 dispatch 报 0.8 us、28 个 3×3 卷积报 49 us），
/// 而「同一张图、同一个段结构、只把一个核缩到 1 个工作组」的差值不需要任何计时器自证。
/// </para>
/// </summary>
public static class VkZero
{
    private static readonly HashSet<string> _names = (Environment.GetEnvironmentVariable("ONNXSHARP_VK_ZERO") ?? "")
        .Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
        .ToHashSet(StringComparer.Ordinal);

    /// <summary>这个核是否被「缩量」。命中时只发 1 个工作组，代价≈0，数值当然是错的。</summary>
    public static bool Hit(string shader) => _names.Count > 0 && _names.Contains(shader);

    /// <summary>
    /// 运行期切换集合（探针在一个进程里交错跑多个配置，避免跨进程的功耗/热漂移把差值淹掉）。
    /// 空集合 = 全开。
    /// </summary>
    public static void Set(params string[] names)
    {
        _names.Clear();
        foreach (string n in names)
            if (!string.IsNullOrWhiteSpace(n)) _names.Add(n.Trim());
    }

    /// <summary>
    /// 轴区间拷贝是否使用 float4 变体。生产默认开；关掉只用于 A/B 对照，
    /// 因为端到端噪声（±1.5 ms）大于单条核的差值，只能靠同结构内的同类聚合来判。
    /// </summary>
    public static bool Vec4Copy = true;

    /// <summary>
    /// 跳过 dispatch 之间的 compute 屏障。**只用于测量**：关掉屏障结果必然是错的，
    /// 但「少掉多少时间」正是屏障本身的成本——这是判断该不该去做精细同步的唯一依据。
    /// </summary>
    public static bool NoBarrier;

    /// <summary>
    /// 归因：退回「读写不分」的旧屏障判定（读到过就算依赖），用来 A/B 出
    /// 「省掉读-读屏障」到底值多少。默认 false。
    /// <para>
    /// 在类型初始化时读一次环境变量，而不是像 <see cref="SkipAll"/> 那样每帧读：
    /// 这个开关改变的是段的**构建结果**（屏障数量），构建只发生一次，
    /// 放进 <see cref="Set"/> 里重置反而会让 A/B 的两边建出同一个段。
    /// </para>
    /// </summary>
    public static readonly bool OldBarrier =
        Environment.GetEnvironmentVariable("ONNXSHARP_VK_OLDBARRIER") == "1";

    /// <summary>
    /// 归因用：非 0 时把每个 dispatch 的元素数压到这个值（1 个工作组）。
    /// <para>
    /// 与 <see cref="Hit"/> 的区别：它在**每帧录制命令时**读，所以能在同一个 Net、
    /// 同一份 arena 布局、同一管线缓存上做 A/B。重建 Net 有 5–7 ms 的系统性偏移
    /// （首个 Net 要现编管线，后面几个又受显存累积影响），足以淹掉任何 µs 级差值。
    /// </para>
    /// </summary>
    public static ulong ClampElements;

    /// <summary>
    /// 归因用：每录制一个真实 dispatch，就在它后面追加这么多个「1 个工作组」的空 dispatch
    /// （不插屏障、不改绑定、不换 pipeline）。
    /// <para>
    /// 这是唯一能测出「真实图里多一个 dispatch 的边际成本」的办法：合成链测出只要 5 us，
    /// 而真实图 166 个 dispatch 摊下来 78 us，差 15 倍——只有把空 dispatch 塞进真实图，
    /// 才能知道那 78 us 是属于「这个 dispatch 本身」还是「这张图的某种全局状态」。
    /// </para>
    /// </summary>
    public static int InjectPerDispatch;

    /// <summary>
    /// 归因用：跳过 Concat/Split/Reshape 的拷贝 dispatch（copy_axis / copy_axis4 / copy_concat4）。
    /// 数值必然错（下游读到脏槽位），换来的是「砍掉这 22 个 dispatch 能省多少」的直接答案。
    /// </summary>
    public static bool SkipCopy;

    /// <summary>归因用：跳过二元逐元素 dispatch（binary_bcast）。同上，只为测收益。</summary>
    public static bool SkipElementwise;

    /// <summary>
    /// 归因用：跳过一个段里的**全部** dispatch（stage in/out、submit、fence 照跑）。
    /// 测出来的是段的地板：4 段 × (搬运 + 一次 WDDM 提交往返)。
    /// 只有知道地板，才能判断「剩下的时间是不是根本不在 dispatch 上」。
    /// </summary>
    public static bool SkipAll;
}
