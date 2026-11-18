// GPU 侧时间戳（P1-5）。存在的意义是把「kernel 真正跑了多久」和「整次提交花了多久」拆开：
// 二者差值就是 CPU 提交 + 驱动开销，P2 判断要不要换后端/换绑定模型时看的就是这个比例。
namespace OnnxSharp.Vulkan;

internal sealed unsafe class VkTimer : IDisposable
{
    /// <summary>默认 2 个槽：0 = 开始，1 = 结束。需要分算子计时就开更多槽。</summary>
    public const int DefaultSlots = 2;

    private readonly VkDevice _dev;
    private readonly IntPtr _pool;
    private readonly uint _slots;
    private readonly ulong[] _results;

    public VkTimer(VkDevice dev, uint slots = DefaultSlots)
    {
        if (dev.TimestampValidBits == 0)
            throw new PlatformNotSupportedException(
                "Vulkan: 选中的计算队列不支持 timestamp（TimestampValidBits = 0），无法计时");
        if (slots < 2) throw new ArgumentOutOfRangeException(nameof(slots), "至少需要 2 个槽");

        _dev = dev;
        _slots = slots;
        _results = new ulong[slots];

        Vk.VkQueryPoolCreateInfo ci = new()
        {
            SType = VkConst.StQueryPoolCreateInfo,
            QueryType = VkConst.QueryTypeTimestamp,
            QueryCount = slots,
        };
        Vk.Check(Vk.vkCreateQueryPool(dev.Device, &ci, null, out _pool), "vkCreateQueryPool");
    }

    /// <summary>在命令缓冲开头调用一次。</summary>
    public void Reset(IntPtr cmd) => Vk.vkCmdResetQueryPool(cmd, _pool, 0, _slots);

    /// <summary>打一个时间点。写时间戳的语义是「此前已提交的 compute 工作全部完成」。</summary>
    public void Mark(IntPtr cmd, uint slot)
    {
        if (slot >= _slots) throw new ArgumentOutOfRangeException(nameof(slot));
        Vk.vkCmdWriteTimestamp(cmd, VkConst.PipelineStageComputeShader, _pool, slot);
    }

    /// <summary>提交并 fence 等待之后调用，取回所有槽的原始 tick。</summary>
    public ReadOnlySpan<ulong> Read()
    {
        fixed (ulong* p = _results)
        {
            Vk.Check(Vk.vkGetQueryPoolResults(_dev.Device, _pool, 0, _slots,
                (nuint)(_slots * sizeof(ulong)), p, sizeof(ulong),
                VkConst.QueryResult64 | VkConst.QueryResultWait), "vkGetQueryPoolResults");
        }
        return _results;
    }

    /// <summary>槽位数。逐 dispatch 计时要求槽数 ≥ dispatch 数 + 1。</summary>
    public uint Slots => _slots;

    /// <summary>两个槽之间的毫秒数（已按 timestampPeriod 换算）。</summary>
    public double ElapsedMs(uint from, uint to)
    {
        ReadOnlySpan<ulong> r = Read();
        if (r[(int)to] < r[(int)from]) return 0; // tick 回绕/跨队列，宁可为 0 也不给负值
        return (r[(int)to] - r[(int)from]) * _dev.TimestampPeriodNs / 1e6;
    }

    /// <summary>
    /// 一次读回、按相邻槽差分（毫秒），长度 = 槽数 − 1。
    /// <para>
    /// 逐 dispatch 计时用：每个 dispatch 后打一个点，第 i 个 dispatch 的时间就是
    /// 槽 i → i+1。逐槽调用 <see cref="ElapsedMs"/> 会把同一个 query pool 读 N 次，
    /// 每次都是一趟驱动往返（几十 us 级），量 60 个 dispatch 就自己污染自己。
    /// </para>
    /// </summary>
    public double[] IntervalMs()
    {
        ReadOnlySpan<ulong> r = Read();
        var ms = new double[_slots - 1];
        double period = _dev.TimestampPeriodNs;
        for (int i = 0; i + 1 < _slots; i++)
            ms[i] = r[i + 1] >= r[i] ? (r[i + 1] - r[i]) * period / 1e6 : 0;
        return ms;
    }

    /// <summary>整段（槽 0 → 最后一个槽）的毫秒数。</summary>
    public double TotalMs => ElapsedMs(0, _slots - 1);

    public void Dispose()
    {
        if (_pool != IntPtr.Zero) Vk.vkDestroyQueryPool(_dev.Device, _pool, null);
    }
}
