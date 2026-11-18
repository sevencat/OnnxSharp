// 缓冲池（P1-9）。
//
// 为什么需要：P1-9 基线实测重建一个分区 499/512 us，其中 1171 KB 裸分配就要 646/570 us ——
// 也就是说瓶颈不是建段逻辑，而是 vkAllocateBuffer + vkAllocateMemory + vkBindBufferMemory
// 这条驱动路径。动态 shape、切换设备、模型重载都会重复走它。
//
// 做法：归还时不真销毁，把 (VkBuffer, VkDeviceMemory) 句柄存进按 (用途, 字节数) 分桶的空闲表；
// 下次申请先租。键用「申请字节数」而不是「分配字节数」，因为形状重复时申请字节数必然重复。
// 找不到精确匹配时退让到「最小的且够大的一块」——形状小幅变化（换 batch）不必重新分配。
//
// 这条路径只用于**短生命周期**的 buffer（段 arena、staging）。权重常驻区不走池：
// 它是模型级长生命周期对象，被复用的风险是「旧 VkBind 静默指向新数据」，比多一次分配贵得多。
//
// 安全前提：本后端所有提交都是「提交 + 等 fence」的同步模型，Dispose 发生时 GPU 已经空闲，
// 所以空闲区间可以立刻复用。将来做异步提交时，这块必须配 fence 回收（P3）。
using System.Runtime.InteropServices;

namespace OnnxSharp.Vulkan;

internal sealed class VkBufferPool : IDisposable
{
    /// <summary>池驻留上限。超过就按桶丢最老的；显存紧张的机器可以调小。</summary>
    public const ulong DefaultBudget = 128UL * 1024 * 1024;

    private readonly VkDevice _dev;
    private ulong _budget;
    private readonly Dictionary<(VkDevice.MemoryKind Kind, ulong Size), List<Entry>> _free = [];

    private ulong _resident;
    private long _rents, _hits, _returns, _drops;
    private bool _disposed;

    private struct Entry
    {
        public IntPtr Buffer, Memory;
        public ulong AllocSize;
        public uint Flags;
    }

    public VkBufferPool(VkDevice dev, ulong budget = DefaultBudget)
    {
        _dev = dev;
        _budget = budget;
    }

    /// <summary>当前驻留字节数（可复用但还没被租走的）。</summary>
    public ulong Resident => _resident;

    /// <summary>驻留上限。下调会立即按桶丢弃，显存紧张的机器可以在运行期调小。</summary>
    public ulong Budget
    {
        get => _budget;
        set
        {
            _budget = value;
            Evict();
        }
    }

    /// <summary>租用次数 / 命中次数 / 归还次数 / 直接丢弃次数。</summary>
    public long Rents => _rents;
    public long Hits => _hits;
    public long Returns => _returns;
    public long Drops => _drops;

    public double HitRate => _rents == 0 ? 0 : (double)_hits / _rents;

    public string Describe() =>
        $"pool: resident={_resident / 1024} KB budget={_budget / 1024} KB " +
        $"rents={_rents} hits={_hits} ({HitRate:P0}) drops={_drops}";

    /// <summary>尝试租一块「至少 <paramref name="bytes"/>」的 buffer。</summary>
    public bool TryRent(ulong bytes, VkDevice.MemoryKind kind, out VkBuffer buffer)
    {
        buffer = null!;
        _rents++;
        if (_disposed || _dev.Lost) return false;

        (VkDevice.MemoryKind, ulong) key = (kind, bytes);
        if (!_free.TryGetValue(key, out List<Entry>? bucket) || bucket.Count == 0)
        {
            // 精确尺寸没货：找最小的、同用途、且不小于申请量的桶
            ulong best = 0;
            foreach (KeyValuePair<(VkDevice.MemoryKind Kind, ulong Size), List<Entry>> kv in _free)
            {
                if (kv.Key.Kind != kind || kv.Value.Count == 0 || kv.Key.Size < bytes) continue;
                if (best == 0 || kv.Key.Size < best) best = kv.Key.Size;
            }
            if (best == 0) return false;

            key = (kind, best);
            bucket = _free[key];
        }

        Entry e = bucket[^1];
        bucket.RemoveAt(bucket.Count - 1);
        _resident -= e.AllocSize;
        _hits++;

        // 句柄搬到新对象上：池里不保留引用，旧 VkBuffer 已经 Released，
        // 再被用到会在 Map/Flush 处抛 ObjectDisposedException 而不是静默读写别人的数据。
        buffer = new VkBuffer
        {
            Dev = _dev.Device,
            Buffer = e.Buffer,
            Memory = e.Memory,
            Size = Math.Max(key.Item2, bytes),
            AllocSize = e.AllocSize,
            Flags = e.Flags,
            Owner = this,
            Kind = kind,
            // 漏了这句就是「租出去、还的时候真销毁」：池只活一轮，实测命中率卡在 41%
            Pooled = true,
        };
        return true;
    }

    /// <summary>归还一块 buffer。返回 true 表示池接管了句柄，调用方不要销毁。</summary>
    public bool Return(VkBuffer buf)
    {
        if (_disposed || _dev.Lost) return false;
        if (buf.Buffer == IntPtr.Zero || buf.Memory == IntPtr.Zero) return false;

        ulong size = buf.Size;
        if (size > _budget)
        {
            _drops++; // 超过整个预算的单块不值得驻留
            return false;
        }

        _returns++;
        _resident += buf.AllocSize;
        _free.TryAdd((buf.Kind, size), []);
        _free[(buf.Kind, size)].Add(new Entry
        {
            Buffer = buf.Buffer,
            Memory = buf.Memory,
            AllocSize = buf.AllocSize,
            Flags = buf.Flags,
        });

        Evict();
        return true;
    }

    /// <summary>丢掉所有驻留块（设备丢失、或显存吃紧时手动调）。</summary>
    public void Trim()
    {
        foreach (List<Entry> bucket in _free.Values)
        {
            foreach (Entry e in bucket) Destroy(e);
            bucket.Clear();
        }
        _free.Clear();
        _resident = 0;
    }

    /// <summary>超预算时从「条目最多的桶」丢最老的。近似 LRU：同桶内按归还顺序排队。</summary>
    private void Evict()
    {
        while (_resident > _budget)
        {
            List<Entry>? victim = null;
            foreach (List<Entry> bucket in _free.Values)
                if (bucket.Count > 0 && (victim is null || bucket.Count > victim.Count)) victim = bucket;
            if (victim is null) return;

            Entry e = victim[0];
            victim.RemoveAt(0);
            _resident -= e.AllocSize;
            _drops++;
            Destroy(e);
        }
    }

    private unsafe void Destroy(Entry e)
    {
        IntPtr dev = _dev.Device;
        if (dev == IntPtr.Zero) return; // 设备已销毁：句柄随之失效，不要再调 API
        if (e.Buffer != IntPtr.Zero) Vk.vkDestroyBuffer(dev, e.Buffer, null);
        if (e.Memory != IntPtr.Zero) Vk.vkFreeMemory(dev, e.Memory, null);
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        Trim();
    }
}
