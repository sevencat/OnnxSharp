// 权重常驻区（P1-7）。
//
// 为什么单独一块：权重是模型级常量，整个推理生命周期不变；激活是每段分配的临时数据。
// 两者生命周期完全不同，混在一个 arena 里会导致「段重分配 → 权重跟着重传」。
//
// 量级差多少：yolo11n 约 2.6M 参数（fp32 约 10 MB）。若按段切并每段重传，搬运量随段数
// 线性放大；而权重只该在加载模型时传一次。
//
// 这块 buffer 用 device-local + 只读绑定，段内 dispatch 直接以外部切片的形式引用它
// （VkBind.External），既不参与段 arena 的分配，也不参与依赖屏障判定。
using OnnxSharp.Tensors;

namespace OnnxSharp.Vulkan;

internal sealed unsafe class VkWeightArena : IDisposable
{
    private readonly VkDevice _dev;
    private readonly string _name;
    private readonly List<Entry> _entries = [];
    private readonly Dictionary<Tensor, int> _index = new(ReferenceEqualityComparer.Instance);

    private ulong _cursor;
    private VkBuffer? _buffer;
    private bool _committed, _disposed;

    private readonly record struct Entry(Tensor Tensor, ulong Offset, ulong Size);

    public VkWeightArena(VkDevice dev, string name = "weights")
    {
        _dev = dev;
        _name = name;
    }

    public string Name => _name;
    public int Count => _entries.Count;

    /// <summary>已登记的权重总字节数（含对齐填充）。</summary>
    public ulong Bytes => Align(_cursor);

    /// <summary>是否已上传到显存。</summary>
    public bool Committed => _committed;

    /// <summary>登记一个权重张量，返回其切片 id（同一 Tensor 引用复用同一个 id）。</summary>
    public int Add(Tensor tensor)
    {
        ThrowIfCommitted();
        ArgumentNullException.ThrowIfNull(tensor);
        if (tensor.ByteLength == 0)
            throw new ArgumentException($"权重张量 {tensor.Shape} 是空的", nameof(tensor));
        if (_index.TryGetValue(tensor, out int existing))
            return existing;

        ulong offset = Align(_cursor);
        // 后端 buffer 的拷贝 region 要求 4 B 对齐；量化权重可能是 1 字节类型，尾部补到 4。
        // 补齐的部分只进 region，不进 memcpy（否则会越过 tensor.Buffer 末尾）。
        ulong size = ((ulong)tensor.ByteLength + 3) & ~3UL;
        _entries.Add(new Entry(tensor, offset, size));
        _cursor = offset + size;

        int id = _entries.Count - 1;
        _index.Add(tensor, id);
        return id;
    }

    /// <summary>该张量是否已登记。</summary>
    public bool Contains(Tensor tensor) => _index.ContainsKey(tensor);

    /// <summary>按 id 取绑定切片。必须在 <see cref="Commit"/> 之后。</summary>
    public VkBind BindOf(int id)
    {
        if (!_committed || _buffer is null)
            throw new InvalidOperationException($"权重区 \"{_name}\" 还没 Commit");
        if ((uint)id >= (uint)_entries.Count)
            throw new ArgumentOutOfRangeException(nameof(id), $"权重切片 {id} 不存在（共 {_entries.Count} 个）");

        Entry e = _entries[id];
        return VkBind.External(_buffer, e.Offset, e.Size);
    }

    public VkBind BindOf(Tensor tensor)
    {
        if (!_index.TryGetValue(tensor, out int id))
            throw new ArgumentException($"张量 {tensor.Shape} 没有登记到这个权重区", nameof(tensor));
        return BindOf(id);
    }

    /// <summary>分配显存并一次性上传全部权重。幂等；之后不能再 Add。</summary>
    public void Commit()
    {
        ThrowIfDisposed();
        if (_committed) return;
        _committed = true;
        if (_entries.Count == 0) return;

        ulong total = Align(_cursor);
        _buffer = _dev.NewBuffer(total, VkDevice.MemoryKind.DeviceLocal);

        // 所有权重走一次 staging + 一次提交：逐个 Upload 就是「一个权重一次 submit」。
        // staging 是短命对象，走池：模型重载/换 shape 重建时省掉一次 300+ us 的 vkAllocateMemory。
        // 常驻的 _buffer 本身不进池——它活得比池长，被复用会让旧切片静默指向新数据。
        using VkBuffer staging = _dev.NewBuffer(total, VkDevice.MemoryKind.HostUpload, pool: true);
        void* p = staging.Map();

        Span<VkDevice.CopyRegion> regions = new VkDevice.CopyRegion[_entries.Count];
        for (int i = 0; i < _entries.Count; i++)
        {
            Entry e = _entries[i];
            fixed (byte* src = e.Tensor.Buffer)
                System.Buffer.MemoryCopy(src + e.Tensor.ByteOffset, (byte*)p + e.Offset,
                    (long)(total - e.Offset), (long)e.Tensor.ByteLength);
            regions[i] = new VkDevice.CopyRegion(e.Offset, e.Offset, e.Size);
        }

        staging.Flush(0, total);
        staging.Unmap();

        _dev.CopyRegions(staging, _buffer, regions);
    }

    private void ThrowIfCommitted()
    {
        if (_committed)
            throw new InvalidOperationException(
                $"权重区 \"{_name}\" 已 Commit；新增权重需要重建（模型加载阶段一次性登记完）");
    }

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(nameof(VkWeightArena));
    }

    private static ulong Align(ulong v) => (v + VkSegment.SlotAlign - 1) & ~(VkSegment.SlotAlign - 1);

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _buffer?.Dispose();
        _buffer = null;
        _index.Clear();
        _entries.Clear();
    }
}
