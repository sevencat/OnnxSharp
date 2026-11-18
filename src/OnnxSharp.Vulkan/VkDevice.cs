// Vulkan 设备封装（P1-3）：instance / 选卡 / 计算队列 / 内存类型 / 命令缓冲 / fence / buffer / pipeline。
// 只保留通用设施。SimdOnnx 里那些针对特定 kernel 的调优（cooperative matrix、固定子组宽度、
// Vec4 元素级 kernel）不在 P1 范围，等 P2 有实际 kernel 数据再按需加。
using System.Diagnostics;
using System.Runtime.InteropServices;

namespace OnnxSharp.Vulkan;

internal unsafe sealed class VkDevice : IDisposable
{
    public IntPtr Instance, PhysDevice, Device, Queue;

    public string DeviceName = "";
    public uint VendorId, DeviceId, ApiVersion, DriverVersion;
    public uint QueueFamily;
    public Vk.VkPhysicalDeviceMemoryProperties MemProps;

    /// <summary>VK_KHR_push_descriptor：descriptor 内联写入，省掉 per-dispatch 的 set 绑定。</summary>
    public bool PushDescriptors;
    /// <summary>VK 1.1 core storageBuffer16BitAccess：P3 的 fp16 激活存储要用。</summary>
    public bool StorageBuffer16;
    /// <summary>DEVICE_LOCAL|HOST_VISIBLE|HOST_COHERENT 存在（ReBAR），可省掉 staging 往返。</summary>
    public bool CoherentDeviceLocal;
    /// <summary>核显：device-local 就是系统内存，CPU/GPU 抢带宽。</summary>
    public bool UnifiedMemory;

    public ulong DeviceLocalHeap;
    public ulong MaxStorageRange = uint.MaxValue;
    public uint MaxSharedMemory;
    public uint SubgroupSize = 32;
    public double TimestampPeriodNs = 1;
    /// <summary>计算队列的 timestamp 有效位；0 表示不能计时（VkTimer 会明确拒绝）。</summary>
    public uint TimestampValidBits;

    /// <summary>P1-5 的 pipeline 模板缓存，键为 "shader|bindings|pushConstBytes"。随设备一起销毁。</summary>
    internal readonly Dictionary<string, VkPipelineTemplate> Templates = new(StringComparer.Ordinal);

    /// <summary>Submit / fence 等待报 ErrorDeviceLost 后置位；此后调用应快速失败，由上层重建设备。</summary>
    internal volatile bool Lost;

    /// <summary>
    /// 进入「设备已丢失」状态：置位 <see cref="Lost"/> 并放弃池里所有驻留块。
    /// 丢失之后 vkAllocateMemory 也开始返回 DEVICE_LOST，留着驻留块既没用又挡住诊断，
    /// 所以在这里一次性清空；后续分配/提交全部快速失败，由上层重建 <see cref="VkDevice"/>。
    /// </summary>
    internal void MarkLost()
    {
        if (Lost) return;
        Lost = true;
        Pool.Trim();
    }

    /// <summary>vkQueueSubmit 在同一队列上需要外部同步。</summary>
    public readonly object QueueLock = new();

    /// <summary>
    /// fence 的忙等预算（us）。0 = 纯阻塞等待。
    /// <para>
    /// 每段的固定成本实测约 120 us，而段内 8 次 dispatch 的边际成本只有 4 us/次——
    /// 说明这笔钱主要花在 submit/等待的往返上，不在 GPU 干活上。WDDM 上
    /// <c>vkWaitForFences</c> 是「挂起线程 + 事件唤醒」，唤醒本身就有几十 us 的延迟；
    /// 先自旋一小段再退回阻塞，通常能把这段延迟吃掉而不烧满一个核（有预算上限）。
    /// 设成 0 就是纯阻塞，用来 A/B 验证「这笔钱到底在不在等待上」。
    /// </para>
    /// </summary>
    public static long SpinWaitUs { get; set; } = ReadSpinDefault();

    private static long ReadSpinDefault()
        => long.TryParse(Environment.GetEnvironmentVariable("ONNXSHARP_VK_SPIN"), out long us) && us >= 0
            ? us : 200;

    private IntPtr _cmdPool;

    /// <summary>零散 Upload/Download 的复用 staging（上传写合并、回读可缓存，两者不能共用）。</summary>
    private VkBuffer? _stagingUp, _stagingDown;

    /// <summary>短生命周期 buffer 的空闲表（段 arena、staging）。见 <see cref="VkBufferPool"/>。</summary>
    internal readonly VkBufferPool Pool;

    /// <summary>VK_KHR_push_descriptor 的入口，只在设备支持时非空。</summary>
    private delegate* unmanaged<IntPtr, uint, IntPtr, uint, uint, Vk.VkWriteDescriptorSet*, void>
        _cmdPushDescriptorSet;

    private VkDevice() => Pool = new VkBufferPool(this);

    static VkDevice() => Vk.RegisterResolver();

    public string ApiVersionText => $"{ApiVersion >> 22}.{(ApiVersion >> 12) & 0x3FFu}.{ApiVersion & 0xFFFu}";

    /// <summary>NVIDIA 用 10/8/8/6 打包驱动版本，与 Vulkan 规范的 22/12/12 不同；直接按规范解会得到
    /// 621.96.0 这种不存在的版本号，benchv3 记录驱动时必须有可核对的值。</summary>
    public string DriverVersionText => VendorId == 0x10DE
        ? $"{DriverVersion >> 22}.{(DriverVersion >> 14) & 0xFFu}.{(DriverVersion >> 6) & 0xFFu}.{DriverVersion & 0x3Fu}"
        : $"{DriverVersion >> 22}.{(DriverVersion >> 12) & 0x3FFu}.{DriverVersion & 0xFFFu}";

    // ---------------------------------------------------------------- 创建

    /// <summary>创建 instance 与设备。设备按「独显优先、核显次之」排序，<paramref name="deviceIndex"/>
    /// 是该排序下的序号（0 = 首选）。失败时把已建的一半资源也清掉，不留悬挂 instance。</summary>
    public static VkDevice Create(uint deviceIndex = 0)
    {
        var d = new VkDevice();
        try
        {
            d.CreateInstance();
            d.PickPhysicalDevice(deviceIndex);
            d.ReadProperties();
            d.PickQueueFamily();
            d.ReadMemoryProperties();
            d.CreateLogicalDevice();
            return d;
        }
        catch
        {
            d.Dispose();
            throw;
        }
    }

    private void CreateInstance()
    {
        Vk.VkApplicationInfo app = new()
        {
            SType = VkConst.StApplicationInfo,
            AppVersion = Vk.MakeVersion(0, 6, 0),
            EngineVersion = Vk.MakeVersion(0, 6, 0),
            ApiVersion = Vk.MakeVersion(1, 1, 0),
        };
        Vk.VkInstanceCreateInfo ci = new() { SType = VkConst.StInstanceCreateInfo, PAppInfo = &app };
        Vk.Check(Vk.vkCreateInstance(&ci, null, out Instance), "vkCreateInstance");
    }

    private void PickPhysicalDevice(uint deviceIndex)
    {
        uint count = 0;
        Vk.Check(Vk.vkEnumeratePhysicalDevices(Instance, &count, null), "enum physical devices");
        if (count == 0)
            throw new PlatformNotSupportedException("Vulkan: no physical device (no Vulkan-capable GPU/driver)");

        IntPtr* devices = stackalloc IntPtr[(int)count];
        Vk.Check(Vk.vkEnumeratePhysicalDevices(Instance, &count, devices), "enum physical devices");

        // 先按「独显 → 核显 → 其它 → 软件」排出稳定顺序，再取第 deviceIndex 个。
        // 不能边扫边数：那样两趟各自的计数会和用户以为的「设备 0/1」错位。
        int n = (int)count;
        uint* types = stackalloc uint[n];
        for (int i = 0; i < n; i++)
        {
            Vk.VkPhysicalDeviceProperties p;
            Vk.vkGetPhysicalDeviceProperties(devices[i], &p);
            types[i] = p.DeviceType;
        }
        int* order = stackalloc int[n];
        int k = 0;
        k = Collect(order, k, types, n, VkConst.PhysDeviceDiscrete);
        k = Collect(order, k, types, n, VkConst.PhysDeviceIntegrated);
        for (int i = 0; i < n; i++)
            if (types[i] != VkConst.PhysDeviceDiscrete
                && types[i] != VkConst.PhysDeviceIntegrated
                && types[i] != VkConst.PhysDeviceCpu)
                order[k++] = i;
        if (k == 0)
            for (int i = 0; i < n; i++)
                if (types[i] == VkConst.PhysDeviceCpu) order[k++] = i;

        if (deviceIndex >= (uint)k)
            throw new PlatformNotSupportedException(
                $"Vulkan: device index {deviceIndex} out of range ({k} usable of {n} enumerated)");
        PhysDevice = devices[order[deviceIndex]];
    }

    private static int Collect(int* order, int k, uint* types, int n, uint want)
    {
        for (int i = 0; i < n; i++)
            if (types[i] == want) order[k++] = i;
        return k;
    }

    private void ReadProperties()
    {
        Vk.VkPhysicalDeviceProperties props;
        Vk.vkGetPhysicalDeviceProperties(PhysDevice, &props);
        unsafe
        {
            byte* pName = props.DeviceName;
            DeviceName = Marshal.PtrToStringUTF8((IntPtr)pName) ?? "?";
        }
        VendorId = props.VendorID;
        DeviceId = props.DeviceID;
        ApiVersion = props.ApiVersion;
        DriverVersion = props.DriverVersion;
        UnifiedMemory = props.DeviceType == VkConst.PhysDeviceIntegrated;
        MaxStorageRange = props.MaxStorageBufferRange;
        MaxSharedMemory = props.MaxComputeSharedMemorySize;
        TimestampPeriodNs = props.TimestampPeriodNs;

        Vk.VkPhysicalDeviceSubgroupProperties sg = new() { SType = VkConst.StPhysicalDeviceSubgroupProperties };
        Vk.VkPhysicalDeviceProperties2 p2 = new() { SType = VkConst.StPhysicalDeviceProperties2, PNext = &sg };
        Vk.vkGetPhysicalDeviceProperties2(PhysDevice, &p2);
        if (sg.SubgroupSize > 0) SubgroupSize = sg.SubgroupSize;

        // storageBuffer16BitAccess（VK 1.1 core）——P3 fp16 存储的前置条件
        Vk.VkPhysicalDevice16BitStorageFeatures f16 = new() { SType = VkConst.StPhysicalDevice16BitStorageFeatures };
        Vk.VkPhysicalDeviceFeatures2 f2 = new() { SType = VkConst.StPhysicalDeviceFeatures2, PNext = &f16 };
        Vk.vkGetPhysicalDeviceFeatures2(PhysDevice, &f2);
        StorageBuffer16 = f16.StorageBuffer16BitAccess != 0;
    }

    private void PickQueueFamily()
    {
        uint n = 0;
        Vk.vkGetPhysicalDeviceQueueFamilyProperties(PhysDevice, &n, null);
        if (n == 0) throw new PlatformNotSupportedException("Vulkan: device exposes no queue family");

        Vk.VkQueueFamilyProperties* qf = stackalloc Vk.VkQueueFamilyProperties[(int)n];
        Vk.vkGetPhysicalDeviceQueueFamilyProperties(PhysDevice, &n, qf);

        const uint Invalid = uint.MaxValue;
        QueueFamily = Invalid;
        // 优先「纯计算 + 带时间戳」的家族：与图形队列分开，避免被呈现阻塞，也能取 timestamp
        for (uint i = 0; i < n && QueueFamily == Invalid; i++)
            if ((qf[i].QueueFlags & VkConst.QueueCompute) != 0
                && (qf[i].QueueFlags & VkConst.QueueGraphics) == 0
                && qf[i].TimestampValidBits > 0)
                QueueFamily = i;
        for (uint i = 0; i < n && QueueFamily == Invalid; i++)
            if ((qf[i].QueueFlags & VkConst.QueueCompute) != 0 && qf[i].TimestampValidBits > 0)
                QueueFamily = i;
        for (uint i = 0; i < n && QueueFamily == Invalid; i++)
            if ((qf[i].QueueFlags & VkConst.QueueCompute) != 0)
                QueueFamily = i;

        if (QueueFamily == Invalid)
            throw new PlatformNotSupportedException("Vulkan: no compute queue family");

        TimestampValidBits = qf[QueueFamily].TimestampValidBits;
    }

    private void ReadMemoryProperties()
    {
        Vk.VkPhysicalDeviceMemoryProperties mp;
        Vk.vkGetPhysicalDeviceMemoryProperties(PhysDevice, &mp);
        MemProps = mp;

        for (int i = 0; i < (int)mp.MemoryHeapCount; i++)
        {
            Vk.VkMemoryHeap h = mp.HeapAt(i);
            if ((h.Flags & VkConst.MemHeapDeviceLocal) != 0 && h.Size > DeviceLocalHeap)
                DeviceLocalHeap = h.Size;
        }
        for (int i = 0; i < (int)mp.MemoryTypeCount; i++)
        {
            uint f = mp.TypeAt(i).PropertyFlags;
            if ((f & (VkConst.MemDeviceLocal | VkConst.MemHostVisible | VkConst.MemHostCoherent))
                == (VkConst.MemDeviceLocal | VkConst.MemHostVisible | VkConst.MemHostCoherent))
                CoherentDeviceLocal = true;
        }
    }

    private void CreateLogicalDevice()
    {
        // 扩展探测：只关心 push descriptor
        uint extCount = 0;
        Vk.vkEnumerateDeviceExtensionProperties(PhysDevice, null, &extCount, null);
        bool hasPush = false;
        if (extCount > 0)
        {
            Vk.VkExtensionProperties* exts = stackalloc Vk.VkExtensionProperties[(int)extCount];
            if (Vk.vkEnumerateDeviceExtensionProperties(PhysDevice, null, &extCount, exts) == VkResult.Success)
                for (int i = 0; i < (int)extCount; i++)
                    if (new string((sbyte*)exts[i].ExtensionName) == "VK_KHR_push_descriptor") hasPush = true;
        }

        float priority = 1f;
        Vk.VkDeviceQueueCreateInfo qci = new()
        {
            SType = VkConst.StDeviceQueueCreateInfo,
            QueueFamilyIndex = QueueFamily,
            QueueCount = 1,
            PQueuePriorities = &priority,
        };

        // 16-bit storage 链进 pNext（VK 1.1 core），P3 才会真正用上，但 feature 要在创建设备时开启
        Vk.VkPhysicalDevice16BitStorageFeatures f16 = new()
        {
            SType = VkConst.StPhysicalDevice16BitStorageFeatures,
            StorageBuffer16BitAccess = StorageBuffer16 ? 1u : 0u,
        };

        byte** extNames = stackalloc byte*[1];
        uint nExt = 0;
        byte* pPush = stackalloc byte[24];
        if (hasPush)
        {
            "VK_KHR_push_descriptor\0"u8.CopyTo(new Span<byte>(pPush, 24));
            extNames[nExt++] = pPush;
        }

        Vk.VkDeviceCreateInfo ci = new()
        {
            SType = VkConst.StDeviceCreateInfo,
            PNext = &f16,
            QueueCreateInfoCount = 1,
            PQueueCreateInfos = &qci,
            EnabledExtensionCount = nExt,
            PpEnabledExtensionNames = extNames,
            // Features2 走 pNext 时必须留空
            PEnabledFeatures = null,
        };
        Vk.Check(Vk.vkCreateDevice(PhysDevice, &ci, null, out Device), "vkCreateDevice");
        PushDescriptors = nExt > 0;

        Vk.vkGetDeviceQueue(Device, QueueFamily, 0, out Queue);

        if (PushDescriptors)
        {
            ReadOnlySpan<byte> fname = "vkCmdPushDescriptorSetKHR\0"u8;
            byte* pname = stackalloc byte[fname.Length];
            fname.CopyTo(new Span<byte>(pname, fname.Length));
            IntPtr p = Vk.vkGetDeviceProcAddr(Device, pname);
            if (p == IntPtr.Zero)
                PushDescriptors = false; // 扩展声明了但入口拿不到：老实地退回普通 descriptor set
            else
                _cmdPushDescriptorSet =
                    (delegate* unmanaged<IntPtr, uint, IntPtr, uint, uint, Vk.VkWriteDescriptorSet*, void>)p;
        }

        Vk.VkCommandPoolCreateInfo pci = new()
        {
            SType = VkConst.StCommandPoolCreateInfo,
            Flags = VkConst.CmdPoolResetCommandBuffer,
            QueueFamilyIndex = QueueFamily,
        };
        Vk.Check(Vk.vkCreateCommandPool(Device, &pci, null, out _cmdPool), "vkCreateCommandPool");
    }

    // ---------------------------------------------------------------- buffer / 内存

    /// <summary>
    /// 分配时对内存类型的意图。PCIe 上的 host 内存有两种，选错方向会让 CPU 侧慢一个数量级：
    /// <list type="bullet">
    /// <item>写合并（WC，无 HOST_CACHED）：CPU 顺序写很快，**读极慢**；适合上传。</item>
    /// <item>可缓存（HOST_CACHED）：CPU 读走 cache line，是回读的唯一正确选择。</item>
    /// </list>
    /// 实测（RTX 5060 Laptop，400 KB）：WC 回读 ≈0.2 GB/s（1974 us），HOST_CACHED 可达数 GB/s。
    /// </summary>
    public enum MemoryKind
    {
        /// <summary>显存内部，CPU 不可映射（激活/权重常驻）。</summary>
        DeviceLocal,
        /// <summary>CPU 写 → GPU 读的 staging：纯 host 类型，不与显存抢空间。</summary>
        HostUpload,
        /// <summary>GPU 写 → CPU 读的 staging：优先 HOST_CACHED。</summary>
        HostReadback,
    }

    /// <summary>
    /// 分配 storage buffer。<paramref name="kind"/> 决定显存还是哪一类 host 内存。
    /// <paramref name="pool"/> 只给短生命周期对象用（段 arena / staging）：Dispose 时不是真销毁，
    /// 而是留在 <see cref="Pool"/> 里给下一个同尺寸的申请。实测 1.2 MB 单次分配要 646 us，
    /// 而重建一个分区一共才 500 us —— 分配就是重建的全部成本。
    /// </summary>
    public VkBuffer NewBuffer(ulong bytes, MemoryKind kind, bool pool = false)
    {
        if (Lost) throw new VkDeviceLostException("NewBuffer");
        if (bytes == 0) throw new ArgumentOutOfRangeException(nameof(bytes), "buffer size must be > 0");
        if (pool && Pool.TryRent(bytes, kind, out VkBuffer rented)) return rented;

        var b = new VkBuffer { Dev = Device, Size = bytes, Pooled = pool, Kind = kind, Owner = pool ? Pool : null };
        bool ok = false;
        try
        {
            Vk.VkBufferCreateInfo bci = new()
            {
                SType = VkConst.StBufferCreateInfo,
                Size = bytes,
                Usage = VkConst.BufferUsageStorageBuffer
                      | VkConst.BufferUsageTransferSrc
                      | VkConst.BufferUsageTransferDst,
                SharingMode = VkConst.SharingExclusive,
            };
            Vk.Check(Vk.vkCreateBuffer(Device, &bci, null, out b.Buffer), "vkCreateBuffer");

            Vk.VkMemoryRequirements req;
            Vk.vkGetBufferMemoryRequirements(Device, b.Buffer, &req);

            uint memType = FindMemoryType(req.MemoryTypeBits, kind);
            Vk.VkMemoryAllocateInfo mai = new()
            {
                SType = VkConst.StMemoryAllocateInfo,
                AllocationSize = req.Size,
                MemoryTypeIndex = memType,
            };
            Vk.Check(Vk.vkAllocateMemory(Device, &mai, null, out b.Memory), "vkAllocateMemory");
            b.AllocSize = req.Size;
            Vk.Check(Vk.vkBindBufferMemory(Device, b.Buffer, b.Memory, 0), "vkBindBufferMemory");
            b.Flags = MemProps.TypeAt((int)memType).PropertyFlags;
            ok = true;
            return b;
        }
        finally
        {
            if (!ok) b.Dispose();
        }
    }

    /// <summary>
    /// 按用途挑内存类型。评分而不是「撞到第一个就用」的原因：一张卡上往往同时有
    /// WC 与 HOST_CACHED 两种 host 类型，方向选错的代价是 10 倍（见 <see cref="MemoryKind"/>）。
    /// </summary>
    private uint FindMemoryType(uint typeBits, MemoryKind kind)
    {
        uint best = uint.MaxValue;
        int bestScore = -1;
        for (int i = 0; i < (int)MemProps.MemoryTypeCount; i++)
        {
            if ((typeBits & (1u << i)) == 0) continue;
            uint f = MemProps.TypeAt(i).PropertyFlags;
            bool host = (f & VkConst.MemHostVisible) != 0;
            bool coherent = (f & VkConst.MemHostCoherent) != 0;
            bool local = (f & VkConst.MemDeviceLocal) != 0;
            bool cached = (f & VkConst.MemHostCached) != 0;

            int score;
            if (kind == MemoryKind.DeviceLocal)
            {
                if (!local) continue;
                // 不可映射的显存优先：ReBAR 那种可映射类型留给别的用途
                score = host ? 1 : 2;
            }
            else
            {
                if (!host) continue;
                // staging 一律放 host heap，不占显存
                score = local ? 0 : 2;
                // 回读要可缓存；上传要写合并（不可缓存）
                score += kind == MemoryKind.HostReadback
                    ? (cached ? 4 : 0)
                    : (cached ? 0 : 1);
                if (coherent) score += 1;
            }

            if (score > bestScore)
            {
                bestScore = score;
                best = (uint)i;
            }
        }

        if (best != uint.MaxValue) return best;
        throw new PlatformNotSupportedException(
            $"Vulkan: no memory type for {kind} (typeBits=0x{typeBits:X})");
    }

    /// <summary>可增长的复用 staging。VkSegment 之外的零散 Upload/Download 也不该每次新分配
    /// （一次 vkAllocateMemory ≈200 us，比 400 KB 的拷贝本身还贵）。</summary>
    private VkBuffer Staging(ref VkBuffer? slot, ulong bytes, MemoryKind kind)
    {
        if (slot is not null && slot.Size >= bytes) return slot;
        slot?.Dispose();
        slot = NewBuffer(bytes, kind, pool: true);
        return slot;
    }

    /// <summary>把 CPU 数据写进 buffer 的指定偏移。device-local 走 staging + vkCmdCopyBuffer；可映射则直接写。</summary>
    public void Upload(VkBuffer dst, void* src, ulong bytes, ulong dstOffset = 0)
    {
        if (dstOffset + bytes > dst.Size)
            throw new ArgumentOutOfRangeException(nameof(bytes), "upload exceeds buffer size");
        if (dst.IsHostVisible)
        {
            void* p = dst.Map();
            Buffer.MemoryCopy(src, (byte*)p + dstOffset, (long)(dst.AllocSize - dstOffset), (long)bytes);
            dst.Flush(dstOffset, bytes);
            dst.Unmap();
            return;
        }

        VkBuffer staging = Staging(ref _stagingUp, bytes, MemoryKind.HostUpload);
        void* sp = staging.Map();
        Buffer.MemoryCopy(src, sp, (long)staging.AllocSize, (long)bytes);
        staging.Flush(0, bytes);
        staging.Unmap();

        CopyRegions(staging, dst, [new CopyRegion(0, dstOffset, bytes)]);
    }

    /// <summary>从 buffer 读回 CPU。device-local 先拷到 staging。</summary>
    public void Download(VkBuffer src, void* dst, ulong bytes, ulong offset = 0)
    {
        if (offset + bytes > src.Size) throw new ArgumentOutOfRangeException(nameof(bytes), "download exceeds buffer size");
        if (src.IsHostVisible)
        {
            void* p = src.Map();
            src.Invalidate(offset, bytes);
            Buffer.MemoryCopy((byte*)p + offset, dst, (long)bytes, (long)bytes);
            src.Unmap();
            return;
        }

        // 回读必须用 HOST_CACHED：WC 内存 CPU 读 400 KB 要 2 ms（见 MemoryKind）
        VkBuffer staging = Staging(ref _stagingDown, bytes, MemoryKind.HostReadback);
        CopyRegions(src, staging, [new CopyRegion(offset, 0, bytes)]);

        void* sp = staging.Map();
        staging.Invalidate(0, bytes);
        Buffer.MemoryCopy(sp, dst, (long)bytes, (long)bytes);
        staging.Unmap();
    }

    /// <summary>一条 <c>vkCmdCopyBuffer</c> 里的一块区域。</summary>
    public readonly struct CopyRegion
    {
        public readonly ulong SrcOffset;
        public readonly ulong DstOffset;
        public readonly ulong Size;

        public CopyRegion(ulong srcOffset, ulong dstOffset, ulong size)
        {
            SrcOffset = srcOffset;
            DstOffset = dstOffset;
            Size = size;
        }
    }

    /// <summary>把多个 region 录进调用方给的命令缓冲（不提交）。</summary>
    public void CmdCopyBuffer(IntPtr cmd, VkBuffer src, VkBuffer dst, ReadOnlySpan<CopyRegion> regions)
    {
        if (regions.Length == 0) return;
        if (regions.Length > MaxCopyRegions)
            throw new ArgumentOutOfRangeException(nameof(regions),
                $"一次最多 {MaxCopyRegions} 个 region，实际 {regions.Length}");

        Vk.VkBufferCopy* p = stackalloc Vk.VkBufferCopy[regions.Length];
        for (int i = 0; i < regions.Length; i++)
        {
            CopyRegion r = regions[i];
            if (r.SrcOffset + r.Size > src.Size || r.DstOffset + r.Size > dst.Size)
                throw new ArgumentOutOfRangeException(nameof(regions), $"region {i} 越界");
            // vkCmdCopyBuffer 要求 offset 与 size 都是 4 的倍数
            if ((r.Size & 3) != 0 || (r.SrcOffset & 3) != 0 || (r.DstOffset & 3) != 0)
                throw new ArgumentException(
                    $"region {i} 需 4 字节对齐：src={r.SrcOffset} dst={r.DstOffset} size={r.Size}", nameof(regions));
            p[i] = new Vk.VkBufferCopy { SrcOffset = r.SrcOffset, DstOffset = r.DstOffset, Size = r.Size };
        }
        Vk.vkCmdCopyBuffer(cmd, src.Buffer, dst.Buffer, (uint)regions.Length, p);
    }

    /// <summary>
    /// 批量拷贝：所有 region 录进同一条命令缓冲，只提交一次。
    /// <para>
    /// stage in/out 必须走这条。逐个张量调 <see cref="Upload"/> 就是「一个张量一次 submit」，
    /// P1-5 实测每次提交约 100 us——一个 16 输入的段光搬运就要 1.6 ms。
    /// </para>
    /// </summary>
    public void CopyRegions(VkBuffer src, VkBuffer dst, ReadOnlySpan<CopyRegion> regions)
    {
        if (regions.Length == 0) return;

        IntPtr cmd = NewCommandBuffer();
        try
        {
            Vk.VkCommandBufferBeginInfo begin = new()
            {
                SType = VkConst.StCommandBufferBeginInfo,
                Flags = VkConst.CmdBufferUsageOneTimeSubmit,
            };
            Vk.Check(Vk.vkBeginCommandBuffer(cmd, &begin), "vkBeginCommandBuffer");
            CmdCopyBuffer(cmd, src, dst, regions);
            Vk.Check(Vk.vkEndCommandBuffer(cmd), "vkEndCommandBuffer");
            SubmitAndWait(cmd);
        }
        finally
        {
            FreeCommandBuffer(cmd);
        }
    }

    // ---------------------------------------------------------------- 命令缓冲 / fence

    public IntPtr NewCommandBuffer()
    {
        if (Lost) throw new VkDeviceLostException("NewCommandBuffer");
        Vk.VkCommandBufferAllocateInfo ai = new()
        {
            SType = VkConst.StCommandBufferAllocateInfo,
            CommandPool = _cmdPool,
            Level = VkConst.CmdBufferLevelPrimary,
            CommandBufferCount = 1,
        };
        IntPtr cmd;
        Vk.Check(Vk.vkAllocateCommandBuffers(Device, &ai, &cmd), "vkAllocateCommandBuffers");
        return cmd;
    }

    public void FreeCommandBuffer(IntPtr cmd)
    {
        if (cmd == IntPtr.Zero) return;
        IntPtr* p = stackalloc IntPtr[1] { cmd };
        Vk.vkFreeCommandBuffers(Device, _cmdPool, 1, p);
    }

    public IntPtr NewFence(bool signaled = false)
    {
        Vk.VkFenceCreateInfo ci = new()
        {
            SType = VkConst.StFenceCreateInfo,
            Flags = signaled ? VkConst.FenceSignaled : 0u,
        };
        Vk.Check(Vk.vkCreateFence(Device, &ci, null, out IntPtr fence), "vkCreateFence");
        return fence;
    }

    public void DestroyFence(IntPtr fence)
    {
        if (fence == IntPtr.Zero) return;
        Vk.vkDestroyFence(Device, fence, null);
    }

    /// <summary>提交一个命令缓冲并等待完成。单次提交 + fence，P1 阶段够用；P1-9 再改环形提交。</summary>
    public void SubmitAndWait(IntPtr cmd)
    {
        IntPtr fence = NewFence();
        try
        {
            Submit(cmd, fence);
            WaitFence(fence);
        }
        finally
        {
            DestroyFence(fence);
        }
    }

    public void Submit(IntPtr cmd, IntPtr fence)
    {
        if (Lost) throw new VkDeviceLostException("vkQueueSubmit");
        // vkQueueSubmit 要求对同一 queue 的外部同步
        lock (QueueLock)
        {
            IntPtr* cmds = stackalloc IntPtr[1] { cmd };
            Vk.VkSubmitInfo si = new()
            {
                SType = VkConst.StSubmitInfo,
                CommandBufferCount = 1,
                PCommandBuffers = cmds,
            };
            VkResult r = Vk.vkQueueSubmit(Queue, 1, &si, fence);
            if (r == VkResult.ErrorDeviceLost) MarkLost();
            Vk.Check(r, "vkQueueSubmit");
        }
    }

    public void WaitFence(IntPtr fence)
    {
        if (Lost) throw new VkDeviceLostException("vkWaitForFences");

        // 有预算的自旋：够快的时候连一次线程挂起/唤醒都不用付
        long budget = SpinWaitUs * (Stopwatch.Frequency / 1_000_000);
        if (budget > 0)
        {
            long start = Stopwatch.GetTimestamp();
            while (Stopwatch.GetTimestamp() - start <= budget)
            {
                VkResult sr = Vk.vkGetFenceStatus(Device, fence);
                if (sr == VkResult.Success) return;
                // NotReady 是**成功码**（正值 = 还在跑），不是错误；只有负值才是失败
                if (sr == VkResult.NotReady) continue;
                if (sr == VkResult.ErrorDeviceLost) MarkLost();
                Vk.Check(sr, "vkGetFenceStatus");
            }
        }

        // 100 ms 一档，避免设备挂死时无限等待
        const ulong Slice = 100_000_000;
        const int MaxSlices = 100; // 10 s
        for (int i = 0; i < MaxSlices; i++)
        {
            VkResult r = Vk.vkWaitForFences(Device, 1, &fence, 1, Slice);
            if (r == VkResult.Success) return;
            if (r == VkResult.Timeout) continue;
            if (r == VkResult.ErrorDeviceLost) MarkLost();
            Vk.Check(r, "vkWaitForFences");
        }
        throw new TimeoutException("Vulkan: fence wait timed out after 10 s");
    }

    // ---------------------------------------------------------------- shader / pipeline

    public IntPtr NewShaderModule(byte[] spirv)
    {
        if ((spirv.Length & 3) != 0) throw new ArgumentException("SPIR-V size must be a multiple of 4");
        fixed (byte* p = spirv)
        {
            Vk.VkShaderModuleCreateInfo ci = new()
            {
                SType = VkConst.StShaderModuleCreateInfo,
                CodeSize = (nuint)spirv.Length,
                PCode = (uint*)p,
            };
            Vk.Check(Vk.vkCreateShaderModule(Device, &ci, null, out IntPtr module), "vkCreateShaderModule");
            return module;
        }
    }

    public void DestroyShaderModule(IntPtr module)
    {
        if (module == IntPtr.Zero) return;
        Vk.vkDestroyShaderModule(Device, module, null);
    }

    /// <summary>一个 set layout：binding 0..<paramref name="bindings"/>-1 全是 storage buffer，
    /// 外加一段 compute push constant。pushLayout=true 时 set 走 PUSH_DESCRIPTOR。</summary>
    public VkPipeline NewPipeline(IntPtr shaderModule, int bindings, int pushConstBytes, bool pushLayout)
    {
        var dslb = stackalloc Vk.VkDescriptorSetLayoutBinding[bindings];
        for (int i = 0; i < bindings; i++)
        {
            dslb[i] = new Vk.VkDescriptorSetLayoutBinding
            {
                Binding = (uint)i,
                DescriptorType = VkConst.DescStorageBuffer,
                DescriptorCount = 1,
                StageFlags = VkConst.StageComputeShader,
            };
        }
        Vk.VkDescriptorSetLayoutCreateInfo dslci = new()
        {
            SType = VkConst.StDescriptorSetLayoutCreateInfo,
            BindingCount = (uint)bindings,
            PBindings = dslb,
            Flags = pushLayout ? 1u : 0u,
        };
        Vk.Check(Vk.vkCreateDescriptorSetLayout(Device, &dslci, null, out IntPtr setLayout),
            "vkCreateDescriptorSetLayout");

        Vk.VkPushConstantRange pcr = new()
        {
            StageFlags = VkConst.StageComputeShader,
            Offset = 0,
            Size = (uint)pushConstBytes,
        };
        Vk.VkPipelineLayoutCreateInfo plci = new()
        {
            SType = VkConst.StPipelineLayoutCreateInfo,
            SetLayoutCount = 1,
            PSetLayouts = &setLayout,
            PushConstantRangeCount = pushConstBytes > 0 ? 1u : 0u,
            PPushConstantRanges = pushConstBytes > 0 ? &pcr : null,
        };
        Vk.Check(Vk.vkCreatePipelineLayout(Device, &plci, null, out IntPtr layout), "vkCreatePipelineLayout");

        byte* entry = stackalloc byte[5] { (byte)'m', (byte)'a', (byte)'i', (byte)'n', 0 };
        Vk.VkComputePipelineCreateInfo cpci = new()
        {
            SType = VkConst.StComputePipelineCreateInfo,
            Stage = new Vk.VkPipelineShaderStageCreateInfo
            {
                SType = VkConst.StPipelineShaderStageCreateInfo,
                Stage = VkConst.StageComputeShader,
                Module = shaderModule,
                PName = entry,
            },
            Layout = layout,
        };
        IntPtr pipeline;
        Vk.Check(Vk.vkCreateComputePipelines(Device, IntPtr.Zero, 1, &cpci, null, &pipeline),
            "vkCreateComputePipelines");

        return new VkPipeline
        {
            Pipeline = pipeline,
            Layout = layout,
            SetLayout = setLayout,
            PushLayout = pushLayout,
            Bindings = bindings,
            PushConstBytes = pushConstBytes,
        };
    }

    /// <summary>vkCmdPipelineBarrier，单个全局内存屏障。</summary>
    public void CmdBarrier(IntPtr cmd, uint srcStage, uint dstStage, uint srcAccess, uint dstAccess)
    {
        Vk.VkMemoryBarrier b = new()
        {
            SType = VkConst.StMemoryBarrier,
            SrcAccessMask = srcAccess,
            DstAccessMask = dstAccess,
        };
        Vk.vkCmdPipelineBarrier(cmd, srcStage, dstStage, 0, 1, &b, 0, null, 0, null);
    }

    /// <summary>compute 写 → 后续 compute 读 / transfer 读。</summary>
    public void CmdComputeBarrier(IntPtr cmd)
        => CmdBarrier(cmd,
            VkConst.PipelineStageComputeShader,
            VkConst.PipelineStageComputeShader | VkConst.PipelineStageTransfer,
            VkConst.AccessShaderWrite,
            VkConst.AccessShaderRead | VkConst.AccessTransferRead);

    /// <summary>transfer 写 → compute 读。staging 拷入后紧接 dispatch 时必须显式加，
    /// 光靠「同队列提交顺序」只保证执行顺序，不保证内存可见性。</summary>
    public void CmdTransferToComputeBarrier(IntPtr cmd)
        => CmdBarrier(cmd,
            VkConst.PipelineStageTransfer,
            VkConst.PipelineStageComputeShader,
            VkConst.AccessTransferWrite,
            VkConst.AccessShaderRead);

    // ---------------------------------------------------------------- 描述符 / 派发

    private IntPtr _descPool;

    /// <summary>惰性创建共享 descriptor pool。P1 按 64 set × 8 binding 预留；
    /// P1-6 的 arena 会改成按段复用，避免每次 dispatch 都新建 set。</summary>
    private IntPtr DescriptorPool()
    {
        if (_descPool != IntPtr.Zero) return _descPool;
        Vk.VkDescriptorPoolSize size = new()
        {
            Type = VkConst.DescStorageBuffer,
            DescriptorCount = 64 * 8,
        };
        Vk.VkDescriptorPoolCreateInfo ci = new()
        {
            SType = VkConst.StDescriptorPoolCreateInfo,
            MaxSets = 64,
            PoolSizeCount = 1,
            PPoolSizes = &size,
        };
        Vk.Check(Vk.vkCreateDescriptorPool(Device, &ci, null, out _descPool), "vkCreateDescriptorPool");
        return _descPool;
    }

    public IntPtr NewDescriptorSet(VkPipeline pipe)
    {
        if (Lost) throw new VkDeviceLostException("NewDescriptorSet");
        IntPtr pool = DescriptorPool();
        IntPtr layout = pipe.SetLayout;
        Vk.VkDescriptorSetAllocateInfo ai = new()
        {
            SType = VkConst.StDescriptorSetAllocateInfo,
            DescriptorPool = pool,
            DescriptorSetCount = 1,
            PSetLayouts = &layout,
        };
        IntPtr set;
        Vk.Check(Vk.vkAllocateDescriptorSets(Device, &ai, &set), "vkAllocateDescriptorSets");
        return set;
    }

    /// <summary>
    /// storage buffer 绑定的最小偏移对齐。Vulkan 保证 minStorageBufferOffsetAlignment ≤ 256，
    /// 直接用上限即可（不必解析 VkPhysicalDeviceLimits 的魔数偏移）。
    /// </summary>
    public const ulong DescriptorOffsetAlign = 256;

    /// <summary>一次批量拷贝的 region 上限。VkBufferCopy 24 B，1024 个约 24 KB 栈空间。</summary>
    private const int MaxCopyRegions = 1024;

    /// <summary>一个 storage buffer 绑定切片：哪个 buffer 的哪一段。</summary>
    public readonly struct BufferRange
    {
        public readonly VkBuffer Buffer;
        public readonly ulong Offset;
        public readonly ulong Size;

        public BufferRange(VkBuffer buffer, ulong offset, ulong size)
        {
            Buffer = buffer;
            Offset = offset;
            Size = size;
        }
    }

    /// <summary>descriptor 的 range 不能超过 maxStorageBufferRange（两块卡都是 4 GB），
    /// 大 buffer 靠 offset/range 切片，不是靠单个 descriptor 覆盖全部。</summary>
    private ulong RangeFor(VkBuffer buffer, ulong range)
        => Math.Min(range == 0 ? buffer.Size : range, MaxStorageRange);

    /// <summary>把 buffer 绑到 set 的某个 binding。<paramref name="range"/>=0 表示整个 buffer。</summary>
    public void UpdateStorageBuffer(IntPtr set, uint binding, VkBuffer buffer, ulong offset = 0, ulong range = 0)
    {
        Vk.VkDescriptorBufferInfo info = new()
        {
            Buffer = buffer.Buffer,
            Offset = offset,
            Range = RangeFor(buffer, range),
        };
        Vk.VkWriteDescriptorSet write = new()
        {
            SType = VkConst.StWriteDescriptorSet,
            DstSet = set,
            DstBinding = binding,
            DescriptorCount = 1,
            DescriptorType = VkConst.DescStorageBuffer,
            PBufferInfo = &info,
        };
        Vk.vkUpdateDescriptorSets(Device, 1, &write, 0, null);
    }

    /// <summary>只绑定 pipeline（后续自己绑 descriptor，或用 push descriptor）。</summary>
    public void CmdBindPipelineOnly(IntPtr cmd, VkPipeline pipe)
        => Vk.vkCmdBindPipeline(cmd, VkConst.BindPointCompute, pipe.Pipeline);

    public void CmdBindDescriptorSet(IntPtr cmd, VkPipeline pipe, IntPtr set)
    {
        IntPtr* sets = stackalloc IntPtr[1] { set };
        Vk.vkCmdBindDescriptorSets(cmd, VkConst.BindPointCompute, pipe.Layout, 0, 1, sets, 0, null);
    }

    /// <summary>bind pipeline + 一个 descriptor set（普通路径）。</summary>
    public void CmdBindPipeline(IntPtr cmd, VkPipeline pipe, IntPtr set)
    {
        CmdBindPipelineOnly(cmd, pipe);
        CmdBindDescriptorSet(cmd, pipe, set);
    }

    /// <summary>
    /// VK_KHR_push_descriptor：binding i 顺序绑定 <paramref name="buffers"/>。
    /// 不分配 descriptor set、不调 vkUpdateDescriptorSets，全部内联进命令流——
    /// 这是 Vulkan 侧对 D3D12 root constant/descriptor table 的对应手段。
    /// 前提：set layout 必须以 PUSH_DESCRIPTOR 标志创建（NewPipeline 的 pushLayout=true）。
    /// </summary>
    public void CmdPushDescriptorSet(IntPtr cmd, VkPipeline pipe, ReadOnlySpan<VkBuffer> buffers)
    {
        CheckPush(pipe, buffers.Length);
        int n = buffers.Length;
        Vk.VkDescriptorBufferInfo* infos = stackalloc Vk.VkDescriptorBufferInfo[n];
        for (int i = 0; i < n; i++)
        {
            infos[i] = new Vk.VkDescriptorBufferInfo
            {
                Buffer = buffers[i].Buffer,
                Offset = buffers[i].Offset,
                Range = RangeFor(buffers[i], 0),
            };
        }
        EmitPushDescriptorSet(cmd, pipe, infos, n);
    }

    /// <summary>
    /// 同一个 arena buffer 的多个区间分别绑到 binding 0..N-1（P1-6 的主路径）。
    /// 所有算子共用一个物理 buffer，绑定差异只体现在 offset/range 上——
    /// 这既是 descriptor 数量最小化，也是「一次提交装 N 个 dispatch」的前提。
    /// </summary>
    public void CmdPushDescriptorSetSlices(IntPtr cmd, VkPipeline pipe, VkBuffer arena,
        ReadOnlySpan<ulong> offsets, ReadOnlySpan<ulong> sizes)
    {
        if (sizes.Length != offsets.Length)
            throw new ArgumentException($"offsets({offsets.Length}) 与 sizes({sizes.Length}) 长度不一致");

        int n = offsets.Length;
        var ranges = new BufferRange[n];
        for (int i = 0; i < n; i++)
            ranges[i] = new BufferRange(arena, offsets[i], sizes[i]);
        CmdPushDescriptorSetRanges(cmd, pipe, ranges);
    }

    /// <summary>
    /// 每个 binding 指向各自的 (buffer, offset, size)——P1-7 的主力路径：
    /// 激活读写在段 arena 上，权重常驻在独立的 persistent buffer 上，两者必须能混绑。
    /// </summary>
    public void CmdPushDescriptorSetRanges(IntPtr cmd, VkPipeline pipe, ReadOnlySpan<BufferRange> ranges)
    {
        CheckPush(pipe, ranges.Length);
        int n = ranges.Length;
        Vk.VkDescriptorBufferInfo* infos = stackalloc Vk.VkDescriptorBufferInfo[n];
        for (int i = 0; i < n; i++)
        {
            BufferRange r = ranges[i];
            if (r.Size == 0)
                throw new ArgumentException($"binding {i} 的 range 为 0", nameof(ranges));
            if (r.Offset + r.Size > r.Buffer.Size)
                throw new ArgumentOutOfRangeException(nameof(ranges),
                    $"binding {i} 越界：offset {r.Offset} + size {r.Size} > {r.Buffer.Size}");
            // 偏移不满足 minStorageBufferOffsetAlignment 时，驱动不会报错，只会读错位置
            if ((r.Offset & (DescriptorOffsetAlign - 1)) != 0)
                throw new ArgumentException(
                    $"binding {i} 的 offset {r.Offset} 未按 {DescriptorOffsetAlign} B 对齐");

            infos[i] = new Vk.VkDescriptorBufferInfo
            {
                Buffer = r.Buffer.Buffer,
                Offset = r.Offset,
                Range = Math.Min(r.Size, MaxStorageRange),
            };
        }
        EmitPushDescriptorSet(cmd, pipe, infos, n);
    }

    private void CheckPush(VkPipeline pipe, int count)
    {
        if (_cmdPushDescriptorSet == null)
            throw new PlatformNotSupportedException("Vulkan: 设备不支持 VK_KHR_push_descriptor");
        if (!pipe.PushLayout)
            throw new InvalidOperationException(
                $"pipeline \"{pipe.Name}\" 的 set layout 不是 push-descriptor 布局（创建时 pushLayout=false）");
        if (count != pipe.Bindings)
            throw new ArgumentException($"pipeline 声明 {pipe.Bindings} 个 binding，实际 {count} 个");
    }

    private void EmitPushDescriptorSet(IntPtr cmd, VkPipeline pipe, Vk.VkDescriptorBufferInfo* infos, int n)
    {
        Vk.VkWriteDescriptorSet* writes = stackalloc Vk.VkWriteDescriptorSet[n];
        for (int i = 0; i < n; i++)
        {
            writes[i] = new Vk.VkWriteDescriptorSet
            {
                SType = VkConst.StWriteDescriptorSet,
                DstBinding = (uint)i,
                DescriptorCount = 1,
                DescriptorType = VkConst.DescStorageBuffer,
                PBufferInfo = &infos[i],
            };
        }
        _cmdPushDescriptorSet(cmd, VkConst.BindPointCompute, pipe.Layout, 0, (uint)n, writes);
    }

    public void DestroyPipeline(VkPipeline pipe)
    {
        if (pipe is null || Device == IntPtr.Zero) return;
        if (pipe.Pipeline != IntPtr.Zero) Vk.vkDestroyPipeline(Device, pipe.Pipeline, null);
        if (pipe.Layout != IntPtr.Zero) Vk.vkDestroyPipelineLayout(Device, pipe.Layout, null);
        if (pipe.SetLayout != IntPtr.Zero) Vk.vkDestroyDescriptorSetLayout(Device, pipe.SetLayout, null);
        pipe.Pipeline = pipe.Layout = pipe.SetLayout = IntPtr.Zero;
    }

    public void CmdPushConstants(IntPtr cmd, VkPipeline pipe, void* data, uint size)
    {
        // 大小对不上时驱动只会给出含糊的结果或静默截断，这里直接拦下
        if (size != (uint)pipe.PushConstBytes)
            throw new ArgumentException(
                $"push constant 大小不匹配：pipeline 声明 {pipe.PushConstBytes} B，传入 {size} B");
        Vk.vkCmdPushConstants(cmd, pipe.Layout, VkConst.StageComputeShader, 0, size, data);
    }

    public void CmdDispatch(IntPtr cmd, uint groupCountX, uint groupCountY = 1, uint groupCountZ = 1)
        => Vk.vkCmdDispatch(cmd, groupCountX, groupCountY, groupCountZ);

    // ---------------------------------------------------------------- 销毁

    public void Dispose()
    {
        if (Device != IntPtr.Zero)
        {
            // 模板持有 pipeline/layout/module，必须在 vkDestroyDevice 之前释放
            foreach (VkPipelineTemplate t in Templates.Values)
            {
                try { t.Dispose(); } catch { /* 设备已经在回收路径上，不让单个模板打断整体清理 */ }
            }
            Templates.Clear();

            if (_descPool != IntPtr.Zero) { Vk.vkDestroyDescriptorPool(Device, _descPool, null); _descPool = IntPtr.Zero; }
            Pool.Dispose(); // 池里驻留的块也要在 vkDestroyDevice 之前释放
            _stagingUp?.Dispose();
            _stagingDown?.Dispose();
            _stagingUp = _stagingDown = null;
            if (_cmdPool != IntPtr.Zero) { Vk.vkDestroyCommandPool(Device, _cmdPool, null); _cmdPool = IntPtr.Zero; }
            Vk.vkDestroyDevice(Device, null);
            Device = IntPtr.Zero;
            Queue = IntPtr.Zero;
        }
        if (Instance != IntPtr.Zero)
        {
            Vk.vkDestroyInstance(Instance, null);
            Instance = IntPtr.Zero;
        }
    }
}

/// <summary>设备内存块。Offset 预留给 P1-6 的 arena 子分配（同一 VkDeviceMemory 内的切片）。</summary>
internal unsafe sealed class VkBuffer : IDisposable
{
    public IntPtr Dev, Buffer, Memory;
    public ulong Size, AllocSize;
    /// <summary>在 VkDeviceMemory 内的起始偏移；P1-6 的 arena 子分配会写它，单 buffer 分配恒为 0。</summary>
    public ulong Offset { get; set; }
    public uint Flags;

    /// <summary>归还时交给谁（null = 直接销毁）。</summary>
    internal VkBufferPool? Owner;
    /// <summary>创建时指定的用途，池按它分桶。</summary>
    internal VkDevice.MemoryKind Kind;
    /// <summary>是否允许进池。权重常驻区显式关掉：它活得比池长，复用会让旧切片静默指向新数据。</summary>
    internal bool Pooled;

    /// <summary>Dispose 之后句柄已归池或已销毁，再用就是 bug，必须炸出来而不是读写别人的数据。</summary>
    public bool Released { get; private set; }

    public bool IsHostVisible => (Flags & VkConst.MemHostVisible) != 0;
    public bool HostCoherent => (Flags & VkConst.MemHostCoherent) != 0;

    public void* Map()
    {
        ThrowIfReleased();
        void* p;
        Vk.Check(Vk.vkMapMemory(Dev, Memory, 0, Vk.WholeSize, 0, &p), "vkMapMemory");
        return p;
    }

    public void Unmap() => Vk.vkUnmapMemory(Dev, Memory);

    /// <summary>nonCoherentAtomSize 的保守上限。实际驱动从未超过 256，按它对扩范围不至于违约。</summary>
    private const ulong AtomAlign = VkDevice.DescriptorOffsetAlign;

    /// <summary>
    /// vkFlushMappedMemoryRanges 要求 offset/size 按 nonCoherentAtomSize 对齐，否则是未定义行为
    /// （写入可能丢一部分，而且不报错）。这里把范围向外扩到对齐边界。
    /// </summary>
    private (ulong Offset, ulong Size) AtomRange(ulong offset, ulong size)
    {
        ulong begin = offset & ~(AtomAlign - 1);
        ulong end = (offset + size + AtomAlign - 1) & ~(AtomAlign - 1);
        if (end > AllocSize) end = AllocSize;
        if (begin == 0 && end == AllocSize) return (0, Vk.WholeSize);
        return (begin, end - begin);
    }

    public void Flush(ulong offset, ulong size)
    {
        ThrowIfReleased();
        if (HostCoherent) return;
        (ulong begin, ulong length) = AtomRange(offset, size);
        Vk.VkMappedMemoryRange r = new()
        {
            SType = VkConst.StMappedMemoryRange,
            Memory = Memory,
            Offset = begin,
            Size = length,
        };
        Vk.Check(Vk.vkFlushMappedMemoryRanges(Dev, 1, &r), "vkFlushMappedMemoryRanges");
    }

    public void Invalidate(ulong offset, ulong size)
    {
        ThrowIfReleased();
        if (HostCoherent) return;
        (ulong begin, ulong length) = AtomRange(offset, size);
        Vk.VkMappedMemoryRange r = new()
        {
            SType = VkConst.StMappedMemoryRange,
            Memory = Memory,
            Offset = begin,
            Size = length,
        };
        Vk.Check(Vk.vkInvalidateMappedMemoryRanges(Dev, 1, &r), "vkInvalidateMappedMemoryRanges");
    }

    private void ThrowIfReleased()
    {
        if (Released) throw new ObjectDisposedException(nameof(VkBuffer), "buffer 已归还/销毁，句柄不再有效");
    }

    public void Dispose()
    {
        if (Released) return;
        Released = true;

        // 进池：句柄搬进空闲表，本对象不再持有它们（再被用到会抛 ObjectDisposedException）
        if (Pooled && Owner is not null && Owner.Return(this))
        {
            Buffer = IntPtr.Zero;
            Memory = IntPtr.Zero;
            return;
        }

        if (Buffer != IntPtr.Zero) { Vk.vkDestroyBuffer(Dev, Buffer, null); Buffer = IntPtr.Zero; }
        if (Memory != IntPtr.Zero) { Vk.vkFreeMemory(Dev, Memory, null); Memory = IntPtr.Zero; }
    }
}

internal sealed class VkPipeline
{
    public IntPtr Pipeline, Layout, SetLayout;
    public bool PushLayout;
    public int Bindings;
    public int PushConstBytes;
    public string Name = "";
}
