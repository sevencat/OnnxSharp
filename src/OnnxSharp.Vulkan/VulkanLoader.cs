using System.Runtime.InteropServices;

namespace OnnxSharp.Vulkan;

/// <summary>
/// Vulkan loader 与物理设备探测。
/// <para>
/// 面向「部署环境可能根本没有 GPU / 没有 Vulkan loader」的场景（精简容器、老工控机），
/// 本类**不抛异常**，失败一律返回 <c>false</c>，调用方据此回退 CPU。
/// </para>
/// </summary>
public static unsafe class VulkanLoader
{
    /// <summary>
    /// 尝试加载 loader 并枚举物理设备。
    /// </summary>
    /// <param name="loaderName">实际加载到的库名（如 <c>vulkan-1.dll</c>），失败为 null。</param>
    /// <param name="devices">物理设备名；失败为空。</param>
    /// <returns>loader 可用且至少有一个物理设备时为 true。</returns>
    public static bool TryProbe(out string? loaderName, out IReadOnlyList<string> devices)
    {
        loaderName = null;
        devices = Array.Empty<string>();

        try
        {
            Vk.RegisterResolver();
            devices = EnumerateDeviceNames();
            // LoadedLibrary 由 resolver 在首次 P/Invoke 时写入，必须等枚举之后再读。
            loaderName = Vk.LoadedLibrary;
            return devices.Count > 0;
        }
        catch (Exception ex) when (ex is DllNotFoundException
                                      or EntryPointNotFoundException
                                      or BadImageFormatException
                                      or InvalidOperationException)
        {
            loaderName = null;
            devices = Array.Empty<string>();
            return false;
        }
    }

    private static List<string> EnumerateDeviceNames()
    {
        // "OnnxSharp" 不能直接 fixed（字符串是 UTF-16），先转 UTF-8 字节。
        byte[] appName = "OnnxSharp"u8.ToArray();

        IntPtr instance = IntPtr.Zero;
        try
        {
            fixed (byte* pAppName = appName)
            {
                var app = new Vk.VkApplicationInfo
                {
                    SType = VkConst.StApplicationInfo,
                    PAppName = pAppName,
                    ApiVersion = Vk.MakeVersion(1, 1, 0),
                };

                var ci = new Vk.VkInstanceCreateInfo
                {
                    SType = VkConst.StInstanceCreateInfo,
                    PAppInfo = &app,
                };

                Vk.Check(Vk.vkCreateInstance(&ci, null, out instance), "vkCreateInstance");
            }

            uint count = 0;
            Vk.Check(Vk.vkEnumeratePhysicalDevices(instance, &count, null), "vkEnumeratePhysicalDevices");
            if (count == 0)
                return new List<string>();

            IntPtr* handles = stackalloc IntPtr[(int)count];
            Vk.Check(Vk.vkEnumeratePhysicalDevices(instance, &count, handles), "vkEnumeratePhysicalDevices");

            var names = new List<string>((int)count);
            for (uint i = 0; i < count; i++)
            {
                Vk.VkPhysicalDeviceProperties props;
                Vk.vkGetPhysicalDeviceProperties(handles[i], &props);

                // props 是栈上局部，fixed buffer 已是固定地址，可直接取指针（fixed 语句会报 CS0213）。
                byte* pName = props.DeviceName;
                names.Add(Marshal.PtrToStringUTF8((IntPtr)pName) ?? string.Empty);
            }

            return names;
        }
        finally
        {
            if (instance != IntPtr.Zero)
                Vk.vkDestroyInstance(instance, null);
        }
    }
}
