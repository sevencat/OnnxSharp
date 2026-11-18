// 加载随程序集嵌入的 .spv（由 build-shaders.ps1 生成并入库）。
// 不依赖文件系统、不依赖 Vulkan SDK：包发出去之后 shader 就在 dll 里。
using System.Reflection;

namespace OnnxSharp.Vulkan;

internal static class VkShaders
{
    private const string Prefix = "OnnxSharp.Vulkan.Shaders.";
    private const string Suffix = ".spv";
    private const uint SpirvMagic = 0x07230203;

    private static readonly object Gate = new();
    private static readonly Dictionary<string, byte[]> Cache = new(StringComparer.Ordinal);

    /// <summary>已嵌入的 shader 名（不含前缀与扩展名），用于报错时列出可选值。</summary>
    public static IReadOnlyList<string> Names
    {
        get
        {
            var list = new List<string>();
            foreach (string n in typeof(VkShaders).Assembly.GetManifestResourceNames())
            {
                string? name = StripName(n);
                if (name is not null) list.Add(name);
            }
            list.Sort(StringComparer.Ordinal);
            return list;
        }
    }

    /// <summary>按名字取 SPIR-V 字节。名字可带或不带 <c>.spv</c>。<c>internal</c>：调用方是 VkDevice。</summary>
    public static byte[] Load(string name)
    {
        ArgumentException.ThrowIfNullOrEmpty(name);

        string key = name.EndsWith(Suffix, StringComparison.OrdinalIgnoreCase)
            ? name[..^Suffix.Length]
            : name;

        lock (Gate)
        {
            if (Cache.TryGetValue(key, out byte[]? cached)) return cached;

            Assembly asm = typeof(VkShaders).Assembly;
            string resource = Prefix + key + Suffix;
            using Stream? s = asm.GetManifestResourceStream(resource)
                ?? throw new InvalidOperationException(
                    $"找不到嵌入 shader \"{key}\"。可用: [{string.Join(", ", Names)}]。" +
                    "改了 .comp 之后需要运行 src/OnnxSharp.Vulkan/build-shaders.ps1 重新生成 .spv。");

            byte[] buffer = new byte[s.Length];
            s.ReadExactly(buffer);
            Validate(key, buffer);
            Cache[key] = buffer;
            return buffer;
        }
    }

    private static void Validate(string key, byte[] spirv)
    {
        // 常见坏情况：忘了重新生成（旧 .spv 对不上新接口）、脚本被跳过导致空文件、
        // 或者把 .comp 文本误当资源嵌进去了。都在这里直接炸掉，而不是让驱动在
        // vkCreateShaderModule 里返回一个含糊的错误码。
        if (spirv.Length < 20 || (spirv.Length & 3) != 0)
            throw new InvalidOperationException($"shader \"{key}\" 的 SPIR-V 长度非法: {spirv.Length} B");

        uint magic = BitConverter.ToUInt32(spirv, 0);
        if (magic != SpirvMagic)
            throw new InvalidOperationException(
                $"shader \"{key}\" 不是 SPIR-V（magic=0x{magic:X8}）。检查 build-shaders.ps1 是否真的编译成功。");
    }

    private static string? StripName(string resourceName)
    {
        if (!resourceName.StartsWith(Prefix, StringComparison.Ordinal)) return null;
        if (!resourceName.EndsWith(Suffix, StringComparison.Ordinal)) return null;
        return resourceName[Prefix.Length..^Suffix.Length];
    }
}
