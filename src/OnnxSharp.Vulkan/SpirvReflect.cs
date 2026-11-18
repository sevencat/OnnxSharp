// 极小的 SPIR-V 反射：只取 compute 模板真正需要的两样东西。
// 目的不是通用反射库，而是把「host 侧声明」和「shader 实际用法」不一致的错误在
// 创建 pipeline 时就暴露出来——这两类错驱动都不会报，只会静默少算或多读。
using System.Runtime.InteropServices;

namespace OnnxSharp.Vulkan;

internal readonly struct SpirvInfo
{
    /// <summary><c>local_size_x/y/z</c>；未声明 LocalSize 时为 0（compute 不可能合法，调用方应报错）。</summary>
    public readonly uint LocalSizeX, LocalSizeY, LocalSizeZ;

    /// <summary>shader 里出现的 descriptor set/binding 对，按 binding 排序。</summary>
    public readonly (uint Set, uint Binding)[] Descriptors;

    public SpirvInfo(uint x, uint y, uint z, (uint, uint)[] descriptors)
    {
        LocalSizeX = x;
        LocalSizeY = y;
        LocalSizeZ = z;
        Descriptors = descriptors;
    }

    public uint LocalSizeTotal => LocalSizeX * LocalSizeY * LocalSizeZ;
}

internal static class SpirvReflect
{
    private const uint Magic = 0x07230203;
    private const int OpExecutionMode = 16;
    private const int OpDecorate = 71;
    private const uint ExecutionModeLocalSize = 17;
    private const uint DecorationBinding = 33;
    private const uint DecorationDescriptorSet = 34;

    public static SpirvInfo Parse(byte[] spirv, string name)
    {
        if (spirv.Length < 20 || (spirv.Length & 3) != 0)
            throw new InvalidOperationException($"shader \"{name}\": SPIR-V 长度非法 ({spirv.Length} B)");

        ReadOnlySpan<uint> w = MemoryMarshal.Cast<byte, uint>(spirv);
        if (w[0] != Magic)
            throw new InvalidOperationException($"shader \"{name}\": 不是 SPIR-V");

        uint bound = w[3];
        if (bound == 0 || bound > 1 << 20)
            throw new InvalidOperationException($"shader \"{name}\": SPIR-V id bound 异常 ({bound})");

        // id → 装饰值；用 -1 表示「未出现」。bound 通常只有几百，直接开数组比字典快
        int[] bindings = new int[bound];
        int[] sets = new int[bound];
        Array.Fill(bindings, -1);
        Array.Fill(sets, -1);

        uint lx = 0, ly = 0, lz = 0;

        int i = 5; // 跳过 5 个 word 的 header
        while (i < w.Length)
        {
            uint first = w[i];
            int wc = (int)(first >> 16);
            int op = (int)(first & 0xFFFF);
            if (wc <= 0 || i + wc > w.Length)
                throw new InvalidOperationException($"shader \"{name}\": SPIR-V 指令流损坏 @ word {i}");

            if (op == OpExecutionMode && wc >= 6 && w[i + 2] == ExecutionModeLocalSize)
            {
                lx = w[i + 3];
                ly = w[i + 4];
                lz = w[i + 5];
            }
            else if (op == OpDecorate && wc >= 4)
            {
                uint id = w[i + 1];
                uint dec = w[i + 2];
                if (id < bound)
                {
                    if (dec == DecorationBinding) bindings[id] = (int)w[i + 3];
                    else if (dec == DecorationDescriptorSet) sets[id] = (int)w[i + 3];
                }
            }

            i += wc;
        }

        var list = new List<(uint, uint)>();
        for (uint id = 0; id < bound; id++)
        {
            int b = bindings[id];
            if (b < 0) continue;
            int s = sets[id];
            list.Add(((uint)(s < 0 ? 0 : s), (uint)b));
        }
        list.Sort((a, b) => a.Item2.CompareTo(b.Item2));

        return new SpirvInfo(lx, ly, lz, list.ToArray());
    }
}
