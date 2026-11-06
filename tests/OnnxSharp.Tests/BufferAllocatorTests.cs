using OnnxSharp.Core;
using OnnxSharp.Engine;

namespace OnnxSharp.Tests;

internal static class BufferAllocatorTests
{
    public static IEnumerable<TestCase> Cases()
    {
        yield return new("allocator: linear chain reuses 2 slots", LinearChainReusesSlots);
        yield return new("allocator: parallel branches get distinct slots", ParallelBranchesDistinct);
        yield return new("allocator: same-op read/write does not reuse (conservative)", SameOpReadWriteNotReused);
        yield return new("allocator: graph output stays live to the end", GraphOutputStaysLive);
        yield return new("allocator: never-written args have no slot", NeverWrittenHasNoSlot);
    }

    private static LayerInfo Op(string name, Arg[] inputs, params Arg[] outputs) => new()
    {
        Name = name,
        Type = "Test",
        Inputs = inputs.ToList(),
        Outputs = outputs.ToList(),
    };

    // op0 -> a, op1: a -> b, op2: b -> c, 输出 c
    private static void LinearChainReusesSlots()
    {
        var a = new Arg(1);
        var b = new Arg(2);
        var c = new Arg(3);

        var prog = new LayerInfo[]
        {
            Op("op0", Array.Empty<Arg>(), a),
            Op("op1", new[] { a }, b),
            Op("op2", new[] { b }, c),
        };

        var slotOf = BufferAllocator.Assign(prog, new[] { c }, 4, out int slotCount);

        Assert.Equal(2, slotCount, "chain should need 2 slots");
        Assert.NotEqual(BufferAllocator.NoSlot, slotOf[a.Idx], "a must be slotted");
        Assert.NotEqual(BufferAllocator.NoSlot, slotOf[b.Idx], "b must be slotted");
        Assert.Equal(slotOf[a.Idx], slotOf[c.Idx], "a and c have disjoint lifetimes -> share");
        Assert.NotEqual(slotOf[a.Idx], slotOf[b.Idx], "a and b overlap -> distinct");
    }

    private static void ParallelBranchesDistinct()
    {
        var a = new Arg(1);
        var b = new Arg(2);
        var c = new Arg(3);

        var prog = new LayerInfo[]
        {
            Op("op0", Array.Empty<Arg>(), a, b),
            Op("op1", new[] { a, b }, c),
        };

        var slotOf = BufferAllocator.Assign(prog, new[] { c }, 4, out int slotCount);

        Assert.NotEqual(slotOf[a.Idx], slotOf[b.Idx], "concurrent outputs must not share");
        Assert.NotEqual(BufferAllocator.NoSlot, slotOf[c.Idx], "c must be slotted");
        Assert.Equal(3, slotCount, "a/b live at op1 while c is written there -> c needs its own slot");
    }

    /// <summary>
    /// 记录分配器的保守策略：若前一个 Arg 的最后读取与后一个 Arg 的首次写入落在同一个 op 上，
    /// 不复用（避免任何「先写输出再读输入」的算子破坏输入）。代价是峰值多一代张量。
    /// </summary>
    private static void SameOpReadWriteNotReused()
    {
        var a = new Arg(1);
        var b = new Arg(2);

        var prog = new LayerInfo[]
        {
            Op("op0", Array.Empty<Arg>(), a),
            Op("op1", new[] { a }, b),
        };

        var slotOf = BufferAllocator.Assign(prog, new[] { b }, 3, out int slotCount);

        Assert.NotEqual(slotOf[a.Idx], slotOf[b.Idx], "touching lifetimes stay separate");
        Assert.Equal(2, slotCount, "conservative: no reuse at the touching index");
    }

    private static void GraphOutputStaysLive()
    {
        var a = new Arg(1);
        var b = new Arg(2);

        // a 是图输出；b 在 a 之后才写，不能被塞进 a 的槽。
        var prog = new LayerInfo[]
        {
            Op("op0", Array.Empty<Arg>(), a),
            Op("op1", Array.Empty<Arg>(), b),
        };

        var slotOf = BufferAllocator.Assign(prog, new[] { a }, 3, out int slotCount);

        Assert.Equal(2, slotCount, "graph output must not be reused by later writes");
        Assert.NotEqual(slotOf[a.Idx], slotOf[b.Idx], "output slot must stay dedicated");
    }

    private static void NeverWrittenHasNoSlot()
    {
        var input = new Arg(1);
        var constant = new Arg(2);
        var outArg = new Arg(3);

        var prog = new LayerInfo[]
        {
            Op("op0", new[] { input, constant }, outArg),
        };

        var slotOf = BufferAllocator.Assign(prog, new[] { outArg }, 4, out int slotCount);

        Assert.Equal(1, slotCount, "only the written arg needs a slot");
        Assert.Equal(BufferAllocator.NoSlot, slotOf[input.Idx], "input has no slot");
        Assert.Equal(BufferAllocator.NoSlot, slotOf[constant.Idx], "const has no slot");
        Assert.Equal(0, slotOf[outArg.Idx], "output gets slot 0");
    }
}
