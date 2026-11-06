using OnnxSharp.Core;
using OnnxSharp.Tensors;

namespace OnnxSharp.Tests;

internal static class EngineTests
{
    public static IEnumerable<TestCase> Cases()
    {
        yield return new("engine: Add + Relu smoke", AddReluSmoke);
        yield return new("engine: const input drives output shape", ConstInputDrivesShape);
        yield return new("engine: temp pool reuses the same tensor", TempPoolReusesTensor);
        yield return new("engine: dynamic op fits its own output", DynamicOpFitsItself);
        yield return new("engine: shared slots are stable across forwards", SlotsStableAcrossForwards);
        yield return new("engine: graph append invalidates buffers", GraphAppendInvalidatesBuffers);
        yield return new("engine: GetLayouts defaults to NCHW", GetLayoutsDefaultsToNchw);
    }

    private static void AddReluSmoke()
    {
        var impl = new NetImpl();
        var x = impl.Declare("x", ArgKind.Input);
        var graph = impl.NewGraph("main", new[] { x }, isMain: true);

        var bias = impl.NewConstArg("bias", Tensor.FromArray(new float[] { 0.5f, 0.5f, -1f }, new TensorShape(3)));

        var add = LayerRegistry.Create("Add", new LayerParams { Type = "Add", Name = "add" });
        add.Inputs.Add(x);
        add.Inputs.Add(bias);
        var y = impl.Declare("y", ArgKind.Temp);
        add.Outputs.Add(y);
        graph.Append(add);

        var relu = LayerRegistry.Create("Relu", new LayerParams { Type = "Relu", Name = "relu" });
        relu.Inputs.Add(y);
        var z = impl.Declare("z", ArgKind.Temp);
        relu.Outputs.Add(z);
        graph.Append(relu);

        graph.SetOutputs(new[] { z });
        impl.SetInput("x", Tensor.FromArray(new float[] { 1, -2, 3, 4, -5, 6 }, new TensorShape(2, 3)));

        var result = impl.Forward();
        float[] expected = { 1.5f, 0f, 2f, 4.5f, 0f, 5f };
        var actual = result.AsSpan<float>().ToArray();

        Assert.Equal(6, actual.Length, "element count");
        for (int i = 0; i < expected.Length; i++)
            Assert.Near(expected[i], actual[i], 1e-6f, $"value[{i}]");
    }

    private static void ConstInputDrivesShape()
    {
        var impl = new NetImpl();
        var x = impl.Declare("x", ArgKind.Input);
        var graph = impl.NewGraph("g", new[] { x }, isMain: true);

        var n = impl.NewConstArg("n", Tensor.FromInt64(new long[] { 4 }));

        var layer = new ReadConstShapeLayer { Name = "rcs", Type = "ReadConstShape" };
        layer.Inputs.Add(x);
        layer.Inputs.Add(n);
        var y = impl.Declare("y", ArgKind.Temp);
        layer.Outputs.Add(y);
        graph.Append(layer);
        graph.SetOutputs(new[] { y });

        impl.SetInput("x", Tensor.FromArray(new float[] { 1, 2, 3, 4, 5, 6 }, new TensorShape(2, 3)));

        var result = impl.Forward();
        Assert.Equal(4, result.Shape.Rank == 1 ? result.Shape[0] : -1, "shape from const input");
        Assert.Equal(new TensorShape(4), result.Shape, "shape");

        var values = result.AsSpan<float>().ToArray();
        for (int i = 0; i < 4; i++)
            Assert.Near(i + 1, values[i], 1e-6f, $"value[{i}]");
    }

    private static void TempPoolReusesTensor()
    {
        var impl = new NetImpl();
        var x = impl.Declare("x", ArgKind.Input);
        var graph = impl.NewGraph("g", new[] { x }, isMain: true);

        var n = impl.NewConstArg("n", Tensor.FromInt64(new long[] { 2 }));
        var layer = new ReadConstShapeLayer { Name = "rcs", Type = "ReadConstShape" };
        layer.Inputs.Add(x);
        layer.Inputs.Add(n);
        var y = impl.Declare("y", ArgKind.Temp);
        layer.Outputs.Add(y);
        graph.Append(layer);
        graph.SetOutputs(new[] { y });

        impl.SetInput("x", Tensor.FromArray(new float[] { 1, 2 }, new TensorShape(2)));

        impl.Forward();
        var firstTemp = layer.LastTemp;
        Assert.True(firstTemp is not null, "layer must receive a temp tensor");

        impl.Forward();
        Assert.Same(firstTemp, layer.LastTemp, "temp tensor object must be pooled across forwards");
        Assert.Equal(1, impl.TempCount, "only one temp object should exist");
    }

    private static void DynamicOpFitsItself()
    {
        var impl = new NetImpl();
        var graph = impl.NewGraph("g", Array.Empty<Arg>(), isMain: true);

        var n = impl.NewConstArg("n", Tensor.FromInt64(new long[] { 3 }));

        var dyn = new DynamicFillLayer(7f) { Name = "dyn", Type = "Dyn" };
        dyn.Inputs.Add(n);
        var d = impl.Declare("d", ArgKind.Temp);
        dyn.Outputs.Add(d);
        graph.Append(dyn);

        var relu = LayerRegistry.Create("Relu", new LayerParams { Type = "Relu", Name = "relu" });
        relu.Inputs.Add(d);
        var o = impl.Declare("o", ArgKind.Temp);
        relu.Outputs.Add(o);
        graph.Append(relu);

        graph.SetOutputs(new[] { o });

        var result = impl.Forward();
        Assert.Equal(new TensorShape(3), result.Shape, "downstream sees the dynamic shape");

        var values = result.AsSpan<float>().ToArray();
        foreach (var v in values)
            Assert.Near(7f, v, 1e-6f, "value");
    }

    private static void SlotsStableAcrossForwards()
    {
        var impl = new NetImpl();
        var x = impl.Declare("x", ArgKind.Input);
        var graph = impl.NewGraph("main", new[] { x }, isMain: true);

        var bias = impl.NewConstArg("bias", Tensor.FromArray(new float[] { 1f, 1f, 1f }, new TensorShape(3)));
        var add = LayerRegistry.Create("Add", new LayerParams { Type = "Add", Name = "add" });
        add.Inputs.Add(x);
        add.Inputs.Add(bias);
        var y = impl.Declare("y", ArgKind.Temp);
        add.Outputs.Add(y);
        graph.Append(add);
        graph.SetOutputs(new[] { y });

        impl.SetInput("x", Tensor.FromArray(new float[] { 1, 2, 3 }, new TensorShape(1, 3)));

        impl.Forward();
        int slotsAfterFirst = impl.SlotCount;

        impl.Forward();
        Assert.Equal(slotsAfterFirst, impl.SlotCount, "slot count must not grow across forwards");

        var values = impl.Forward().AsSpan<float>().ToArray();
        float[] expected = { 2f, 3f, 4f };
        for (int i = 0; i < expected.Length; i++)
            Assert.Near(expected[i], values[i], 1e-6f, $"value[{i}]");
    }

    private static void GraphAppendInvalidatesBuffers()
    {
        var impl = new NetImpl();
        var x = impl.Declare("x", ArgKind.Input);
        var graph = impl.NewGraph("main", new[] { x }, isMain: true);

        var neg = LayerRegistry.Create("Neg", new LayerParams { Type = "Neg", Name = "neg" });
        neg.Inputs.Add(x);
        var n = impl.Declare("n", ArgKind.Temp);
        neg.Outputs.Add(n);
        graph.Append(neg);
        graph.SetOutputs(new[] { n });

        impl.SetInput("x", Tensor.FromArray(new float[] { 1f, -2f }, new TensorShape(2)));
        var first = impl.Forward().AsSpan<float>().ToArray();
        Assert.Near(-1f, first[0], 1e-6f, "neg");

        int versionBefore = graph.Version;

        // 追加一个 Relu，图结构变化 → 缓冲槽需要重算。
        var relu = LayerRegistry.Create("Relu", new LayerParams { Type = "Relu", Name = "relu" });
        relu.Inputs.Add(n);
        var r = impl.Declare("r", ArgKind.Temp);
        relu.Outputs.Add(r);
        graph.Append(relu);
        graph.SetOutputs(new[] { r });

        Assert.True(graph.Version > versionBefore, "graph version must advance");

        var second = impl.Forward().AsSpan<float>().ToArray();
        Assert.Near(0f, second[0], 1e-6f, "relu(neg(x))[0]");
        Assert.Near(2f, second[1], 1e-6f, "relu(neg(x))[1]");
    }

    private static void GetLayoutsDefaultsToNchw()
    {
        var layer = LayerRegistry.Create("Relu", new LayerParams { Type = "Relu", Name = "relu" });
        layer.Outputs.Add(new Arg(1));

        var outputs = new List<Layout>();
        layer.GetLayouts(new[] { Layout.NCHW }, outputs);

        Assert.Equal(1, outputs.Count, "one layout per output");
        Assert.Equal(Layout.NCHW, outputs[0], "default layout");
    }
}
