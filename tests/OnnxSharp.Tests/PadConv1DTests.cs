using OnnxSharp.Core;
using OnnxSharp.Tensors;

namespace OnnxSharp.Tests;

/// <summary>
/// M9 新算子：<c>Pad</c>（Paraformer 的 pad_fn）与 1-D <c>Conv</c>（FSMN 记忆块）。
/// 两者的正确性直接决定 encoder 段能不能用。
/// </summary>
internal static class PadConv1DTests
{
    public static IEnumerable<TestCase> Cases()
    {
        yield return new("pad: constant mode fills and copies", ConstantMode);
        yield return new("pad: negative pads crop", NegativePads);
        yield return new("pad: edge mode replicates borders", EdgeMode);
        yield return new("conv: 1-D depthwise over time", Conv1D);
    }

    private static float[] Run(LayerInfo layer, Tensor input, out TensorShape shape)
    {
        var impl = new NetImpl();
        var arg = impl.Declare("x", ArgKind.Input);
        var graph = impl.NewGraph("g", new[] { arg }, isMain: true);
        layer.Inputs.Add(arg);
        var output = impl.Declare("y", ArgKind.Temp);
        layer.Outputs.Add(output);
        graph.Append(layer);
        graph.SetOutputs(new[] { output });

        impl.SetInput("x", input);
        var result = impl.Forward();
        shape = result.Shape;
        return result.AsSpan<float>().ToArray();
    }

    private static void ConstantMode()
    {
        var parameters = new LayerParams { Type = "Pad", Name = "pad" };
        parameters.Set("pads", new[] { 0, 1, 0, 1 });
        parameters.Set("value", 9.0);

        var values = Run(
            LayerRegistry.Create("Pad", parameters),
            Tensor.FromArray([1f, 2f, 3f, 4f], new TensorShape(2, 2)),
            out var shape);

        Assert.Equal(4, shape.Dims[1], "columns");
        Assert.Equal(2, shape.Dims[0], "rows");

        // [[9,1,2,9],[9,3,4,9]]
        Assert.Equal(9f, values[0], "[0,0]");
        Assert.Equal(1f, values[1], "[0,1]");
        Assert.Equal(2f, values[2], "[0,2]");
        Assert.Equal(9f, values[3], "[0,3]");
        Assert.Equal(9f, values[4], "[1,0]");
        Assert.Equal(4f, values[6], "[1,2]");
        Assert.Equal(9f, values[7], "[1,3]");
    }

    private static void NegativePads()
    {
        var parameters = new LayerParams { Type = "Pad", Name = "pad" };
        parameters.Set("pads", new[] { -1, 0, 0, 0 });      // 砍掉第一行

        var values = Run(
            LayerRegistry.Create("Pad", parameters),
            Tensor.FromArray([1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f], new TensorShape(3, 3)),
            out var shape);

        Assert.Equal(2, shape.Dims[0], "rows");
        Assert.Equal(3, shape.Dims[1], "columns");
        Assert.Equal(4f, values[0], "[0,0]");
        Assert.Equal(9f, values[5], "[1,2]");
    }

    private static void EdgeMode()
    {
        var parameters = new LayerParams { Type = "Pad", Name = "pad" };
        parameters.Set("mode", "edge");
        parameters.Set("pads", new[] { 0, 2, 0, 2 });

        var values = Run(
            LayerRegistry.Create("Pad", parameters),
            Tensor.FromArray([1f, 2f, 3f], new TensorShape(1, 3)),
            out var shape);

        Assert.Equal(7, shape.Dims[1], "columns");
        Assert.Equal(1f, values[0], "[0,0]");
        Assert.Equal(1f, values[1], "[0,1]");
        Assert.Equal(1f, values[2], "[0,2]");
        Assert.Equal(2f, values[3], "[0,3]");
        Assert.Equal(3f, values[4], "[0,4]");
        Assert.Equal(3f, values[6], "[0,6]");
    }

    private static void Conv1D()
    {
        var parameters = new LayerParams { Type = "Conv", Name = "conv1d" };
        parameters.Set("kernel_shape", new[] { 2 });

        var weight = Tensor.FromArray([1f, 1f], new TensorShape(1, 1, 2));

        // Conv 需要 2 个输入：用一个 NetImpl 手动接上常量权重
        var impl = new NetImpl();
        var arg = impl.Declare("x", ArgKind.Input);
        var graph = impl.NewGraph("g", new[] { arg }, isMain: true);

        var conv = LayerRegistry.Create("Conv", parameters);
        conv.Inputs.Add(arg);
        conv.Inputs.Add(impl.NewConstArg("w", weight));

        var output = impl.Declare("y", ArgKind.Temp);
        conv.Outputs.Add(output);
        graph.Append(conv);
        graph.SetOutputs(new[] { output });

        impl.SetInput("x", Tensor.FromArray([1f, 2f, 3f, 4f], new TensorShape(1, 1, 4)));
        var result = impl.Forward();

        Assert.Equal(3, result.Shape.Dims[2], "length");
        var span = result.AsSpan<float>();
        Assert.Near(3f, span[0], 1e-6f, "[0]");
        Assert.Near(5f, span[1], 1e-6f, "[1]");
        Assert.Near(7f, span[2], 1e-6f, "[2]");
    }
}
