using OnnxSharp.Core;
using OnnxSharp.Tensors;

namespace OnnxSharp.Tests;

/// <summary>
/// 0.6.0 的 <see cref="LayerPlan"/> 不变量：形状缓存与 Finalize 失效判定读**同一份**输入签名。
/// <para>
/// 这里守的是那个已经踩过的坑 —— 两套签名不同步时，动态输出形状的算子每帧都会判定
/// 「签名变了」，每帧重跑 <c>PrepackWeights + Finalize</c>（rfdetr-small 上 52 MB/帧）。
/// </para>
/// </summary>
internal static class LayerPlanTests
{
    public static IEnumerable<TestCase> Cases()
    {
        yield return new("layerplan: identical signature matches", MatchesIdentical);
        yield return new("layerplan: changed shape misses", RejectsChangedShape);
        yield return new("layerplan: changed dtype misses", RejectsChangedType);
        yield return new("layerplan: invalidate drops state but keeps lists", InvalidateKeepsLists);
        yield return new("engine: dynamic layer finalizes once per signature", DynamicFinalizesOnce);
        yield return new("engine: shape change re-infers and re-finalizes", ShapeChangeRefinalizes);
    }

    private static LayerPlan NewPlan()
    {
        var plan = new LayerPlan();
        plan.WriteInputs(new[] { new TensorShape(1, 3, 640, 640) }, new[] { DType.F32 });
        plan.State = PlanState.Shaped;
        return plan;
    }

    private static void MatchesIdentical()
    {
        var plan = NewPlan();
        Assert.True(
            plan.Matches(new[] { new TensorShape(1, 3, 640, 640) }, new[] { DType.F32 }),
            "same shape + same dtype must hit");
    }

    private static void RejectsChangedShape()
    {
        var plan = NewPlan();

        Assert.False(
            plan.Matches(new[] { new TensorShape(1, 3, 512, 512) }, new[] { DType.F32 }),
            "different dims must miss");

        Assert.False(
            plan.Matches(
                new[] { new TensorShape(1, 3, 640, 640), new TensorShape(1) },
                new[] { DType.F32, DType.F32 }),
            "extra input must miss");
    }

    private static void RejectsChangedType()
    {
        var plan = NewPlan();
        Assert.False(
            plan.Matches(new[] { new TensorShape(1, 3, 640, 640) }, new[] { DType.F16 }),
            "same shape but different dtype must miss");
    }

    private static void InvalidateKeepsLists()
    {
        var plan = NewPlan();
        plan.Invalidate();

        Assert.Equal(PlanState.None, plan.State, "state after invalidate");
        Assert.Equal(1, plan.InShapes.Count, "signature list keeps its content");
    }

    /// <summary>
    /// 动态形状层连续 3 帧：<c>Finalize</c> 必须只跑一次，且从不做形状推断。
    /// （0.5.x 这里是 3 次 —— 就是每帧白分配 52 MB 的那条路径。）
    /// </summary>
    private static void DynamicFinalizesOnce()
    {
        var (impl, layer) = Build(dynamicOutput: true);
        impl.SetInput("x", Tensor.FromArray(new float[] { 1, 2, 3, 4 }, new TensorShape(2, 2)));

        for (int frame = 0; frame < 3; frame++)
            Assert.Equal(new TensorShape(2, 2), impl.Forward().Shape, $"frame {frame} output shape");

        Assert.Equal(1, layer.FinalizeCalls, "dynamic layer must finalize once per signature");
        Assert.Equal(0, layer.ShapeCalls, "dynamic layer must not infer shapes");
    }

    /// <summary>静态层：第二帧走缓存，形状变化后必须重新推断并重新 Finalize。</summary>
    private static void ShapeChangeRefinalizes()
    {
        var (impl, layer) = Build(dynamicOutput: false);
        impl.SetInput("x", Tensor.FromArray(new float[] { 1, 2, 3, 4 }, new TensorShape(2, 2)));
        impl.Forward();
        impl.Forward();

        Assert.Equal(1, layer.ShapeCalls, "second forward reuses the cached shapes");
        Assert.Equal(1, layer.FinalizeCalls, "second forward reuses the finalized state");

        impl.SetInput("x", Tensor.FromArray(new float[] { 1, 2, 3, 4, 5, 6 }, new TensorShape(2, 3)));
        var output = impl.Forward();

        Assert.Equal(new TensorShape(2, 3), output.Shape, "output follows the new input shape");
        Assert.Equal(2, layer.ShapeCalls, "shape change re-infers");
        Assert.Equal(2, layer.FinalizeCalls, "shape change re-finalizes");
    }

    private static (NetImpl Net, CountingLayer Layer) Build(bool dynamicOutput)
    {
        var impl = new NetImpl();
        var x = impl.Declare("x", ArgKind.Input);
        var graph = impl.NewGraph("g", new[] { x }, isMain: true);

        var layer = new CountingLayer { Name = "probe", Type = "Probe", Dynamic = dynamicOutput };
        layer.Inputs.Add(x);

        var y = impl.Declare("y", ArgKind.Temp);
        layer.Outputs.Add(y);

        graph.Append(layer);
        graph.SetOutputs(new[] { y });

        return (impl, layer);
    }

    /// <summary>原样拷贝输入，并记录形状推断/Finalize 的调用次数。</summary>
    private sealed class CountingLayer : Layer
    {
        public bool Dynamic;
        public int ShapeCalls;
        public int FinalizeCalls;

        public override bool DynamicOutputShapes() => Dynamic;

        public override void GetMemoryShapes(
            IReadOnlyList<TensorShape> inShapes,
            int requiredOutputs,
            List<TensorShape> outShapes,
            List<TensorShape> tempShapes)
        {
            ShapeCalls++;
            outShapes.Add(inShapes[0]);
        }

        public override void Finalize(IReadOnlyList<Tensor> inputs, IReadOnlyList<Tensor> outputs)
            => FinalizeCalls++;

        public override void Forward(
            IReadOnlyList<Tensor> inputs,
            IReadOnlyList<Tensor> outputs,
            IReadOnlyList<Tensor> temps)
        {
            if (Dynamic)
                outputs[0].Fit(inputs[0].Shape, DType.F32);

            inputs[0].AsSpan<float>().CopyTo(outputs[0].AsSpan<float>());
        }
    }
}
