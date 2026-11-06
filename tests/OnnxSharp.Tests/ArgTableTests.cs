using OnnxSharp.Core;
using OnnxSharp.Tensors;

namespace OnnxSharp.Tests;

internal static class ArgTableTests
{
    public static IEnumerable<TestCase> Cases()
    {
        yield return new("args: Declare rejects duplicate names", DeclareRejectsDuplicate);
        yield return new("args: Require reports missing names", RequireReportsMissing);
        yield return new("args: TryGetArg does not create", TryGetArgDoesNotCreate);
        yield return new("args: const arg carries shape/dtype/layout", ConstArgCarriesMetadata);
    }

    private static void DeclareRejectsDuplicate()
    {
        var impl = new NetImpl();
        impl.Declare("x", ArgKind.Input);

        Assert.Throws<OnnxSharpException>(() => impl.Declare("x", ArgKind.Temp),
            "declaring the same name twice");
    }

    private static void RequireReportsMissing()
    {
        var impl = new NetImpl();
        Assert.Throws<OnnxSharpException>(() => impl.Require("nope"), "requiring a missing arg");
    }

    private static void TryGetArgDoesNotCreate()
    {
        var impl = new NetImpl();
        int before = impl.ArgCount;

        Assert.False(impl.TryGetArg("ghost", out _), "TryGetArg must report false");
        Assert.Equal(before, impl.ArgCount, "TryGetArg must not add args");

        impl.Declare("x", ArgKind.Input);
        Assert.True(impl.TryGetArg("x", out var arg), "TryGetArg must find declared arg");
        Assert.Equal(ArgKind.Input, impl.GetArgKind(arg), "kind preserved");
    }

    private static void ConstArgCarriesMetadata()
    {
        var impl = new NetImpl();
        var arg = impl.NewConstArg("c", Tensor.FromArray(new float[] { 1, 2, 3, 4 }, new TensorShape(2, 2)));
        var data = impl.GetArgData(arg);

        Assert.Equal(ArgKind.Const, data.Kind, "kind");
        Assert.Equal(DType.F32, data.DType, "dtype");
        Assert.Equal(new TensorShape(2, 2), data.Shape, "shape");
        Assert.Equal(Layout.NCHW, data.Layout, "layout");
        Assert.True(data.Const is not null, "const tensor present");
    }
}
