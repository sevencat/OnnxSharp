namespace OnnxSharp.Core;

/// <summary>
/// 算子工厂表。对应 OpenCV 的 <c>LayerFactory</c>
/// （<c>CV_DNN_REGISTER_LAYER_CLASS</c>）。
/// </summary>
public static class LayerRegistry
{
    private static readonly Dictionary<string, Func<LayerParams, LayerInfo>> Factories =
        new(StringComparer.Ordinal);

    private static bool _builtinsRegistered;

    public static void Register(string type, Func<LayerParams, LayerInfo> factory)
    {
        ArgumentException.ThrowIfNullOrEmpty(type);
        Factories[type] = factory ?? throw new ArgumentNullException(nameof(factory));
    }

    public static bool IsRegistered(string type) => Factories.ContainsKey(type);

    public static LayerInfo Create(string type, LayerParams parameters)
    {
        EnsureBuiltins();
        if (!Factories.TryGetValue(type, out var factory))
            throw new OnnxSharpException($"op type '{type}' is not registered");
        LayerInfo op = factory(parameters);
        // 属性集合是算子定义的一部分，不只是导入器的中间变量：GPU 发射器要靠它读
        // stride/pad/group，融合 pass 要靠它读 perm/axis。导入器本来就会再赋一次（同一对象），
        // 这里是让「手工建的图」（测试、探针）行为与导入的图一致。
        op.Parameters ??= parameters;
        return op;
    }

    public static void EnsureBuiltins()
    {
        if (_builtinsRegistered)
            return;
        _builtinsRegistered = true;

        Ops.NaryEltwiseLayer.Register();
        Ops.ActivationLayer.Register();
        Ops.ShapeOps.Register();
        Ops.DnnOps.Register();
        Ops.IndexOps.Register();
        Ops.BroadcastOps.Register();
        Ops.MiscOps.Register();
        Ops.NormOps.Register();
        Ops.QuantOps.Register();
        Ops.QLinearConvOps.Register();
        Ops.QLinearElementOps.Register();
        Ops.ArgMaxOps.Register();
        Ops.MatMulArgMaxOps.Register();
        Ops.QuantizedMatMulOps.Register();
        Ops.AttentionOps.Register();
        Ops.DflOps.Register();
        Ops.AttnOps.Register();
        Ops.FusedEltwiseLayer.Register();
    }
}
