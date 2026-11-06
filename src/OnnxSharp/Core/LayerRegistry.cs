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
        return factory(parameters);
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
    }
}
