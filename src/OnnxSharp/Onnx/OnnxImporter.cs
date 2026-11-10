using OnnxSharp.Core;
using OnnxSharp.Tensors;

namespace OnnxSharp.Onnx;

/// <summary>
/// ONNX → 图 的通用导入器（对应 OpenCV <c>onnx_importer.cpp</c> 的骨架部分）。
/// <para>
/// 策略与上游一致但更简单：<b>不做常量折叠</b>，常量以 <see cref="ArgKind.Const"/> 直接挂在图上，
/// 需要读常量值的算子通过 <see cref="LayerInfo.In"/> 反查（M3 再补 folding pass）。
/// </para>
/// </summary>
internal static class OnnxImporter
{
    /// <summary>把已解析的 ONNX 模型构建成可执行的图。</summary>
    public static NetImpl Build(OnnxFile file)
    {
        var net = new NetImpl();

        // 1) initializer → Const Arg
        foreach (var (name, tensor) in file.Initializers)
            net.NewConstArg(name, tensor);

        // 2) graph input（已在 initializer 里出现的会被 OnnxFile 过滤掉）
        var valueInfos = CollectValueInfos(file);

        var inputs = new List<Arg>(file.Inputs.Count);
        foreach (var info in file.Inputs)
        {
            if (info.Type == DType.Invalid)
                throw new OnnxSharpException($"input '{info.Name}' is not a tensor type (sequence/map unsupported)");
            var arg = net.Declare(info.Name, ArgKind.Input);
            ApplyValueInfo(net, arg, info);
            inputs.Add(arg);
        }

        var graph = net.NewGraph("main", inputs, isMain: true);

        // 3) 按拓扑序逐节点建层
        foreach (var node in file.Nodes)
        {
            if (node.OpType == "Constant")
            {
                RegisterConstant(net, node);
                continue;
            }

            if (node.Domain.Length != 0 && node.Domain != "ai.onnx" && node.Domain != "ai.onnx.ml")
                throw new OnnxSharpException(
                    $"node '{node.Name}' ({node.OpType}) uses unsupported domain '{node.Domain}'");

            if (!LayerRegistry.IsRegistered(node.OpType))
                throw new OnnxSharpException(
                    $"op type '{node.OpType}' (node '{node.Name}') is not implemented yet");

            var parameters = BuildParameters(node);
            var layer = LayerRegistry.Create(node.OpType, parameters);

            foreach (var name in node.Inputs)
                layer.Inputs.Add(name.Length == 0 ? default : net.Require(name));

            foreach (var name in node.Outputs)
            {
                if (name.Length == 0)
                {
                    layer.Outputs.Add(default);   // 可选输出（如 MaxPool 的 Indices）
                    continue;
                }

                if (net.TryGetArg(name, out var existing))
                {
                    layer.Outputs.Add(existing);
                    continue;
                }

                var arg = net.Declare(name, ArgKind.Temp);
                if (valueInfos.TryGetValue(name, out var info))
                    ApplyValueInfo(net, arg, info);
                layer.Outputs.Add(arg);
            }

            graph.Append(layer);
        }

        // 4) 图输出
        var outputs = new List<Arg>(file.Outputs.Count);
        foreach (var info in file.Outputs)
            outputs.Add(net.Require(info.Name));
        graph.SetOutputs(outputs);

        return net;
    }

    // ---------------- value_info → Arg 元信息 ----------------

    /// <summary>
    /// 汇总 <c>graph.value_info</c>（以及输入/输出声明）。ONNX 导出器通常会把**所有**中间张量的
    /// shape/dtype 写进 value_info，这是纯 shape 推断（不跑 kernel）的唯一来源。
    /// </summary>
    private static Dictionary<string, OnnxValueInfo> CollectValueInfos(OnnxFile file)
    {
        var result = new Dictionary<string, OnnxValueInfo>(StringComparer.Ordinal);

        foreach (var info in file.ValueInfos)
            result[info.Name] = info;
        foreach (var info in file.Outputs)
            result[info.Name] = info;

        return result;
    }

    /// <summary>
    /// 把 value_info 的静态 shape/dtype 预置到 Arg 上。
    /// 这样未执行 <c>Forward</c> 时 <see cref="NetImpl.GetArgTensor"/> 也能给出正确形状，
    /// 且运行时 <see cref="Interpreter"/> 推断出的形状会覆盖它（动态维度一律跳过）。
    /// </summary>
    private static void ApplyValueInfo(NetImpl net, Arg arg, OnnxValueInfo info)
    {
        if (info.Type == DType.Invalid)
            return;

        bool dynamic = false;
        for (int i = 0; i < info.Dims.Length; i++)
        {
            if (info.Dims[i] < 0)
            {
                dynamic = true;
                break;
            }
        }

        var data = net.GetArgData(arg);

        // dtype 总要记住：标量输入（rank 0，Dims.Length == 0）也要有正确类型，
        // 否则 int64 的 `max_label_len` 会被当成 F32（decoder 里当 Slice 的 starts 用）。
        data.DType = info.Type;

        if (dynamic || info.Dims.Length == 0)
            return;   // 动态维度 / 标量：形状交给运行时或调用方

        data.Shape = new TensorShape(info.Dims);
        data.HasShape = true;
        data.Layout = Layout.NCHW;
    }

    // ---------------- Constant 节点 ----------------

    private static void RegisterConstant(NetImpl net, OnnxNodeProto node)
    {
        if (node.Outputs.Length == 0 || node.Outputs[0].Length == 0)
            throw new OnnxSharpException($"Constant node '{node.Name}' has no output");

        var name = node.Outputs[0];
        var tensor = ReadConstantValue(node);

        if (net.TryGetArg(name, out _))
            throw new OnnxSharpException($"Constant node '{node.Name}' collides with existing arg '{name}'");

        net.NewConstArg(name, tensor);
    }

    private static Tensor ReadConstantValue(OnnxNodeProto node)
    {
        if (node.Tensor("value") is { } tensor)
            return tensor;

        if (node.Ints("value_ints") is { Length: > 0 } ints)
            return Tensor.FromInt64(ints);

        if (node.Floats("value_floats") is { Length: > 0 } floats)
            return Tensor.FromArray(floats, new TensorShape(floats.Length));

        if (node.Attributes.ContainsKey("value_int"))
        {
            var scalar = Tensor.Allocate(new TensorShape(Array.Empty<int>()), DType.I64);
            scalar.AsSpan<long>()[0] = node.Int("value_int");
            return scalar;
        }

        if (node.Attributes.ContainsKey("value_float"))
        {
            var scalar = Tensor.Allocate(new TensorShape(Array.Empty<int>()), DType.F32);
            scalar.AsSpan<float>()[0] = node.Float("value_float");
            return scalar;
        }

        throw new OnnxSharpException(
            $"Constant node '{node.Name}': no supported value attribute " +
            $"(have: {string.Join(",", node.Attributes.Keys)})");
    }

    // ---------------- 属性 → LayerParams ----------------

    private static LayerParams BuildParameters(OnnxNodeProto node)
    {
        var parameters = new LayerParams { Name = node.Name, Type = node.OpType };

        foreach (var (key, attribute) in node.Attributes)
        {
            // AttributeProto.type：0=UNDEFINED 1=FLOAT 2=INT 3=STRING 4=TENSOR 6=FLOATS 7=INTS 8=STRINGS
            int type = (int)attribute.Int(20, 0);
            if (type == 0)
                type = InferType(attribute);

            switch (type)
            {
                case 1:
                    parameters.Set(key, ReadSingleFloat(attribute));
                    break;
                case 2:
                    parameters.Set(key, attribute.Int(3));
                    break;
                case 3:
                    parameters.Set(key, attribute.Text(4));
                    break;
                case 4:
                    if (node.Tensor(key) is { } tensor)
                        parameters.Set(key, tensor);
                    break;
                case 6:
                    parameters.Set(key, attribute.Floats(7));
                    break;
                case 7:
                    parameters.Set(key, attribute.Ints(8));
                    break;
                case 8:
                    parameters.Set(key, ReadStrings(attribute));
                    break;
                case 5:
                case 9:
                case 10:
                    // 子图/张量列表：首版不需要
                    break;
                default:
                    throw new OnnxSharpException(
                        $"node '{node.Name}' ({node.OpType}): unsupported attribute type {type} for '{key}'");
            }
        }

        return parameters;
    }

    private static int InferType(Proto attribute)
    {
        if (attribute.Has(2))
            return 1;
        if (attribute.Has(3))
            return 2;
        if (attribute.Has(4))
            return 3;
        if (attribute.Has(5))
            return 4;
        if (attribute.Has(7))
            return 6;
        if (attribute.Has(8))
            return 7;
        if (attribute.Has(9))
            return 8;
        return 0;
    }

    private static float ReadSingleFloat(Proto attribute)
    {
        var buffer = attribute.Buffer(2);
        if (buffer is not { Length: >= 4 } value)
            throw new OnnxSharpException("FLOAT attribute has no 'f' field");
        return BitConverter.ToSingle(value.Span[..4]);
    }

    private static string[] ReadStrings(Proto attribute)
    {
        var list = new List<string>();
        foreach (var buffer in attribute.Buffers(9))
            list.Add(System.Text.Encoding.UTF8.GetString(buffer.Span));
        return list.ToArray();
    }
}
