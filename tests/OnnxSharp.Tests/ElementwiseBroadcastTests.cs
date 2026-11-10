using OnnxSharp.Core;
using OnnxSharp.Tensors;

namespace OnnxSharp.Tests;

/// <summary>
/// 广播逐元素算子的回归：
/// ① <c>[1,3,1] op [1,3,4]</c> 走「每段一个标量」的新路径（<c>ApplyInnerScalar</c>）；
/// ② <c>[1,3,4] op [1,1,4]</c> 走尾部广播；③ 双向都不等于输出形状时只能走通用循环。
/// 之前加了 ① 却漏了「a、b 都不等于 c」的情形，直接把 ASR encoder 跑崩（越界读），所以这里逐形状钉死。
/// </summary>
internal static class ElementwiseBroadcastTests
{
    public static IEnumerable<TestCase> Cases()
    {
        yield return new("broadcast: inner-constant operand (per-row scalar)", InnerConstant);
        yield return new("broadcast: tail operand (per-column vector)", TailVector);
        yield return new("broadcast: neither side equals the output", BothBroadcast);
    }

    private static void InnerConstant()
    {
        // [1,3,1] op [1,3,4]，三种算子各测一遍
        var left = Tensor.FromArray([1f, 2f, 3f], new TensorShape(1, 3, 1));
        var right = Tensor.FromArray(
            [10f, 20f, 30f, 40f, 1f, 2f, 3f, 4f, 100f, 200f, 300f, 400f],
            new TensorShape(1, 3, 4));

        Check(BinaryOpKind.Sub, left, right);
        Check(BinaryOpKind.Mul, left, right);
        Check(BinaryOpKind.Div, right, left);
    }

    private static void TailVector()
    {
        var left = Tensor.FromArray([1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f], new TensorShape(1, 2, 4));
        var right = Tensor.FromArray([2f, 4f, 8f, 16f], new TensorShape(1, 1, 4));
        Check(BinaryOpKind.Mul, left, right);
    }

    private static void BothBroadcast()
    {
        // 两边都不是输出形状：只能走通用循环，但也必须正确
        var left = Tensor.FromArray([1f, 2f, 3f], new TensorShape(1, 3, 1));
        var right = Tensor.FromArray([10f, 20f, 30f, 40f], new TensorShape(1, 1, 4));
        Check(BinaryOpKind.Sub, left, right);
    }

    private static void Check(BinaryOpKind op, Tensor a, Tensor b)
    {
        var impl = new NetImpl();
        var left = impl.Declare("a", ArgKind.Input);
        var right = impl.Declare("b", ArgKind.Input);
        var graph = impl.NewGraph("g", new[] { left, right }, isMain: true);

        string type = op switch
        {
            BinaryOpKind.Sub => "Sub",
            BinaryOpKind.Mul => "Mul",
            _ => "Div",
        };

        var layer = LayerRegistry.Create(type, new LayerParams { Type = type, Name = type });
        layer.Inputs.Add(left);
        layer.Inputs.Add(right);

        var c = impl.Declare("c", ArgKind.Temp);
        layer.Outputs.Add(c);
        graph.Append(layer);
        graph.SetOutputs(new[] { c });

        impl.SetInput("a", a);
        impl.SetInput("b", b);
        var actual = impl.Forward();

        // 参考实现：按 ONNX 广播规则逐元素算
        var outDims = actual.Shape.ToArray();
        var expected = new float[actual.Shape.Total()];
        for (int i = 0; i < expected.Length; i++)
        {
            var index = new int[outDims.Length];
            int rest = i;
            for (int axis = outDims.Length - 1; axis >= 0; axis--)
            {
                index[axis] = rest % outDims[axis];
                rest /= outDims[axis];
            }

            float x = a.AsSpan<float>()[Offset(a.Shape.ToArray(), index)];
            float y = b.AsSpan<float>()[Offset(b.Shape.ToArray(), index)];
            expected[i] = op switch
            {
                BinaryOpKind.Sub => x - y,
                BinaryOpKind.Mul => x * y,
                _ => x / y,
            };
        }

        var span = actual.AsSpan<float>();
        for (int i = 0; i < expected.Length; i++)
            Assert.Near(expected[i], span[i], 1e-6f, $"{type} value[{i}]");
    }

    private static int Offset(int[] dims, int[] index)
    {
        int offset = 0;
        int stride = 1;
        for (int axis = dims.Length - 1; axis >= 0; axis--)
        {
            int coordinate = dims[axis] == 1 ? 0 : index[index.Length - dims.Length + axis];
            offset += coordinate * stride;
            stride *= dims[axis];
        }

        return offset;
    }
}
