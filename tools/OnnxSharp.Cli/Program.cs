using OnnxSharp.Core;
using OnnxSharp.Tensors;

// M0 冒烟测试：手工建图 Input[2,3] --Add(bias[3])--> Relu --> output
// 用于验证 Arg 表 / shape 推断 / 广播 / 解释执行链路。

var impl = new NetImpl();

var x = impl.NewArg("x", ArgKind.Input);
var graph = impl.NewGraph("main", new[] { x }, isMain: true);

var bias = impl.NewConstArg("bias", Tensor.FromArray(new float[] { 0.5f, 0.5f, -1f }, new TensorShape(3)));

var add = LayerRegistry.Create("Add", new LayerParams { Type = "Add", Name = "add" });
add.Inputs.Add(x);
add.Inputs.Add(bias);
var y = impl.NewArg("y", ArgKind.Temp);
add.Outputs.Add(y);
graph.Append(add);

var relu = LayerRegistry.Create("Relu", new LayerParams { Type = "Relu", Name = "relu" });
relu.Inputs.Add(y);
var z = impl.NewArg("z", ArgKind.Temp);
relu.Outputs.Add(z);
graph.Append(relu);

graph.SetOutputs(new[] { z });

impl.SetInput("x", Tensor.FromArray(new float[] { 1, -2, 3, 4, -5, 6 }, new TensorShape(2, 3)));

var result = impl.Forward();

// 期望：add 后 [[1.5, -1.5, 2], [4.5, -4.5, 5]]，Relu 后 [[1.5, 0, 2], [4.5, 0, 5]]
float[] expected = { 1.5f, 0f, 2f, 4.5f, 0f, 5f };
var actual = result.AsSpan<float>().ToArray();

Console.WriteLine($"graph      : {graph}");
Console.WriteLine($"output     : {result}");
Console.WriteLine($"values     : {string.Join(", ", actual)}");

bool ok = actual.Length == expected.Length;
for (int i = 0; ok && i < expected.Length; i++)
    ok = MathF.Abs(actual[i] - expected[i]) < 1e-6f;

Console.WriteLine(ok ? "SMOKE TEST: PASS" : "SMOKE TEST: FAIL");
return ok ? 0 : 1;
