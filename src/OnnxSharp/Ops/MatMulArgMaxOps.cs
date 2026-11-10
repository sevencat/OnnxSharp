using OnnxSharp.Core;
using OnnxSharp.Kernels;
using OnnxSharp.Tensors;

namespace OnnxSharp.Ops;

/// <summary>
/// 图融合产物：<c>ArgMax(MatMul(a, b) [+ bias])</c> 合成一层，**不物化 logits**。
/// <para>
/// 由 <see cref="GraphFusion"/> 把 <c>MatMul → ArgMax</c>（或中间的常量 bias <c>Add</c>）
/// 替换成这个算子；语义等价于「算完 logits 再取沿最后一维的最大值下标」，
/// 但省掉 <c>[m,n]</c> 的写回与再读一遍（CTC 那种 <c>m = 帧数</c> 的形状能省几 MB 流量）。
/// </para>
/// </summary>
internal static class MatMulArgMaxOps
{
    public static void Register() => LayerRegistry.Register("MatMulArgMax", p => new MatMulArgMaxLayer(p));

    private sealed class MatMulArgMaxLayer : Layer
    {
        public MatMulArgMaxLayer(LayerParams parameters)
        {
            Type = parameters.Type;
            Name = parameters.Name;
        }

        private static (int M, int K, int N) Resolve(IReadOnlyList<TensorShape> inShapes)
        {
            var a = inShapes[0];
            var b = inShapes[1];

            if (a.Rank != 2 || b.Rank != 2)
                throw new OnnxSharpException($"MatMulArgMax: only 2-D is supported, got {a} x {b}");

            int m = a[0];
            int k = a[1];
            int n = b[1];
            if (b[0] != k)
                throw new OnnxSharpException($"MatMulArgMax: K mismatch ({a} x {b})");

            return (m, k, n);
        }

        public override void GetMemoryShapes(
            IReadOnlyList<TensorShape> inShapes,
            int requiredOutputs,
            List<TensorShape> outShapes,
            List<TensorShape> tempShapes)
        {
            var (m, _, _) = Resolve(inShapes);
            outShapes.Add(new TensorShape(m));
        }

        public override void GetTypes(
            IReadOnlyList<DType> inTypes,
            int requiredOutputs,
            int requiredTemps,
            List<DType> outTypes,
            List<DType> tempTypes)
        {
            outTypes.Clear();
            for (int i = 0; i < requiredOutputs; i++)
                outTypes.Add(DType.I64);

            tempTypes.Clear();
            for (int i = 0; i < requiredTemps; i++)
                tempTypes.Add(DType.F32);
        }

        public override void Forward(
            IReadOnlyList<Tensor> inputs,
            IReadOnlyList<Tensor> outputs,
            IReadOnlyList<Tensor> temps)
        {
            var a = inputs[0];
            var b = inputs[1];

            var shapes = new List<TensorShape>(2) { a.Shape, b.Shape };
            var (m, k, n) = Resolve(shapes);

            var bias = Inputs.Count > 2 && Inputs[2].Idx != 0 ? inputs[2].AsSpan<float>() : default;

            MatMulArgMaxKernel.Compute(
                a.AsSpan<float>(), b.AsSpan<float>(), bias,
                m, k, n,
                outputs[0].AsSpan<long>(),
                default);
        }
    }
}
