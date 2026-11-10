using System.Buffers;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using OnnxSharp.Tensors;

namespace OnnxSharp.Kernels;

/// <summary>
/// 无 VNNI 机器上 <c>MatMulInteger</c> 的替代实现：把「整数矩阵乘」交给 F32 GEMM 做。
/// <para>
/// <b>为什么</b>：本机（Zen4）<c>AvxVnni.IsSupported = False</c>，<see cref="Int8Gemm"/> 只能走标量
/// int32 路径 —— 实测 **~0.9 GFLOPS**（Paraformer encoder 全段 10.9 s）；而同样形状的
/// <see cref="GemmKernel"/> F32 路径有 **73 GFLOPS**。差 80× 的根源是指令，不是数据布局。
/// </para>
/// <para>
/// <b>为什么 F32 能做整数乘**而且结果精确**</b>：反量化后的因子都是小整数
/// （<c>|a - a_zp| ≤ 255</c>、<c>|b - b_zp| ≤ 255</c>），每个乘积 &lt; 2^16 精确可表示；
/// 累加的绝对值通常在 <c>k·|term|</c> 量级，而量化后 <c>a</c> 以零点为中心、正负相消，
/// 实测 <c>k = 1280</c> 时和只有几千量级 —— 远在 2^24 之内，F32 加法逐步精确。
/// 于是「F32 GEMM → 四舍五入回 int32」与整数点积逐位相同（单元测试按 <c>Assert.Equal</c> 校验）。
/// </para>
/// <para>
/// 代价：权重需要一次性展开成 F32（体积 ×4，由 <c>Ops.QuantOps</c> 按层缓存，
/// 可用环境变量 <c>ONNXSHARP_INT8_F32GEMM=0</c> 关掉）。
/// </para>
/// </summary>
public static class Int8F32Gemm
{
    /// <summary>
    /// <c>C[m,n] = A[m,k] · B[k,n]</c>，输入是**整数小值**的 F32 视图，输出取整回 <c>int32</c>。
    /// </summary>
    public static void MultiplyInteger(
        ReadOnlySpan<float> activation,
        ReadOnlySpan<float> weights,
        Span<int> destination,
        int m,
        int k,
        int n)
    {
        int count = checked(m * n);
        if (destination.Length < count)
            throw new ArgumentException($"destination has {destination.Length} values, [{m},{n}] needed", nameof(destination));

        float[] buffer = ArrayPool<float>.Shared.Rent(count);
        try
        {
            var accumulator = buffer.AsSpan(0, count);
            GemmKernel.Multiply(activation, weights, accumulator, m, k, n);

            for (int i = 0; i < count; i++)
                destination[i] = (int)MathF.Round(accumulator[i], MidpointRounding.ToEven);
        }
        finally
        {
            ArrayPool<float>.Shared.Return(buffer);
        }
    }

    /// <summary>
    /// 同 <see cref="MultiplyInteger(ReadOnlySpan{float}, ReadOnlySpan{float}, Span{int}, int, int, int)"/>，
    /// 但权重是**预打包面板**（常量权重每帧零拷贝，省掉每次 <c>Multiply</c> 内部的 B 重排）。
    /// </summary>
    internal static void MultiplyInteger(
        ReadOnlySpan<float> activation,
        GemmKernel.PackedB weights,
        Span<int> destination,
        int m)
    {
        int count = checked(m * weights.N);
        if (destination.Length < count)
            throw new ArgumentException($"destination has {destination.Length} values, {count} needed", nameof(destination));

        float[] buffer = ArrayPool<float>.Shared.Rent(count);
        try
        {
            var accumulator = buffer.AsSpan(0, count);
            GemmKernel.MultiplyPacked(activation, weights, accumulator, m);

            for (int i = 0; i < count; i++)
                destination[i] = (int)MathF.Round(accumulator[i], MidpointRounding.ToEven);
        }
        finally
        {
            ArrayPool<float>.Shared.Return(buffer);
        }
    }

    /// <summary>u8 → <c>a - zeroPoint</c>。</summary>
    public static void WidenActivation(ReadOnlySpan<byte> source, Span<float> destination, int zeroPoint)
    {
        int i = 0;
        if (Avx2.IsSupported)
        {
            var vzero = Vector256.Create((float)zeroPoint);
            for (; i + 8 <= source.Length; i += 8)
            {
                var bytes = Vector128.LoadUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(source[i..]));
                var value = Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(bytes)) - vzero;
                value.StoreUnsafe(ref destination[i]);
            }
        }

        for (; i < source.Length; i++)
            destination[i] = source[i] - zeroPoint;
    }

    /// <summary>s8 → <c>a - zeroPoint</c>。</summary>
    public static void WidenActivation(ReadOnlySpan<sbyte> source, Span<float> destination, int zeroPoint)
    {
        int i = 0;
        if (Avx2.IsSupported)
        {
            var vzero = Vector256.Create((float)zeroPoint);
            for (; i + 8 <= source.Length; i += 8)
            {
                var bytes = Vector128.LoadUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(source[i..]));
                var value = Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(bytes)) - vzero;
                value.StoreUnsafe(ref destination[i]);
            }
        }

        for (; i < source.Length; i++)
            destination[i] = source[i] - zeroPoint;
    }

    /// <summary>u8 权重 <c>[k,n]</c> → <c>b - zeroPoint[col]</c>（per-column 或 per-tensor）。</summary>
    public static void WidenWeights(
        ReadOnlySpan<byte> source,
        Span<float> destination,
        int k,
        int n,
        ReadOnlySpan<int> zeroPoints)
    {
        for (int row = 0; row < k; row++)
        {
            var input = source.Slice(row * n, n);
            var output = destination.Slice(row * n, n);

            if (zeroPoints.Length == 1)
            {
                WidenActivation(input, output, zeroPoints[0]);
                continue;
            }

            for (int column = 0; column < n; column++)
                output[column] = input[column] - zeroPoints[column];
        }
    }

    /// <summary>s8 权重 <c>[k,n]</c> → <c>b - zeroPoint[col]</c>。</summary>
    public static void WidenWeights(
        ReadOnlySpan<sbyte> source,
        Span<float> destination,
        int k,
        int n,
        ReadOnlySpan<int> zeroPoints)
    {
        for (int row = 0; row < k; row++)
        {
            var input = source.Slice(row * n, n);
            var output = destination.Slice(row * n, n);

            if (zeroPoints.Length == 1)
            {
                WidenActivation(input, output, zeroPoints[0]);
                continue;
            }

            for (int column = 0; column < n; column++)
                output[column] = input[column] - zeroPoints[column];
        }
    }
}
