using System.Runtime.InteropServices;
using OnnxSharp.Tensors;

namespace OnnxSharp.Tests;

/// <summary>
/// Transpose 内核的三档实现（块拷贝 / 单轴旋转矩阵转置 / 通用兜底）必须与朴素实现逐元素一致。
/// 覆盖：rank 2..5 的全部 perm × 多组 dims（含 1 维退化），以及非 F32 元素宽度。
/// </summary>
internal static class TransposeKernelTests
{
    public static IEnumerable<TestCase> Cases()
    {
        yield return new("transpose: all perms rank 2..5 match naive", AllPermsMatchNaive);
        yield return new("transpose: rfdetr shapes match naive", RfDetrShapesMatchNaive);
        yield return new("transpose: non-f32 element sizes match naive", NonFloatMatchNaive);
    }

    private static void AllPermsMatchNaive()
    {
        var dimSets = new List<int[]>
        {
            new[] { 2, 3 },
            new[] { 5, 7 },
            new[] { 1, 4, 3 },
            new[] { 4, 1, 6 },
            new[] { 3, 5, 2, 4 },
            new[] { 1, 8, 3, 1 },
            new[] { 2, 2, 3, 2, 2 },
            new[] { 1, 1, 4, 1, 3 },
        };

        foreach (var dims in dimSets)
        {
            foreach (var perm in Permutations(dims.Length))
            {
                var src = RandomFloats(Count(dims));
                var expected = NaiveTranspose(src, dims, perm);
                var actual = Run(src, dims, perm);

                for (int i = 0; i < expected.Length; i++)
                {
                    if (expected[i] != actual[i])
                    {
                        throw new TestFailureException(
                            $"dims={Join(dims)} perm={Join(perm)} at {i}: expected {expected[i]}, got {actual[i]}");
                    }
                }
            }
        }
    }

    /// <summary>RF-DETR 里实际出现的 perm + shape 组合（profile 热点来源）。</summary>
    private static void RfDetrShapesMatchNaive()
    {
        var cases = new (int[] Dims, int[] Perm)[]
        {
            (new[] { 4, 257, 6, 64 }, new[] { 0, 2, 1, 3 }),
            (new[] { 4, 257, 6, 64 }, new[] { 0, 2, 3, 1 }),
            (new[] { 300, 8, 32 }, new[] { 1, 0, 2 }),
            (new[] { 1, 32, 32, 128 }, new[] { 0, 3, 1, 2 }),
            (new[] { 1, 256, 300 }, new[] { 0, 2, 1 }),
            (new[] { 2, 2, 16, 16, 384 }, new[] { 0, 2, 1, 3, 4 }),
            (new[] { 1, 8, 300, 32 }, new[] { 0, 1, 3, 2 }),
            (new[] { 1, 8, 300, 32 }, new[] { 2, 0, 1, 3 }),
            (new[] { 3, 5, 7, 11, 13 }, new[] { 1, 4, 3, 0, 2 }),
        };

        foreach (var (dims, perm) in cases)
        {
            var src = RandomFloats(Count(dims));
            var expected = NaiveTranspose(src, dims, perm);
            var actual = Run(src, dims, perm);

            for (int i = 0; i < expected.Length; i++)
            {
                if (expected[i] != actual[i])
                {
                    throw new TestFailureException(
                        $"dims={Join(dims)} perm={Join(perm)} at {i}: expected {expected[i]}, got {actual[i]}");
                }
            }
        }
    }

    private static void NonFloatMatchNaive()
    {
        // elementSize = 1 / 2 / 8：走 Transpose2DScalar 与块拷贝路径。
        foreach (int size in new[] { 1, 2, 8 })
        {
            var dims = new[] { 3, 4, 5 };
            foreach (var perm in Permutations(dims.Length))
            {
                int count = Count(dims);
                var src = new byte[count * size];
                Random.Shared.NextBytes(src);

                var dst = new byte[src.Length];
                TransposeKernel.Transpose(src, dst, dims, perm, size);

                var expected = NaiveTransposeBytes(src, dims, perm, size);
                for (int i = 0; i < expected.Length; i++)
                {
                    if (expected[i] != dst[i])
                    {
                        throw new TestFailureException(
                            $"size={size} dims={Join(dims)} perm={Join(perm)} byte {i}: expected {expected[i]}, got {dst[i]}");
                    }
                }
            }
        }
    }

    private static float[] Run(float[] src, int[] dims, int[] perm)
    {
        var dst = new float[src.Length];
        TransposeKernel.Transpose(
            MemoryMarshal.AsBytes(src.AsSpan()),
            MemoryMarshal.AsBytes(dst.AsSpan()),
            dims,
            perm,
            4);

        return dst;
    }

    private static float[] NaiveTranspose(float[] src, int[] dims, int[] perm)
    {
        int rank = dims.Length;
        var outDims = new int[rank];
        var inStride = new int[rank];

        int run = 1;
        for (int i = rank - 1; i >= 0; i--)
        {
            inStride[i] = run;
            run *= dims[i];
        }

        for (int i = 0; i < rank; i++)
            outDims[i] = dims[perm[i]];

        var dst = new float[run];
        var coord = new int[rank];

        for (long n = 0; n < run; n++)
        {
            long source = 0;
            for (int i = 0; i < rank; i++)
                source += (long)coord[i] * inStride[perm[i]];

            dst[n] = src[source];

            for (int i = rank - 1; i >= 0; i--)
            {
                if (++coord[i] < outDims[i])
                    break;
                coord[i] = 0;
            }
        }

        return dst;
    }

    private static byte[] NaiveTransposeBytes(byte[] src, int[] dims, int[] perm, int elementSize)
    {
        int rank = dims.Length;
        var outDims = new int[rank];
        var inStride = new int[rank];

        int run = 1;
        for (int i = rank - 1; i >= 0; i--)
        {
            inStride[i] = run;
            run *= dims[i];
        }

        for (int i = 0; i < rank; i++)
            outDims[i] = dims[perm[i]];

        var dst = new byte[run * elementSize];
        var coord = new int[rank];

        for (long n = 0; n < run; n++)
        {
            long source = 0;
            for (int i = 0; i < rank; i++)
                source += (long)coord[i] * inStride[perm[i]];

            Buffer.BlockCopy(src, checked((int)(source * elementSize)), dst,
                checked((int)(n * elementSize)), elementSize);

            for (int i = rank - 1; i >= 0; i--)
            {
                if (++coord[i] < outDims[i])
                    break;
                coord[i] = 0;
            }
        }

        return dst;
    }

    private static IEnumerable<int[]> Permutations(int rank)
    {
        var current = new int[rank];
        for (int i = 0; i < rank; i++)
            current[i] = i;

        return Walk(current, 0);

        static List<int[]> Walk(int[] perm, int start)
        {
            if (start == perm.Length - 1)
                return new List<int[]> { (int[])perm.Clone() };

            var result = new List<int[]>();
            for (int i = start; i < perm.Length; i++)
            {
                (perm[start], perm[i]) = (perm[i], perm[start]);
                result.AddRange(Walk(perm, start + 1));
                (perm[start], perm[i]) = (perm[i], perm[start]);
            }

            return result;
        }
    }

    private static float[] RandomFloats(int count)
    {
        var values = new float[count];
        for (int i = 0; i < count; i++)
            values[i] = Random.Shared.Next(-1000, 1000) * 0.25f;

        return values;
    }

    private static int Count(int[] dims)
    {
        int product = 1;
        foreach (int d in dims)
            product *= d;

        return product;
    }

    private static string Join(int[] values) => string.Join(",", values);
}
