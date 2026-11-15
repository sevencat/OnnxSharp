using System.Runtime.InteropServices;
using OnnxSharp.Core;
using OnnxSharp.Tensors;

namespace OnnxSharp.Ops;

/// <summary>
/// 广播类算子：Expand / Tile / Where / Not / Equal / Less / Greater / And。
/// <para>对应 OpenCV 的 <c>expand_layer.cpp</c>、<c>eltwise_layer.cpp</c>（比较部分）。</para>
/// </summary>
internal static class BroadcastOps
{
    public static void Register()
    {
        LayerRegistry.Register("Expand", p => new ExpandLayer(p));
        LayerRegistry.Register("Tile", p => new TileLayer(p));
        LayerRegistry.Register("Where", p => new WhereLayer(p));
        LayerRegistry.Register("Not", p => new NotLayer(p));
        LayerRegistry.Register("Equal", p => new CompareLayer(p, CompareKind.Equal));
        LayerRegistry.Register("Less", p => new CompareLayer(p, CompareKind.Less));
        LayerRegistry.Register("Greater", p => new CompareLayer(p, CompareKind.Greater));
        LayerRegistry.Register("GreaterOrEqual", p => new CompareLayer(p, CompareKind.GreaterOrEqual));
        LayerRegistry.Register("And", p => new CompareLayer(p, CompareKind.And));
        LayerRegistry.Register("IsNaN", p => new IsNaNNLayer(p));
    }

    internal enum CompareKind
    {
        Equal,
        Less,
        Greater,
        GreaterOrEqual,
        And,
    }

    // ---------------- IsNaN ----------------

    /// <summary>
    /// <c>IsNaN(x)</c>：输出 Bool。SAM3 的编码器在 softmax 之后挂了「NaN → 0」的保护
    /// （<c>IsNaN → Where</c>），文本 / 几何 / 解码器图里一共有 40 处。
    /// </summary>
    private sealed class IsNaNNLayer : Layer
    {
        public IsNaNNLayer(LayerParams parameters)
        {
            Type = parameters.Type;
            Name = parameters.Name;
        }

        public override void GetMemoryShapes(
            IReadOnlyList<TensorShape> inShapes,
            int requiredOutputs,
            List<TensorShape> outShapes,
            List<TensorShape> tempShapes)
            => outShapes.Add(inShapes[0]);

        public override void GetTypes(
            IReadOnlyList<DType> inTypes,
            int requiredOutputs,
            int requiredTemps,
            List<DType> outTypes,
            List<DType> tempTypes)
        {
            outTypes.Clear();
            for (int i = 0; i < requiredOutputs; i++)
                outTypes.Add(DType.Bool);

            tempTypes.Clear();
            for (int i = 0; i < requiredTemps; i++)
                tempTypes.Add(DType.F32);
        }

        public override void Forward(
            IReadOnlyList<Tensor> inputs,
            IReadOnlyList<Tensor> outputs,
            IReadOnlyList<Tensor> temps)
        {
            var input = inputs[0];
            if (input.DType != DType.F32)
                throw new OnnxSharpException($"IsNaN '{Name}': only F32 is supported (got {input.DType})");

            var source = input.AsSpan<float>();
            var target = outputs[0].AsSpan<byte>();

            int i = 0;
            if (System.Runtime.Intrinsics.X86.Avx.IsSupported)
            {
                Span<float> buffer = stackalloc float[System.Runtime.Intrinsics.Vector256<float>.Count];
                for (; i + buffer.Length <= source.Length; i += buffer.Length)
                {
                    var v = System.Runtime.Intrinsics.Vector256.LoadUnsafe(ref source[i]);
                    // NaN 与自身比较「无序」，OrderedNotEqualNonSignaling 正好筛出 NaN
                    var mask = System.Runtime.Intrinsics.X86.Avx.Compare(
                        v, v, System.Runtime.Intrinsics.X86.FloatComparisonMode.OrderedNotEqualNonSignaling);
                    System.Runtime.Intrinsics.Vector256.StoreUnsafe(mask, ref buffer[0]);

                    for (int j = 0; j < buffer.Length; j++)
                        target[i + j] = buffer[j] != 0f ? (byte)1 : (byte)0;
                }
            }

            for (; i < source.Length; i++)
                target[i] = float.IsNaN(source[i]) ? (byte)1 : (byte)0;
        }
    }

    // ---------------- Expand ----------------

    private sealed class ExpandLayer : Layer
    {
        public ExpandLayer(LayerParams parameters)
        {
            Type = parameters.Type;
            Name = parameters.Name;
        }

        public override void GetMemoryShapes(
            IReadOnlyList<TensorShape> inShapes,
            int requiredOutputs,
            List<TensorShape> outShapes,
            List<TensorShape> tempShapes)
        {
            var target = OpUtil.ReadInts(OpUtil.RequireConstInput(this, 1));
            outShapes.Add(new TensorShape(
                OpUtil.BroadcastDims(new[] { inShapes[0], new TensorShape(target) })));
        }

        public override void Forward(
            IReadOnlyList<Tensor> inputs,
            IReadOnlyList<Tensor> outputs,
            IReadOnlyList<Tensor> temps)
        {
            var data = inputs[0];
            var output = outputs[0];

            if (data.Shape == output.Shape)
            {
                OpUtil.CopyBytes(data, 0, output, 0, data.Count);
                return;
            }

            var dims = output.Shape.ToArray();
            var strides = OpUtil.BroadcastStrides(data.Shape, dims);
            var cursor = new int[dims.Length];
            long offset = 0;

            for (long n = 0; n < output.Count; n++)
            {
                OpUtil.CopyBytes(data, offset, output, n, 1);

                for (int axis = dims.Length - 1; axis >= 0; axis--)
                {
                    cursor[axis]++;
                    offset += strides[axis];
                    if (cursor[axis] < dims[axis])
                        break;
                    offset -= (long)strides[axis] * dims[axis];
                    cursor[axis] = 0;
                }
            }
        }
    }

    // ---------------- Tile ----------------

    private sealed class TileLayer : Layer
    {
        public TileLayer(LayerParams parameters)
        {
            Type = parameters.Type;
            Name = parameters.Name;
        }

        public override void GetMemoryShapes(
            IReadOnlyList<TensorShape> inShapes,
            int requiredOutputs,
            List<TensorShape> outShapes,
            List<TensorShape> tempShapes)
        {
            var input = inShapes[0];
            var repeats = OpUtil.ReadInts(OpUtil.RequireConstInput(this, 1));

            if (repeats.Length != input.Rank)
                throw new OnnxSharpException(
                    $"Tile '{Name}': repeats length {repeats.Length} != input rank {input.Rank}");

            var dims = new int[input.Rank];
            for (int i = 0; i < dims.Length; i++)
                dims[i] = checked(input[i] * repeats[i]);

            outShapes.Add(new TensorShape(dims));
        }

        public override void Forward(
            IReadOnlyList<Tensor> inputs,
            IReadOnlyList<Tensor> outputs,
            IReadOnlyList<Tensor> temps)
        {
            var data = inputs[0];
            var output = outputs[0];

            var sourceDims = data.Shape.ToArray();
            var dims = output.Shape.ToArray();
            var sourceStrides = OpUtil.Strides(sourceDims);

            var cursor = new int[dims.Length];
            long offset = 0;

            for (long n = 0; n < output.Count; n++)
            {
                OpUtil.CopyBytes(data, offset, output, n, 1);

                // 源坐标对每个轴取模：等价于把输入块按 repeats 平铺
                for (int axis = dims.Length - 1; axis >= 0; axis--)
                {
                    cursor[axis]++;
                    if (cursor[axis] < dims[axis])
                    {
                        int wrapped = cursor[axis] % sourceDims[axis];
                        offset += wrapped == 0
                            ? -(long)sourceStrides[axis] * (sourceDims[axis] - 1)
                            : sourceStrides[axis];
                        break;
                    }

                    offset -= (long)sourceStrides[axis] * (sourceDims[axis] - 1);
                    cursor[axis] = 0;
                }
            }
        }
    }

    // ---------------- Where ----------------

    private sealed class WhereLayer : Layer
    {
        public WhereLayer(LayerParams parameters)
        {
            Type = parameters.Type;
            Name = parameters.Name;
        }

        public override void GetMemoryShapes(
            IReadOnlyList<TensorShape> inShapes,
            int requiredOutputs,
            List<TensorShape> outShapes,
            List<TensorShape> tempShapes)
            => outShapes.Add(new TensorShape(OpUtil.BroadcastDims(inShapes)));

        public override void GetTypes(
            IReadOnlyList<DType> inTypes,
            int requiredOutputs,
            int requiredTemps,
            List<DType> outTypes,
            List<DType> tempTypes)
        {
            // input[0] 是 Bool 条件，输出类型跟着 x（input[1]）走
            var t = inTypes.Count > 1 ? inTypes[1] : DType.F32;

            outTypes.Clear();
            for (int i = 0; i < requiredOutputs; i++)
                outTypes.Add(t);

            tempTypes.Clear();
            for (int i = 0; i < requiredTemps; i++)
                tempTypes.Add(DType.F32);
        }

        public override void Forward(
            IReadOnlyList<Tensor> inputs,
            IReadOnlyList<Tensor> outputs,
            IReadOnlyList<Tensor> temps)
        {
            var condition = inputs[0];
            var x = inputs[1];
            var y = inputs[2];
            var output = outputs[0];

            var dims = output.Shape.ToArray();
            var stridesC = OpUtil.BroadcastStrides(condition.Shape, dims);
            var stridesX = OpUtil.BroadcastStrides(x.Shape, dims);
            var stridesY = OpUtil.BroadcastStrides(y.Shape, dims);

            // 按行快路径：条件在尾维上是广播的（一行共享一个 bool），而 x/y 在尾维上要么连续、
            // 要么也是广播（标量/整行同值）—— 于是每行只需一次判定 + 一次 memcpy 或填充。
            // rfdetr 的 Where 是 cond[1,1024,1] + 标量 + [1,1024,256]：逐元素要 2.6 ms，按行约 0.05 ms。
            if (dims.Length > 0)
            {
                int last = dims.Length - 1;
                if (dims[last] > 1
                    && stridesC[last] == 0
                    && (stridesX[last] == 0 || stridesX[last] == 1)
                    && (stridesY[last] == 0 || stridesY[last] == 1))
                {
                    ForwardRows(condition, x, y, output, dims, stridesC, stridesX, stridesY, dims[last]);
                    return;
                }
            }

            var mask = OpUtil.ReadBools(condition);
            var cursor = new int[dims.Length];
            long offsetC = 0, offsetX = 0, offsetY = 0;

            for (long n = 0; n < output.Count; n++)
            {
                if (mask[(int)offsetC])
                    OpUtil.CopyBytes(x, offsetX, output, n, 1);
                else
                    OpUtil.CopyBytes(y, offsetY, output, n, 1);

                for (int axis = dims.Length - 1; axis >= 0; axis--)
                {
                    cursor[axis]++;
                    offsetC += stridesC[axis];
                    offsetX += stridesX[axis];
                    offsetY += stridesY[axis];
                    if (cursor[axis] < dims[axis])
                        break;

                    offsetC -= (long)stridesC[axis] * dims[axis];
                    offsetX -= (long)stridesX[axis] * dims[axis];
                    offsetY -= (long)stridesY[axis] * dims[axis];
                    cursor[axis] = 0;
                }
            }
        }

        /// <summary>
        /// 尾维整段同条件的循环：外层走 dims[0..^1]，内层一次写 <paramref name="inner"/> 个元素。
        /// </summary>
        private static void ForwardRows(
            Tensor condition,
            Tensor x,
            Tensor y,
            Tensor output,
            int[] dims,
            int[] stridesC,
            int[] stridesX,
            int[] stridesY,
            int inner)
        {
            var mask = OpUtil.ReadBools(condition);
            int es = output.DType.SizeOf();
            var xBytes = x.Buffer.AsSpan(x.ByteOffset, x.ByteLength);
            var yBytes = y.Buffer.AsSpan(y.ByteOffset, y.ByteLength);
            var outBytes = output.Buffer.AsSpan(output.ByteOffset, output.ByteLength);

            int colX = stridesX[^1];
            int colY = stridesY[^1];
            int outerRank = dims.Length - 1;
            var cursor = new int[outerRank];
            long outer = output.Count / inner;
            long offsetC = 0, offsetX = 0, offsetY = 0, offsetOut = 0;

            for (long row = 0; row < outer; row++)
            {
                var dst = outBytes.Slice((int)(offsetOut * es), inner * es);
                if (mask[(int)offsetC])
                    WriteRun(xBytes, (int)(offsetX * es), colX, dst, inner, es);
                else
                    WriteRun(yBytes, (int)(offsetY * es), colY, dst, inner, es);

                offsetOut += inner;

                for (int axis = outerRank - 1; axis >= 0; axis--)
                {
                    cursor[axis]++;
                    offsetC += stridesC[axis];
                    offsetX += stridesX[axis];
                    offsetY += stridesY[axis];
                    if (cursor[axis] < dims[axis])
                        break;

                    offsetC -= (long)stridesC[axis] * dims[axis];
                    offsetX -= (long)stridesX[axis] * dims[axis];
                    offsetY -= (long)stridesY[axis] * dims[axis];
                    cursor[axis] = 0;
                }
            }
        }

        /// <summary>
        /// 写一行：<paramref name="colStride"/> 为 1 时整块拷贝，为 0 时把单个元素重复填满。
        /// 填充按「等宽整数」做，位模式与 float/int64 一致，省掉按 dtype 分派。
        /// </summary>
        private static void WriteRun(
            ReadOnlySpan<byte> src,
            int byteOffset,
            int colStride,
            Span<byte> dst,
            int count,
            int es)
        {
            if (colStride != 0)
            {
                src.Slice(byteOffset, count * es).CopyTo(dst);
                return;
            }

            var one = src.Slice(byteOffset, es);
            switch (es)
            {
                case 1:
                    dst.Fill(one[0]);
                    return;
                case 2:
                    MemoryMarshal.Cast<byte, short>(dst).Fill(MemoryMarshal.Read<short>(one));
                    return;
                case 4:
                    MemoryMarshal.Cast<byte, int>(dst).Fill(MemoryMarshal.Read<int>(one));
                    return;
                case 8:
                    MemoryMarshal.Cast<byte, long>(dst).Fill(MemoryMarshal.Read<long>(one));
                    return;
                default:
                    for (int i = 0; i < count; i++)
                        one.CopyTo(dst.Slice(i * es, es));
                    return;
            }
        }
    }

    // ---------------- Not ----------------

    private sealed class NotLayer : Layer
    {
        public NotLayer(LayerParams parameters)
        {
            Type = parameters.Type;
            Name = parameters.Name;
        }

        public override void GetMemoryShapes(
            IReadOnlyList<TensorShape> inShapes,
            int requiredOutputs,
            List<TensorShape> outShapes,
            List<TensorShape> tempShapes)
            => outShapes.Add(inShapes[0]);

        public override void GetTypes(
            IReadOnlyList<DType> inTypes,
            int requiredOutputs,
            int requiredTemps,
            List<DType> outTypes,
            List<DType> tempTypes)
        {
            outTypes.Clear();
            for (int i = 0; i < requiredOutputs; i++)
                outTypes.Add(DType.Bool);

            tempTypes.Clear();
            for (int i = 0; i < requiredTemps; i++)
                tempTypes.Add(DType.F32);
        }

        public override void Forward(
            IReadOnlyList<Tensor> inputs,
            IReadOnlyList<Tensor> outputs,
            IReadOnlyList<Tensor> temps)
        {
            var source = OpUtil.ReadBools(inputs[0]);
            var target = outputs[0].AsSpan<byte>();

            for (int i = 0; i < source.Length; i++)
                target[i] = source[i] ? (byte)0 : (byte)1;
        }
    }

    // ---------------- Equal / Less / Greater / And ----------------

    private sealed class CompareLayer : Layer
    {
        private readonly CompareKind _kind;

        public CompareLayer(LayerParams parameters, CompareKind kind)
        {
            Type = parameters.Type;
            Name = parameters.Name;
            _kind = kind;
        }

        public override void GetMemoryShapes(
            IReadOnlyList<TensorShape> inShapes,
            int requiredOutputs,
            List<TensorShape> outShapes,
            List<TensorShape> tempShapes)
            => outShapes.Add(new TensorShape(OpUtil.BroadcastDims(inShapes)));

        public override void GetTypes(
            IReadOnlyList<DType> inTypes,
            int requiredOutputs,
            int requiredTemps,
            List<DType> outTypes,
            List<DType> tempTypes)
        {
            outTypes.Clear();
            for (int i = 0; i < requiredOutputs; i++)
                outTypes.Add(DType.Bool);

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
            var target = outputs[0].AsSpan<byte>();

            var dims = outputs[0].Shape.ToArray();
            var stridesA = OpUtil.BroadcastStrides(a.Shape, dims);
            var stridesB = OpUtil.BroadcastStrides(b.Shape, dims);

            var cursor = new int[dims.Length];
            long offsetA = 0, offsetB = 0;

            if (a.DType == DType.F32 && b.DType == DType.F32 && _kind != CompareKind.And)
            {
                var sa = a.AsSpan<float>();
                var sb = b.AsSpan<float>();

                for (int n = 0; n < target.Length; n++)
                {
                    target[n] = CompareValue(sa[(int)offsetA], sb[(int)offsetB]) ? (byte)1 : (byte)0;

                    for (int axis = dims.Length - 1; axis >= 0; axis--)
                    {
                        cursor[axis]++;
                        offsetA += stridesA[axis];
                        offsetB += stridesB[axis];
                        if (cursor[axis] < dims[axis])
                            break;

                        offsetA -= (long)stridesA[axis] * dims[axis];
                        offsetB -= (long)stridesB[axis] * dims[axis];
                        cursor[axis] = 0;
                    }
                }

                return;
            }

            if (a.DType == DType.I64 && b.DType == DType.I64 && _kind != CompareKind.And)
            {
                // 形状链里的比较（如 Equal(shape_a, shape_b)）走整数路径
                var la = a.AsSpan<long>();
                var lb = b.AsSpan<long>();

                for (int n = 0; n < target.Length; n++)
                {
                    long x = la[(int)offsetA];
                    long y = lb[(int)offsetB];
                    target[n] = _kind switch
                    {
                        CompareKind.Equal => x == y ? (byte)1 : (byte)0,
                        CompareKind.Less => x < y ? (byte)1 : (byte)0,
                        CompareKind.Greater => x > y ? (byte)1 : (byte)0,
                        CompareKind.GreaterOrEqual => x >= y ? (byte)1 : (byte)0,
                        _ => (byte)0,
                    };

                    for (int axis = dims.Length - 1; axis >= 0; axis--)
                    {
                        cursor[axis]++;
                        offsetA += stridesA[axis];
                        offsetB += stridesB[axis];
                        if (cursor[axis] < dims[axis])
                            break;

                        offsetA -= (long)stridesA[axis] * dims[axis];
                        offsetB -= (long)stridesB[axis] * dims[axis];
                        cursor[axis] = 0;
                    }
                }

                return;
            }

            if (_kind != CompareKind.And)
                throw new OnnxSharpException(
                    $"'{Type}' '{Name}': only F32/I64 inputs are supported (got {a.DType}/{b.DType})");

            var maskA = OpUtil.ReadBools(a);
            var maskB = OpUtil.ReadBools(b);

            for (int n = 0; n < target.Length; n++)
            {
                target[n] = maskA[(int)offsetA] && maskB[(int)offsetB] ? (byte)1 : (byte)0;

                for (int axis = dims.Length - 1; axis >= 0; axis--)
                {
                    cursor[axis]++;
                    offsetA += stridesA[axis];
                    offsetB += stridesB[axis];
                    if (cursor[axis] < dims[axis])
                        break;

                    offsetA -= (long)stridesA[axis] * dims[axis];
                    offsetB -= (long)stridesB[axis] * dims[axis];
                    cursor[axis] = 0;
                }
            }
        }

        private bool CompareValue(float a, float b) => _kind switch
        {
            CompareKind.Equal => a == b,
            CompareKind.Less => a < b,
            CompareKind.Greater => a > b,
            CompareKind.GreaterOrEqual => a >= b,
            _ => throw new OnnxSharpException($"'{Type}': unsupported float comparison"),
        };
    }
}
