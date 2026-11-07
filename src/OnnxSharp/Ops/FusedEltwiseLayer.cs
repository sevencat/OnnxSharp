using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using OnnxSharp.Core;
using OnnxSharp.Tensors;
using OnnxSharp.Threading;

namespace OnnxSharp.Ops;

/// <summary>融合链里的单个运算。</summary>
internal enum EltwiseKind : byte
{
    Add,
    Sub,
    Mul,
    Div,
    Min,
    Max,
    Sqrt,
    Erf,
    Exp,
    Neg,
    Abs,
    Relu,
    Sigmoid,
}

/// <summary>
/// 一条「寄存器机」指令：<c>reg[dst] = op(reg[a], reg[b])</c>，一元算子忽略 <see cref="B"/>。
/// <para>
/// 寄存器 0..n-1 对应本层的 <see cref="LayerInfo.Inputs"/>（外部输入，每块只从内存读一次），
/// n 之后的寄存器是链内中间结果，<b>全程留在 YMM 寄存器里，不落内存</b>。
/// </para>
/// </summary>
internal readonly struct EltwiseStep
{
    public readonly EltwiseKind Kind;
    public readonly int A;
    public readonly int B;
    public readonly int Dst;

    public EltwiseStep(EltwiseKind kind, int a, int b, int dst)
    {
        Kind = kind;
        A = a;
        B = b;
        Dst = dst;
    }
}

/// <summary>
/// 一条同形状逐元素链融合后的算子（不是 ONNX 算子，由 <see cref="GraphFusion"/> 生成）。
/// <para>
/// <b>为什么要它</b>：ONNX 导出的 GELU 是 <c>Div → Erf → Add → Mul → Mul</c> 五个节点，
/// 每个节点都要把整张张量读一遍、写一遍。RF-DETR 有 12 条这样的链，每条张量 1.58 M 元素（6.3 MB），
/// 合计每推理白白搬约 600 MB。融合后整条链只读一遍外部输入、写一遍最终输出。
/// </para>
/// <para>
/// <b>只处理同形状</b>，不做一般广播；链上的「标量常量」操作数按 splat 处理
/// （GELU 里的 <c>+1</c>、<c>×0.5</c> 就是这种）。是否为标量在运行时按实际长度判定（长度 1）。
/// </para>
/// </summary>
internal sealed class FusedEltwiseLayer : Layer
{
    private readonly EltwiseStep[] _steps;

    /// <summary>链内需用到的寄存器总数（外部输入 + 中间结果）。</summary>
    private readonly int _regCount;

    /// <summary>最终结果所在的寄存器。</summary>
    private readonly int _outReg;

    /// <summary>命中 GELU 模式时的专用计划，走手写内核。</summary>
    private readonly GeluPlan? _gelu;

    public FusedEltwiseLayer(LayerParams parameters, EltwiseStep[] steps, int regCount, int outReg)
    {
        Type = parameters.Type;
        Name = parameters.Name;
        _steps = steps;
        _regCount = regCount;
        _outReg = outReg;
        // 构造时 Inputs 还没填，外部输入个数 = 寄存器总数 - 步骤数（每个步骤占一个中间寄存器）。
        _gelu = GeluPlan.TryMatch(steps, regCount - steps.Length, outReg);
    }

    /// <summary>
    /// GELU 的固定形状：<c>Div(x, √2) → Erf → Add(_, 1) → Mul(x, _) → Mul(_, 0.5)</c>。
    /// <para>
    /// 为什么单独给它一条路径：通用寄存器机内核里 <c>regs</c> 是运行时下标访问的数组，
    /// JIT 没法把它提升到 YMM 寄存器，每步都要「写回栈、再读出来」，加上每步一次 <c>switch</c> 分发，
    /// 实测吃掉全部带宽收益（1.65 ms vs 未融合的 1.50 ms）。
    /// 手写内核里中间值是本地变量，JIT 全程留在寄存器里。
    /// </para>
    /// <para>
    /// 只认「一个张量 + 一个标量」的操作数组合；运行时不满足就回退通用路径。
    /// </para>
    /// </summary>
    private sealed class GeluPlan
    {
        /// <summary>step0 <c>Div</c> 的两个操作数寄存器（哪个是标量运行时才能定）。</summary>
        public int DivA, DivB;

        /// <summary>step2 <c>Add</c> 的标量寄存器。</summary>
        public int One;

        /// <summary>step3 <c>Mul</c> 的两个操作数寄存器（一个是 <c>x</c>，一个是上一步结果）。</summary>
        public int MulA, MulB;

        /// <summary>step3 里「上一步结果」所在的寄存器，用来区分哪个操作数是 <c>x</c>。</summary>
        public int MulCdf;

        /// <summary>step4 <c>Mul</c> 的两个操作数寄存器（一个是上一步结果，一个是标量 0.5）。</summary>
        public int ScaleA, ScaleB;

        /// <summary>step4 里「上一步结果」所在的寄存器。</summary>
        public int ScaleSrc;

        public static GeluPlan? TryMatch(EltwiseStep[] steps, int nIn, int outReg)
        {
            if (steps.Length != 5)
                return null;

            ref readonly var s0 = ref steps[0];
            ref readonly var s1 = ref steps[1];
            ref readonly var s2 = ref steps[2];
            ref readonly var s3 = ref steps[3];
            ref readonly var s4 = ref steps[4];

            if (s0.Kind != EltwiseKind.Div || s0.A < 0 || s0.B < 0 || s0.A >= nIn || s0.B >= nIn)
                return null;
            if (s1.Kind != EltwiseKind.Erf || s1.A != s0.Dst || s1.B >= 0)
                return null;
            if (s2.Kind != EltwiseKind.Add || s2.A != s1.Dst || s2.B < 0 || s2.B >= nIn)
                return null;
            if (s3.Kind != EltwiseKind.Mul)
                return null;
            if (s4.Kind != EltwiseKind.Mul || s4.Dst != outReg)
                return null;

            // step3: Mul 一侧是 cdf（step2 的结果）、另一侧是 x
            int xReg;
            if (s3.A == s2.Dst && s3.B >= 0 && s3.B < nIn)
                xReg = s3.B;
            else if (s3.B == s2.Dst && s3.A >= 0 && s3.A < nIn)
                xReg = s3.A;
            else
                return null;

            // step4: Mul 一侧是上一步结果、另一侧是 0.5 之类的标量
            int scaleReg;
            if (s4.A == s3.Dst && s4.B >= 0 && s4.B < nIn)
                scaleReg = s4.B;
            else if (s4.B == s3.Dst && s4.A >= 0 && s4.A < nIn)
                scaleReg = s4.A;
            else
                return null;

            return new GeluPlan
            {
                DivA = s0.A, DivB = s0.B,
                One = s2.B,
                MulA = s3.A, MulB = s3.B, MulCdf = s2.Dst,
                ScaleA = s4.A, ScaleB = s4.B, ScaleSrc = s3.Dst,
            };
        }
    }

    public static void Register()
    {
        // 占位：防止 ONNX 文件里出现同名算子被当成融合层使用。
        LayerRegistry.Register("FusedEltwise", _ => throw new OnnxSharpException(
            "FusedEltwise 只能由 GraphFusion 构造，不能从 ONNX 导入"));
    }

    public override void GetMemoryShapes(
        IReadOnlyList<TensorShape> inShapes,
        int requiredOutputs,
        List<TensorShape> outShapes,
        List<TensorShape> tempShapes)
    {
        if (inShapes.Count != Inputs.Count)
            throw new OnnxSharpException($"FusedEltwise '{Name}': input count changed ({inShapes.Count} != {Inputs.Count})");

        // 输出形状 = 第一个「非标量」输入的形状；其余输入要么同形，要么元素数为 1。
        TensorShape outShape = default;
        bool found = false;

        for (int i = 0; i < inShapes.Count; i++)
        {
            var s = inShapes[i];
            if (s.Total() == 1)
                continue;   // 标量常量（含 rank 0）：splat

            if (!found)
            {
                outShape = s;
                found = true;
            }
            else if (!s.Equals(outShape))
            {
                throw new OnnxSharpException(
                    $"FusedEltwise '{Name}': input #{i} shape {s} does not match {outShape}");
            }
        }

        if (!found)
            outShape = inShapes.Count > 0 ? inShapes[0] : TensorShape.Scalar;

        outShapes.Add(outShape);
    }

    public override unsafe void Forward(
        IReadOnlyList<Tensor> inputs,
        IReadOnlyList<Tensor> outputs,
        IReadOnlyList<Tensor> temps)
    {
        var dstSpan = outputs[0].AsSpan<float>();
        int n = dstSpan.Length;
        if (n <= 0)
            return;

        int nIn = inputs.Count;

        // 标量操作数（GELU 的 +1 / ×0.5）整条链恒定，先 splat 好。
        var isScalar = new bool[nIn];
        var splat = new float[nIn];
        for (int i = 0; i < nIn; i++)
        {
            var t = inputs[i];
            if (t.DType != DType.F32 || outputs[0].DType != DType.F32)
                throw new NotSupportedException("FusedEltwise supports F32 only");

            var span = t.AsSpan<float>();
            if (span.Length == 1 && n != 1)
            {
                isScalar[i] = true;
                splat[i] = span[0];
            }
        }

        // 一次把输入输出都 pin 住：最多 4~6 个操作数，句柄开销远小于省下的两遍流量。
        var handles = new GCHandle[nIn + 1];
        try
        {
            var src = new nint[nIn];
            for (int i = 0; i < nIn; i++)
            {
                var t = inputs[i];
                handles[i] = GCHandle.Alloc(t.Buffer, GCHandleType.Pinned);
                src[i] = handles[i].AddrOfPinnedObject() + t.ByteOffset;
            }

            handles[nIn] = GCHandle.Alloc(outputs[0].Buffer, GCHandleType.Pinned);
            nint dst = handles[nIn].AddrOfPinnedObject() + outputs[0].ByteOffset;

            // GELU 走手写内核：中间值全程是本地变量，JIT 能留在 YMM 寄存器里。
            if (_gelu is { } plan && TryResolveGelu(plan, src, isScalar, splat,
                    out nint geluX0, out float geluDiv, out float geluHalf, out float geluOne))
            {
                if (n >= ParallelRange.SerialThreshold && ParallelRange.Threads > 1)
                    ParallelRange.For(n, (begin, end) =>
                        RunGeluRange(geluX0, dst, begin, end, geluDiv, geluHalf, geluOne));
                else
                    RunGeluRange(geluX0, dst, 0, n, geluDiv, geluHalf, geluOne);

                return;
            }

            if (n >= ParallelRange.SerialThreshold && ParallelRange.Threads > 1)
                ParallelRange.For(n, (begin, end) => RunRange(src, dst, n, begin, end, isScalar, splat));
            else
                RunRange(src, dst, n, 0, n, isScalar, splat);
        }
        finally
        {
            for (int i = 0; i < handles.Length; i++)
            {
                if (handles[i].IsAllocated)
                    handles[i].Free();
            }
        }
    }

    /// <summary>
    /// 把 GELU 计划解析成「一个张量指针 + 三个标量」。任一侧不是「一张量一标量」就返回 false，
    /// 交给通用路径（例如有人把常数扩成同形张量）。
    /// </summary>
    private bool TryResolveGelu(
        GeluPlan plan, nint[] src, bool[] isScalar, float[] splat,
        out nint x0, out float div, out float half, out float one)
    {
        x0 = default;
        div = 0;
        half = 0;
        one = splat[plan.One];

        // step0: Div(tensor, scalar)
        if (isScalar[plan.DivA] == isScalar[plan.DivB])
            return false;

        int xReg;
        if (isScalar[plan.DivA])
        {
            xReg = plan.DivB;
            div = splat[plan.DivA];
        }
        else
        {
            xReg = plan.DivA;
            div = splat[plan.DivB];
        }

        // step3 的被乘数必须就是同一个 x（GELU 的 0.5x·(1+erf(x/√2)) 才对得上）
        int mulX = plan.MulA == plan.MulCdf ? plan.MulB : plan.MulA;
        if (mulX != xReg || isScalar[mulX])
            return false;

        // step4 的另一个操作数必须是标量
        int scaleArg = plan.ScaleA == plan.ScaleSrc ? plan.ScaleB : plan.ScaleA;
        if (!isScalar[scaleArg])
            return false;

        x0 = src[xReg];
        half = splat[scaleArg];
        return true;
    }

    /// <summary>
    /// GELU 专用内核：<c>(x · (1 + erf(x / √2))) · 0.5</c>。
    /// 读 x 一遍、写结果一遍，中间值全部留在 YMM 寄存器里。
    /// </summary>
    private static unsafe void RunGeluRange(
        nint xN, nint dstN, int begin, int end, float div, float half, float one)
    {
        var src = (float*)xN;
        var dst = (float*)dstN;

        int i = begin;

        if (Avx.IsSupported)
        {
            var vdiv = Vector256.Create(div);
            var vhalf = Vector256.Create(half);
            var vone = Vector256.Create(one);
            int last = end - 7;

            for (; i < last; i += 8)
            {
                var x = Avx.LoadVector256(src + i);
                var q = Avx.Divide(x, vdiv);
                var cdf = Avx.Add(VectorMath.Erf(q), vone);
                var h = Avx.Multiply(x, cdf);
                Avx.Store(dst + i, Avx.Multiply(h, vhalf));
            }
        }

        // 尾部不足 8 个元素（1.58M 元素时每层最多 7 个），标量算完即可。
        for (; i < end; i++)
        {
            float q = src[i] / div;
            float cdf = TensorOps.Erf(q) + one;
            dst[i] = src[i] * cdf * half;
        }
    }

    /// <summary>
    /// 跑 <c>[begin, end)</c> 这一段。整条链的所有中间值都在 YMM 寄存器里，
    /// 所以一次循环只读外部输入、写最终输出，没有中间张量落内存。
    /// </summary>
    private unsafe void RunRange(
        nint[] srcN, nint dstN, int n, int begin, int end, bool[] isScalar, float[] splat)
    {
        int nIn = Inputs.Count;
        var steps = _steps;
        var regs = new Vector256<float>[_regCount];

        var vsplat = new Vector256<float>[nIn];
        for (int r = 0; r < nIn; r++)
        {
            if (isScalar[r])
                vsplat[r] = Vector256.Create(splat[r]);
        }

        var dst = (float*)dstN;
        int last = end - 7;

        int i = begin;
        if (Avx.IsSupported)
        {
            for (; i < last; i += 8)
            {
                for (int r = 0; r < nIn; r++)
                {
                    regs[r] = isScalar[r]
                        ? vsplat[r]
                        : Avx.LoadVector256((float*)srcN[r] + i);
                }

                for (int s = 0; s < steps.Length; s++)
                {
                    ref readonly var step = ref steps[s];
                    regs[step.Dst] = ApplyVector(step.Kind, regs[step.A],
                        step.B >= 0 ? regs[step.B] : default);
                }

                Avx.Store(dst + i, regs[_outReg]);
            }
        }

        if (i < end)
        {
            var regsS = new float[_regCount];
            for (; i < end; i++)
            {
                for (int r = 0; r < nIn; r++)
                    regsS[r] = isScalar[r] ? splat[r] : ((float*)srcN[r])[i];

                for (int s = 0; s < steps.Length; s++)
                {
                    ref readonly var step = ref steps[s];
                    regsS[step.Dst] = ApplyScalar(step.Kind, regsS[step.A],
                        step.B >= 0 ? regsS[step.B] : 0f);
                }

                dst[i] = regsS[_outReg];
            }
        }
    }

    /// <summary>尾部不足 8 个元素时的标量回退。</summary>
    private static float ApplyScalar(EltwiseKind kind, float a, float b) => kind switch
    {
        EltwiseKind.Add => a + b,
        EltwiseKind.Sub => a - b,
        EltwiseKind.Mul => a * b,
        EltwiseKind.Div => a / b,
        EltwiseKind.Min => MathF.Min(a, b),
        EltwiseKind.Max => MathF.Max(a, b),
        EltwiseKind.Sqrt => MathF.Sqrt(a),
        EltwiseKind.Erf => TensorOps.Erf(a),
        EltwiseKind.Exp => MathF.Exp(a),
        EltwiseKind.Neg => -a,
        EltwiseKind.Abs => MathF.Abs(a),
        EltwiseKind.Relu => a > 0f ? a : 0f,
        EltwiseKind.Sigmoid => 1f / (1f + MathF.Exp(-a)),
        _ => throw new OnnxSharpException($"unknown fused elementwise op {kind}"),
    };

    private static Vector256<float> ApplyVector(EltwiseKind kind, Vector256<float> a, Vector256<float> b)
        => kind switch
        {
            EltwiseKind.Add => Avx.Add(a, b),
            EltwiseKind.Sub => Avx.Subtract(a, b),
            EltwiseKind.Mul => Avx.Multiply(a, b),
            EltwiseKind.Div => Avx.Divide(a, b),
            EltwiseKind.Min => Avx.Min(a, b),
            EltwiseKind.Max => Avx.Max(a, b),
            EltwiseKind.Sqrt => Avx.Sqrt(a),
            EltwiseKind.Erf => VectorMath.Erf(a),
            EltwiseKind.Exp => VectorMath.Exp(a),
            EltwiseKind.Neg => Avx.Xor(a, Vector256.Create(-0.0f)),
            EltwiseKind.Abs => Avx.AndNot(Vector256.Create(-0.0f), a),
            EltwiseKind.Relu => Avx.Max(a, Vector256<float>.Zero),
            EltwiseKind.Sigmoid => VectorMath.Sigmoid(a),
            _ => throw new OnnxSharpException($"unknown fused elementwise op {kind}"),
        };
}
