namespace OnnxSharp.Tensors;

/// <summary>张量元素类型。首版只实现 F32，其余为后续扩展预留。</summary>
public enum DType : byte
{
    Invalid = 0,
    F32 = 1,
    F16 = 2,
    I64 = 3,
    I32 = 4,
    I8 = 5,
    U8 = 6,
    Bool = 7,
}

public static class DTypeExtensions
{
    /// <summary>单个元素占用的字节数。</summary>
    public static int SizeOf(this DType t) => t switch
    {
        DType.F32 => 4,
        DType.F16 => 2,
        DType.I64 => 8,
        DType.I32 => 4,
        DType.I8 => 1,
        DType.U8 => 1,
        DType.Bool => 1,
        _ => throw new ArgumentOutOfRangeException(nameof(t), t, "unknown dtype"),
    };

    public static bool IsInteger(this DType t) =>
        t is DType.I64 or DType.I32 or DType.I8 or DType.U8 or DType.Bool;

    public static bool IsFloat(this DType t) => t is DType.F32 or DType.F16;
}
