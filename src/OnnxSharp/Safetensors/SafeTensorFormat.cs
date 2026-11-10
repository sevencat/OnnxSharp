using System.Buffers.Binary;
using System.Runtime.InteropServices;
using OnnxSharp.Core;
using OnnxSharp.Tensors;

namespace OnnxSharp.Safetensors;

/// <summary>
/// safetensors 的 dtype 字符串 ↔ ONNX <c>TensorProto.DataType</c> 的桥，以及两者之间的**数值转换**。
/// <para>
/// <b>为什么要转换</b>：safetensors 只描述「权重是什么类型」，ONNX 的 initializer 还带着
/// 「图期望什么类型」。训练产物常是 F16 / BF16，而骨架（以及本引擎）要 F32；
/// 直接按字节搬运会把 2 字节当 4 字节读，静默出错。
/// </para>
/// <para>
/// 支持：F64 / F32 / F16 / BF16 / I64 / I32 / I16 / I8 / U8 / U16 / U32 / U64 / BOOL。
/// FP8（<c>F8_E4M3</c> / <c>F8_E5M2</c>）不做解码 —— 报错让调用方先转成 F16/BF16。
/// </para>
/// </summary>
internal static partial class SafeTensorFormat
{
    /// <summary>单个元素占用的字节数。</summary>
    public static int SizeOf(string dtype, string path, string name) => dtype switch
    {
        "F64" or "I64" or "U64" => 8,
        "F32" or "I32" or "U32" => 4,
        "F16" or "BF16" or "I16" or "U16" => 2,
        "I8" or "U8" or "BOOL" or "F8_E4M3" or "F8_E5M2" => 1,
        _ => throw new OnnxSharpException(
            $"safetensors '{path}': tensor '{name}' has unsupported dtype '{dtype}'; "
            + "supported: F64 F32 F16 BF16 I64 I32 I16 I8 U8 U16 U32 U64 BOOL"),
    };

    /// <summary>safetensors dtype 对应的 ONNX <c>TensorProto.DataType</c>。</summary>
    public static int ToOnnxDataType(string dtype, string path, string name) => dtype switch
    {
        "F32" => 1,
        "U8" => 2,
        "I8" => 3,
        "U16" => 4,
        "I16" => 5,
        "I32" => 6,
        "I64" => 7,
        "BOOL" => 9,
        "F16" => 10,
        "F64" => 11,
        "U32" => 12,
        "U64" => 13,
        "BF16" => 16,
        "F8_E4M3" => 17,
        "F8_E5M2" => 19,
        _ => throw new OnnxSharpException(
            $"safetensors '{path}': tensor '{name}' has unsupported dtype '{dtype}'"),
    };

    /// <summary>
    /// ONNX <c>TensorProto.DataType</c> → safetensors 的 dtype 字符串（<see cref="ToOnnxDataType"/> 的逆）。
    /// 用于把 ONNX 里的权重抽成 safetensors。
    /// </summary>
    public static string ToSafeTensorDType(int onnxType, string name) => onnxType switch
    {
        1 => "F32",
        2 => "U8",
        3 => "I8",
        4 => "U16",
        5 => "I16",
        6 => "I32",
        7 => "I64",
        9 => "BOOL",
        10 => "F16",
        11 => "F64",
        12 => "U32",
        13 => "U64",
        16 => "BF16",
        _ => throw new OnnxSharpException(
            $"tensor '{name}': ONNX data_type {onnxType} has no safetensors equivalent "
            + "(FLOAT8 / COMPLEX / STRING 不支持)"),
    };

    /// <summary>
    /// <see cref="SafeTensorFile.GetTensor"/> 的目标类型：浮点一律 F32、整型取引擎支持的位宽
    /// （I16/U16 → I32，U32/U64 → I64）。这些类型 <see cref="DType"/> 里都有。
    /// </summary>
    public static int LoadTargetDType(string dtype, string path, string name) => dtype switch
    {
        "F64" or "F32" or "F16" or "BF16" => 1,
        "U8" => 2,
        "I8" => 3,
        "I16" or "U16" or "I32" => 6,
        "I64" or "U32" or "U64" => 7,
        "BOOL" => 9,
        _ => throw new OnnxSharpException(
            $"safetensors '{path}': tensor '{name}' has dtype '{dtype}' that cannot be loaded "
            + "(FP8 needs an explicit conversion to F16/BF16 first)"),
    };

    /// <summary>把原始字节转成目标 ONNX 类型的 <c>raw_data</c> 字节（同类型时直接返回原数组）。</summary>
    public static byte[] Encode(byte[] source, string sourceDtype, int onnxType, SafeTensorInfo info)
    {
        long count = info.ElementCount;
        int sourceSize = SizeOf(sourceDtype, info.Name, info.Name);
        if (source.Length != checked(count * sourceSize))
            throw new OnnxSharpException(
                $"safetensors '{info.Name}': {source.Length} B does not match {count} x {sourceDtype}");

        if (ToOnnxDataType(sourceDtype, info.Name, info.Name) == onnxType)
            return source;

        if (IsFloatTarget(onnxType))
        {
            var values = new float[count];
            DecodeFloat(source, sourceDtype, values, info.Name);
            return EncodeFloat(values, onnxType, info.Name);
        }

        if (!IsIntTarget(onnxType))
            throw new OnnxSharpException(
                $"tensor '{info.Name}': cannot convert {sourceDtype} to ONNX data_type {onnxType}");

        var integers = new long[count];
        DecodeInt(source, sourceDtype, integers, info.Name);
        return EncodeInt(integers, onnxType, info.Name);
    }

    /// <summary>ONNX 类型的字节数（<c>TensorProto.DataType</c>）。</summary>
    public static int SizeOfOnnxType(int onnxType, string name) => onnxType switch
    {
        1 or 6 or 12 => 4,     // FLOAT / INT32 / UINT32
        2 or 3 or 9 => 1,      // UINT8 / INT8 / BOOL
        4 or 5 or 10 or 16 => 2, // UINT16 / INT16 / FLOAT16 / BFLOAT16
        7 or 11 or 13 => 8,    // INT64 / DOUBLE / UINT64
        _ => throw new OnnxSharpException($"tensor '{name}': unsupported ONNX data_type {onnxType}"),
    };

    /// <summary>把 <c>raw_data</c> 包成引擎张量。</summary>
    public static Tensor ToTensor(byte[] raw, int onnxType, SafeTensorInfo info)
    {
        DType dtype = onnxType switch
        {
            1 => DType.F32,
            2 => DType.U8,
            3 => DType.I8,
            6 => DType.I32,
            7 => DType.I64,
            9 => DType.Bool,
            _ => throw new OnnxSharpException(
                $"tensor '{info.Name}': data_type {onnxType} cannot be represented as a Tensor"),
        };

        var dims = new int[info.Shape.Length];
        for (int i = 0; i < dims.Length; i++)
            dims[i] = checked((int)info.Shape[i]);

        var tensor = Tensor.Allocate(new TensorShape(dims), dtype);
        raw.AsSpan(0, Math.Min(raw.Length, tensor.Buffer.Length)).CopyTo(tensor.Buffer);
        return tensor;
    }

    private static bool IsFloatTarget(int onnxType) => onnxType is 1 or 10 or 11 or 16;

    private static bool IsIntTarget(int onnxType) => onnxType is 2 or 3 or 4 or 5 or 6 or 7 or 9 or 12 or 13;
}
