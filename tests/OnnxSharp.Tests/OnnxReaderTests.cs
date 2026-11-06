using System.Buffers.Binary;
using OnnxSharp.Core;
using OnnxSharp.Onnx;
using OnnxSharp.Tensors;

namespace OnnxSharp.Tests;

internal static class OnnxReaderTests
{
    private const string Yolo11n = @"F:\ai\models\yolo\yolo11n.onnx";
    private const string RfDetrSmall = @"F:\ai\rfdetr\onnx_model\rfdetr-small.onnx";

    public static IEnumerable<TestCase> Cases()
    {
        yield return new("proto: varint multi-byte and negative int64", VarintRoundTrip);
        yield return new("proto: packed int32_data / bool", PackedInt32Data);
        yield return new("proto: packed float_data", PackedFloatData);
        yield return new("proto: truncated message throws", TruncatedThrows);
        yield return new("tensor: raw_data int64", RawDataInt64);
        yield return new("onnx: yolo11n structure", Yolo11nStructure);
        yield return new("onnx: rfdetr-small structure", RfDetrStructure);
    }

    // ---------------- 手工构造 protobuf ----------------

    private static void WriteVarint(List<byte> sink, ulong value)
    {
        while (value >= 0x80)
        {
            sink.Add((byte)(value | 0x80));
            value >>= 7;
        }
        sink.Add((byte)value);
    }

    private static void AddVarintField(List<byte> sink, int field, ulong value)
    {
        WriteVarint(sink, (ulong)((field << 3) | 0));
        WriteVarint(sink, value);
    }

    private static void AddBytesField(List<byte> sink, int field, ReadOnlySpan<byte> payload)
    {
        WriteVarint(sink, (ulong)((field << 3) | 2));
        WriteVarint(sink, (ulong)payload.Length);
        sink.AddRange(payload.ToArray());
    }

    private static Proto BuildTensorProto(int[] dims, int dataType, Action<List<byte>> body)
    {
        var bytes = new List<byte>(32);
        foreach (var dim in dims)
            AddVarintField(bytes, 1, (ulong)dim);
        AddVarintField(bytes, 2, (ulong)dataType);
        body(bytes);
        return Proto.Parse(bytes.ToArray());
    }

    // ---------------- 测试 ----------------

    private static void VarintRoundTrip()
    {
        // 300 需要两字节；-1 走 10 字节补码
        var bytes = new List<byte>();
        AddVarintField(bytes, 1, 300);
        WriteVarint(bytes, (1 << 3) | 0);
        WriteVarint(bytes, unchecked((ulong)-1));

        var proto = Proto.Parse(bytes.ToArray());
        var values = proto.Ints(1);

        Assert.Equal(2, values.Length, "two varint values");
        Assert.Equal(300L, values[0], "multi-byte varint");
        Assert.Equal(-1L, values[1], "negative int64 via two's complement");
    }

    private static void PackedInt32Data()
    {
        var tensor = BuildTensorProto(new[] { 3 }, 9 /* BOOL */, bytes =>
        {
            var packed = new List<byte>();
            WriteVarint(packed, 0);
            WriteVarint(packed, 1);
            WriteVarint(packed, 2);
            AddBytesField(bytes, 5, packed.ToArray());
        });

        var decoded = OnnxFile.CreateBare(".").DecodeTensor(tensor, "mask");

        Assert.Equal(DType.Bool, decoded.DType, "dtype");
        Assert.Equal(new TensorShape(3), decoded.Shape, "shape");
        Assert.Equal(3, decoded.ByteLength, "bool is 1 byte each");
        var bytes = decoded.Buffer.AsSpan(0, 3).ToArray();
        Assert.Equal((byte)0, bytes[0], "0 -> false");
        Assert.Equal((byte)1, bytes[1], "1 -> true");
        Assert.Equal((byte)1, bytes[2], "2 -> true (non-zero)");
    }

    private static void PackedFloatData()
    {
        var tensor = BuildTensorProto(new[] { 2, 2 }, 1 /* FLOAT */, bytes =>
        {
            var packed = new List<byte>();
            Span<byte> buffer = stackalloc byte[4];
            foreach (var v in new[] { 1.5f, -2f, 3f, 4.25f })
            {
                BinaryPrimitives.WriteSingleLittleEndian(buffer, v);
                packed.AddRange(buffer.ToArray());
            }
            AddBytesField(bytes, 4, packed.ToArray());
        });

        var decoded = OnnxFile.CreateBare(".").DecodeTensor(tensor, "w");

        Assert.Equal(new TensorShape(2, 2), decoded.Shape, "shape");
        var values = decoded.AsSpan<float>().ToArray();
        Assert.Near(1.5f, values[0], 1e-7f, "value[0]");
        Assert.Near(-2f, values[1], 1e-7f, "value[1]");
        Assert.Near(3f, values[2], 1e-7f, "value[2]");
        Assert.Near(4.25f, values[3], 1e-7f, "value[3]");
    }

    private static void TruncatedThrows()
    {
        // 声明 16 字节的 raw_data，实际只给 4 字节
        var bytes = new List<byte>();
        AddVarintField(bytes, 1, 4);
        AddVarintField(bytes, 2, 1 /* FLOAT */);
        WriteVarint(bytes, (9 << 3) | 2);
        WriteVarint(bytes, 16);
        bytes.AddRange(new byte[4]);

        Assert.Throws<OnnxSharpException>(() => Proto.Parse(bytes.ToArray()), "truncated length-delimited field");
    }

    private static void RawDataInt64()
    {
        var raw = new byte[24];
        for (int i = 0; i < 3; i++)
            BinaryPrimitives.WriteInt64LittleEndian(raw.AsSpan(i * 8, 8), (i + 1) * 10L);

        var tensor = BuildTensorProto(new[] { 3 }, 7 /* INT64 */, bytes => AddBytesField(bytes, 9, raw));

        var decoded = OnnxFile.CreateBare(".").DecodeTensor(tensor, "shape");

        Assert.Equal(DType.I64, decoded.DType, "dtype");
        Assert.Equal(new TensorShape(3), decoded.Shape, "shape");
        Assert.Equal(24, decoded.ByteLength, "byte length");
        var values = decoded.AsSpan<long>().ToArray();
        Assert.Equal(10L, values[0], "value[0]");
        Assert.Equal(20L, values[1], "value[1]");
        Assert.Equal(30L, values[2], "value[2]");
    }

    private static void Yolo11nStructure()
    {
        RequireModel(Yolo11n);
        var file = OnnxFile.Load(Yolo11n);

        Console.WriteLine($"        yolo11n: opset={file.Opset} nodes={file.Nodes.Count} " +
                          $"init={file.Initializers.Count} producer='{file.Producer}'");

        Assert.Equal(1, file.Inputs.Count, "single input");
        Assert.Equal("images", file.Inputs[0].Name, "input name");
        Assert.Equal(DType.F32, file.Inputs[0].Type, "input dtype");
        Assert.Equal(new TensorShape(1, 3, 640, 640), ToShape(file.Inputs[0].Dims), "input shape");

        Assert.Equal(1, file.Outputs.Count, "single output");
        Assert.Equal("output0", file.Outputs[0].Name, "output name");
        Assert.Equal(new TensorShape(1, 84, 8400), ToShape(file.Outputs[0].Dims), "output shape");

        Assert.True(file.Nodes.Count > 100, "node count");
        Assert.True(file.Initializers.Count > 100, "initializer count");
        Assert.True(file.Opset >= 13, "opset must be modern enough");

        foreach (var node in file.Nodes)
            Assert.False(string.IsNullOrEmpty(node.OpType), "every node needs op_type");

        // 每个 initializer 都已成功解码（Load 不抛即证明），抽查非空
        foreach (var (name, tensor) in file.Initializers)
            Assert.True(tensor.DType != DType.Invalid, $"initializer '{name}' dtype");
    }

    private static void RfDetrStructure()
    {
        RequireModel(RfDetrSmall);
        var file = OnnxFile.Load(RfDetrSmall);

        Console.WriteLine($"        rfdetr-small: opset={file.Opset} nodes={file.Nodes.Count} " +
                          $"init={file.Initializers.Count} input='{file.Inputs[0].Name}'");

        Assert.Equal(1, file.Inputs.Count, "single input");
        Assert.True(file.Nodes.Count > 500, "node count");

        // RF-DETR 的 I64/Bool 常量多以 Constant 节点属性形式出现（而非 initializer），
        // 是 GridSample / Where / Gather 的形状与索引来源。
        var histogram = new Dictionary<DType, int>();
        int constants = 0;
        foreach (var node in file.Nodes)
        {
            if (node.OpType != "Constant")
                continue;
            constants++;
            var value = node.Tensor("value");
            if (value is null)
                continue;
            histogram[value.DType] = histogram.GetValueOrDefault(value.DType) + 1;
        }

        int ints = histogram.GetValueOrDefault(DType.I64)
                 + histogram.GetValueOrDefault(DType.I32)
                 + histogram.GetValueOrDefault(DType.Bool);

        Console.WriteLine($"        rfdetr-small: constant nodes={constants} dtypes=" +
                          string.Join(",", histogram.Select(kv => $"{kv.Key}:{kv.Value}")));

        Assert.True(constants > 0, "expected Constant nodes");
        Assert.True(ints > 0, "expected integer/bool constant tensors");
    }

    // ---------------- 工具 ----------------

    private static TensorShape ToShape(int[] dims) => new(dims);

    private static void RequireModel(string path)
    {
        if (!File.Exists(path))
            throw new TestSkipException($"model not found: {path}");
    }
}
