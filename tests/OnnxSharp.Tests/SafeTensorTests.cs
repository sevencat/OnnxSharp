using System.Buffers.Binary;
using System.Text;
using OnnxSharp.Core;
using OnnxSharp.Onnx;
using OnnxSharp.Safetensors;
using OnnxSharp.Tensors;

namespace OnnxSharp.Tests;

/// <summary>
/// safetensors 读取 + ONNX 权重补丁（形态 A：骨架 ONNX + safetensors 权重 → 新 ONNX）。
/// <para>
/// 这里<b>完全离线自足</b>：ONNX 骨架、safetensors 文件都在测试里手工拼字节，
/// 因此可以拿「已知权重」跑真前向，逐值验证补丁确实生效，而不是只检查文件大小。
/// </para>
/// </summary>
internal static class SafeTensorTests
{
    public static IEnumerable<TestCase> Cases()
    {
        yield return new("safetensors: header / dtype / values", HeaderAndValues);
        yield return new("safetensors: metadata and scalar", MetadataAndScalar);
        yield return new("safetensors: broken header reports file and tensor", BrokenHeader);
        yield return new("patcher: patch replaces weights (numeric)", PatchReplacesWeights);
        yield return new("patcher: strip prefix matches suffixed names", StripPrefixMatches);
        yield return new("patcher: shape mismatch is actionable", ShapeMismatchMessage);
        yield return new("patcher: external data path round-trips", ExternalDataRoundTrip);
        yield return new("extract: weights + skeleton round-trip through patch", ExtractRoundTrip);
        yield return new("extract: typed int64_data is decoded", ExtractInt64Data);
    }

    // ---------------- 用例 ----------------

    private static void HeaderAndValues()
    {
        string directory = NewTempDirectory();
        try
        {
            // F32 [2,2] + F16 [2] + I64 [1]
            byte[] f32 = Floats(1f, 2f, 3f, 4f);
            byte[] f16 = Halves(0.5f, -1.5f);
            byte[] i64 = Int64s(7L);

            string path = BuildSafeTensorFile(
                directory,
                "w.safetensors",
                new TensorSpec("a.weight", "F32", new long[] { 2, 2 }, f32),
                new TensorSpec("b.bias", "F16", new long[] { 2 }, f16),
                new TensorSpec("c.steps", "I64", new long[] { 1 }, i64));

            using var file = SafeTensorFile.Open(path);

            Assert.True(file.Tensors.Count == 3, $"expected 3 tensors, got {file.Tensors.Count}");
            Assert.True(file.Names[0] == "a.weight", $"unexpected first name '{file.Names[0]}'");

            var weight = file.Info("a.weight");
            Assert.True(weight.ByteLength == 16, $"a.weight bytes {weight.ByteLength} != 16");
            Assert.True(weight.ElementCount == 4, $"a.weight count {weight.ElementCount} != 4");

            var tensor = file.GetTensor("a.weight");
            Assert.True(tensor.DType == DType.F32, $"a.weight dtype {tensor.DType} != F32");
            Assert.True(tensor.Shape.Dims.SequenceEqual(new[] { 2, 2 }), "a.weight shape mismatch");
            var values = tensor.AsSpan<float>().ToArray();
            Assert.True(values.SequenceEqual(new[] { 1f, 2f, 3f, 4f }), "a.weight values mismatch");

            // F16 → F32 的转换是「加载权重」的常见路径
            var half = file.GetTensor("b.bias");
            Assert.True(half.DType == DType.F32, $"b.bias should load as F32, got {half.DType}");
            var halfValues = half.AsSpan<float>().ToArray();
            Assert.True(Math.Abs(halfValues[0] - 0.5f) < 1e-6 && Math.Abs(halfValues[1] + 1.5f) < 1e-6,
                $"b.bias values [{halfValues[0]}, {halfValues[1]}] mismatch");

            var steps = file.GetTensor("c.steps");
            Assert.True(steps.DType == DType.I64, $"c.steps dtype {steps.DType} != I64");
            Assert.True(steps.AsSpan<long>()[0] == 7, "c.steps value mismatch");
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    private static void MetadataAndScalar()
    {
        string directory = NewTempDirectory();
        try
        {
            string path = BuildSafeTensorFile(
                directory,
                "meta.safetensors",
                new TensorSpec("scale", "F32", Array.Empty<long>(), Floats(2.5f)),
                metadata: new Dictionary<string, string>(StringComparer.Ordinal) { ["format"] = "pt" });

            using var file = SafeTensorFile.Open(path);
            Assert.True(file.Metadata.TryGetValue("format", out string? format) && format == "pt",
                "metadata 'format' missing");
            Assert.True(file.Tensors.Count == 1, $"expected 1 tensor, got {file.Tensors.Count}");

            var scalar = file.GetTensor("scale");
            Assert.True(scalar.Shape.Dims.Length == 0, "scalar should have rank 0");
            Assert.True(Math.Abs(scalar.AsSpan<float>()[0] - 2.5f) < 1e-6f, "scalar value mismatch");
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    private static void BrokenHeader()
    {
        string directory = NewTempDirectory();
        try
        {
            // data_offsets 与 dtype×元素数 不一致：必须报出文件名和张量名
            string path = Path.Combine(directory, "bad.safetensors");
            byte[] header = Encoding.UTF8.GetBytes(
                "{\"w\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0,4]}}");
            var bytes = new List<byte>();
            bytes.AddRange(BitConverter.GetBytes((ulong)header.Length));
            bytes.AddRange(header);
            bytes.AddRange(new byte[8]);
            File.WriteAllBytes(path, bytes.ToArray());

            bool threw = false;
            try
            {
                using var file = SafeTensorFile.Open(path);
            }
            catch (OnnxSharpException e)
            {
                threw = true;
                Assert.True(e.Message.Contains("bad.safetensors", StringComparison.Ordinal)
                            && e.Message.Contains("'w'", StringComparison.Ordinal),
                    $"error message should name file and tensor, got: {e.Message}");
            }

            Assert.True(threw, "mismatched data_offsets should throw");
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    private static void PatchReplacesWeights()
    {
        string directory = NewTempDirectory();
        try
        {
            string skeleton = BuildSkeleton(Path.Combine(directory, "skeleton.onnx"));
            string weights = BuildSafeTensorFile(
                directory,
                "trained.safetensors",
                new TensorSpec("W", "F32", new long[] { 2, 3 }, Floats(1f, 2f, 3f, 4f, 5f, 6f)),
                new TensorSpec("B", "F32", new long[] { 3 }, Floats(0.5f, 0.6f, 0.7f)));

            string output = Path.Combine(directory, "patched.onnx");
            using (var file = SafeTensorFile.Open(weights))
            {
                var report = OnnxWeightPatcher.Apply(skeleton, file, output, new OnnxWeightPatchOptions { Strict = true });
                Assert.True(report.Patched == 2, $"patched {report.Patched} != 2");
                Assert.True(report.KeptOriginal == 0, $"kept {report.KeptOriginal} != 0");
                Assert.True(report.MissingInWeights.Count == 0, "unexpected missing initializers");
                Assert.True(report.UnusedWeights.Count == 0, "unexpected unused weights");
                Assert.True(report.InlineBytes == 36, $"inline bytes {report.InlineBytes} != 36");
            }

            // 骨架里 W/B 都是 0 → 不打补丁时输出必须全 0（证明下面 5.5/7.6/9.7 来自补丁）
            Expect(RunOnce(skeleton, 1f, 1f), 0f, 0f, 0f);

            // x=[1,1]：x·W = [5,7,9]，再 + B
            Expect(RunOnce(output, 1f, 1f), 5.5f, 7.6f, 9.7f);

            // 另一个输入再核一次，避免「恰好对上」
            Expect(RunOnce(output, 2f, 0f), 2.5f, 4.6f, 6.7f);
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    private static void StripPrefixMatches()
    {
        string directory = NewTempDirectory();
        try
        {
            // 训练框架的 state_dict 带 model. 前缀，骨架里的名字不带 → 靠 StripPrefixes 补上
            string skeleton = BuildSkeleton(Path.Combine(directory, "skeleton.onnx"));
            string weights = BuildSafeTensorFile(
                directory,
                "prefixed.safetensors",
                new TensorSpec("model.W", "F32", new long[] { 2, 3 }, Floats(1f, 0f, 0f, 0f, 1f, 0f)),
                new TensorSpec("model.B", "F32", new long[] { 3 }, Floats(0f, 0f, 1f)));

            string output = Path.Combine(directory, "patched.onnx");
            using (var file = SafeTensorFile.Open(weights))
            {
                var report = OnnxWeightPatcher.Apply(skeleton, file, output);
                Assert.True(report.Patched == 2, $"patched {report.Patched} != 2 (prefix matching failed)");
            }

            // W 只保留对角 → x=[1,2] 得 [1,2,1]
            Expect(RunOnce(output, 1f, 2f), 1f, 2f, 1f);
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    private static void ShapeMismatchMessage()
    {
        string directory = NewTempDirectory();
        try
        {
            string skeleton = BuildSkeleton(Path.Combine(directory, "skeleton.onnx"));
            string weights = BuildSafeTensorFile(
                directory,
                "transposed.safetensors",
                new TensorSpec("W", "F32", new long[] { 3, 2 }, Floats(1f, 2f, 3f, 4f, 5f, 6f)));

            using var file = SafeTensorFile.Open(weights);
            bool threw = false;
            try
            {
                OnnxWeightPatcher.Apply(skeleton, file, Path.Combine(directory, "out.onnx"));
            }
            catch (OnnxSharpException e)
            {
                threw = true;
                Assert.True(e.Message.Contains("fuse=False", StringComparison.Ordinal),
                    $"error should mention the fuse=False hint, got: {e.Message}");
            }

            Assert.True(threw, "shape mismatch should throw");
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    private static void ExternalDataRoundTrip()
    {
        string directory = NewTempDirectory();
        try
        {
            string skeleton = BuildSkeleton(Path.Combine(directory, "skeleton.onnx"));
            string weights = BuildSafeTensorFile(
                directory,
                "trained.safetensors",
                new TensorSpec("W", "F32", new long[] { 2, 3 }, Floats(1f, 2f, 3f, 4f, 5f, 6f)),
                new TensorSpec("B", "F32", new long[] { 3 }, Floats(0.5f, 0.6f, 0.7f)));

            string output = Path.Combine(directory, "external.onnx");
            using (var file = SafeTensorFile.Open(weights))
            {
                // 阈值设成 1 字节：36 字节权重必须走外部数据文件
                var options = new OnnxWeightPatchOptions { ExternalDataThreshold = 1 };
                var report = OnnxWeightPatcher.Apply(skeleton, file, output, options);

                Assert.True(report.ExternalDataPath is not null, "expected an external data file");
                Assert.True(File.Exists(report.ExternalDataPath!), $"missing {report.ExternalDataPath}");
                Assert.True(report.ExternalBytes == 36, $"external bytes {report.ExternalBytes} != 36");
                Assert.True(report.InlineBytes == 0, $"inline bytes {report.InlineBytes} != 0");
            }

            // 引擎读 external_data（含路径逃逸校验）应当照常工作
            Expect(RunOnce(output, 1f, 1f), 5.5f, 7.6f, 9.7f);

            // 换目录也能用：外部数据按 location 相对模型目录解析
            string moved = Path.Combine(directory, "moved");
            Directory.CreateDirectory(moved);
            string movedModel = Path.Combine(moved, Path.GetFileName(output));
            File.Copy(output, movedModel);
            File.Copy(output + ".data", movedModel + ".data");
            Expect(RunOnce(movedModel, 1f, 1f), 5.5f, 7.6f, 9.7f);
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    private static void ExtractRoundTrip()
    {
        string directory = NewTempDirectory();
        try
        {
            byte[] w = Floats(1f, 2f, 3f, 4f, 5f, 6f);
            byte[] b = Floats(0.5f, 0.6f, 0.7f);
            string model = BuildSkeleton(Path.Combine(directory, "model.onnx"), w, b);

            // 原模型可直接推理 → 作为基准
            Expect(RunOnce(model, 1f, 1f), 5.5f, 7.6f, 9.7f);

            string weights = Path.Combine(directory, "extracted.safetensors");
            string skeleton = Path.Combine(directory, "skeleton.onnx");
            var report = OnnxWeightPatcher.Extract(model, weights, skeleton);

            Assert.True(report.Tensors == 2, $"extracted {report.Tensors} tensors, expected 2");
            Assert.True(report.WeightBytes == 36, $"extracted {report.WeightBytes} B, expected 36");
            Assert.True(report.SkeletonBytes > 0 && report.SkeletonBytes < new FileInfo(model).Length,
                $"skeleton {report.SkeletonBytes} B should be smaller than the model");

            using var file = SafeTensorFile.Open(weights);

            // 抽出来的字节必须与 ONNX 里的权重逐字节一致，dtype 也要还原对
            Assert.True(file.ReadRaw("W").SequenceEqual(w), "extracted W bytes differ from the ONNX tensor");
            Assert.True(file.ReadRaw("B").SequenceEqual(b), "extracted B bytes differ from the ONNX tensor");
            Assert.True(file.Info("W").DType == "F32", $"W dtype should be F32, got {file.Info("W").DType}");
            Assert.True(file.Metadata.TryGetValue("source", out string? source) && source == "model.onnx",
                "metadata 'source' should point at the model file");

            // 骨架 + 抽出来的权重 → 打回去，必须与原模型数值一致
            string restored = Path.Combine(directory, "restored.onnx");
            var patchReport = OnnxWeightPatcher.Apply(skeleton, file, restored, new OnnxWeightPatchOptions { Strict = true });
            Assert.True(patchReport.Patched == 2, $"patched {patchReport.Patched} != 2");
            Expect(RunOnce(restored, 1f, 1f), 5.5f, 7.6f, 9.7f);

            // 剥掉数据的骨架本身不能推理（这正是「骨架」的定义）
            bool threw = false;
            try
            {
                RunOnce(skeleton, 1f, 1f);
            }
            catch (Exception)
            {
                threw = true;
            }

            Assert.True(threw, "a weight-stripped skeleton must not be runnable without weights");
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    private static void ExtractInt64Data()
    {
        string directory = NewTempDirectory();
        try
        {
            // int64 走 typed 字段（int64_data）而不是 raw_data，且含负数（varint 符号扩展）
            var graph = new List<byte>();
            Field(graph, 1, Node(new[] { "x", "T" }, new[] { "y" }, "Identity", "id"));
            StrField(graph, 2, "g");
            Field(graph, 5, TensorInitInt64("T", new long[] { 3 }, new long[] { 1, -2, 3 }));
            Field(graph, 11, ValueInfo("x", 7, new long[] { 3 }));
            Field(graph, 12, ValueInfo("y", 7, new long[] { 3 }));

            var model = new List<byte>();
            Field(model, 1, 8);
            StrField(model, 2, "onnxsharp-tests");
            Field(model, 7, graph.ToArray());
            Field(model, 8, OpsetId(17));

            string modelPath = Path.Combine(directory, "int64.onnx");
            File.WriteAllBytes(modelPath, model.ToArray());

            string weightsPath = Path.Combine(directory, "int64.safetensors");
            var report = OnnxWeightPatcher.Extract(modelPath, weightsPath);
            Assert.True(report.Tensors == 1, $"extracted {report.Tensors} tensors, expected 1");

            using var file = SafeTensorFile.Open(weightsPath);
            Assert.True(file.Info("T").DType == "I64", $"T dtype should be I64, got {file.Info("T").DType}");
            Assert.True(file.ReadRaw("T").SequenceEqual(Int64s(1, -2, 3)),
                "typed int64_data should decode to little-endian int64 bytes");
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    // ---------------- 工具 ----------------

    private sealed record TensorSpec(string Name, string DType, long[] Shape, byte[] Data);

    private static string NewTempDirectory()
    {
        string directory = Path.Combine(Path.GetTempPath(), "onnxsharp-sft-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(directory);
        return directory;
    }

    private static void Expect(float[] actual, params float[] expected)
    {
        Assert.True(actual.Length == expected.Length, $"expected {expected.Length} values, got {actual.Length}");
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(actual[i] - expected[i]) < 1e-5f,
                $"value[{i}] expected {expected[i]}, got {actual[i]}");
    }

    private static float[] RunOnce(string modelPath, params float[] input)
    {
        var net = Net.ReadModel(modelPath);
        net.SetInput(net.InputNames[0], Tensor.FromArray(input, new TensorShape(1, input.Length)));
        return net.Forward().AsSpan<float>().ToArray();
    }

    private static byte[] Floats(params float[] values)
    {
        var bytes = new byte[values.Length * 4];
        for (int i = 0; i < values.Length; i++)
            BinaryPrimitives.WriteSingleLittleEndian(bytes.AsSpan(i * 4, 4), values[i]);
        return bytes;
    }

    private static byte[] Halves(params float[] values)
    {
        var bytes = new byte[values.Length * 2];
        for (int i = 0; i < values.Length; i++)
            BinaryPrimitives.WriteUInt16LittleEndian(bytes.AsSpan(i * 2, 2), BitConverter.HalfToUInt16Bits((Half)values[i]));
        return bytes;
    }

    private static byte[] Int64s(params long[] values)
    {
        var bytes = new byte[values.Length * 8];
        for (int i = 0; i < values.Length; i++)
            BinaryPrimitives.WriteInt64LittleEndian(bytes.AsSpan(i * 8, 8), values[i]);
        return bytes;
    }

    private static string BuildSafeTensorFile(string directory, string fileName, params TensorSpec[] tensors)
        => BuildSafeTensorFileCore(directory, fileName, tensors, null);

    private static string BuildSafeTensorFile(
        string directory, string fileName, TensorSpec tensor, Dictionary<string, string> metadata)
        => BuildSafeTensorFileCore(directory, fileName, new[] { tensor }, metadata);

    private static string BuildSafeTensorFileCore(
        string directory, string fileName, TensorSpec[] tensors, Dictionary<string, string>? metadata)
    {
        var header = new StringBuilder("{");
        var data = new List<byte>();
        long offset = 0;
        bool first = true;

        foreach (var tensor in tensors)
        {
            if (!first)
                header.Append(',');
            first = false;

            header.Append('"').Append(tensor.Name).Append("\":{");
            header.Append("\"dtype\":\"").Append(tensor.DType).Append("\",");
            header.Append("\"shape\":[").Append(string.Join(",", tensor.Shape)).Append("],");
            header.Append("\"data_offsets\":[").Append(offset).Append(',')
                .Append(offset + tensor.Data.Length).Append("]}");

            offset += tensor.Data.Length;
            data.AddRange(tensor.Data);
        }

        if (metadata is not null)
        {
            if (!first)
                header.Append(',');
            header.Append("\"__metadata__\":{");

            bool firstMeta = true;
            foreach (var (key, value) in metadata)
            {
                if (!firstMeta)
                    header.Append(',');
                firstMeta = false;
                header.Append('"').Append(key).Append("\":\"").Append(value).Append('"');
            }

            header.Append('}');
        }

        header.Append('}');

        // spec：头部用空格填充，使数据区从 8 字节对齐处开始
        var headerBytes = new List<byte>(Encoding.UTF8.GetBytes(header.ToString()));
        while ((8 + headerBytes.Count) % 8 != 0)
            headerBytes.Add((byte)' ');

        var file = new List<byte>();
        file.AddRange(BitConverter.GetBytes((ulong)headerBytes.Count));
        file.AddRange(headerBytes);
        file.AddRange(data);

        string path = Path.Combine(directory, fileName);
        File.WriteAllBytes(path, file.ToArray());
        return path;
    }

    /// <summary>造一个最小骨架：<c>y = MatMul(x, W) + B</c>，W=[2,3]、B=[3]；默认权重全 0。</summary>
    private static string BuildSkeleton(string path, byte[]? weightBytes = null, byte[]? biasBytes = null)
    {
        var graph = new List<byte>();
        Field(graph, 1, Node(new[] { "x", "W" }, new[] { "t" }, "MatMul", "mm"));
        Field(graph, 1, Node(new[] { "t", "B" }, new[] { "y" }, "Add", "add"));
        StrField(graph, 2, "skeleton");
        Field(graph, 5, TensorInit("W", 1, new long[] { 2, 3 }, weightBytes ?? new byte[24]));
        Field(graph, 5, TensorInit("B", 1, new long[] { 3 }, biasBytes ?? new byte[12]));
        Field(graph, 11, ValueInfo("x", 1, new long[] { 1, 2 }));
        Field(graph, 12, ValueInfo("y", 1, new long[] { 1, 3 }));

        var model = new List<byte>();
        Field(model, 1, 8);                    // ir_version
        StrField(model, 2, "onnxsharp-tests"); // producer_name
        Field(model, 7, graph.ToArray());      // graph
        Field(model, 8, OpsetId(17));          // opset_import

        File.WriteAllBytes(path, model.ToArray());
        return path;
    }

    private static byte[] OpsetId(long version)
    {
        var opset = new List<byte>();
        Field(opset, 2, (ulong)version);
        return opset.ToArray();
    }

    private static byte[] Node(string[] inputs, string[] outputs, string opType, string name)
    {
        var node = new List<byte>();
        foreach (string input in inputs)
            StrField(node, 1, input);
        foreach (string output in outputs)
            StrField(node, 2, output);
        StrField(node, 3, name);
        StrField(node, 4, opType);
        return node.ToArray();
    }

    private static byte[] TensorInit(string name, int dataType, long[] dims, byte[] raw)
    {
        var tensor = new List<byte>();

        var packed = new List<byte>();
        foreach (long dim in dims)
            Varint(packed, (ulong)dim);
        if (packed.Count > 0)
            Field(tensor, 1, packed.ToArray());

        Field(tensor, 2, (ulong)dataType);
        StrField(tensor, 8, name);
        Field(tensor, 9, raw);
        return tensor.ToArray();
    }

    /// <summary>int64 initializer，数据放在 typed 字段（<c>int64_data</c>）而不是 <c>raw_data</c>。</summary>
    private static byte[] TensorInitInt64(string name, long[] dims, long[] values)
    {
        var tensor = new List<byte>();

        var packedDims = new List<byte>();
        foreach (long dim in dims)
            Varint(packedDims, (ulong)dim);
        if (packedDims.Count > 0)
            Field(tensor, 1, packedDims.ToArray());

        Field(tensor, 2, 7); // INT64
        StrField(tensor, 8, name);

        var packedValues = new List<byte>();
        foreach (long value in values)
            Varint(packedValues, unchecked((ulong)value));
        Field(tensor, 7, packedValues.ToArray());

        return tensor.ToArray();
    }

    private static byte[] ValueInfo(string name, int elemType, long[] dims)
    {
        var tensorType = new List<byte>();
        Field(tensorType, 1, (ulong)elemType);

        var shape = new List<byte>();
        foreach (long dim in dims)
        {
            var dimension = new List<byte>();
            Field(dimension, 1, (ulong)dim);
            Field(shape, 1, dimension.ToArray());
        }

        Field(tensorType, 2, shape.ToArray());

        var type = new List<byte>();
        Field(type, 1, tensorType.ToArray());

        var info = new List<byte>();
        StrField(info, 1, name);
        Field(info, 2, type.ToArray());
        return info.ToArray();
    }

    private static void Varint(List<byte> sink, ulong value)
    {
        while (value >= 0x80)
        {
            sink.Add((byte)(value | 0x80));
            value >>= 7;
        }

        sink.Add((byte)value);
    }

    private static void Field(List<byte> sink, int number, ulong value)
    {
        Varint(sink, (ulong)((number << 3) | 0));
        Varint(sink, value);
    }

    private static void Field(List<byte> sink, int number, ReadOnlySpan<byte> payload)
    {
        Varint(sink, (ulong)((number << 3) | 2));
        Varint(sink, (ulong)payload.Length);
        foreach (byte b in payload)
            sink.Add(b);
    }

    private static void StrField(List<byte> sink, int number, string text)
        => Field(sink, number, Encoding.UTF8.GetBytes(text));
}
