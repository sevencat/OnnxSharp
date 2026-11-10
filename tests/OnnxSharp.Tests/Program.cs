using OnnxSharp.Tests;

// M0.5 验证套件。零第三方依赖，直接 dotnet run 即可。
// 退出码：0 = 全通过，1 = 有失败。

var cases = new List<TestCase>();
cases.AddRange(ArgTableTests.Cases());
cases.AddRange(BufferAllocatorTests.Cases());
cases.AddRange(EngineTests.Cases());
cases.AddRange(OnnxReaderTests.Cases());
cases.AddRange(OnnxShapeTests.Cases());
cases.AddRange(TransposeKernelTests.Cases());
cases.AddRange(VisionTests.Cases());
cases.AddRange(ImagingTests.Cases());
cases.AddRange(Int8Tests.Cases());
cases.AddRange(QuantOpsTests.Cases());
cases.AddRange(PadConv1DTests.Cases());
cases.AddRange(ElementwiseBroadcastTests.Cases());
cases.AddRange(MatMulArgMaxTests.Cases());
cases.AddRange(CifTests.Cases());
cases.AddRange(FbankTests.Cases());
cases.AddRange(AsrEndToEndTests.Cases());
cases.AddRange(AsrEncoderTests.Cases());
cases.AddRange(OcrTests.Cases());
cases.AddRange(Sam3Tests.Cases());
cases.AddRange(YoloGoldenTests.Cases());

return Harness.RunAll(cases);
