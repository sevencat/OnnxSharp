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
cases.AddRange(OcrTests.Cases());
cases.AddRange(YoloGoldenTests.Cases());

return Harness.RunAll(cases);
