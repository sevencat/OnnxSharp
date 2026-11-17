# 它比 ORT、SimdYolo、SimdPaddleOCR 都慢：OnnxSharp 使用说明

先摆事实，再谈用法。

`OnnxSharp` 是一个纯 C# 手写的 ONNX 推理引擎（无 P/Invoke、无第三方依赖、只有 CPU 后端）。它**能跑**，但
**同样的模型、同一台机器上，它比所有能对照的成熟实现都慢** —— 这不是"某天负载高"的个例，而是这个项目的基本面。

作者写它不是为了做一个更快的引擎，而是为了搞懂"一个 ONNX 运行时到底是怎么跑起来的"。所以本文前半部分
先说清楚它慢在哪、慢多少，以及作者的水平差在哪；后半部分才是怎么用。

---

## 一、慢多少：三个对手，全部比它快

数据来自 `bench.md §37.3`（2026-11-17 四引擎同会话、每轮轮转、各取最好一轮）。

### 1. 目标检测（yolo11n，640×640，纯 forward，4 线程，median）

| 引擎 | median (ms) | 相对 OnnxSharp |
|---|---:|---|
| **OnnxSharp** | **30.4** | — |
| ONNXRuntime CPU | 23.8 | **快 1.28×** |
| SimdYolo | 31.2 | 我们略快（0.97×） |
| OpenCV 5 `dnn` | 48.8 | 慢 1.6× |

四个引擎都在 4 线程（本项目的基准口径）。早先 OnnxSharp 默认只给 4 个线程、SimdYolo 吃到 16 个，
差距被放大到 1.8×；现在默认线程是**物理核数（16）**，这个类型差已经不存在。
`s` 档同样打平：yolo11s 76.6 vs SimdYolo 77.8、yolov8s 90.2 vs 83.2（我们慢 8%）。

对 ORT 仍稳定落后 1.05~1.28×（F32 图），int8 的 decoder 落后 **1.67×**（它用 `vpdpbusd`，
OnnxSharp 走的是 F32 路径）。

### 2. OCR（PP-OCRv6 tiny，同一套 det/cls/rec 模型、同一份像素）

对手是 `Sdcb.SimdPaddleOCR 2.0.0`，4 线程，同会话 median：

| 图 | OnnxSharp | SimdPaddleOCR | 慢 | 2026-11-16 时 |
|---|---:|---:|---:|---:|
| 900×300，2 行 | 40.5 ms | 33.0 ms | **1.23×** | 1.9×（34.8 / 18.2，min） |
| 大图，3 行 | 53.3 ms | 47.7 ms | **1.12×** | 1.7×（59.7 / 34.3，min） |

分段（安静窗口 min）：det 13.2 / cls 1.6 / rec 2.9，端到端 24.2 ms —— 11-16 时是 det 17.9 / rec 4.4、端到端 34.3。
此前「光 `det` 一项就比对方整条管线还慢」已经不再成立，剩下的差距没有单项超过 5%（`bench.md` §37.2）。

### 3. 语音（sherpa Paraformer int8，中文）

| 段 | OnnxSharp | ONNXRuntime | 慢 |
|---|---:|---:|---:|
| encoder 纯 forward | 30.3 ms | 24.1 ms | 1.26× |
| decoder 纯 forward | 6.0 ms | 3.60 ms | **1.67×** |

### 4. 连 int8 都不如别人的 fp32

YuNet 人脸检测的 int8 版本能跑，但比同一模型的 fp32 版**慢 3.5×**（20.3 vs 5.7 ms，每层多两趟 f32 物化）。
别家上 int8 是加速，这个项目上 int8 是自残 —— 至今没修好。

**一句话总结**：想快，用 ONNXRuntime（CPU）、SimdYolo（YOLO 系）、`Sdcb.SimdPaddleOCR`（OCR）、
OpenCV 5 `dnn`（通用对照基线）。想读一份能调试的纯托管实现，再来这边。

---

## 二、作者水平不行，具体表现在哪

这不是客套式的自谦。作者是写业务代码出身，没系统学过编译器、数值线性代数和 SIMD 指令调度，
这个引擎基本是"照着一个能跑的目标，一边查资料一边拼出来的"。下面每一条都能在仓库里查到出处。

**1. SIMD 内核是照抄的。**
`Resize` / `GridSample` / `Pad` / `Slice` 的插值方式、`INTER_LINEAR` 的半像素对齐，全部按 OpenCV 5 `dnn`
的行为逐位对齐；`Conv` 的权重打包思路来自 SimdYolo 的 `CpuPackedGemm`。照抄之外自己写的那部分，
通常恰好就是慢的那部分。

**2. GEMM 分块参数是试出来的，不是算出来的。**
`RowBlock=4 / Panel=16 / SliceDepth` 这组值来自反复跑 `gemmbench` 的排列组合，作者说不出为什么是这几个数，
也解释不了 `s` 档（通道 128~256）为什么对 SimdYolo 掉到 1.63~1.78×。这是整个 YOLO 侧最大的短板，至今没解决。

**3. 结论被数据打脸过不止一次。**
一次是 OpenCV 那天跑出 45.8 ms（明显是对方自己的坏状态），被当成"已领先"写进结论；一次是 rfdetr-medium
对方的 304.4 ms，同样被当成基准。两次都在 `bench.md` 里自我更正，后来才定下规矩：只有双方绝对值都落在
各自历史区间内，比值才能写进结论。

**4. 连"哪边快"这个判断本身都错过。**
早期结论写着"OnnxSharp 稳定快于 ONNXRuntime CPU"，后来发现是 **median 口径 + 引擎连跑的顺序偏置**
共同造成的假象 —— 本机是功耗受限的 APU，排在前面的引擎总能吃到更高的 boost 档。改成每轮轮转之后，
ORT 从"更慢"变成"快 1.4×"。也就是说，作者曾经拿着一个方向相反的结论做了一轮优化。

**5. 知道慢，但定位不到。**
OCR 的 `det` 单项就慢过对方整条管线，作者能给的解释只有"小通道卷积的固定开销 + 逐行串行 rec"这种层面的话，
没定位到具体哪几行；`rec` 直到现在还是逐行串行跑，多线程只用在 `det` 上。

**6. 遇到不支持的图结构就绕道。**
引擎不支持 `Loop` / `Sequence` / 子图，于是语音识别要先用两个 Python 脚本
（`split_before_control_flow.py` / `split_decoder.py`）把 int8 模型切成 encoder、decoder 两段，
CIF 再手工移植一遍。这不叫设计，这叫躲。

**7. int8 做成负优化。**
YuNet 的 int8 比 fp32 慢 3.5×，三条优化尝试（见 `bench.md` §36）全部中性或负收益，最后只能关掉。
别人做 int8 是为了快 2~4×，这里做 int8 是为了多一条"支持"的条目。

**8. 微基准救了几次，说明 review 时看不出来。**
`ParallelRange.For` 一次空调用 14.4 µs、yolo11n 有 200+ 个并行区、累计 2~3 ms —— 这些问题都是靠
`parbench` / `profile` 量出来的，作者读代码时一个都没发现。

**9. SAM3 一次前向约 26 秒**（1008² ViT-L，FP32），只能离线批处理；4 线程预算也是刻意的
（服务器上这进程不该独占整机），代价就是第一节表里的 1.4× / 1.8×。

所以：**这是一个学习项目，不是产品**。它一定还有不少 bug，少见算子、边缘 dtype、异常路径的覆盖都不完整，
版本之间也可能有行为变化。踩到问题欢迎开 [issue](https://github.com/sevencat/OnnxSharp/issues)，
但请不要指望它明天就比 ORT 快。

---

## 三、用法：从装上到跑出结果

### 3.1 安装

```powershell
dotnet add package Sevencat.OnnxSharp --version 0.4.0
```

目标框架 `net8.0` / `net10.0` / `net11.0`，无运行时依赖。`net11.0` 才有 AVX-VNNI（int8 内核快 2.3×），
其它 TFM 自动回落到 F32 路径 —— 而这个项目的 int8 本来就慢，所以不必为它专门换 TFM。

从源码构建：

```powershell
dotnet build OnnxSharp.slnx -c Release
dotnet run --project tests/OnnxSharp.Tests -c Release
```

### 3.2 命令行：先确认模型能被读进来

```powershell
$cli = "tools\OnnxSharp.Cli\bin\Release\net10.0\OnnxSharp.Cli.exe"

& $cli info yolo11n.onnx                     # 算子统计、输入输出、opset
& $cli dump-graph yolo11n.onnx --type Conv   # 每个节点的推断形状
& $cli bench yolo11n.onnx --input in.f32 --iters 40 --warmup 5
& $cli profile yolo11n.onnx --input in.f32 --top 15   # 逐算子耗时排行
```

`--iters` 默认 1（第一帧真实延迟）。要稳态数字必须 `--iters 40`：.NET 分层编译大约 30 次调用后才升到 tier-1，
少于此数会虚高 1.5~2 倍 —— 这条也是作者被自己坑过之后才写进文档的。

一张图进、框出（内建预处理 + 解码，坐标是原图像素）：

```powershell
& $cli detect yolo11n.onnx photo.jpg --conf 0.25 --out marked.jpg
& $cli face yunet.onnx group.jpg --conf 0.6 --iou 0.3 --out faces.jpg
& $cli ocr page.jpg --det det.onnx --rec rec.onnx --dict keys.txt --cls cls.onnx --out page.ocr.jpg
```

`--json out.json` 会把预处理 / forward / decode 的分段耗时一起落盘，排查慢在哪比看总时间有用。

### 3.3 库用法：自己管张量

```csharp
using OnnxSharp.Core;
using OnnxSharp.Tensors;

var net = Net.ReadModel("yolo11n.onnx");      // 加载 + 常量折叠 + 融合
var input = Tensor.Allocate(new TensorShape(1, 3, 640, 640), DType.F32);
input.AsSpan<float>().Fill(0.5f);

net.SetInput(net.InputNames[0], input);
Tensor output = net.Forward();                 // [1, 84, 8400]

List<Tensor> all = net.ForwardAll();           // 多输出模型（如 RF-DETR）
Tensor dets = net.Forward("dets");
```

全动态输入（OCR 的 `x[N,3,H,W]`）必须先特化，否则 `Shape → Gather → Concat → Reshape` 链折不掉：

```csharp
var net = Net.ReadModel("det.onnx", new TensorShape(1, 3, 960, 960));
```

每个 `(H,W)` 都是一份特化图，流水线内部按形状缓存（检测 2 份 / 识别 8 份）。

### 3.4 库用法：一张图进、框出

```csharp
using OnnxSharp.Vision;

// 库不内置 JPEG/PNG 解码：像素由调用方给
byte[] rgb = LoadRgb24("photo.jpg", out int width, out int height);

var detector = new YoloDetector("yolo11n.onnx", new DetectorOptions
{
    Confidence = 0.25f,
    IouThreshold = 0.45f,
    MaxDetections = 300,
});

foreach (var d in detector.Detect(rgb, width, height))
    Console.WriteLine($"{d.ClassId} {d.Score:F3} [{d.X:F1},{d.Y:F1},{d.Width:F1}x{d.Height:F1}]");

// OpenCV Mat / 相机缓冲：BGR + 行跨度
foreach (var d in detector.Detect(matData, width, height, bgr: true, rowStride: stride))
    Console.WriteLine(d);
```

`bgr` 决定 R/B 顺序，`rowStride` 是每行字节数（0 = `width * 3`），带行尾填充的缓冲不会被复制压紧。

人脸（YuNet）注意三点：**整图拉伸而非 letterbox**、**0..255 原始值不归一化**、**默认 BGR**。

```csharp
var faces = new YuNetDetector("yunet.onnx", new FaceDetectorOptions { ScoreThreshold = 0.6f });
foreach (var f in faces.Detect(bgr, width, height))
    Console.WriteLine($"{f.Score:F3} [{f.X:F1},{f.Y:F1},{f.Width:F1}x{f.Height:F1}] eye {f.Landmarks.RightEye}");
```

OCR：

```csharp
using OnnxSharp.Ocr.PaddleOCR;

var model = PaddleOcrModel.FromFiles("det.onnx", "rec.onnx", "keys.txt", "cls.onnx");
var ocr = new PaddleOcrPipeline(model, new PaddleOcrOptions { DetLimitSide = 1600 });
foreach (OcrLine line in ocr.Run(pixels, width, height, OcrPixelFormat.Bgra32, stride))
    Console.WriteLine($"{line.Text} ({line.Score:F3})");
```

语音（Paraformer，需先用脚本把 int8 模型切成 encoder + decoder 两段）：

```csharp
var asr = new ParaformerRecognizer(new ParaformerOptions
{
    EncoderPath = @"sherpa\encoder.onnx",
    DecoderPath = @"sherpa\decoder.onnx",
    TokensPath = @"sherpa\tokens.txt",
});
Console.WriteLine(asr.TranscribeWav("test.wav").Text);
```

### 3.5 线程数

```powershell
$env:ONNXSHARP_THREADS = "4"     # 默认 = 物理核数（不是逻辑核数）
```

基准里的性能数字都是 4 线程下给的，因为目标场景是"这进程只是众多服务之一"。想要好看的跑分可以设大，
但那不是这个项目的目标，也没系统性测过。

---

## 四、遇到慢的时候怎么自查

1. `profile --top 15` 看热点是不是落在某一层卷积上（YuNet 的 `Conv_0` 一层就占 forward 的 15%）；
2. 确认模型没走 int8：int8 在这个项目里不但不加速，反而可能慢 3.5×；
3. 确认 `--iters 40`，否则量到的是 JIT 冷启动；
4. 确认没有别的进程抢核（`bench.md` 里整轮数据作废过一次，就是 `SC2_x64` 满载）；
5. 还慢就用 ORT。真的，不丢人。

---

## 五、它适合谁

- 想读一份**能改、能断点、能打印中间张量**的 ONNX 实现：仓库里 protobuf 解析、图优化、SIMD 内核、
  算子注册全在一个 sln 里，纯 C#，没有黑盒 native 库；
- 教学 / 实验：加一个算子、换一种打包方式、A/B 两次 `bench` 就能看到结果；
- 零依赖部署：目标机上不能有 CUDA、不能有 native so/dll 的场合（代价就是慢）。

**不适合**：任何对延迟有硬要求的生产场景。那里请用 ONNXRuntime、SimdYolo、
`Sdcb.SimdPaddleOCR`、OpenCV 5 `dnn`。它们都比这个快。

相关文档：`README.md`（完整用法与性能表）、`bench.md`（全部实测与复现命令）、`plan.md` / `progressv2.md`
（设计与进度，含做错又改回来的记录）。
