# OnnxSharp

纯 C# 实现的 ONNX 推理引擎（CPU / FP32，零第三方依赖），可直接跑 YOLO11 / YOLOv8 / RF-DETR 的 ONNX 导出，并附带「一张图进、检测框出」的可选门面。

**中文** ｜ [English](#english)

---

## 中文

### 这是什么

`OnnxSharp` 是一个自己实现的 ONNX 运行时：读 ONNX 文件 → 建图 → 常量折叠 + 算子融合 → 缓冲池 + 多线程 SIMD 内核 → 按节点执行。语义对齐 OpenCV 5 `dnn` 与 SimdYolo 的做法，目标是「纯托管、无 P/Invoke、纯 CPU 也能跑检测模型」。

实测与本机 OpenCV 5 dnn 相差 1.15~1.30×（见 [性能](#性能)）。

### 特性

| 项 | 说明 |
|---|---|
| 依赖 | 零第三方包（不引用 `Google.Protobuf`、`Microsoft.ML.OnnxRuntime`），全部手写 |
| 目标框架 | `net8.0` / `net10.0`，AOT 友好（不用反射、不用 `dynamic`） |
| 算子 | 42 个，覆盖 YOLO11/v8 与 RF-DETR 的导出图（清单见 [支持算子](#支持算子)） |
| 图处理 | 常量折叠（Shape→Gather→Concat→Reshape 链）、算子融合（Conv+SiLU、逐元素短链等） |
| 执行 | 静态内存池（输出脱离池）、按层形状推断、`Parallel.For` + AVX2/AVX-512 向量化 |
| 线程 | 默认 `Environment.ProcessorCount`，可用环境变量 `ONNXSHARP_THREADS` 覆盖 |
| 视觉门面 | `OnnxSharp.Vision`：letterbox / stretch 预处理、YOLO 解码 + NMS、RF-DETR 解码，返回**原图坐标**的框 |
| 工具 | `OnnxSharp.Cli`：结构 dump、跑一次、计时、逐算子 profile、直接图片检测 |
| 图像解码 | **不在库里**（避免引入图像库）；CLI 自带最小 PPM/BMP 解码，仅作演示 |

### 快速开始

需要 .NET 8 或 .NET 10 SDK。

```powershell
dotnet build OnnxSharp.slnx -c Release
dotnet run --project tests/OnnxSharp.Tests -c Release    # 49 个用例，全通过退出码 0
```

准备一个 ONNX 模型（例如 `yolo11n.onnx`）和一张图，用 CLI 直接检测：

```powershell
$cli = "tools\OnnxSharp.Cli\bin\Release\net10.0\OnnxSharp.Cli.exe"
& $cli detect yolo11n.onnx photo.bmp --conf 0.25
```

```
model      : yolo11n.onnx
image      : photo.bmp 1568x1068
family     : yolo
input      : [1, 3, 640, 640]  (1568x1068 -> 640x640 (scale 0.4082x0.4082, pad 0,102))
preprocess : 2.75 ms
forward    : 137.16 ms  (min of 3; 稳态吞吐加 --iters 40)
decode     : 1.06 ms  (min of 3)
detections : 24
  #0   class0           0.7995  [1347.3, 671.9, 116.6 x 331.3]
  #1   class0           0.7525  [1080.5, 732.2, 127.8 x 325.7]
```

> CLI 只解 PPM(P6) / 未压缩 BMP。要跑 JPEG/PNG，先用任意工具转一下：
> `python -c "import cv2;cv2.imwrite('photo.bmp',cv2.imread('photo.jpg'))"`

### 库用法 1：自己管张量

```csharp
using OnnxSharp.Core;
using OnnxSharp.Tensors;

var net = Net.ReadModel("yolo11n.onnx");          // 加载 + 常量折叠 + 融合

var input = Tensor.Allocate(new TensorShape(1, 3, 640, 640), DType.F32);
input.AsSpan<float>().Fill(0.5f);

net.SetInput(net.InputNames[0], input);
Tensor output = net.Forward();                    // 等价于 Forward(OutputNames[0])
Console.WriteLine(output.Shape);                  // [1, 84, 8400]

// 多输出模型（如 RF-DETR 的 dets + labels）
List<Tensor> all = net.ForwardAll();
Tensor dets = net.Forward("dets");
```

`Tensor` 是托管 buffer + 形状/步长（`TensorShape` / `Strides`），支持 `Tensor.FromArray(float[], shape)`、`Clone()`，索引/形状类张量用 `DType.I64` / `I32` / `Bool`。

### 库用法 2：YOLO 一张图进、框出

```csharp
using OnnxSharp.Vision;

// pixels：紧凑 RGB24（长度 = width * height * 3）；OpenCV 的 Mat 传 bgr: true
// LoadRgb24 是调用方自己的解码代码 —— 库不内置 JPEG/PNG 解码
byte[] pixels = LoadRgb24("photo.jpg", out int width, out int height);

var detector = new YoloDetector("yolo11n.onnx", new DetectorOptions
{
    Confidence = 0.25f,      // 默认 0.25
    IouThreshold = 0.45f,    // 默认 0.45
    MaxDetections = 300,     // 默认 300
});

foreach (var d in detector.Detect(pixels, width, height))
    Console.WriteLine($"{d.ClassId} {d.Score:F3} [{d.X:F1},{d.Y:F1},{d.Width:F1}x{d.Height:F1}]");
```

内部完成：letterbox（等比缩放 + 填 114 + 双线性）→ 推理 → 逐锚点 argmax → 阈值 → 同类 NMS → 反 letterbox → 裁剪到图内。框坐标原点在原图左上角，单位像素。

`DetectorOptions` 还可指定 `Layout`（`Auto` / `ChannelsFirst` / `ChannelsLast` / `EndToEnd`）、`ClassCount`、`ClassAgnosticNms`，以及动态输入模型的 `TargetWidth/TargetHeight`。

### 库用法 3：RF-DETR 一张图进、框出

```csharp
using OnnxSharp.Vision;

var detector = new RfDetrDetector("rfdetr-small.onnx", new RfDetrOptions
{
    Confidence = 0.3f,       // 默认 0.3
    MaxDetections = 300,     // 对应官方 num_select
    BackgroundClassId = -1,  // 最后一列是背景；无背景列时设 null
});

foreach (var d in detector.Detect(pixels, width, height))
    Console.WriteLine(d);
```

与 YOLO 的差异（都由门面内部处理）：

| 项 | YOLO | RF-DETR |
|---|---|---|
| 预处理 | letterbox：等比缩放 + 填 114 | **直接拉伸**（不填边，`antialias=false`） |
| 归一化 | `/255` | **ImageNet**：`(v/255 - mean) / std` |
| 输出 | 单张量 `[1, 4+C, N]` | 两张量 `dets [1,Q,4]` + `labels [1,Q,C]` |
| 后处理 | argmax → 阈值 → NMS | 逐类 **sigmoid** → 丢背景列 → `Q×C` top-k → 阈值 |
| 坐标还原 | 反 letterbox（减 pad、除 scale） | 乘原图宽高 |

> RF-DETR 用的是逐类 sigmoid 而不是 softmax，一个查询可以同时在多个类别上过阈值 —— 门面已按官方 `decode_detections` 的语义实现，不需要自己做 NMS。

### 库用法 4：分步预处理 / 后处理

需要在两个阶段之间自己插计时或缓存时，可以只调一半：

```csharp
// yolo / rfdetr 即上面两个检测器实例
// YOLO：letterbox 参数自己拿着，解码时还要用
var (tensor, transform) = yolo.Preprocess(pixels, width, height);
yolo.Net.SetInput(yolo.InputName, tensor);
var detections = yolo.Decode(yolo.Net.Forward(), transform);

// RF-DETR：拉伸不需要还原参数，坐标只需原图宽高
var stretched = rfdetr.Preprocess(pixels, width, height);
detections = rfdetr.Decode(rfdetr.Net.ForwardAll(), width, height);
```

只想要张量、自己写解码，直接用静态方法：

```csharp
var (input, box) = ImagePreprocessor.Letterbox(pixels, width, height, 640, 640);      // YOLO
var stretched  = ImagePreprocessor.Stretch(pixels, width, height, 512, 512,
                                           normalization: ChannelNormalization.ImageNet); // RF-DETR
var kept = DetectionNms.Suppress(candidates, iouThreshold: 0.45f, maxDetections: 300);
```

### 命令行

```powershell
$cli = "tools\OnnxSharp.Cli\bin\Release\net10.0\OnnxSharp.Cli.exe"
```

| 命令 | 用途 |
|---|---|
| `OnnxSharp.Cli` | 冒烟测试：手工建图 Add+Relu，验证引擎链路 |
| `info <model.onnx>` | 解析 ONNX，打印算子统计、输入输出、opset |
| `dump-graph <model.onnx> [--type Conv] [--limit N]` | 导入为图，打印每个节点的推断输出形状 |
| `run <model.onnx> --input in.f32 [--input name=file.bin] [--compare golden.bin] [--repeat N] [--dump dir]` | 跑一次前向，打印输出统计与耗时 |
| `bench <model.onnx> --input in.f32 [--iters N] [--warmup N]` | 计时迭代，输出 median / min |
| `profile <model.onnx> --input in.f32 [--iters N] [--top N]` | 逐算子耗时排行（找热点） |
| `detect <model.onnx> <image.ppm\|bmp> [...]` | 内建预处理 + 解码，直接给出原图坐标的框 |
| `gemmbench` / `convbench` / `flopsbench` / `parbench` | 微基准：GEMM、卷积、FLOPS、并行扩展性 |

`detect` 的常用参数：

| 参数 | 说明 |
|---|---|
| `--family auto\|yolo\|rfdetr` | 检测头类型；默认按输出个数判断（RF-DETR 是双输出） |
| `--conf 0.25` / `--iou 0.45` | 置信度 / NMS 阈值（`--iou` 仅 YOLO） |
| `--max-det 300` / `--classes N` | 最多保留框数 / 类别数（0 = 从输出推断） |
| `--layout auto\|cf\|cl\|e2e` / `--class-agnostic` | 检测头布局 / 跨类 NMS |
| `--background -1\|none` / `--norm imagenet\|255` | RF-DETR 背景列 / 归一化方式 |
| `--labels names.txt` / `--json out.json` | 类别名（每行一个）/ 结果落盘 |
| `--input in.bin --shape 1,3,640,640` | 跳过预处理，直接喂已预处理好的 F32 |
| `--iters N` / `--size N` | 计时迭代数（稳态吞吐用 40）/ 动态输入模型的方形尺寸 |

`--json` 输出示例：

```json
{
  "model": "yolo11n.onnx",
  "family": "yolo",
  "image": "photo.bmp",
  "width": 1568,
  "height": 1068,
  "input": "[1, 3, 640, 640]  (1568x1068 -> 640x640 (scale 0.4082x0.4082, pad 0,102))",
  "confidence": 0.25,
  "iou": 0.45,
  "preprocessMs": 2.75,
  "inferenceMs": 137.16,
  "decodeMs": 1.06,
  "detections": [
    { "classId": 0, "name": "person", "score": 0.7995, "x": 1347.3, "y": 671.9, "width": 116.6, "height": 331.3 }
  ]
}
```

> 计时口径：`--iters` 默认 1，即「第一帧」的真实延迟（含 JIT）。要稳态数字用 `--iters 40` —— .NET 分层编译约 30 次调用后才升到 tier-1，少于此数会高 1.5~2 倍。

### 支持算子

42 个，按实现批次排列：

| 批次 | 算子 |
|---|---|
| 形状/元数据 (9) | `Constant` `Shape` `ConstantOfShape` `Expand` `Tile` `Unsqueeze` `Reshape` `Slice` `Split` |
| 搬运/类型 (3) | `Concat` `Transpose` `Cast` |
| 逐元素/比较 (17) | `Add` `Sub` `Mul` `Div` `Sqrt` `Exp` `Erf` `Sin` `Cos` `Sigmoid` `Relu` `Not` `Equal` `Less` `Greater` `And` `Where` |
| 归约/索引 (5) | `ReduceSum` `ReduceMax` `TopK` `Gather` `GatherElements` |
| 重算子 (8) | `Conv` `MatMul` `Gemm` `MaxPool` `Softmax` `LayerNormalization` `Resize` `GridSample` |

### 性能

仅模型前向、batch=1、FP32、16 线程、40 帧取 median（Ryzen 7 8845HS，Windows，.NET 10）：

| 模型 | OnnxSharp | OpenCV 5 dnn | 比值 |
|---|---|---|---|
| yolo11n 640² | 36.7 ms | 28.6 ms | 1.28× |
| yolov8n 640² | 35.8 ms | 30.9 ms | 1.16× |
| yolo11s 640² | 85.1 ms | 73.1 ms | 1.16× |
| yolov8s 640² | 92.4 ms | 77.4 ms | 1.19× |
| rfdetr-small 512² | 231.5 ms | 191.7 ms | 1.21× |
| rfdetr-medium 576² | 299.4 ms | 243.1 ms | 1.23× |

> 唯一事实源是 [bench.md](bench.md)（含四引擎对照、median/min 双指标与复现命令）。上表不含预处理与后处理，且对后台负载敏感，同代码重跑 ±8% 属正常。

### 已验证的模型

| 模型 | 验证方式 |
|---|---|
| yolo11n | 端到端输出与 ONNXRuntime 逐值对齐（回归用例）；检测结果与 SimdYolo 逐框一致（最大 0.005 px） |
| yolov8n / yolo11s / yolov8s | 跑通并纳入基准 |
| rfdetr-small / rfdetr-medium | 跑通并纳入基准；解码结果与官方 `rfdetr.export._runtime.decode_detections` 逐框一致（41/41 框，坐标 0.00 px） |
| 单算子微模型 | GEMM / attention / 逐元素链，用于内核定点调优 |

### 限制

- **只做 CPU**，没有 CUDA / DirectML / OpenVINO 等执行后端。
- **计算主体是 FP32**；`Cast` 与整数张量只服务于索引/形状路径，未实现 INT8 量化（QDQ）反量化。
- 动态维只做「最小可用」：以 batch=1 为主，动态 H/W 需要构造时给 `TargetWidth/TargetHeight`。
- 不针对 WASM / IL2CPP，未验 NativeAOT（但代码保持 AOT 友好）。
- **库内不做图像解码**（JPEG/PNG），像素由调用方给；CLI 只解 PPM/BMP。
- `OnnxSharp.Vision` 只覆盖**单图、三通道、目标检测**；分割 / 姿态 / OBB / 关键点未实现。

### 目录结构

```
src/OnnxSharp/
  Core/      Net / NetImpl / Graph / Arg / 解释器 / 内存池 / 融合 / 常量折叠
  Onnx/      手写 protobuf reader、ONNX importer
  Tensors/   Tensor / TensorShape / DType / GEMM / Transpose 内核
  Ops/       42 个算子的实现
  Kernels/   直接卷积、打包卷积、Winograd、MaxPool
  Threading/ 并行工具（ONNXSHARP_THREADS）
  Vision/    可选门面：ImagePreprocessor / YoloDetector / YoloDecoder
             / RfDetrDetector / RfDetrDecoder / Detection（可整体删除，不影响推理）
tools/OnnxSharp.Cli/   info / run / bench / profile / detect / 微基准 + 最小 PPM/BMP 解码
tests/OnnxSharp.Tests/ 零依赖测试运行器（49 用例），直接 dotnet run
bench/                四引擎 × 6 模型的对照脚本与产物
```

### 文档

| 文件 | 内容 |
|---|---|
| [plan.md](plan.md) | 设计文档：架构、里程碑、算子清单、性能目标、开放问题 |
| [misc.md](misc.md) | 决策记录与坑：语义细节、对照实验、踩过的雷 |
| [bench.md](bench.md) | 性能唯一事实源：口径、复现命令、每轮结果 |
| [upstream.lock](upstream.lock) | 上游 OpenCV 5 dnn 跟踪基线（机器可读） |

---

## English

[中文](#中文) ｜ **English**

### What it is

`OnnxSharp` is a from-scratch ONNX inference engine in pure C# — **CPU only, FP32, zero third-party dependencies**. It runs ONNX exports of YOLO11 / YOLOv8 / RF-DETR, and ships an optional vision facade that turns an image into detections in source-image coordinates.

Semantics follow OpenCV 5 `dnn` and SimdYolo; the engine is within **1.15–1.30×** of OpenCV on the reference machine (see [Performance](#performance)).

### Features

| Item | Notes |
|---|---|
| Dependencies | None — protobuf reader, kernels and runtime are all hand-written |
| Targets | `net8.0` / `net10.0`, AOT-friendly (no reflection, no `dynamic`) |
| Operators | 42, covering YOLO11/v8 and RF-DETR exports |
| Graph passes | Constant folding (Shape→Gather→Concat→Reshape chains), operator fusion |
| Runtime | Static buffer pool (outputs detach), shape inference, `Parallel.For` + AVX2/AVX-512 |
| Threads | `Environment.ProcessorCount` by default, overridable via `ONNXSHARP_THREADS` |
| Vision | `OnnxSharp.Vision`: letterbox / stretch preprocessing, YOLO decode + NMS, RF-DETR decode |
| Image decoding | **Not in the library** (keeps it dependency-free); the CLI ships a minimal PPM/BMP decoder |

### Quick start

Requires .NET 8 or .NET 10 SDK.

```powershell
dotnet build OnnxSharp.slnx -c Release
dotnet run --project tests/OnnxSharp.Tests -c Release     # 49 cases, exit code 0 on success
```

Detect objects in one command:

```powershell
$cli = "tools\OnnxSharp.Cli\bin\Release\net10.0\OnnxSharp.Cli.exe"
& $cli detect yolo11n.onnx photo.bmp --conf 0.25 --json out.json
& $cli detect rfdetr-small.onnx photo.bmp --conf 0.3
```

The CLI reads PPM(P6) / uncompressed BMP only; convert JPEG/PNG first.

### Library usage

**Raw tensors**

```csharp
using OnnxSharp.Core;
using OnnxSharp.Tensors;

var net = Net.ReadModel("yolo11n.onnx");        // load + constant folding + fusion

var input = Tensor.Allocate(new TensorShape(1, 3, 640, 640), DType.F32);
input.AsSpan<float>().Fill(0.5f);
net.SetInput(net.InputNames[0], input);

Tensor output = net.Forward();                   // or Forward(net.OutputNames[0])
List<Tensor> all = net.ForwardAll();             // multi-output models (RF-DETR)
```

**YOLO — image in, boxes out**

```csharp
using OnnxSharp.Vision;

// pixels: packed RGB24 (length = width * height * 3); pass bgr: true for OpenCV Mats
// LoadRgb24 is your own decoder — the library does not decode JPEG/PNG
byte[] pixels = LoadRgb24("photo.jpg", out int width, out int height);

var detector = new YoloDetector("yolo11n.onnx", new DetectorOptions
{
    Confidence = 0.25f,
    IouThreshold = 0.45f,
    MaxDetections = 300,
});

foreach (var d in detector.Detect(pixels, width, height))
    Console.WriteLine($"{d.ClassId} {d.Score:F3} [{d.X:F1},{d.Y:F1},{d.Width:F1}x{d.Height:F1}]");
```

The facade does letterbox (keep aspect ratio, pad 114, bilinear) → forward → per-anchor argmax → threshold → per-class NMS → un-letterbox → clip. Boxes are in source pixels.

**RF-DETR — image in, boxes out**

```csharp
var detector = new RfDetrDetector("rfdetr-small.onnx", new RfDetrOptions
{
    Confidence = 0.3f,
    MaxDetections = 300,        // == official num_select
    BackgroundClassId = -1,     // last column is background; use null when there is none
});

foreach (var d in detector.Detect(pixels, width, height))
    Console.WriteLine(d);
```

Differences from YOLO (all handled internally):

| | YOLO | RF-DETR |
|---|---|---|
| Preprocess | letterbox (pad 114) | **plain stretch**, no padding |
| Normalization | `/255` | **ImageNet** `(v/255 - mean) / std` |
| Output | `[1, 4+C, N]` | `dets [1,Q,4]` + `labels [1,Q,C]` |
| Decode | argmax → threshold → NMS | per-class **sigmoid** → drop background → `Q×C` top-k → threshold |
| Box mapping | invert letterbox | multiply by source width/height |

**Stage-by-stage**

```csharp
// yolo / rfdetr are the detector instances from above
// YOLO: keep the letterbox transform for decoding
var (tensor, transform) = yolo.Preprocess(pixels, width, height);
yolo.Net.SetInput(yolo.InputName, tensor);
var detections = yolo.Decode(yolo.Net.Forward(), transform);

// RF-DETR: no transform needed, just the source size
var stretched = rfdetr.Preprocess(pixels, width, height);
detections = rfdetr.Decode(rfdetr.Net.ForwardAll(), width, height);
```

Or use the static helpers: `ImagePreprocessor.Letterbox(...)`, `ImagePreprocessor.Stretch(...)` (with `ChannelNormalization.ImageNet`), `DetectionNms.Suppress(...)`.

### CLI commands

| Command | Purpose |
|---|---|
| *(no args)* / `smoke` | Build a tiny Add+Relu graph, verify the engine path |
| `info <model.onnx>` | Dump operator statistics, inputs/outputs, opset |
| `dump-graph <model.onnx> [--type Conv] [--limit N]` | Print inferred output shape per node |
| `run <model.onnx> --input in.f32 [--compare golden.bin] [--repeat N] [--dump dir]` | One forward pass with output statistics |
| `bench <model.onnx> --input in.f32 [--iters N] [--warmup N]` | Timed iterations (median / min) |
| `profile <model.onnx> --input in.f32 [--iters N] [--top N]` | Per-operator timing ranking |
| `detect <model.onnx> <image.ppm\|bmp> [...]` | Preprocess + forward + decode, boxes in source pixels |
| `gemmbench` / `convbench` / `flopsbench` / `parbench` | Micro-benchmarks for tuning |

Key `detect` flags: `--family auto|yolo|rfdetr`, `--conf`, `--iou` (YOLO), `--max-det`, `--classes`, `--layout`, `--class-agnostic`, `--background -1|none` and `--norm imagenet|255` (RF-DETR), `--labels`, `--json`, `--input <f32.bin> --shape 1,3,640,640`, `--iters N`, `--size N`.

> Timing note: `--iters` defaults to 1, i.e. real first-frame latency including JIT. For steady-state numbers use `--iters 40` — .NET tiered compilation promotes hot methods only after ~30 calls, otherwise results are 1.5–2× too slow.

### Performance

Forward pass only, batch=1, FP32, 16 threads, median of 40 frames (Ryzen 7 8845HS, Windows, .NET 10):

| Model | OnnxSharp | OpenCV 5 dnn | Ratio |
|---|---|---|---|
| yolo11n 640² | 36.7 ms | 28.6 ms | 1.28× |
| yolov8n 640² | 35.8 ms | 30.9 ms | 1.16× |
| yolo11s 640² | 85.1 ms | 73.1 ms | 1.16× |
| yolov8s 640² | 92.4 ms | 77.4 ms | 1.19× |
| rfdetr-small 512² | 231.5 ms | 191.7 ms | 1.21× |
| rfdetr-medium 576² | 299.4 ms | 243.1 ms | 1.23× |

[bench.md](bench.md) is the single source of truth (four engines, median/min, reproduction commands). Numbers above exclude pre/post-processing and are sensitive to background load (±8% between runs).

### Verified models

| Model | How it is verified |
|---|---|
| yolo11n | End-to-end outputs match ONNXRuntime value-by-value (regression test); detections match SimdYolo within 0.005 px |
| yolov8n / yolo11s / yolov8s | Runs; included in benchmarks |
| rfdetr-small / rfdetr-medium | Runs; decoding matches official `rfdetr.export._runtime.decode.decode_detections` (41/41 boxes, 0.00 px) |
| Micro models | Single-operator graphs for kernel-level tuning |

### Limitations

- **CPU only** — no CUDA / DirectML / OpenVINO backends.
- **FP32 compute**; integer/bool tensors only serve index and shape paths. INT8 quantized (QDQ) graphs are not supported.
- Dynamic dimensions are minimally supported (batch=1 focus); dynamic H/W needs `TargetWidth/TargetHeight`.
- WASM / IL2CPP are not targeted; NativeAOT is not validated (code stays AOT-friendly).
- **No image decoding in the library**; bring your own pixels. The CLI decodes PPM/BMP only.
- `OnnxSharp.Vision` covers single-image, 3-channel object detection only — no segmentation, pose, OBB or keypoints.

### Layout

```
src/OnnxSharp/   Core (graph/runtime) · Onnx (protobuf reader) · Tensors · Ops · Kernels
                 Threading · Vision (optional facade: image in, boxes out)
tools/OnnxSharp.Cli/   info / run / bench / profile / detect / micro-benchmarks
tests/OnnxSharp.Tests/ zero-dependency test runner (49 cases)
bench/                 four-engine × six-model comparison scripts
```

### Docs

| File | Content |
|---|---|
| [plan.md](plan.md) | Design: architecture, milestones, operator list, performance targets |
| [misc.md](misc.md) | Decision log and pitfalls |
| [bench.md](bench.md) | Performance source of truth: methodology, commands, results |
| [upstream.lock](upstream.lock) | Pinned OpenCV 5 dnn baseline |
