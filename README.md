<p align="center">
  <img src="assets/icon.png" width="112" height="112" alt="OnnxSharp">
</p>

<h1 align="center">OnnxSharp</h1>

<p align="center">
  <a href="https://github.com/sevencat/OnnxSharp"><img src="https://img.shields.io/github/stars/sevencat/OnnxSharp?color=181717&logo=github&logoColor=white" alt="GitHub"></a>
  <a href="https://www.nuget.org/packages/Sevencat.OnnxSharp"><img src="https://img.shields.io/nuget/v/Sevencat.OnnxSharp?color=004880" alt="NuGet version"></a>
  <a href="https://www.nuget.org/packages/Sevencat.OnnxSharp"><img src="https://img.shields.io/nuget/dt/Sevencat.OnnxSharp?color=004880" alt="NuGet downloads"></a>
  <img src="https://img.shields.io/badge/backend-CPU%20now-2ea043" alt="CPU now">
  <img src="https://img.shields.io/badge/GPU%20backends-Vulkan%20planned-orange" alt="Vulkan compute backend planned">
  <img src="https://img.shields.io/badge/dependencies-zero-blue" alt="Zero dependencies">
  <img src="https://img.shields.io/badge/code-AI--written-8b5cf6" alt="Code written by AI">
</p>

<p align="center">
  纯 C# 实现的 ONNX 推理引擎 —— <b>当前只有 CPU</b>（FP32，零第三方依赖），<b>没有</b> CUDA / OpenCL / DirectML 后端，<b>Vulkan 计算后端在规划中</b>（工控机 / 跨厂商 GPU）。<br>
  可直接跑 YOLO11 / YOLOv8 / RF-DETR / PP-OCRv6 的 ONNX 导出，并附带「一张图进、检测框出 / 文本行出」的可选门面。
</p>

<p align="center">
  <b>中文</b> ｜ <a href="#english">English</a>
</p>

---

## 中文

### 这是什么

`OnnxSharp` 是一个自己实现的 ONNX 运行时：读 ONNX 文件 → 建图 → 常量折叠 + 算子融合 → 缓冲池 + 多线程 SIMD 内核 → 按节点执行。语义对齐 OpenCV 5 `dnn` 与 SimdYolo 的做法，目标是「纯托管、无 P/Invoke、纯 CPU 也能跑检测模型」。

实测（四引擎同会话，见 [性能](#性能)）：4 线程下 **对 ONNXRuntime 还差 1.05~1.28×**（F32 图；int8 decoder 1.67×），
**对 SimdYolo 打平**（0.97~1.14×），**对 OpenCV 5 `dnn` 全面领先**（0.59~0.70×）。默认线程数 = 物理核数。

这个引擎能写出来，主要是站在 **SimdYolo** 和 **OpenCV 5 `dnn`** 的肩膀上：算子语义、预处理约定、性能基线和验证方法都大量借鉴了它们。感谢两个项目的作者与贡献者，具体借鉴点见 [致谢](#致谢)。

> **声明**：这是一个**用来学习的项目**，目的是搞懂「一个 ONNX 运行时到底是怎么跑起来的」，不是生产级推理框架。
> 所以**一定会有不少 bug** —— 少见算子、边缘 dtype、异常路径的覆盖都不完整，版本之间也可能有行为变化。
> 拿它跑通模型、做对照实验没问题；上生产请优先用 ONNXRuntime、SimdYolo、SimdPaddleOCR（`Sdcb.SimdPaddleOCR`）、OpenCV 5 `dnn` 这类成熟实现。踩到问题欢迎开 [issue](https://github.com/sevencat/OnnxSharp/issues)。

### 以 CPU 为主，且只用 4 个核

| | |
|---|---|
| **支持** | CPU，FP32，AVX2 / AVX-512 向量化 + 多线程（`ONNXSHARP_THREADS` 可调） |
| **规划中** | **Vulkan 计算后端**——面向工控机 / 跨厂商显卡（NVIDIA / AMD / Intel 核显），以独立可选包提供，主包保持纯托管 |
| **不支持** | CUDA、TensorRT、OpenCL、DirectML、WebGPU、ROCm —— 这些不打算做 |
| **不需要** | CUDA toolkit、cuDNN、任何 native 库或 P/Invoke（Vulkan 包只需系统自带的 Vulkan loader） |
| **线程预算** | **所有基准与结论都限定在 4 线程**，不是「有多少核吃多少核」。真实服务器上这个进程只是众多服务之一，能拿到的常常就是 2~4 个核，而且要长期跑、不能把整机占满 |
| **为什么** | 目标场景是「部署环境没有 CUDA 栈、也不想带 vendor 原生依赖」—— 很多服务器和工控机本来就**没有独显**，CPU 线程又是共享资源。所以优化目标是「**4 线程内的单帧延迟**」，而不是「跑分时的峰值吞吐」。CUDA / OpenCL 后端必然引入 vendor 原生库与驱动版本矩阵，和「纯托管、零依赖」直接冲突；Vulkan 放在后置的可选包里，主包不受影响 |

> 模型必须跑 NVIDIA GPU 的话，请直接用 ONNXRuntime 的 CUDA / TensorRT EP；本项目只规划 Vulkan 计算后端（工控机 / 跨厂商 GPU），不做 CUDA 兼容层，也不做 half-baked 的 GPU 兼容层。

> **口径**：下面所有表格（含 `733 GFLOPS`、`150 GMAC/s` 这类峰值）都是 **4 线程**下测的，
> 单核数字会单独标注；只有明确写「16 线程」的历史对照数据是例外。
> 多给几个核只会更快，但那不是本项目的优化目标，也没有系统性测过 —— 这里追求的是
> 「**在不能独占机器的前提下，用 4 个核把单帧做快**」。

### 代码是 AI 写的

先把归属说清楚：**这个仓库的代码几乎全部由 AI 编码代理生成**（Claude / CodeBuddy 这一类），作者的工作只有三件：

1. 提需求、定验收标准；
2. 搜集参考资料（SimdYolo、OpenCV 5 `dnn` 的源码与文档，见 [致谢](#致谢)）；
3. 跑对照实验、判断结果、决定采纳还是推翻。

由此带来几条你应该知道的约定：

- 设计取舍的依据都落在 `plan.md` / `misc.md` 里，落到「为什么这样写」，而不是「AI 说这样对」；
- 性能与正确性的结论一律以可复现的对照数据为准（`bench.md`、`tests/`），不看主观描述；
- **AI 生成的代码同样需要 review**。遇到看不懂的实现，优先怀疑它，而不是相信注释。

### 作者

<p align="center">
  <img src="author.jpg" width="132" alt="king"><br>
  <b>king</b> ｜ <a href="https://github.com/sevencat">GitHub @sevencat</a>
</p>

引擎的**需求、参考资料与验收标准**都由作者定（见上一节 [代码是 AI 写的](#代码是-ai-写的)）：从 SimdYolo 与 OpenCV 5 `dnn` 抠出算子语义与性能基线，逐轮跑对照实验，判断哪个方向采纳、哪个推翻；代码由 AI 编码代理按这些约束实现。报 bug、提需求或讨论某个优化方向，走 [Issues](https://github.com/sevencat/OnnxSharp/issues) 即可。

### 特性

| 项 | 说明 |
|---|---|
| **后端** | **当前只有 CPU**（FP32，AVX2 / AVX-512 + 多线程）；**没有** CUDA / TensorRT / OpenCL / DirectML 后端；**Vulkan 计算后端规划中**（工控机 / 跨厂商 GPU，独立可选包） |
| 依赖 | 零第三方包（不引用 `Google.Protobuf`、`Microsoft.ML.OnnxRuntime`），全部手写 |
| 目标框架 | `net8.0` / `net10.0` / `net11.0`，AOT 友好（不用反射、不用 `dynamic`）。**`net11.0` 才有 AVX-VNNI**（int8 内核快 2.3×），其它 TFM 自动回落 F32 路径 |
| 算子 | 76 个注册算子，覆盖 YOLO11/v8、RF-DETR、PP-OCRv6 与 sherpa Paraformer 的 encoder 段导出图（清单见 [支持算子](#支持算子)） |
| 图处理 | 常量折叠（Shape→Gather→Concat→Reshape 链）、算子融合（Conv+SiLU、逐元素短链、`MatMul+ArgMax`、`MatMul+Add(bias)`→带按列 bias 的 GEMM、手写 LayerNorm→`LayerNormalization`、`DQL+MatMulInteger+Cast+Mul+Add`→`DynamicQuantizeMatMul`、整条 multi-head attention→`FusedAttention`（序列长度 ≤128 时）、YOLO 的 DFL→`DflSoftmax`）、按输入 shape 签名缓存形状/类型推断、常驻 intra-op 线程池（调用线程参与 + 自旋领任务，不再每次 `Parallel.For`） |
| 执行 | 静态内存池（输出脱离池）、按层形状推断、`Parallel.For` + AVX2/AVX-512 向量化 |
| 线程 | 默认 **物理核数**（Windows 上取 `GetLogicalProcessorInformationEx`，其余平台退回 `ProcessorCount`），可用 `ONNXSHARP_THREADS` 覆盖。超线程对这些 FP 密集内核没有吞吐收益，多出来的自旋线程只会互相抢时间片。**基准与调优一律按 4 线程** —— 目标不是占满整机，而是少核下也够快 |
| 视觉门面 | `OnnxSharp.Vision`：letterbox / stretch 预处理、YOLO 解码 + NMS（类别名自动从模型 metadata 的 `names` 解析）、RF-DETR 解码、YuNet 人脸（框 + 5 点关键点 + NMS），返回**原图坐标**的框 |
| OCR 门面 | `OnnxSharp.Ocr.PaddleOCR`：PP-OCRv6 det + 可选 cls + rec，DB 后处理 + CTC 解码，返回**原图坐标**的文本行 |
| 车牌门面 | `OnnxSharp.Ocr.Plate`：同一份 det/rec 上的**解码侧特化** —— 受限字符集（~70 类）+ 逐位规则 + CTC Viterbi 逐字后验 + 号牌形态校验；`DetectAndRead`（整图）/ `Read`（已给框）/ `ReadWhole`（输入本身就是车牌裁剪图） |
| 动态图 | `Net.ReadModel(path, 输入形状)` 把全动态输入（`x[N,3,H,W]`）**特化**成静态图，导入期即可折叠形状链 |
| 像素输入 | RGB 或 BGR（`bgr` 开关）+ 任意行跨度（`rowStride`，字节）—— OpenCV `Mat`、相机 SDK 的对齐缓冲可直接喂，不必先压成紧凑数组 |
| 工具 | `OnnxSharp.Cli`：结构 dump、跑一次、计时、逐算子 profile、直接图片检测 |
| 图像解码 | **不在库里**（避免引入图像库）；CLI 用 ImageFacade 软后端 + 额外引入的 `Sdcb.SimdCodecs.Jpeg/Png/Bmp`（显式注册） |
| 代码来源 | 由 AI 编码代理生成；需求、参考资料与验收由作者负责（见 [代码是 AI 写的](#代码是-ai-写的)） |

### 安装

**仓库**：[github.com/sevencat/OnnxSharp](https://github.com/sevencat/OnnxSharp) ｜ **NuGet**：[`Sevencat.OnnxSharp`](https://www.nuget.org/packages/Sevencat.OnnxSharp) ｜ **版本**：`0.5.0` —— 包名带前缀，命名空间仍是 `OnnxSharp`。

```powershell
dotnet add package Sevencat.OnnxSharp --version 0.5.0
```

包的版本列表、依赖与各 TFM 的产物见 NuGet 页面：[nuget.org/packages/Sevencat.OnnxSharp](https://www.nuget.org/packages/Sevencat.OnnxSharp#versions-body-tab)。

目标框架 `net8.0` / `net10.0` / `net11.0`，无任何运行时依赖。下面是从源码构建与直接使用的说明。

### 快速开始

需要 .NET 8 或 .NET 10 SDK。

```powershell
dotnet build OnnxSharp.slnx -c Release
dotnet run --project tests/OnnxSharp.Tests -c Release    # 124 个用例（含 yolo26 / OCR 的 golden 对比，缺本地模型则 SKIP），全通过退出码 0
```

准备一个 ONNX 模型（例如 `yolo11n.onnx`）和一张图，用 CLI 直接检测：

```powershell
$cli = "tools\OnnxSharp.Cli\bin\Release\net10.0\OnnxSharp.Cli.exe"
& $cli detect yolo11n.onnx photo.bmp --conf 0.25
```

```
model      : yolo11n.onnx
image      : photo.jpg 3840x2160
family     : yolo
labels     : 80 classes  (model metadata)     # 类别表来源，见下表
input      : [1, 3, 640, 640]  (3840x2160 -> 640x640 (scale 0.1667x0.1667, pad 0,140))
preprocess : 1.32 ms
forward    : 21.24 ms  (min of 40; 首帧加 --iters 1)
decode     : 0.73 ms  (min of 40)
detections : 1
  #0   car              0.4323  [1847.3, 11.9, 951.6 x 978.3]
```

> CLI 的图像 IO 走 [ImageFacade](https://github.com/sevencat/ImageFacade) 的纯托管软后端：画框与中文标签由它自绘；
> 图片格式则是**额外引入**的 `Sdcb.SimdCodecs.Jpeg` / `Png` / `Bmp`，在 `ImageHost.RegisterCodecs()` 里显式注册
> （软后端本身不带任何编解码，要 PPM/WebP 就再加一个格式包）。**不引任何原生库**，产品库仍然「只吃像素」。

OCR（PaddleOCR）用同一套 CLI，直接吃 JPG，并把框画回去：

```powershell
# 模型从 nuget 模型包解出（PP-OCRv6 tiny 的 det/rec/字典 + 方向分类）
powershell -File tools\scripts\extract_ocr_models.ps1
& $cli ocr page.jpg --det .tmp\ocr-models\det.onnx --rec .tmp\ocr-models\rec.onnx `
                    --dict .tmp\ocr-models\dict.txt --cls .tmp\ocr-models\cls.onnx `
                    --limit 1600 --out page.ocr.jpg
```

```
page.jpg 3200x1800
  output : F:\...\page.ocr.jpg
  result : 3 行，849.4 ms
    浙A ES8381  (0.883)  [1752,199,198x68]
    浙A：:3M15X  (0.802)  [1769,1482,202x72]
```

车牌用**同一份** det/rec，换 `plate` 子命令（受限字符集 + 逐位约束 + 号牌形态校验）：

```powershell
& $cli plate page.jpg --det .tmp\ocr-models\det.onnx --rec .tmp\ocr-models\rec.onnx `
                      --dict .tmp\ocr-models\dict.txt --limit 960
```

```
page.jpg 3200x1800
  result : 2 张，850.7 ms（limit 960）
    浙AES8381    97.6%  NewEnergy  [1755,203 187x64]
        浙  91% A 100% E 100% S  98% 8 100% 3 100% 8  94% 1 100%
    新A3M15X     62.9%  Fuel       [1766,1478 212x85]
        新  17% A  44% 3 100% M  94% 1  68% 5 100% X  81%
```

如果车牌已经被别的办法（YOLO / 颜色分割 / 手工）抠成一张图，加 `--direct` 就跳过检测直接识别：

```powershell
& $cli plate plate-crop.png ... --direct     # 输入就是车牌：浙ABM770 100.0%
```

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

// LoadRgb24 是调用方自己的解码代码 —— 库不内置 JPEG/PNG 解码
byte[] pixels = LoadRgb24("photo.jpg", out int width, out int height);

var detector = new YoloDetector("yolo11n.onnx", new DetectorOptions
{
    Confidence = 0.25f,      // 默认 0.25
    IouThreshold = 0.45f,    // 默认 0.45
    MaxDetections = 300,     // 默认 300
});

// ① 紧凑 RGB24（行跨度 = width * 3）；Name 来自模型 metadata 的 names
foreach (var d in detector.Detect(pixels, width, height))
    Console.WriteLine($"{d.ClassId} {d.Label} {d.Score:F3} [{d.X:F1},{d.Y:F1},{d.Width:F1}x{d.Height:F1}]");

// ② BGR + 行尾填充：OpenCV 的 Mat、相机 SDK 的缓冲都长这样
//    matData = mat.data 的 byte 视图（Marshal.Copy / unsafe 指针），stride = (int)mat.Step()
foreach (var d in detector.Detect(matData, width, height, bgr: true, rowStride: stride))
    Console.WriteLine(d);
```

**像素输入约定**：`bgr` 决定 R/B 顺序（OpenCV、多数相机/采集卡是 BGR）；`rowStride` 是**每行字节数**（含行尾对齐填充），0 表示 `width * 3`。两个参数在 `Detect` 和只做预处理的 `Preprocess` 上都可传，内部只按跨度取像素，不会复制或压紧整幅图。

内部完成：letterbox（等比缩放 + 填 114 + 双线性）→ 推理 → 逐锚点 argmax → 阈值 → 同类 NMS → 反 letterbox → 裁剪到图内。框坐标原点在原图左上角，单位像素。

`DetectorOptions` 还可指定 `Layout`（`Auto` / `ChannelsFirst` / `ChannelsLast` / `EndToEnd`）、`ClassCount`、`ClassAgnosticNms`，以及动态输入模型的 `TargetWidth/TargetHeight`。

**类别名**：ultralytics 导出的 ONNX 把类别表写在 metadata 的 `names` 里（`{0: 'person', 1: 'bicycle', …}`，是 Python dict 字符串而不是 JSON），`imgsz` 也在同一处。`YoloDetector` 构造时自动解析，坏格式 / 缺号都不会抛：

```csharp
var detector = new YoloDetector("yolo11n.onnx");

detector.ClassNames                  // 80 条，下标即类别 id；模型没带 names 时是空表
detector.NameOf(0)                   // "person"；缺号或没表时为 null
detector.Metadata.InputHeight        // imgsz 的高/宽，int?（yolo11n 是 640 / 640；没写时 null）
detector.Metadata.Count              // 表长度（最大 id + 1，缺号槽位是空串）

// Detect / Decode 返回的 Detection 已经填好名字：Name 缺号时是 null，Label 退回 "class{id}"
foreach (var d in detector.Detect(pixels, width, height))
    Console.WriteLine($"{d.Label} {d.Score:F3}");   // "car 0.432"

// 不用 YoloDetector（自己前向）时：手动传表，或直接解析 metadata
var boxes = YoloDecoder.Decode(output, transform, options, names);
var parsed = YoloMetadata.Parse(net.Metadata);       // 原样给 metadata 也行，null → Empty
```

优先级：CLI 的 `detect` 是「`--labels` 文件 > 模型 `names` > COCO 猜测（仅当检测头字段数 == 84）」，并在 `labels :` 那行打印来源；
库里的 `YoloDetector` **不猜**——模型没带 `names` 就留空，想要 COCO 名字请自己传 `CocoLabels.Names`。

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

RF-DETR 门面的 `Detect` / `Preprocess` 与 YOLO 一样接受 `bgr` 与 `rowStride`。

### 库用法 4：分步预处理 / 后处理

需要在两个阶段之间自己插计时或缓存时，可以只调一半：

```csharp
// yolo / rfdetr 即上面两个检测器实例
// 预处理同样接受像素约定：Preprocess(pixels, w, h, bgr: true, rowStride: stride)
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

### 库用法 5：OCR 一张图进、文本行出（PaddleOCR）

```csharp
using OnnxSharp.Ocr.PaddleOCR;

// 模型来自 PP-OCRv6 的 ONNX 导出；也可以直接吃 nuget 模型包的嵌入资源（见「OCR 模型从哪来」）：
//   PaddleOcrModel.FromStreams(ChineseV6TinyModels.Default.Detection.OpenRead(), ...)
var model = PaddleOcrModel.FromFiles("det.onnx", "rec.onnx", "ppocr_keys.txt", "cls.onnx");

var ocr = new PaddleOcrPipeline(model, new PaddleOcrOptions
{
    DetLimitSide = 960,     // 大图小字调到 1600 能明显提升小字识别
    BoxThreshold = 0.6f,
    UnclipRatio = 1.6f,
});

// pixels 是调用方解码好的交错像素：RGB24 / BGR24 / RGBA32 / BGRA32，rowStride=0 表示紧凑排列
foreach (OcrLine line in ocr.Run(pixels, width, height, OcrPixelFormat.Bgra32, stride))
    Console.WriteLine($"{line.Text}  ({line.Score:F3})  [{line.X:F0},{line.Y:F0}]");
```

内部完成：检测尺寸按 32 对齐（最长边 ≤ `DetLimitSide`）→ DB 概率图 → 8 邻域连通域 + unclip → 逐行裁剪
→（可选）PP-LCNet 判断是否倒置并旋转 180° → 等比缩到高 48、右侧补 0 → CTC 贪心解码 + 字典。

两个必须知道的点：

- **模型是全动态导出**（`x[N,3,H,W]`），引擎用 `Net.ReadModel(path, 输入形状)` 把输入维**特化**成具体值才能折叠
  `Shape → Slice → Concat → Reshape` 链。所以每个 `(H,W)` 对应一份特化图：流水线按形状缓存
  （检测 2 份 / 识别 8 份，见 `PaddleOcrOptions.DetShapeCacheSize` / `RecShapeCacheSize`）。
- 模型包里**只有 ONNX 与字典、没有推理代码**，所以本库不依赖 `Sdcb.SimdPaddleOCR` 的引擎，只借它的模型资源
  （`tools/scripts/extract_ocr_models.ps1` 把资源原样解出）。它作为**外部对照基准**跑过一次同模型同图的
  4 线程同会话 median 慢 1.12~1.23×（`bench.md` §35 时 1.7~2.1×，已收窄，见 §37.2），基准代码在 `.tmp/ocrbench`（不进仓库）。

#### 车牌：受限解码 + 形态校验（`OnnxSharp.Ocr.Plate`）

车牌**不需要专门模型** —— 通用 PP-OCRv6 的 rec 输出 6906 类，车牌字符都在里面，特化全在解码侧：

| 层 | 做法 |
|---|---|
| 字符集 | 31 省 + 24 字母 + 10 数字 + 6 后缀 ≈ 70 类，字母与数字排除 `I` / `O` |
| 逐位 | `PlateCharset.AllowedAt(position, length)`：首位省、次位发牌机关字母、其余字母/数字、尾位可 学·警·挂·港·澳 |
| 解码 | CTC 前缀 beam（log 域），前缀打包进 `long`（每字符 7 位 rank + 高 8 位长度），非法长度直接剪 |
| 置信度 | 对最终序列做 CTC Viterbi 对齐（2L+1 状态）取**逐字后验**，几何平均 |
| 形态 | `PlateNumber.TryParse`：7 位燃油 / 8 位新能源 / 尾字 学·警·挂·港·澳 / 首字 使·领 |

为什么不能直接复用通用解码：通用 `CtcDecoder` 每帧要在 6906 类上排序取 top-k，`S`/`5`、`B`/`8`、
`0`/`O` 这些同形竞争会白吃 beam 宽度；砍到 70 类之后 beam 8 就够用。

```csharp
using OnnxSharp.Ocr.Plate;

var pipeline = new PaddleOcrPipeline(model);         // 与上面同一个 PaddleOCR 流水线
var reader = new PlateReader(pipeline, new PlateReaderOptions
{
    MinScore = 0.5f,                                 // 逐字几何平均分的下限
    MinAspect = 1.6f, MaxAspect = 6.5f,              // 只对「像车牌」的框做解码
    MaxPlates = 4,
});

// a) 整图：DB 找行 → 宽高比筛 → 逐行车牌解码
foreach (PlateResult plate in reader.DetectAndRead(pixels, width, height, OcrPixelFormat.Bgra32, stride))
    Console.WriteLine($"{plate.Text} {plate.Score:P1} {plate.Plate?.Kind} [{plate.X},{plate.Y} {plate.Width}x{plate.Height}]");

// b) 上游检测器给了框（YOLO / 颜色分割）——只跑 rec
PlateResult? one = reader.Read(pixels, width, height, OcrPixelFormat.Bgra32, stride, x, y, w, h);

// c) 上游已经把车牌抠成一张图——不跑 DB，整幅图当一张牌
PlateResult? whole = reader.ReadWhole(pixels, width, height, OcrPixelFormat.Bgra32, stride);
```

`PlateResult.Decoding.CharScores` 是逐字后验，直接指出哪个字不可信（不用猜「为什么这张不准」）：

```
浙  91% A 100% E 100% S  98% 8 100% 3 100% 8  94% 1 100%
新  17% A  44% 3 100% M  94% 1  68% 5 100% X  81%     ← 低分框，肉眼也糊
```

三种入口的实测（同一台 4 线程机器）：整图 DB 通路 825 ms；`--crop` 单框 rec 稳态 3.6~5.1 ms；
`--direct` 对 135×46 / 175×49 的裁剪图分别 100.0% / 99.6%，稳态 4.45 / 7.45 ms。
`--direct` 误用到整张大图不会乱出结果（形态校验 + 分数下限会拦住，返回空）。

### 库用法 6：语音识别（sherpa-onnx，不在引擎里）

语音**不在**本引擎范围内，原因很具体：sherpa-onnx 的 Paraformer / Zipformer 导出图里带
`Loop` / `NonZero` / `GatherND` / `Sequence`，并且权重是 **int8**（`MatMulInteger`），
而本引擎是静态图 + FP32 + 无子图/循环 —— 不是「补两个算子」能解决的。

需要离线 ASR 时直接用 [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) 的 .NET 包
（`org.k2fsa.sherpa.onnx`，注意它**自带原生 onnxruntime**，与「纯托管、零原生依赖」冲突）：

```csharp
using SherpaOnnx;

var config = new OfflineRecognizerConfig
{
    FeatConfig = new FeatureConfig { SampleRate = 16000, FeatureDim = 80 },
    ModelConfig = new OfflineModelConfig
    {
        Tokens = @"sherpa\tokens.txt",
        NumThreads = 2,
        Provider = "cpu",
        Paraformer = new OfflineParaformerModelConfig { Model = @"sherpa\model.int8.onnx" },
        ModelType = "paraformer",
    },
    DecodingMethod = "greedy_search",
};

using var recognizer = new OfflineRecognizer(config);
using OfflineStream stream = recognizer.CreateStream();
stream.AcceptWaveform(sampleRate, samples);   // float[]，单声道；sherpa 自己重采样到 16 kHz
recognizer.Decode(stream);
Console.WriteLine(stream.Result.Text);
```

WAV 解析（PCM 8/16/24/32 与 float32、任意声道下混）是纯托管代码，约 100 行，不需要额外依赖；
本仓库不内置，避免为一个可选能力引入原生运行库。想自己做纯托管 ASR 的话，
`F:\code\SimdOnnxExt` 的 `docs/PLAN.md`「阶段 6」记录了把 sherpa 的 Paraformer 权重反量化成 fp32、
再用引擎 kernel 重写 fbank/CIF/decoder 的路线与进度。

### 库用法 7：SAM3 框提示分段（image + box → mask）

```csharp
using OnnxSharp.Sam3;

var segmenter = new Sam3Segmenter(new Sam3Options
{
    ModelDirectory = @"models\sam3-fp32",   // vision-encoder / geometry-encoder / decoder
    ScoreThreshold = 0.3f,
    IouThreshold = 0.5f,                    // 200 个 query 里同目标会重复命中，必须做框级 NMS
    MaxItems = 5,
});

var result = segmenter.Segment(pixels, width, height, [new Sam3Box(1347, 672, 1464, 1003)]);
foreach (Sam3Item item in result.Items)
    Console.WriteLine($"{item.Score:F3}  {item.Box}  mask {item.MaskWidth}x{item.MaskHeight}");
```

- 模型要先从官方 **FP16 导出转成 FP32**：`python tools/scripts/onnx_fp16_to_f32.py in.onnx out.onnx`
  （脚本会改 `Cast`/`ConstantOfShape` 的 F16、修 `value_info` 的类型，并把权重写成**每个模型独立**的外部数据文件）。
- 框提示约定：**归一化 cxcywh + label=1**（`Sam3PromptBoxFormat` 可切 `xyxy` / 模型像素）。
- 掩码是 288² 概率，源图坐标 = `mask × result.MaskScaleX/Y`；框已经是源图像素。
- 一次前向约 **26 s**（1008² ViT-L，FP32，16 线程 —— 早期测量，未按 4 线程口径重测），整链 30 s；纯 CPU 只适合离线批处理。
- 轻量版 **EfficientSAM3 EV-M** 是另一套三模型导出（没有 geometry-encoder，提示走 `exemplar_*` + CLIP 文本），
  本库没做门面、直接用 `Net` 串，见 [库用法 11](#库用法-11efficientsam3ev-m三模型分割图--框提示--掩码)。

### 库用法 8：共享的图像预处理（Vision / OCR / SAM3 同一套）

```csharp
using OnnxSharp.Imaging;

var source = new ImagePixels(bgra, width, height, PixelFormat.Bgra32, stride);
ImagePlacement fit = ImageGeometry.Contain(width, height, 1008, 1008, centered: false);  // 左上对齐

float[] input = new float[3 * 1008 * 1008];
ImageTensor.ToChw(source, 0, 0, width, height,
    outWidth: fit.ResizedWidth, outHeight: fit.ResizedHeight,
    offsetX: fit.OffsetX, offsetY: fit.OffsetY,
    destinationHeight: 1008, destinationStride: 1008,
    padValue: 1f,                                   // 白边（归一化后的值）
    destination: input,
    normalization: ChannelNormalization.Divide255);
```

`ImageTensor.ToChw` 是**唯一**的像素 → NCHW 实现：裁剪 + 双线性/最近邻 + 归一化 + 填边 + 可选 180° 旋转，
三处调用方只是参数不同：

| 调用方 | 几何 | 填边 | 归一化 |
|---|---|---|---|
| `ImagePreprocessor.Letterbox`（YOLO） | `ImageGeometry.Contain` 居中 | 114/255 | `/255` |
| `ImagePreprocessor.Stretch`（RF-DETR） | `ImageGeometry.Stretch` | 无 | ImageNet |
| `OcrImage.ToChw`（PaddleOCR） | 裁剪文本行 + 高 48 | 0 | `(v/255-0.5)/0.5` |
| `Sam3Segmenter`（SAM3） | `Contain(centered:false)` | 1.0（白） | `/255` |

### 库用法 9：语音识别一条龙（Paraformer，纯 C#）

```csharp
var recognizer = new ParaformerRecognizer(new ParaformerOptions
{
    EncoderPath = @"sherpa\encoder.onnx",   // split_before_control_flow.py 切出来的 encoder 段
    DecoderPath = @"sherpa\decoder.onnx",   // split_decoder.py 切出来的 decoder 段
    TokensPath = @"sherpa\tokens.txt",
});

var result = recognizer.TranscribeWav("test.wav");      // 16 kHz 单声道
Console.WriteLine(result.Text);        // 今天气真不错我们出去走一走吧
Console.WriteLine(result.TokenIds);    // [5308, 8301, 4474, ...]  (16 个 token)
```

整条链都在 C# 里：`wav → fbank/LFR/CMVN → encoder → CIF → decoder → argmax → 文本`。
其中 CIF 是手工移植的（模型里是 80 节点的 `Loop` + 两个 `Sequence*` 输出），
token 解码规则与 sherpa 一致：argmax → 跳过 `<blank>` → **折叠连续重复** → 遇 `</s>` 停。

命令行等价写法：

```
OnnxSharp.Cli asr test.wav --models sherpa --tokens sherpa\tokens.txt [--json] [--out text.txt]
```

模型只需切一次：

```
python tools/scripts/split_before_control_flow.py model.int8.onnx encoder.onnx /predictor/Concat_5_output_0,/predictor/Add_output_0
python tools/scripts/split_decoder.py model.int8.onnx decoder.onnx
```

### 库用法 10：人脸检测，一张图进、脸 + 5 点关键点出（YuNet）

```csharp
using OnnxSharp.Vision;

// LoadBgr24 是调用方自己的解码 —— 库不内置 JPEG/PNG
byte[] bgr = LoadBgr24("group.jpg", out int width, out int height);

var detector = new YuNetDetector("yunet.onnx", new FaceDetectorOptions
{
    ScoreThreshold = 0.6f,   // 默认 0.6（sqrt(cls×obj)，与 OpenCV 同口径）
    IouThreshold = 0.3f,     // 默认 0.3（人脸互相重叠很正常，别调太高）
    MaxFaces = 10,           // 默认不限
});

// ① 紧凑 BGR24：OpenCV imread、相机 SDK 的默认顺序（bgr 默认就是 true）
foreach (var face in detector.Detect(bgr, width, height))
    Console.WriteLine($"{face.Score:F3} [{face.X:F1},{face.Y:F1},{face.Width:F1}x{face.Height:F1}] " +
                      $"right-eye {face.Landmarks.RightEye} nose {face.Landmarks.Nose}");

// ② RGB24，或带行尾填充的缓冲（rowStride = 每行字节数，0 表示 width * 3）
foreach (var face in detector.Detect(rgb, width, height, bgr: false, rowStride: stride))
    Console.WriteLine(face);

// ③ 复用已加载的模型：多个检测器共享权重，不重复解析 + 折叠
var shared = new YuNetDetector(net);
```

**与 YOLO / RF-DETR 的关键差异**（门面内部都已处理，但换模型时容易踩）：

| 项 | YOLO / RF-DETR | YuNet |
|---|---|---|
| 预处理 | letterbox：等比缩放 + 填 114 | **整图拉伸**到模型静态输入（640×640），不填边 |
| 归一化 | `/255`（RF-DETR 是 ImageNet） | **0..255 原始值**，不减均值、不除 255 |
| 通道序 | RGB | **BGR**（`bgr` 默认 `true`，与 OpenCV `blobFromImage` 一致） |
| 输出 | 单张量 / 双张量 | 12 个张量：`cls` / `obj` / `bbox` / `kps` × stride 8/16/32 |
| 坐标还原 | 反 letterbox | 乘回原图宽高（因为输入是拉伸的） |

5 点关键点顺序与 OpenCV 一致：**右眼、左眼、鼻尖、右嘴角、左嘴角**（"左右"指图中的左右）。
`Face` 还带 `Right` / `Bottom` / `Area`，`Face.IoU(a, b)` 可直接比较两个框。

命令行等价写法：

```
OnnxSharp.Cli face yunet.onnx group.jpg [--conf 0.6] [--iou 0.3] [--max-faces 10] [--json out.json] [--out marked.jpg]
```

int8 模型（`yunet_int8.onnx`）API 完全一样，换个路径就能跑 —— 但当前比 fp32 **慢 3.5×**（20.3 vs 5.7 ms，原因见[性能](#性能)与 `bench.md` §36 / §37.4）。

### 库用法 11：EfficientSAM3（EV-M）三模型分割（图 + 框提示 → 掩码）

官方 EfficientSAM3 的 **EV-M** 导出拆成三个 ONNX（和「库用法 7」的 SAM3 门面不是同一套：没有 geometry-encoder，
提示走 `exemplar_*` 槽位 + CLIP 文本），本库没有为它单独做门面，直接用 `Net` 串：

| 模型 | 输入 | 输出 |
|---|---|---|
| `image_encoder` | `image[1,3,1008,1008]`（拉伸 + ImageNet 归一化） | `fpn_0[1,256,288,288]` / `fpn_1[1,256,144,144]` / `fpn_2[1,256,72,72]` + `pos_0/1/2` |
| `text_encoder` | `tokens[1,16]`（int64，CLIP BPE，BOS 49406 / EOS 49407） | `language_features[16,1,256]` / `language_padding[1,16]` / `language_embeds[16,1,512]` |
| `decoder` | `fpn_0..2` + `pos_2` + `language_*` + `exemplar_boxes[8,1,4]` / `exemplar_labels[8,1]` / `exemplar_mask[1,8]` | `pred_logits[1,200,1]` / `pred_boxes[1,200,4]` / `pred_masks[1,200,288,288]` / `presence_logit[1,1]` |

```csharp
using OnnxSharp.Core;
using OnnxSharp.Tensors;
using OnnxSharp.Vision;

const int Size = 1008, MaskSize = 288, Slots = 8;

// 1) image encoder：整图拉伸到 1008²（不填边）+ ImageNet 归一化
var encoder = Net.ReadModel(@"esam3\efficientsam3_ev_m_image_encoder.onnx");
encoder.SetInput("image", ImagePreprocessor.Stretch(pixels, width, height, Size, Size,
                                                    normalization: ChannelNormalization.ImageNet));

var feats = encoder.ForwardAll();
Tensor fpn0 = Output(encoder, feats, "fpn_0").Clone();   // 输出指向引擎缓冲槽，下一次 Forward 会覆盖
Tensor fpn1 = Output(encoder, feats, "fpn_1").Clone();
Tensor fpn2 = Output(encoder, feats, "fpn_2").Clone();
Tensor pos2 = Output(encoder, feats, "pos_2").Clone();

// 2) text encoder：16 个 CLIP token，BOS/EOS 之后补 0；只给框时就是空文本
var text = Net.ReadModel(@"esam3\efficientsam3_ev_m_text_encoder.onnx");
long[] tokens = new long[16];
tokens[0] = 49406;
tokens[1] = 49407;
text.SetInput("tokens", Tensor.FromInt64(tokens, new TensorShape(1, 16)));

var language = text.ForwardAll();
Tensor languageFeatures = Output(text, language, "language_features").Clone();
Tensor languagePadding = Output(text, language, "language_padding").Clone();

// 3) decoder：8 个 exemplar 槽，槽 0 放提示框（cxcywh、按原图归一化）
var decoder = Net.ReadModel(@"esam3\efficientsam3_ev_m_decoder.onnx");
var boxes = new float[Slots * 4];
var labels = new long[Slots];
var slots = new float[Slots];

float x1 = 1951.8f, y1 = 1055.4f, w = 1019.4f, h = 1092.9f;   // 提示框：YOLO 检出或人工标注，原图像素
boxes[0] = (x1 + w / 2) / width;
boxes[1] = (y1 + h / 2) / height;
boxes[2] = w / width;
boxes[3] = h / height;
labels[0] = 1;                        // 1 = 目标框，0 = 空槽
slots[0] = 1;                         // 1 = 该槽有效

decoder.SetInput("fpn_0", fpn0);
decoder.SetInput("fpn_1", fpn1);
decoder.SetInput("fpn_2", fpn2);
decoder.SetInput("pos_2", pos2);
decoder.SetInput("language_features", languageFeatures);
decoder.SetInput("language_padding", languagePadding);
decoder.SetInput("exemplar_boxes", Tensor.FromArray(boxes, new TensorShape(Slots, 1, 4)));
decoder.SetInput("exemplar_labels", Tensor.FromInt64(labels, new TensorShape(Slots, 1)));
decoder.SetInput("exemplar_mask", Tensor.FromArray(slots, new TensorShape(1, Slots)));

// 4) 后处理：score = sigmoid(pred_logits) × sigmoid(presence_logit)，取最高分的 query
var outputs = decoder.ForwardAll();
ReadOnlySpan<float> logits = Output(decoder, outputs, "pred_logits").AsSpan<float>();
float presence = Sigmoid(Output(decoder, outputs, "presence_logit").AsSpan<float>()[0]);

int best = ArgMax(logits);
float score = Sigmoid(logits[best]) * presence;

ReadOnlySpan<float> maskLogits = Output(decoder, outputs, "pred_masks")
    .AsSpan<float>()
    .Slice(best * MaskSize * MaskSize, MaskSize * MaskSize);
// 掩码是 logits：> 0 即前景；要概率图就逐点 sigmoid

static Tensor Output(Net net, IReadOnlyList<Tensor> outputs, string name)
{
    for (int i = 0; i < outputs.Count; i++)
        if (net.OutputNames[i] == name) return outputs[i];
    throw new InvalidOperationException($"{name} 不在输出里");
}

static int ArgMax(ReadOnlySpan<float> values)
{
    int best = 0;
    for (int i = 1; i < values.Length; i++)
        if (values[i] > values[best]) best = i;
    return best;
}

static float Sigmoid(float value) => 1f / (1f + MathF.Exp(-value));
```

- **掩码坐标**：预处理是整图拉伸，所以 288² 的掩码直接覆盖原图 —— 放大时按 `x * width / 288`、`y * height / 288`，
  不需要反 letterbox。**要框就从掩码算 bbox**：这一版导出的 `pred_boxes` 与同 query 的掩码对不上（实测框只有掩码的 1/2~1/7），
  乘原图宽高会得到错框。
- **输出缓冲槽**：`ForwardAll` 返回的张量指向引擎缓冲，下一次 `Forward` 会覆盖，串模型时必须 `Clone()`（上面 fpn / language_* 都做了）。
- **耗时**（EV-M，1008²，16 线程，FP32）：image encoder 0.8–2.0 s、decoder 2.9–4.2 s、text encoder 6–11 ms；
  同会话 onnxruntime 是 0.44–1.4 s / 2.3–3.7 s / 3–4 ms，比值 **1.5–2.6× / 1.1–1.3× / 2.0–2.6×**
  （机器有 ~50% 后台负载，绝对值波动可达 2×；热点与逐轮原始值见 `bench.md` §41.4）。纯 CPU 只适合离线批处理。
- **实测结果**（提示框 = yolo26s 检出的一辆车）：`presence 0.9998`、best query 157（`score 0.9636`）、前景占 11.04%；
  掩码与 onnxruntime 的 **IoU = 1.00000**（概率图 `max|d| = 5.5e-3`，差异来自预处理双线性与 OpenCV `INTER_LINEAR` 的细微不同）。
- 本库只算 fp32：官方导出若是 fp16，先用 `python tools/scripts/onnx_fp16_to_f32.py in.onnx out.onnx` 转换。

命令行等价写法（三个模型依次跑，中间结果落盘后再喂下一个）：

```powershell
$cli = 'tools\OnnxSharp.Cli\bin\Release\net10.0\OnnxSharp.Cli.exe'

& $cli run esam3\efficientsam3_ev_m_image_encoder.onnx --input image=image.f32   --dump ours_img
& $cli run esam3\efficientsam3_ev_m_text_encoder.onnx  --input tokens=tokens.i64 --dump ours_txt

& $cli run esam3\efficientsam3_ev_m_decoder.onnx `
    --input fpn_0=ours_img/out0.f32 --input fpn_1=ours_img/out1.f32 --input fpn_2=ours_img/out2.f32 `
    --input pos_2=ours_img/out5.f32 `
    --input language_features=ours_txt/out0.f32 --input language_padding=ours_txt/out1.f32 `
    --input exemplar_boxes=exemplar_boxes.f32 --input exemplar_labels=exemplar_labels.i64 `
    --input exemplar_mask=exemplar_mask.f32 --dump ours_dec
```

`--dump` 按输出顺序写成 `out0.f32`、`out1.f32`…，image encoder 的 `out5` 就是 `pos_2`。

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
| `detect <model.onnx> <image> [--out marked.jpg]` | 内建预处理 + 解码，直接给出原图坐标的框（可画出标注图） |
| `face <yunet.onnx> <image> [--conf 0.6] [--iou 0.3] [--max-faces N] [--json out.json] [--out marked.jpg]` | YuNet 人脸检测：框 + 5 点关键点，坐标是原图像素（可画出标注图） |
| `features <model.onnx> <wav> [--out feats.f32]` | wav → fbank → LFR → CMVN（Paraformer 的 `speech` 输入），参数取自模型 metadata |
| `asr <wav> --models <dir> [--tokens f] [--json] [--out text.txt]` | 端到端语音识别（fbank → encoder → CIF → decoder → 文本），<dir> 里放 `encoder.onnx` / `decoder.onnx` / `tokens.txt` |
| `ocr <image> --det det.onnx --rec rec.onnx --dict keys.txt [--cls cls.onnx] [--out marked.jpg]` | PaddleOCR：DB 检测 → 方向分类 → CTC 识别，给出原图坐标的文本行（可画出标注图） |
| `plate <image> --det det.onnx --rec rec.onnx --dict keys.txt [--direct \| --crop x,y,w,h]` | 车牌专用 OCR：受限字符集 + 逐位约束 + 号牌形态校验；`--direct` = 输入本身就是车牌裁剪图，跳过检测 |
| `sam3 <image> --models <fp32 dir> --box x1,y1,x2,y2 [--box ...] [--out marked.jpg]` | SAM3 框提示分段：`--conf 0.3` / `--iou 0.5` 做框级 NMS / `--max-items 5` / `--format cxcywh\|xyxy\|px` |
| `gemmbench` / `convbench` / `flopsbench` / `parbench` | 微基准：GEMM、卷积、FLOPS、并行扩展性 |

`detect` 的常用参数：

| 参数 | 说明 |
|---|---|
| `--family auto\|yolo\|rfdetr` | 检测头类型；默认按输出个数判断（RF-DETR 是双输出） |
| `--conf 0.25` / `--iou 0.45` | 置信度 / NMS 阈值（`--iou` 仅 YOLO） |
| `--max-det 300` / `--classes N` | 最多保留框数 / 类别数（0 = 从输出推断） |
| `--layout auto\|cf\|cl\|e2e` / `--class-agnostic` | 检测头布局 / 跨类 NMS |
| `--background -1\|none` / `--norm imagenet\|255` | RF-DETR 背景列 / 归一化方式 |
| `--labels names.txt` / `--json out.json` | 类别名（每行一个，覆盖模型 metadata 的 `names`）/ 结果落盘 |
| `--input in.bin --shape 1,3,640,640` | 跳过预处理，直接喂已预处理好的 F32 |
| `--iters N` / `--size N` | 计时迭代数（稳态吞吐用 40）/ 动态输入模型的方形尺寸 |

`plate` 的关键参数：

| 参数 | 说明 |
|---|---|
| `--det` / `--rec` / `--dict` | 与 `ocr` 完全相同的 PP-OCRv6 模型，车牌不另需模型 |
| `--beam 8` / `--greedy` | 车牌前缀 beam 宽度 / 贪心解码（对照用，实测差异落在噪声内） |
| `--no-structure` / `--loose` | 关逐位规则 / 关号牌形态校验（只剩字符集约束，用来做对照） |
| `--score 0.5` / `--max 4` | 逐字几何平均分下限 / 最多返回几张 |
| `--direct` | **输入本身就是一张车牌裁剪图**：跳过 DB，整幅图当一张牌 |
| `--crop x,y,w,h` | 在整图里指定一块车牌区域：同样跳过 DB（上游车牌检测器已在跑的场景） |
| `--limit` / `--det-conf` / `--box` / `--unclip` / `--no-aspect` | DB 检测侧参数（`--direct` / `--crop` 时无效） |
| `--iters N` | 只与 `--direct` / `--crop` 一起用：量单张 rec 的稳态耗时（首次含图导入） |
| `--json out.json` | 落盘（含 `mode: detect\|crop\|direct`、逐字分数与号牌各段） |

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

`tools\OnnxSharp.Sft2Onnx`（独立小工具：把 safetensors 权重打进一份现成骨架，**纯 C#，不需要 Python**）：

```powershell
$sft = "tools\OnnxSharp.Sft2Onnx\bin\Release\net10.0\sft2onnx.exe"
```

| 命令 | 用途 |
|---|---|
| `sft2onnx patch <skeleton.onnx> <weights.safetensors> -o out.onnx [--verify]` | 只替换 initializer 的数据字段，图、metadata、未知字段按原始字节保留 |
| `sft2onnx extract <model.onnx> -o w.safetensors [--graph-out skeleton.onnx]` | 反向：把 ONNX 的权重抽成 safetensors，并（可选）写出只含图的骨架 |
| `sft2onnx list <model.onnx\|weights.safetensors>` | 列出骨架的 initializer / 权重文件的张量（按扩展名自动识别） |

`patch` 的选项：`--map <骨架名>=<权重名>`（可重复） / `--strip a,b`（默认 `module.,model.,_orig_mod.,backbone.`） / `--strict` / `--external-threshold <GiB>`（默认 1，超过就写外部数据文件） / `--data-name` / `--verify`（写完用引擎加载一遍）。

> **为什么必须有骨架**：ONNX = 计算图 + 权重；safetensors 只有权重。架构没法从权重反推，所以图由骨架提供 —— 训练侧产出 `best.safetensors`，部署侧用发布时附带的那份骨架生成最终模型。
>
> 实测（yolo11n，196 个 initializer / 10.2 MiB）：替换耗时 **67 ms**，ORT 跑补丁前后 **max\|diff\| = 0.0**（逐位一致）；权重存成 F16、骨架是 F32 时自动转换，max\|diff\| 1.28（F16 量化误差，相对 2e-3）。
>
> ⚠️ 骨架要用 **`fuse=False`** 导出（Ultralytics 的 `export()` 默认 `fuse=True` 会把 BatchNorm 折进 Conv，`bn.*` 权重就没有对应项）。工具检测到未匹配的 `running_mean/running_var` 会直接提示这一点。
>
> **两个方向不对称**：`ONNX → safetensors + 骨架` 是无损搬运（图和权重都在文件里，实测 yolo11n 拆出 86 KB 骨架 + 10.15 MiB 权重，再打回去**字节数完全相同、ORT 逐位一致**）；反过来 `safetensors + config → ONNX` 才难，因为图不存在，需要架构代码。注意 `extract` 出来的骨架**不能直接推理**（数据被剥掉了），它只配合 `patch` 使用。

`extract` 的实测（yolo11n）：

| | 大小 |
|---|---|
| `yolo11n.onnx` | 10,720,227 B |
| `extract` → 骨架 | 86,042 B（320 节点 + 196 个 initializer 头） |
| `extract` → safetensors | 10,651,800 B（196 张量） |
| 骨架 + 权重 → `patch` 回来 | 10,720,227 B（**与原文件字节数相同**），ORT max\|diff\| = **0.0** |

### OCR 模型从哪来（含车牌）

OCR 是**双模型**：det（DB 检测）+ rec（CTC 识别），外加可选的 cls（文本行方向）与字典文件。
车牌复用同一套 det/rec，**不需要额外模型**。四条路径，按「省事 → 可控」排列。

**A. 仓库脚本：从 nuget 模型包原样解出（最省事）**

```powershell
powershell -File tools\scripts\extract_ocr_models.ps1                            # → .tmp\ocr-models
powershell -File tools\scripts\extract_ocr_models.ps1 -Destination D:\models\ocr
```

脚本做的事很笨但可靠：`Sdcb.SimdPaddleOCR.Models.ChineseV6Tiny`（PP-OCRv6 tiny 的 det / rec / 字典）与
`Sdcb.SimdPaddleOCR.Models.TextLineOrientation`（cls）这两个包**只把 ONNX 与字典作为嵌入资源**，
脚本用反射把资源原样 copy 出来，得到 `det.onnx` / `rec.onnx` / `dict.txt` / `cls.onnx`。
除了 `dotnet restore` 不需要任何别的东西，也不引入 native 依赖。

**B. 不落盘：代码里直接读模型包的资源流**

这两个包的传递依赖只有 `Sdcb.SimdPaddleOCR.ModelProvider`（纯托管 + 嵌入资源，**不含推理引擎、不含 native**）：

```powershell
dotnet add package Sdcb.SimdPaddleOCR.Models.ChineseV6Tiny --version 1.0.0
dotnet add package Sdcb.SimdPaddleOCR.Models.TextLineOrientation --version 1.0.0
```

```csharp
using OnnxSharp.Ocr.PaddleOCR;
using Sdcb.SimdPaddleOCR.Models.ChineseV6Tiny;
using Sdcb.SimdPaddleOCR.Models.TextLineOrientation;

var bundle = ChineseV6TinyModels.Default;                             // det / rec / 字典 三件套

using Stream detection = bundle.Detection.OpenRead();
using Stream recognition = bundle.Recognition.OpenRead();
using var dictionary = new StreamReader(bundle.Dictionary.OpenRead());
using Stream classification = TextLineOrientationModel.OpenRead();    // 可选：方向分类

// 全程在内存里，不写临时文件（FromFiles / FromBytes 也都有）
var model = PaddleOcrModel.FromStreams(detection, recognition, dictionary, classification);
Console.WriteLine($"{model.Classes.Count} 类，cls {model.HasClassification}");   // 6906 类，cls True

var ocr = new PaddleOcrPipeline(model, new PaddleOcrOptions { DetLimitSide = 960 });
```

（上面这段在仓库里实际编译运行过，输出就是注释里那行。）

**C. 已经在用 Sdcb 的引擎？模型文件可以共用**

- `Sdcb.PaddleOCR`（`PaddleOcrAll`）走的是 `Sdcb.PaddleInference` + 原生 `paddle_inference`；
  本库**不跑** `paddle_inference`，要的是 **ONNX**（det / rec / dict，可选 cls）。
- 手上只有 Paddle 格式（`inference.pdmodel` + `inference.pdiparams`）时，先 `paddle2onnx` 转一次（见 D）。
- 已经是 ONNX 的（包括 `Sdcb.SimdPaddleOCR` 生态里的模型包），**同一份文件两边都能用**：
  本仓库就用同模型同图与 `Sdcb.SimdPaddleOCR` 做过逐字对照（`bench.md` §35 / §37.2）。

**D. 自己导出（模型最自由）**

```powershell
paddle2onnx --model_dir ch_PP-OCRv4_det --model_filename inference.pdmodel `
            --params_filename inference.pdiparams --save_file det.onnx --opset_version 11
paddle2onnx --model_dir ch_PP-OCRv4_rec --model_filename inference.pdmodel `
            --params_filename inference.pdiparams --save_file rec.onnx --opset_version 11
```

几条从实际踩坑里来的约束：

- **字典必须与 rec 配套**：`dict.txt` 每行一个字符，中间空行要保留，索引 0 是 blank，
  末尾补一个空格（`use_space_char`）；rec 的类别数应为「字典行数 + 2」（PP-OCRv6 tiny 是 6906）。
- **det / rec 都是全动态导出**（`x[N,3,H,W]`），要 `Net.ReadModel(path, shape)` 特化；流水线按形状缓存
  （det 2 份 / rec 8 份，`DetShapeCacheSize` / `RecShapeCacheSize`），换尺寸不会无限增长。
- 只支持 **PP-OCR 系列**（DB 检测 + CTC 识别 + PP-LCNet 方向分类）；SAST / 表格 / 版面识别不在范围内。
- 检测后处理是**轴对齐框**（不做 pyclipper 多边形外扩）：竖排 / 大角度倾斜能框住，但框是正的。
- 车牌用通用 rec 即可，特化在 `OnnxSharp.Ocr.Plate`（把 6906 类压到车牌字符集 + 逐位规则 + 形态校验）。

### 支持算子

76 个注册算子（另有 `Constant` 在导入期内联为常量 Arg，`Silu` / `FusedEltwise` / `MatMulArgMax` / `DynamicQuantizeMatMul` / `FusedAttention` / `DflSoftmax` 是图融合的产物）：

| 类别 | 算子 |
|---|---|
| 形状/元数据 (12) | `Shape` `ConstantOfShape` `Expand` `Tile` `Unsqueeze` `Squeeze` `Reshape` `Slice` `Split` `Flatten` `Range` `Pad` |
| 搬运/类型 (4) | `Concat` `Transpose` `Cast` `Identity` |
| 逐元素 (26) | `Add` `Sub` `Mul` `Div` `Mod` `Pow` `Max` `Min` `Abs` `Neg` `Sqrt` `Exp` `Log` `Erf` `Sin` `Cos` `Reciprocal` `Round` `Floor` `Ceil` `Sign` `Clip` `Relu` `HardSigmoid` `Sigmoid` `Tanh` |
| 比较/逻辑 (8) | `Equal` `Less` `Greater` `GreaterOrEqual` `And` `Not` `Where` `IsNaN` |
| 归约/索引 (7) | `ReduceSum` `ReduceMax` `ReduceMean` `TopK` `Gather` `GatherElements` `GatherND` |
| 量化 (2) | `DynamicQuantizeLinear` `MatMulInteger` |
| 索引补充 (1) | `ArgMax`（axis / keepdims / select_last_index；融合成 `MatMulArgMax` 时不物化 logits） |
| 重算子 (16) | `Conv` `ConvTranspose` `MatMul` `Gemm` `Einsum` `RoiAlign` `MaxPool` `AveragePool` `GlobalAveragePool` `GlobalMaxPool` `Softmax` `LayerNormalization` `BatchNormalization` `InstanceNormalization` `Resize` `GridSample` |

> 与目标模型的关系：YOLO11/v8 用前五类里的大部分；RF-DETR 额外要 `GridSample` / `TopK` / `GatherElements` /
> `Reduce*` / `Sin` / `Cos` / `Where`；PP-OCRv6 额外要 `HardSigmoid` / `GlobalAveragePool` / `ReduceMean` /
> `ConvTranspose` / `BatchNormalization` / `Identity`（OCR 补的）；SAM3 又补了 `IsNaN` / `GreaterOrEqual` /
> `InstanceNormalization` / `Einsum`(bqc,bchw-&gt;bqhw) / `RoiAlign`；sherpa Paraformer 的 encoder 段又补了
> `DynamicQuantizeLinear` / `MatMulInteger` / `Pad` 和 **1-D `Conv`**（FSMN 记忆块，输入 `[N,C,T]`）。
> **动态维**不属于算子问题：`Net.ReadModel(path, 输入形状)` 会把全动态图特化成静态图，否则 `Shape` 链折不出常量。

### 量化推理（int8）

支持 ORT 旧式量化（QOperator）的两个核心算子：`DynamicQuantizeLinear` 与 `MatMulInteger`
（u8 / i8 输入、标量或 per-column 零点），与 ONNXRuntime **逐位一致**（回归用例
`quant: quantized graph matches onnxruntime` 用 `tools/scripts/make_quant_model.py` 生成模型 + ORT 金标准）。

「int8 权重常驻 + int8 MAC」这条加速路径三条设想都实测过，结论都是否定的：

| 设想 | 实测 | 结论 |
|---|---|---|
| int8 权重 + F32 计算（`vpmovzxbw` 解包后走现有 FMA 内核），把权重带宽降到 1/4 | `96×320×1280`：int8 权重 10.88 ms vs FP32 1.07 ms（**0.10×**，同在单线程测试进程里） | 否定。按列块反量化把每个列块的 GEMM 做到了 96×320×64 这种「低于并行阈值」的尺寸，20 次串行调用 |
| 瓶颈是权重带宽 | 16 线程实测 `96×320×1280` 的 FP32 GEMM 有 **496 GFLOPS**（机器峰值 400~600） | 否定。带宽不是瓶颈；之前记的「73 GFLOPS」是测试进程里的单线程数字，已更正 |
| VNNI（`vpdpbusd`，理论 4× 整数吞吐） | `net10.0` 上 `AvxVnni.IsSupported = False`（当年的结论）；**`net11.0` 下同一台 Zen4 为 `True`** | **已落地**：256 位 VNNI 内核，4 线程 733 GFLOPS、单核 183（= 理论峰值），见 [优化与结果](#优化与结果) |

**真正有效的替代**（已实现并回归）：没有 VNNI 时 `MatMulInteger` 不走向量化的标量 int32 内核，
而是把「整数矩阵乘」交给 F32 GEMM —— 权重一次性展开成 `(w - w_zp)` 的 F32 并缓存，
激活按 `(a - a_zp)` 展开，算完四舍五入回 int32。反量化后因子都是小整数、累加和远在 2^24 内，
**结果与整数点积逐位相同**。所以现在两条路都在，按运行时能力自动选：

| 运行时 | `MatMulInteger` 走什么 | decoder 纯 forward |
|---|---|---|
| `net11.0` 且 `AvxVnni.IsSupported`（Zen4 / Zen5 / Intel 12 代+） | **4 行 × 24 列 VNNI 整数内核**（256 位 `vpdpbusd`），4 线程 733 GFLOPS | **6.0 ms** |
| 其它（`net8.0` / `net10.0` 或老 CPU） | F32 GEMM 展开（逐位等价） | 6.9 ms |

下面这张表是**当年**（.NET 10、无 VNNI）把 encoder 段从 10.9 s 打到 104 ms 的演进链，保留作对照：

| 阶段 | 耗时（`speech[1,96,560]`，97 帧） |
|---|---|
| 标量 int32 `MatMulInteger` | 10,871 ms |
| + F32 GEMM 替代 | 578 ms（18.8×） |
| + `Cast` 内核化（170 个 i32→f32，占 33% → 0.8%） | 411 ms |
| + `Pow(x,2)` 改写成 `Mul`（占 12% → 0.1%） | 443 ms |
| + 「每段一个标量」广播快路径（`EwInner` 思路） | 171 ms |
| + 1-D `Conv` 向量化 + `DynamicQuantizeLinear` 的 min/max 向量化 | **104 ms（累计 104×）** |
| ONNXRuntime（同一切图；ORT 把 `DynamicQuantizeLinear + MatMulInteger` 融成 `DynamicQuantizeMatMul`） | 30.9 ms |

即：**量化图的正确性 + 可用的吞吐都有了**。这段当年的结论是「剩下的差距不是 VNNI」，
**那条已被推翻**（见 [优化与结果](#优化与结果) 与 [bench.md](bench.md) §32 / §34）：
差距的一半恰恰就是 VNNI 整数内核，补上之后 decoder 对 ORT 由 2.97× 收到 1.67×。
剩下的 1.67× 里，`DynamicQuantizeMatMul`（占 decoder 的 39%）仍是 F32 展开 ——
它的成本就是 GEMM 本身（≈512 GFLOPS），换整数内核要多吃一趟 `int32 → float`，实测为负收益，所以维持现状。

### 语音前端（fbank → LFR → CMVN）

Paraformer 的 `speech` 输入是 <c>[T, 560]</c> 的 560 维特征，不是原始波形。这一层现在是纯 C#：

```csharp
// 从模型 metadata 自动读 LFR / CMVN 参数（lfr_window_size / neg_mean / inv_stddev）
var extractor = SpeechFeatureExtractor.FromModel(@"sherpa\model.int8.onnx");
var feats = extractor.ComputeFromWav("test.wav");     // [Frames, 560]
```

或者直接命令行（产出可以直接喂给 `run` / ORT）：

```
OnnxSharp.Cli features sherpa\model.int8.onnx test.wav --out feats.f32
# wav/meta/feats 三行：feats : [100, 560]  (22.6 ms)
```

实现细节（都实测对齐过 kaldi-native-fbank，sherpa 内部用的同一个库）：

- 分帧 25/10 ms、snip_edges、去直流、预加重 0.97、povey 窗、补零到 512、实 FFT、功率谱
- 梅尔三角滤波 **全程 float**（用 double 会让「恰好落在三角边上的 FFT bin」包含关系翻转，响亮 bin 上能差 3e-4）
- 取对数下限 `1.1920929e-7`（`float epsilon`）；只算到 `FftSize/2`（不含 Nyquist）
- LFR：`T_lfr = ceil(T/6)`，左侧补 3 帧、右侧重复最后一帧（funasr `WavFrontend` 语义）

实测：**median \|diff\| 4.8e-7、54.9% 逐位相同、有声 bin ≤ 9.7e-5**；剩下的差异只在全静音的 bin（能量贴着 log 下限，对数会把 1e-8 的绝对差放大成 1e-2），对 ASR 无影响。

### 性能

**全部数据都是「纯 CPU + 4 线程」**（`ONNXSHARP_THREADS=4`），这是刻意的口径而不是机器上限：
服务器多数没有显卡，这个进程也不会独占整机（见上一节）。
仅模型前向、batch=1、FP32、输入全 0.5（Ryzen 7 8845HS / Zen4 / Windows 11 / .NET 10）。
四引擎在同一会话内轮转（同样 4 线程）、各取中位（括号里是最小）：

| 模型 | OnnxSharp | ONNXRuntime CPU | OpenCV 5 dnn | SimdYolo |
|---|---|---|---|---|
| yolo11n 640² | 29.2（28.7） | 23.3（23.2） | 45.8（45.0） | 32.2（29.3） |
| yolo11s 640² | 79.7（77.1） | 64.6（63.8） | 132.3（126.2） | 78.7（77.5） |
| rfdetr-small 512² | 198.0（196.3） | 187.5（187.2） | 315.2（311.2） | 跑不了 |

| 模型 | ÷ ONNXRuntime | ÷ OpenCV | ÷ SimdYolo |
|---|---|---|---|
| yolo11n | **1.25×** | 0.64× | 0.91× |
| yolo11s | **1.23×** | 0.60× | 1.01× |
| rfdetr-small | **1.06×** | 0.63× | — |

语音（sherpa Paraformer int8，中文，`huihui.wav` → 76 帧特征 / 16 token，ms）：

| 段 | OnnxSharp | ONNXRuntime | 倍率 |
|---|---|---|---|
| encoder 纯 forward | 30.3 | 24.1 | **1.26×** |
| decoder 纯 forward | **6.0** | 3.60 | **1.67×** |
| 端到端（min） | 48（feat 4 / enc 37 / dec 6） | enc+dec 27.5 | — |

OCR（PP-OCRv6 tiny，**同一套 det/cls/rec 模型、同一份像素**，对手是 `Sdcb.SimdPaddleOCR 2.0.0`，4 线程）：

| 图（median） | OnnxSharp | SimdPaddleOCR | 倍率 | §35 当时（min 口径） |
|---|---|---:|---:|---:|
| 900×300，2 行 | 40.5 | 33.0 | **1.23×** | 1.9×（34.8 / 18.2） |
| 大图，3 行 | 53.3 | 47.7 | **1.12×** | 1.7×（59.7 / 34.3） |

> 口径从 min 换成 median：这台机器偶发百毫秒级尖刺（偶发时 min 也会被污染），同一会话内两张图的
> min 倍率是 0.93× 与 1.42×（自相矛盾），median 倍率则一致（1.23× / 1.12×）。绝对值偏高是因为本次
> 会话有 ~30% 常驻负载。安静窗口下的分段（min）：**det 13.2 / cls 1.6 / rec 2.9 ms，端到端 24.2 ms**
> —— 优化前是 det 17.9 / cls 1.9 / rec 4.4、端到端 34.3。`det` 已经不再慢于对方整条管线。
> 根因与改动见 [bench.md](bench.md) §35 / §37.2。

人脸（YuNet，640²，0.22 MB；体积太小，没进上表的四引擎同会话对照）：

| 段 | min | median |
|---|---:|---:|
| 纯 forward（53 层 Conv） | 5.4~5.7 | 5.9~6.1 |
| 预处理（拉伸到 640² + BGR→CHW） | 1.7 | 2.8 |
| 解码（12 输出 + NMS） | ~0.5 | ~0.5 |
| **端到端** | **8.4** | **10.9** |

> 53 层 Conv 占 forward 的 94%，`Conv_0`（3→16，3×3，s2）一层就 15%。
> int8 版能跑但**慢 3.5×**（每层多两趟 f32 物化，20.3 vs 5.7 ms；池化/Resize 行切分后已比 §36 的 22.95 快 11%）；
> 三条优化尝试均为中性或负收益，见 [bench.md](bench.md) §36 / §37.4。

**结论**：ORT 仍是最快的 CPU 引擎，差距按计算域分层 —— **F32 图上落后 1.05~1.28×**，
int8 的 decoder 落后 1.67×（还剩 ORT 的 `vpdpbusd` 峰值与我们 F32 路径的差），
**OCR 落后 1.1~1.3×**（不同对手）；OpenCV 已全面领先 0.59~0.70×，SimdYolo 打平（0.97~1.14×）。
OpenCV 那一天本身处于坏状态，跨会话只比倍率方向，历史口径见 [bench.md](bench.md) §4 的 1.21~1.23×。

> 口径提醒：不含预处理与后处理；对后台负载敏感，同代码重跑 ±8% 属正常；
> 跨会话的绝对值不作数，只比同会话内的倍率与方向（理由见 `bench.md` §31.5）。
> 全部数字与复现命令在 [bench.md](bench.md)（本表是 §28 + §34.5 + §37）。

### 优化与结果

这一节是**优化史**：按「图 → 内存 → 内核 → 线程 → 前端」列做过什么、量到什么，
以及哪些直觉被数据否掉。单项的完整推导、逐轮 A/B 与复现命令都在
[bench.md](bench.md)（唯一事实源），这里只留结论。

> **所有收益都是 4 线程下的**：机器有 8 核 16 线程，但口径固定为 4（`ONNXSHARP_THREADS=4`）——
> 多开线程当然更快，可那样测出来的数字在真实部署里兑现不了。同理，内核峰值也只拿单核/4 线程说话
> （`183 GFLOPS/核`、`733 GFLOPS/4 线程`），不去凑满核的跑分。

**有单点实测收益的**

| 改动 | 实测 |
|---|---|
| int8 权重改走 F32 GEMM（权重展开成 `w - zp` 缓存、激活同理），取代标量 int32 内核 | Paraformer encoder 段 10,871 → 578 ms（18.8×），结果与整数点积**逐位相同** |
| `Cast` / `Pow(x,2)` / 1-D `Conv` / `DynamicQuantizeLinear` 逐个内核化 + 每段一个标量广播快路径 | 578 → **104 ms**（累计 104×；同图 ORT 30.9） |
| `DQL + MatMulInteger + Cast + Mul + Add` 融成 `DynamicQuantizeMatMul` | 少掉 u8/int32 的物化与 3~5 趟独立访存 |
| 张量缓冲 **64 字节对齐**（POH 分配 + 偏移推到对齐边界，`Tensor.Alignment`） | yolo11n 29.9 → **28.4 ms**；1×1 GEMM 124 → **138 GMAC/s**（图内对齐率 A 18% → 全对齐） |
| int8 GEMM 按 MLAS 结构重写成 **4 行 × 24 列** VNNI 块（12 累加 + 3 B + 1 A = 占满 16 个 ymm）+ 向量化 epilogue | 76×320×1280 **314 → 732 GFLOPS**（单核 183 = 256 位 `vpdpbusd` 理论上限）；decoder 10.6 → **6.0 ms**，对 ORT 由 2.97× 收到 **1.67×** |
| 常驻 intra-op 线程池（调用线程参与 + 自旋领任务，不再每次 `Parallel.For`） | 去掉每帧的派发开销；窄 m（16 行）不再被派发吃掉 |
| 视图式 `Reshape` / `Transpose`（零拷贝） | 折叠后 36 个 `Transpose` 合计只占 **0.5%**，`Reshape` 0.0% |
| 语音前端手写：kaldi fbank / LFR / CMVN | 与 funasr 语义一致，median \|diff\| **4.8e-7**、54.9% 逐位相同 |

**结构性改动（作用是让差距可定位 / 让后续优化有的放矢）**

| 改动 | 作用 |
|---|---|
| 算子融合：Conv+SiLU、逐元素短链、`MatMul+ArgMax`、`MatMul+Add(bias)`→带列 bias 的 GEMM、手写 LayerNorm→`LayerNormalization`、整条 MHA→`FusedAttention`、YOLO DFL→`DflSoftmax` | 减少中间张量物化与多趟访存；注意力/DFL 段不再走 8~10 个算子 |
| 按输入 shape 签名缓存形状与类型推断 | 同尺寸重复推理不再重走 shape 链 |
| `Net.ReadModel(path, 输入形状)` 静态特化 | 全动态图（PaddleOCR）在导入期就能折叠形状链 |
| 卷积四路径：NCHWc 打包 / Im2Col / 1×1 专用 GEMM / Winograd | 3×3 走 NCHWc：内核 9.44 ms、reorder 仅 9.8%、**150 GMAC/s = 4 线程峰值的 81%** |
| `profile` 逐算子 + 一组 `ONNXSHARP_*` 诊断开关（`GEMM_ALIGN` / `DQM_BREAKDOWN` / `INT8_TIER` / `CONV_PROF`） | 每轮优化都是「先量再改」，否则改不动也证明不了 |

**被实测否掉的路线**（比成功项更省时间，别再走一遍）

| 设想 | 实测 | 结论 |
|---|---|---|
| int8 权重 + F32 计算，权重带宽降到 1/4 | `96×320×1280`：int8 权重 10.88 ms vs FP32 1.07 ms（**0.10×**） | 否。按列块反量化把 GEMM 切成低于并行阈值的尺寸 |
| 瓶颈是权重带宽 | 16 线程 F32 GEMM 实测 **496 GFLOPS**（机器峰值 400~600） | 否。带宽不是瓶颈 |
| i16 展开 + `vpmaddwd` | 比 F32 慢 **4×** | 否（吞吐与 FMA 相同） |
| 标量 int32 内核 | 比 F32 慢 **40×**（352~356 ms） | 只留作正确性 oracle |
| 更窄行块（MR 6 / 4）适配 m=16 | 与 MR=12 持平或更差 | 否。小 m 的瓶颈是固定开销 |
| 再融合 170 个 `Transpose` / `Concat` / `Reshape` | 折叠后三者合计 **0.5%** | 前提错误（那是 raw 图的数字） |
| Winograd 3×3 | **0 次命中**（空间都超 `MaxSpatialPixels`） | 现实现是标量变换，打不过 NCHWc |
| direct conv 省掉 im2col 打包 | 全部打包合计只占帧 **5.7%** | 上限太小 |
| 首层（3 通道）沿输出宽向量化 | YuNet `Conv_0` **0.81 → 1.08 ms**（更慢）：每条 `vfmadd` 要配一条 `vbroadcastss`，而面板式 4 行 × 16 列能复用广播 | 否。代码留着但默认关（`ONNXSHARP_FIRSTCONV`） |
| `DynamicQuantizeMatMul` 改走整数内核 | 它的成本就是 F32 GEMM 本身（等效 ≈512 GFLOPS），换过去还多一趟 `int32 → float` | 负收益 |
| 自写 native int8 内核 / P-Invoke | —— | **已不需要**：`net11.0` 下 `AvxVnni` 可用，256 位路径已到理论峰值 |

**当前最弱的一环（由数据指认，不是感觉）**：同模型同图对上 `Sdcb.SimdPaddleOCR`，OCR 仍落后 **1.1~1.3×**
（§35 时是 1.7~2.1×）。§35 定位到的三项「小通道 + 小图」固定开销已吃掉大半（det 17.9 → 13.2 ms，
rec 4.4 → 2.9 ms，见 [bench.md](bench.md) §37.2）；剩下的量级是 det 的 Conv（6.5 ms，实测 87~195 GMAC/s）
与逐元素链（约 2.5 ms，纯内存受限），**没有单项超过 5%**，所以这一环暂时不再作为主线。

**一条经验**：所有有效改动都来自「先量再改」。§29 证明 3×3 的打包浪费只有 5.7%；
§30 用 `GEMM_ALIGN` 量出图内对齐率 A 18% / B 8% / C 0%，才找到 1.2 ms；
§32 用 F32 峰值反推 ORT 在跑整数内核，§34 顺着这条线索补齐内核、把 decoder 差距砍掉一半。
反过来，**每一次「直觉上这里该优化」的猜测，大约有一半被数据直接否掉**（见上表）。

### 已验证的模型

| 模型 | 验证方式 |
|---|---|
| yolo11n | 端到端输出与 ONNXRuntime 逐值对齐（回归用例）；检测结果与 SimdYolo 逐框一致（最大 0.005 px） |
| yolov8n / yolo11s / yolov8s | 跑通并纳入基准 |
| rfdetr-small | 跑通并纳入基准；解码结果与官方 `rfdetr.export._runtime.decode_detections` 逐框一致（41/41 框，坐标 0.00 px）；原始输出与 ONNXRuntime 对齐（512² 随机输入：dets `2.3e-4`、labels `5.0e-5`）。*（rfdetr-medium 的本地导出已丢失，历史数字见 bench.md，当前无法复测）* |
| PP-OCRv6 tiny（det / rec / cls） | det 与 rec 的**原始输出**与 ONNXRuntime 对齐（64² + 960² 输入：max\|diff\| 1.4e-7 / 7.6e-6）；900×300 合成图两行文字全对（`Hello OnnxSharp 123` 0.957、`第二行文字 PaddleOCR` 0.982），与 `Sdcb.SimdPaddleOCR` 同模型同图的输出**逐字一致**；车牌大图上主行 `浙A ES8381` 一致，低置信度小框因 1~2 px 检测差异给出不同字符（`e`↔`8`、`A3M15X`↔`浙A：3M15X`），属模型噪声。性能：同会话 median 比它慢 **1.12~1.23×**（§35 时 1.7~2.1×，见 `bench.md` §35 / §37.2） |
| **PP-OCRv6 tiny + 车牌受限解码**（`Ocr/Plate`） | 车牌专用通路：字符集从 6906 压到 ~70 类 + 逐位规则 + 号牌形态校验。实测（3200×1800）：`浙AES8381` 97.6% NewEnergy、`新A3M15X` 62.9% Fuel；纯车牌裁剪图直读（`--direct`）`浙ABM770` **100.0%**、`浙ACP5872` **99.6%**。与同图普通 `ocr` 对照：普通 OCR 输出 `H浙ACP5872`（框里混进贴字），受限解码把它剪掉；把整张大图误喂给 `--direct` 返回 **0 张**（形态校验 + 分数下限兜底） |
| 单算子微模型 | GEMM / attention / 逐元素链，用于内核定点调优 |
| sherpa Paraformer encoder + CIF（int8） | 用手写脚本把 `model.int8.onnx` 在第一个 `Loop`（CIF）前切开，5715 节点的 encoder 子图**直接跑通**（`speech[1,96,560]` → `[1,97,320]`）；`DynamicQuantizeLinear` + `MatMulInteger`(+s8 权重、per-column 零点) 与 ONNXRuntime **逐位一致**（max\|diff\| = 0）。CIF 用 `Speech/Paraformer/ContinuousIntegrateAndFire` 重写（绕开 `Loop`/`Sequence*`），与 ORT 录的 `frames`/`fires`/`frame.7`/`integrate.7` **逐位一致**，token 数 21 = ORT 21。当前 encoder 纯 forward（76 帧、4 线程）**30.3 ms**，同图 ORT 24.1 ms |
| **Paraformer 端到端（中文）** | Windows 自带的 Microsoft Huihui 合成 4.54 s 中文 wav → 76 帧特征 → 16 个 token → **「今天气真不错我们出去走一走吧」**，与 ORT 全模型参考文本**逐字一致**；整链纯 C#。**热态 min 48 ms / 中位 65 ms**（feat 4 / encoder 37 / decoder 6，4 线程），冷启动 ~1.1 s；ORT 两段合计 27.5 ms（不含特征前端）。演进：81~87 → 68~70 → **48** |
| **YuNet（人脸检测，fp32）** | 官方 `face_detection_yunet` 静态导出（`[1,3,640,640]`，106 节点 / 8 种算子）。**端到端与 OpenCV `cv2.FaceDetectorYN` 逐值对齐**：同一个 640×640 输入下框差 **0.00 px**、5 点关键点差 **3e-5 px**、分数差 5e-7（`YuNetTests`）；512×512 lena 检出 1 张脸 score 0.9135，坐标换回原图空间；预处理是「整图拉伸 + BGR + 0..255 原值」。`face` 子命令可直接出框 / JSON / 画框图。稳态（640×640，**4 线程**）：纯 forward **5.4 ms 最小 / 6.1 ms 中位**；端到端（拉伸预处理 1.7 + forward + 解码 0.5）**8.4 ms 最小 / 10.9 ms 中位**。53 层 Conv 占 forward 的 94%，`Conv_0`（3→16，3×3，s2）一层就占 15%，已试过的三条优化路都是中性或负收益（[bench.md](bench.md) §36） |
| YuNet int8（`yunet_int8.onnx`） | ✅ 支持：53 `QLinearConv` + 16 `DequantizeLinear` + 5 `QuantizeLinear` + 6 `QLinearSigmoid` + 2 `QLinearAdd` + 4 `MaxPool` 全匹配，跑通。同图对 fp32：score 0.8944 vs 0.9135、框中心差 ~1 px；但**慢 3.5×**（20.3 vs 5.7 ms/帧，4 线程；比 §36 的 22.95 快 11%）—— `QLinearConv` 每层多两趟 f32 物化（RQ→DQ 搬运），小模型上省下的带宽抵不过开销。根因与三条尝试见 [bench.md](bench.md) §36 / §37.4 |
| SAM3（vision / geometry / text 编码器 + decoder） | 框提示分段跑通，decoder 与 ONNXRuntime **逐值一致**；行人框提示 top-1 **0.932** |
| **EfficientSAM3 EV-M**（image / text encoder + decoder） | 三模型手工串联（导出没有 geometry-encoder，提示走 `exemplar_*` + CLIP 文本，见 [库用法 11](#库用法-11efficientsam3ev-m三模型分割图--框提示--掩码)）。与 ONNXRuntime 对齐：`fpn_0/1/2` `max\|d\|` 6.3e-7 / 2.1e-5 / 8e-8，`language_*` 4.1e-6（`padding`/`embeds` 逐位一致），`pred_masks` 2.0e-3（值域 ±32）；端到端把 yolo26s 的车辆框当提示，**presence 0.9998、best query 157、掩码与 ORT IoU 1.00000**（概率图 `max\|d\| = 2.5e-5`；换成本库 `Stretch` 预处理是 5.5e-3，差异来自缩放的插值方式）。性能见 [bench.md](bench.md) §41 |

> 跨引擎性能对照（OnnxSharp / ONNXRuntime CPU / OpenCV 5 dnn / SimdYolo）见 [bench.md](bench.md)。
> 同类引擎 **Sdcb.SimdOnnx 目前跑不了 YOLO**：它的 `Softmax` 只支持最后一维，而 YOLO 的 DFL 是
> `axis=1`，`GraphCompiler.EmitSoftmax` 会直接抛 `NotSupportedException`（yolo11n / v8n / v8s / 11s 都撞）。

### 限制

- **当前只做 CPU**，没有 CUDA / DirectML / OpenVINO 等执行后端；Vulkan 计算后端在规划中，以独立可选包提供，不改变主包的零依赖。
- **计算主体是 FP32**，但支持 ORT 旧式量化（QOperator）的 `DynamicQuantizeLinear` + `MatMulInteger`
  （含 s8 权重与 per-column 零点）；`QuantizeLinear` / `DequantizeLinear` / `QLinearConv`（QDQ 格式）未实现。
- 动态维只做「最小可用」：全动态图（如 PaddleOCR）要用 `Net.ReadModel(path, 输入形状)` **特化**成静态图，
  每个尺寸一份图；纯运行期动态形状（同一份图换尺寸）不支持。
- 不针对 WASM / IL2CPP，未验 NativeAOT（但代码保持 AOT 友好）。
- **库内不做图像解码**（JPEG/PNG），像素由调用方给；CLI 用 ImageFacade 解 JPEG/PNG/BMP 并负责画框/保存。
- `OnnxSharp.Vision` 只覆盖**单图、三通道、目标检测**；分割 / 姿态 / OBB / 关键点未实现。
- `OnnxSharp.Ocr.PaddleOCR` 只覆盖 **PP-OCRv6 系列**（DB 检测 + CTC 识别 + 文本行方向分类）；表格识别、
  版面分析、SAST 等其它 OCR 结构未实现。检测后处理是**轴对齐框**（不做 pyclipper 的多边形外扩），
  竖排 / 大角度倾斜文本能框住但框是正的。
- `OnnxSharp.Ocr.Plate` 是**解码侧特化**：只认中国大陆号牌形态（省 + 发牌机关字母 + 序号，7/8 位，
  后缀 学·警·挂·港·澳、首字 使·领）。不看车牌颜色，不拼双层黄牌的上下两行，严重倾斜时因为检测框
  是轴对齐的会掉分；`ReadWhole` / `--direct` 要求「整幅图就是一块牌」，否则应该走 `DetectAndRead`。
- **语音识别已打通（Paraformer）**：`Speech/Paraformer/ParaformerRecognizer` 走 `wav → fbank/LFR/CMVN → encoder → CIF → decoder → 文本`，
  与 ONNXRuntime 全模型的参考文本逐字一致。**前提**：模型要先按
  `tools/scripts/split_*.py` 切成 `encoder.onnx` / `decoder.onnx`（整图里的两个 `Loop` 引擎不执行）。
  只验证过这一个中文模型（`speech_paraformer_asr_nat-zh-cn-16k-common-vocab8358`），
  流式 / 热词 / 标点 / 时间戳都没有；英文模型、其它语种未验证。

### 目录结构

```
src/OnnxSharp/
  Core/      Net / NetImpl / Graph / Arg / 解释器 / 内存池 / 融合 / 常量折叠
  Onnx/      手写 protobuf reader、ONNX importer
  Tensors/   Tensor / TensorShape / DType / GEMM / Transpose 内核 / Quantization（int8 饱和转换）
  Ops/       算子的实现（含 NormOps：BatchNormalization / Identity；QuantOps：DynamicQuantizeLinear / MatMulInteger）
  Kernels/   直接卷积、打包卷积、Winograd、MaxPool、Int8Gemm（VNNI + 标量 oracle）、Int8DequantGemm
  Threading/ 并行工具（ONNXSHARP_THREADS）
  Vision/    可选门面：ImagePreprocessor / YoloDetector / YoloDecoder
             / RfDetrDetector / RfDetrDecoder / Detection（可整体删除，不影响推理）
  Ocr/       可选门面（可整体删除，不影响推理）
    PaddleOCR/  OcrLine / OcrPixelFormat / 像素→NCHW / CTC 解码
                + PaddleOcrModel / PaddleOcrPipeline / DbPostprocess / PaddleOcrOptions
    Plate/      车牌特化：PlateCharset（逐位规则）/ RankTable / PlateBeamSearch / PlateAlignment
                / PlateDecoder / PlateNumber（形态校验）/ PlateReader（DetectAndRead / Read / ReadWhole）
  Sam3/      可选门面：Sam3Segmenter（vision → geometry → decoder）+ Sam3Box / Sam3Item / Sam3Nms
  Speech/    语音（可整体删除，不影响推理）
    通用前端：WavReader / FbankOptions / FbankExtractor（kaldi fbank）/ LfrCmvn / SpeechFeatureExtractor
              / TokenTable（tokens.txt + argmax 解码规则）
    Paraformer/  ContinuousIntegrateAndFire（CIF，替代图里的 Loop）/ ParaformerRecognizer（端到端）
                其它 ASR 模型（Whisper / Zipformer / streaming paraformer）以后放同级的兄弟目录
  Imaging/   共享图像预处理：ImagePixels / ImageGeometry / AxisMap / ImageTensor（Vision / OCR / SAM3 共用）
tools/OnnxSharp.Cli/   info / run / bench / profile / detect / ocr / plate / face / asr / sam3 / 微基准 + ImageFacade 图像 IO
tools/OnnxSharp.Sft2Onnx/  safetensors → ONNX 权重补丁（patch / list），纯 C#
tools/scripts/         dump_shapes / gen_golden / extract_ocr_models.ps1 / onnx_fp16_to_f32 / fbank_probe
                       / make_quant_model / split_before_control_flow / sherpa_encoder_probe
                       / probe_tensors / inspect_loop / compare_outputs
tests/OnnxSharp.Tests/ 零依赖测试运行器（124 用例），直接 dotnet run
bench/                四引擎 × 6 模型的对照脚本与产物
```

### 文档

| 文件 | 内容 |
|---|---|
| [plan.md](plan.md) | 设计文档：架构、里程碑、算子清单、性能目标、开放问题 |
| [misc.md](misc.md) | 决策记录与坑：语义细节、对照实验、踩过的雷 |
| [bench.md](bench.md) | **优化史 + 性能唯一事实源**：口径、复现命令、每轮 A/B、被否证的路线（§1~§40，§40 是车牌特化与 yolo26） |
| [upstream.lock](upstream.lock) | 上游 OpenCV 5 dnn 跟踪基线（机器可读） |

### 致谢

这个引擎不是凭空发明的，更多是把两个优秀项目的经验重新实现了一遍。**感谢 SimdYolo 与 [OpenCV](https://opencv.org/) 5 `dnn` 的作者与贡献者**，本项目的很多「正确答案」都来自它们：

**SimdYolo —— 纯 C# 路线的可行性证明**

- 「纯托管 + SIMD 也能跑好 YOLO」这个前提，本身就是它验证过的结论，本项目只是把范围扩展到通用 ONNX 图。
- letterbox 的**半像素中心**采样公式 `(dst + 0.5) * src / dst - 0.5`、填边值 114、以及「用实际取整后的 scale 而不是理论比例还原坐标」的取舍，全部沿用它的约定 —— 否则同一张图在不同引擎里会有 1~2 像素的系统偏移。
- 解码与 NMS 的组织方式（逐锚点取 max、同类别抑制、按分数排序后截断）也参考了它的实现。
- 验证方法同样来自它：先把检测框与官方实现对齐到亚像素，再谈优化。本项目的 YOLO 结果与它逐框一致（最大偏差 0.005 px）。

**OpenCV 5 `dnn` —— 语义与性能的对照基准**

- 算子边界行为的**事实标准**：`Resize` / `GridSample` / `Pad` / `Slice` 的插值方式与越界处理、`INTER_LINEAR` 的半像素对齐、`Conv` 的 auto_pad 语义，都按它的行为对齐；有疑问时以它的输出为准。
- ONNX Importer 的组织方式（算子注册、常量折叠时机、形状推断的位置）提供了可参照的骨架。
- **性能基线**：每一轮优化都以「不劣于本机 OpenCV 5 dnn」为目标，`upstream.lock` 里锁定的就是这条对照线；`bench/` 下的对照脚本与 [bench.md](bench.md) 的结论都以它为参照。

---

## English

[中文](#中文) ｜ **English**

### What it is

`OnnxSharp` is a from-scratch ONNX inference engine in pure C# — **CPU first, FP32, zero third-party dependencies**, with a **Vulkan compute backend planned** as a separate package for industrial PCs and cross-vendor GPUs. It runs ONNX exports of YOLO11 / YOLOv8 / RF-DETR, and ships an optional vision facade that turns an image into detections in source-image coordinates.

Semantics follow OpenCV 5 `dnn` and SimdYolo. Measured on four engines in the same session (see [Performance](#performance)) at 4 threads: **still 1.05–1.28× behind ONNXRuntime** on F32 graphs (1.67× on the int8 decoder), **on par with SimdYolo** (0.97–1.14×), and **ahead of OpenCV 5 `dnn`** across the board (0.59–0.70×). The default thread count is the physical core count.

This engine stands on the shoulders of **SimdYolo** and **OpenCV 5 `dnn`**: operator semantics, preprocessing conventions, performance baselines and the validation methodology are heavily borrowed from both. Thanks to their authors and contributors — details in [Acknowledgements](#acknowledgements).

> **Disclaimer**: this is a **learning project** — its purpose is to understand how an ONNX runtime actually works, not to be a production inference framework.
> So **it certainly has plenty of bugs**: rare operators, edge-case dtypes and error paths are not fully covered, and behaviour may change between versions.
> Running models and comparison experiments is fine; for production, prefer mature implementations such as ONNXRuntime, SimdYolo, SimdPaddleOCR (`Sdcb.SimdPaddleOCR`) or OpenCV 5 `dnn`. Bug reports are welcome via [Issues](https://github.com/sevencat/OnnxSharp/issues).

### CPU first, and only 4 cores

| | |
|---|---|
| **Supported** | CPU, FP32, AVX2 / AVX-512 vectorisation + multithreading (`ONNXSHARP_THREADS` to tune) |
| **Planned** | A **Vulkan compute backend** for industrial PCs and cross-vendor GPUs (NVIDIA / AMD / Intel iGPU), shipped as a separate optional package so the core stays managed-only |
| **Not supported** | CUDA, TensorRT, OpenCL, DirectML, WebGPU, ROCm — these are not planned |
| **Not required** | CUDA toolkit, cuDNN, any native library or P/Invoke (the Vulkan package only needs the system Vulkan loader) |
| **Thread budget** | **Every benchmark and conclusion here is measured on 4 threads**, not "use every core you can find". On real servers this process is one of many services: it often gets 2–4 cores, runs for a long time, and must not hog the box |
| **Why** | The target is deploying where no CUDA stack exists and vendor-native dependencies are unwelcome — **most servers and industrial PCs have no discrete GPU anyway**, and CPU threads are a shared resource. So the goal is **single-frame latency inside 4 threads**, not peak throughput on a benchmark run. A CUDA / OpenCL backend would drag in vendor native libraries and a driver version matrix, which contradicts "managed only, zero dependencies" outright; Vulkan goes into a later optional package instead |

> If your model must run on an NVIDIA GPU, use the ONNXRuntime CUDA / TensorRT EPs. This project only plans a Vulkan compute backend (industrial PCs / cross-vendor GPUs) — no CUDA shim, and no half-baked GPU compatibility layer.

> **Caveat**: every table below (including peaks such as `733 GFLOPS` or `150 GMAC/s`) is measured on
> **4 threads**; per-core figures are called out separately. Only the historical rows that explicitly say
> "16 threads" are the exception. More cores is simply faster — that is not what this project optimises
> for, and it has not been measured systematically. The goal is *make one frame fast with 4 cores,
> on a machine you do not own*.

### The code is AI-written

Ownership first: **nearly all code in this repository was generated by AI coding agents** (Claude / CodeBuddy style). The maintainer's job is limited to three things:

1. stating requirements and acceptance criteria;
2. collecting reference material (SimdYolo, OpenCV 5 `dnn` sources and docs — see [Acknowledgements](#acknowledgements));
3. running the comparison experiments, judging results, and deciding what to keep or throw away.

A few conventions follow from that:

- Design trade-offs are recorded in `plan.md` / `misc.md` as *why it is written this way*, not as "the AI said so";
- Correctness and performance claims rest on reproducible measurements (`bench.md`, `tests/`), never on prose;
- **AI-generated code still needs review.** If an implementation looks wrong to you, suspect it first rather than trusting the comment above it.

### Author

<p align="center">
  <img src="author.jpg" width="132" alt="king"><br>
  <b>king</b> ｜ <a href="https://github.com/sevencat">GitHub @sevencat</a>
</p>

The **requirements, reference material and acceptance criteria** come from the maintainer (see [The code is AI-written](#the-code-is-ai-written)): operator semantics and performance baselines were mined from SimdYolo and OpenCV 5 `dnn`, and every decision to keep or roll back a change came from an A/B run. AI coding agents implement the code under those constraints. Bugs, feature requests and optimisation ideas: [Issues](https://github.com/sevencat/OnnxSharp/issues).

### Features

| Item | Notes |
|---|---|
| **Backend** | **CPU today** (FP32, AVX2 / AVX-512 + threads); **no** CUDA / TensorRT / OpenCL / DirectML backend; **Vulkan compute backend planned** (industrial PCs / cross-vendor GPUs, separate optional package) |
| Dependencies | None — protobuf reader, kernels and runtime are all hand-written |
| Targets | `net8.0` / `net10.0` / `net11.0`, AOT-friendly (no reflection, no `dynamic`). **`net11.0` is the one with AVX-VNNI** (int8 kernels 2.3× faster); other TFMs fall back to the F32 path automatically |
| Operators | 76 registered ops, covering YOLO11/v8, RF-DETR, PP-OCRv6 and the encoder part of sherpa Paraformer exports |
| Graph passes | Constant folding (Shape→Gather→Concat→Reshape chains), operator fusion (Conv+SiLU, element-wise chains, `MatMul`+`ArgMax`, manual LayerNorm → `LayerNormalization`) |
| Runtime | Static buffer pool (outputs detach), shape inference, `Parallel.For` + AVX2/AVX-512 |
| Threads | **Physical core count** by default (via `GetLogicalProcessorInformationEx` on Windows, falling back to `ProcessorCount`), overridable via `ONNXSHARP_THREADS`. SMT buys nothing on these FP-bound kernels, the extra spinning workers only steal time slices. **All benchmarks and tuning assume 4 threads** — the goal is not to saturate the machine |
| Vision | `OnnxSharp.Vision`: letterbox / stretch preprocessing, YOLO decode + NMS (class names parsed from the model metadata `names`), RF-DETR decode, YuNet faces (box + 5 landmarks + NMS) |
| OCR | `OnnxSharp.Ocr.PaddleOCR`: PP-OCRv6 det + optional cls + rec, DB post-processing, CTC — text lines in source pixels |
| Plates | `OnnxSharp.Ocr.Plate`: a **decoder-side** specialisation on the same det/rec — ~70-class charset, positional rules, CTC Viterbi per-character posteriors, plate shape validation; `DetectAndRead` (whole image) / `Read` (known box) / `ReadWhole` (the input *is* a cropped plate) |
| Dynamic graphs | `Net.ReadModel(path, inputShape)` **specialises** a fully dynamic input (`x[N,3,H,W]`) so shape chains fold at import |
| Image decoding | **Not in the library** (keeps it dependency-free); the CLI uses the ImageFacade software backend plus opt-in `Sdcb.SimdCodecs.Jpeg/Png/Bmp` format packages (registered explicitly) |
| Written by | AI coding agents; requirements, references and review by the maintainer (see [The code is AI-written](#the-code-is-ai-written)) |

### Install

**Repo**: [github.com/sevencat/OnnxSharp](https://github.com/sevencat/OnnxSharp) | **NuGet**: [`Sevencat.OnnxSharp`](https://www.nuget.org/packages/Sevencat.OnnxSharp) | **Version**: `0.5.0` — the package carries the prefix, the namespace stays `OnnxSharp`.

```powershell
dotnet add package Sevencat.OnnxSharp --version 0.5.0
```

Version history, dependencies and per-TFM assets are on the NuGet page: [nuget.org/packages/Sevencat.OnnxSharp](https://www.nuget.org/packages/Sevencat.OnnxSharp#versions-body-tab).

Targets `net8.0` / `net10.0` / `net11.0` with no runtime dependencies. Building from source is described below.

### Quick start

Requires .NET 8 or .NET 10 SDK.

```powershell
dotnet build OnnxSharp.slnx -c Release
dotnet run --project tests/OnnxSharp.Tests -c Release     # 124 cases (yolo26 / OCR goldens skip when the models are absent)
```

Detect objects in one command:

```powershell
$cli = "tools\OnnxSharp.Cli\bin\Release\net10.0\OnnxSharp.Cli.exe"
& $cli detect yolo11n.onnx photo.bmp --conf 0.25 --json out.json
& $cli detect rfdetr-small.onnx photo.bmp --conf 0.3
```

Detect text lines with PaddleOCR:

```powershell
powershell -File tools\scripts\extract_ocr_models.ps1   # det/rec/dict + cls from the nuget model packages
& $cli ocr page.jpg --det .tmp\ocr-models\det.onnx --rec .tmp\ocr-models\rec.onnx `
                    --dict .tmp\ocr-models\dict.txt --cls .tmp\ocr-models\cls.onnx `
                    --limit 1600 --out page.ocr.jpg
```

```
page.jpg 3200x1800
  output : F:\...\page.ocr.jpg
  result : 3 行，849.4 ms
    浙A ES8381  (0.883)  [1752,199,198x68]
    浙A：:3M15X  (0.802)  [1769,1482,202x72]
```

Licence plates use the **same** det/rec through the `plate` subcommand (restricted charset + positional
rules + plate shape validation), and `--direct` when the input already *is* a cropped plate:

```powershell
& $cli plate page.jpg --det .tmp\ocr-models\det.onnx --rec .tmp\ocr-models\rec.onnx `
                      --dict .tmp\ocr-models\dict.txt --limit 960
# 浙AES8381 97.6% NewEnergy / 新A3M15X 62.9% Fuel — with per-character posteriors

& $cli plate plate-crop.png ... --direct      # 浙ABM770 100.0% (135×46, 4.45 ms steady state)
```

The CLI decodes images with ImageFacade (pure managed software backend): picture formats are **opt-in**
`Sdcb.SimdCodecs.Jpeg` / `Png` / `Bmp` packages registered in `ImageHost.RegisterCodecs()`, and `--out`
writes an annotated copy.

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

// LoadRgb24 is your own decoder — the library does not decode JPEG/PNG
byte[] pixels = LoadRgb24("photo.jpg", out int width, out int height);

var detector = new YoloDetector("yolo11n.onnx", new DetectorOptions
{
    Confidence = 0.25f,
    IouThreshold = 0.45f,
    MaxDetections = 300,
});

// (1) packed RGB24, row stride == width * 3; Label comes from the model metadata `names`
foreach (var d in detector.Detect(pixels, width, height))
    Console.WriteLine($"{d.ClassId} {d.Label} {d.Score:F3} [{d.X:F1},{d.Y:F1},{d.Width:F1}x{d.Height:F1}]");

// (2) BGR with row padding: OpenCV Mat, camera SDK buffers
//     matData = byte view of mat.data (Marshal.Copy / unsafe pointer), stride = (int)mat.Step()
foreach (var d in detector.Detect(matData, width, height, bgr: true, rowStride: stride))
    Console.WriteLine(d);
```

**Pixel input contract**: `bgr` swaps R/B (OpenCV and most camera/capture SDKs are BGR); `rowStride` is the **byte count per row** (including alignment padding), `0` means `width * 3`. Both flags exist on `Detect` and on the preprocess-only `Preprocess`, and pixels are read in place — the image is never copied or repacked.

The facade does letterbox (keep aspect ratio, pad 114, bilinear) → forward → per-anchor argmax → threshold → per-class NMS → un-letterbox → clip. Boxes are in source pixels.

**Class names**: ultralytics exports store the label table in the ONNX metadata as `names` — a Python dict literal (`{0: 'person', 1: 'bicycle', …}`), not JSON — together with `imgsz`. `YoloDetector` parses it at construction time; malformed entries or gaps never throw:

```csharp
detector.ClassNames              // 80 names, indexed by class id; empty when the model has no `names`
detector.NameOf(0)               // "person"; null for a missing / unknown id
detector.Metadata.InputHeight    // imgsz height/width as int? (640/640 for yolo11n; null when absent)

var boxes = YoloDecoder.Decode(output, transform, options, names);  // own forward pass: pass a table
YoloMetadata.Parse(net.Metadata);                                   // or parse metadata yourself
```

Every `Detection` from `Detect` / `Decode` carries `Name` (null when the model has no table) and `Label` (falls back to `class{id}`). `YoloDetector` never guesses: with no `names` you get `null`, pass `CocoLabels.Names` explicitly if you want COCO labels. The CLI's `detect` does guess in one case — `--labels` file > model `names` > COCO when the head has 84 fields — and prints the source on its `labels :` line.

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

Both detectors accept the same pixel contract (`bgr` / `rowStride`) on `Detect` and `Preprocess`.

**YuNet — image in, faces out**

```csharp
using OnnxSharp.Vision;

// LoadBgr24 is your own decoder — the library does not decode JPEG/PNG
byte[] bgr = LoadBgr24("group.jpg", out int width, out int height);

var detector = new YuNetDetector("yunet.onnx", new FaceDetectorOptions
{
    ScoreThreshold = 0.6f,   // default 0.6 (sqrt(cls×obj), same as OpenCV)
    IouThreshold = 0.3f,     // default 0.3 (faces overlap a lot, do not raise this)
    MaxFaces = 10,           // default unlimited
});

// (1) packed BGR24 — OpenCV imread and most camera SDKs (bgr defaults to true)
foreach (var face in detector.Detect(bgr, width, height))
    Console.WriteLine($"{face.Score:F3} [{face.X:F1},{face.Y:F1},{face.Width:F1}x{face.Height:F1}] " +
                      $"right-eye {face.Landmarks.RightEye} nose {face.Landmarks.Nose}");

// (2) RGB24, or a buffer with row padding (rowStride = bytes per row, 0 means width * 3)
foreach (var face in detector.Detect(rgb, width, height, bgr: false, rowStride: stride))
    Console.WriteLine(face);

// (3) reuse an already-loaded model: many detectors, one set of weights
var shared = new YuNetDetector(net);
```

Differences from YOLO / RF-DETR (all handled internally, but easy to trip over):

| | YOLO / RF-DETR | YuNet |
|---|---|---|
| Preprocess | letterbox (pad 114) | **plain stretch** to the model's static input (640×640), no padding |
| Normalization | `/255` (ImageNet for RF-DETR) | **raw 0..255**, no mean subtraction, no `/255` |
| Channels | RGB | **BGR** (`bgr` defaults to `true`, matching OpenCV `blobFromImage`) |
| Output | one tensor / two tensors | 12 tensors: `cls` / `obj` / `bbox` / `kps` × stride 8/16/32 |
| Box mapping | invert letterbox | multiply by source width/height (the input was stretched) |

The 5 landmarks keep OpenCV's order: **right eye, left eye, nose, right mouth corner, left mouth corner**
("left/right" are image-side, not subject-side). `Face` also exposes `Right` / `Bottom` / `Area`, and
`Face.IoU(a, b)` compares two boxes.

CLI equivalent:

```
OnnxSharp.Cli face yunet.onnx group.jpg [--conf 0.6] [--iou 0.3] [--max-faces 10] [--json out.json] [--out marked.jpg]
```

The int8 model (`yunet_int8.onnx`) uses the exact same API — just point it at the other file. It is
currently **3.5× slower** than fp32, 20.3 vs 5.7 ms (see [Performance](#performance) and `bench.md` §36 / §37.4).

**Stage-by-stage**

```csharp
// yolo / rfdetr are the detector instances from above
// preprocessing takes the same pixel contract: Preprocess(pixels, w, h, bgr: true, rowStride: stride)
// YOLO: keep the letterbox transform for decoding
var (tensor, transform) = yolo.Preprocess(pixels, width, height);
yolo.Net.SetInput(yolo.InputName, tensor);
var detections = yolo.Decode(yolo.Net.Forward(), transform);

// RF-DETR: no transform needed, just the source size
var stretched = rfdetr.Preprocess(pixels, width, height);
detections = rfdetr.Decode(rfdetr.Net.ForwardAll(), width, height);
```

Or use the static helpers: `ImagePreprocessor.Letterbox(...)`, `ImagePreprocessor.Stretch(...)` (with `ChannelNormalization.ImageNet`), `DetectionNms.Suppress(...)`.

**OCR — image in, text lines out (PaddleOCR)**

```csharp
using OnnxSharp.Ocr.PaddleOCR;

// Models come from the PP-OCRv6 ONNX export; the nuget model packages can be fed directly
// (see "Where do the OCR models come from" below):
//   PaddleOcrModel.FromStreams(ChineseV6TinyModels.Default.Detection.OpenRead(), ...)
var model = PaddleOcrModel.FromFiles("det.onnx", "rec.onnx", "ppocr_keys.txt", "cls.onnx");

var ocr = new PaddleOcrPipeline(model, new PaddleOcrOptions
{
    DetLimitSide = 960,     // raise to 1600 when the text is small relative to the image
    BoxThreshold = 0.6f,
    UnclipRatio = 1.6f,
});

// pixels are interleaved RGB24 / BGR24 / RGBA32 / BGRA32; rowStride = 0 means packed rows
foreach (OcrLine line in ocr.Run(pixels, width, height, OcrPixelFormat.Bgra32, stride))
    Console.WriteLine($"{line.Text}  ({line.Score:F3})  [{line.X:F0},{line.Y:F0}]");
```

Internally: detection size aligned to 32 (longest side ≤ `DetLimitSide`) → DB probability map → 8-connected
components + unclip → per-line crop → (optional) PP-LCNet upside-down check with a 180° rotation →
aspect-preserving resize to height 48 with zero padding → CTC greedy decode + dictionary.

Two things worth knowing:

- **The models are fully dynamic** (`x[N,3,H,W]`); the engine uses `Net.ReadModel(path, inputShape)` to
  **specialise** the input dimensions so the `Shape → Slice → Concat → Reshape` chains can fold at import.
  One specialised graph per `(H,W)`, cached per shape (2 for detection, 8 for recognition —
  `PaddleOcrOptions.DetShapeCacheSize` / `RecShapeCacheSize`).
- The model packages contain **only ONNX + dictionary, no inference code**, so the pipeline borrows just the
  model resources (`tools/scripts/extract_ocr_models.ps1` writes them out unchanged).

**Licence plates — same det/rec, a specialised decoder (`OnnxSharp.Ocr.Plate`)**

No plate-specific model is needed: the stock PP-OCRv6 recogniser already emits every class the plate
characters live in, so the work is entirely on the decoding side — the character set is cut to ~70
classes (31 provinces + 24 letters + 10 digits + 6 suffixes, `I`/`O` excluded), each position is
constrained (`PlateCharset.AllowedAt`), the final sequence is re-scored with a CTC Viterbi alignment
(per-character posteriors, geometric mean), and `PlateNumber.TryParse` validates the shape
(7-digit fuel / 8-digit new-energy / 学·警·挂·港·澳 suffix / 使·领 prefix).

```csharp
using OnnxSharp.Ocr.Plate;

var reader = new PlateReader(pipeline, new PlateReaderOptions { MinScore = 0.5f, MaxPlates = 4 });

// a) whole image: DB detection → aspect filter (1.6–6.5) → per-box plate decode
foreach (PlateResult plate in reader.DetectAndRead(pixels, width, height, OcrPixelFormat.Bgra32, stride))
    Console.WriteLine($"{plate.Text} {plate.Score:P1} {plate.Plate?.Kind}");

// b) a box from your own detector — one rec pass
PlateResult? one = reader.Read(pixels, width, height, OcrPixelFormat.Bgra32, stride, x, y, w, h);

// c) the input *is* one plate (already cropped upstream) — no DB
PlateResult? whole = reader.ReadWhole(pixels, width, height, OcrPixelFormat.Bgra32, stride);
```

`PlateResult.Decoding.CharScores` carries the per-character posterior, so a weak character is called
out instead of guessed (`浙 91% A 100% … 8 94%`; a blurry plate reads `新 17% A 44%`). Measured:
whole-image DB path 825 ms, `--crop` single-box rec 3.6–5.1 ms steady state, `--direct` on 135×46 /
175×49 crops 100.0% / 99.6% at 4.45 / 7.45 ms. Feeding a full-size photo to `--direct` returns
**0 plates** — the shape check and the score floor catch the misuse.

**Where do the OCR models come from**

OCR needs a det + a rec model, an optional cls model and a dictionary; plates reuse the same det/rec.

- **Script (simplest)**: `powershell -File tools\scripts\extract_ocr_models.ps1` copies
  `det.onnx` / `rec.onnx` / `dict.txt` / `cls.onnx` out of the `Sdcb.SimdPaddleOCR.Models.ChineseV6Tiny`
  and `...TextLineOrientation` packages — they embed the ONNX files as resources and contain **no
  inference code**, so the script just streams them to disk.
- **In memory**: reference those two packages (their only transitive dependency is
  `Sdcb.SimdPaddleOCR.ModelProvider` — managed, no native runtime) and read the resources directly:

```csharp
using Sdcb.SimdPaddleOCR.Models.ChineseV6Tiny;
using Sdcb.SimdPaddleOCR.Models.TextLineOrientation;

var bundle = ChineseV6TinyModels.Default;                             // det / rec / dictionary
using Stream detection = bundle.Detection.OpenRead();
using Stream recognition = bundle.Recognition.OpenRead();
using var dictionary = new StreamReader(bundle.Dictionary.OpenRead());
using Stream classification = TextLineOrientationModel.OpenRead();    // optional

var model = PaddleOcrModel.FromStreams(detection, recognition, dictionary, classification);
// 6906 classes, cls True — no temp files, no native dependency
```

- **Already on Sdcb?** `Sdcb.PaddleOCR` (`PaddleOcrAll`) runs `paddle_inference` (native) on Paddle-format
  models, while this library takes **ONNX**; the same ONNX files can be shared, and Paddle-format models
  convert once with `paddle2onnx --model_dir ... --save_file det.onnx`.
- **Model constraints**: the dictionary must match the recogniser (one character per line, blank lines
  preserved, index 0 = blank, trailing space for `use_space_char`, classes = lines + 2); det/rec are
  exported fully dynamic and get specialised per input shape; only the PP-OCR family (DB + CTC +
  PP-LCNet orientation) is supported.

**Speech (sherpa-onnx) is not in the engine**

sherpa-onnx Paraformer / Zipformer graphs contain `Loop` / `NonZero` / `GatherND` / `Sequence` and quantised
(**int8**) weights, while this engine is static-graph, FP32, no subgraphs — that is not "add two operators".

For offline ASR use the sherpa-onnx .NET package (`org.k2fsa.sherpa.onnx`; note it **ships a native
onnxruntime**, which conflicts with "pure managed, zero native dependencies"):

```csharp
using SherpaOnnx;

var config = new OfflineRecognizerConfig
{
    FeatConfig = new FeatureConfig { SampleRate = 16000, FeatureDim = 80 },
    ModelConfig = new OfflineModelConfig
    {
        Tokens = @"sherpa\tokens.txt",
        NumThreads = 2,
        Provider = "cpu",
        Paraformer = new OfflineParaformerModelConfig { Model = @"sherpa\model.int8.onnx" },
        ModelType = "paraformer",
    },
    DecodingMethod = "greedy_search",
};

using var recognizer = new OfflineRecognizer(config);
using OfflineStream stream = recognizer.CreateStream();
stream.AcceptWaveform(sampleRate, samples);   // float[] mono; sherpa resamples to 16 kHz itself
recognizer.Decode(stream);
Console.WriteLine(stream.Result.Text);
```

The WAV reader (PCM 8/16/24/32 and float32, any channel count down-mixed to mono) is ~100 lines of managed
code; it is deliberately not bundled here so an optional feature does not drag in a native runtime.

### CLI commands

| Command | Purpose |
|---|---|
| *(no args)* / `smoke` | Build a tiny Add+Relu graph, verify the engine path |
| `info <model.onnx>` | Dump operator statistics, inputs/outputs, opset |
| `dump-graph <model.onnx> [--type Conv] [--limit N]` | Print inferred output shape per node |
| `run <model.onnx> --input in.f32 [--compare golden.bin] [--repeat N] [--dump dir]` | One forward pass with output statistics |
| `bench <model.onnx> --input in.f32 [--iters N] [--warmup N]` | Timed iterations (median / min) |
| `profile <model.onnx> --input in.f32 [--iters N] [--top N]` | Per-operator timing ranking |
| `detect <model.onnx> <image> [--out marked.jpg]` | Preprocess + forward + decode, boxes in source pixels |
| `ocr <image> --det det.onnx --rec rec.onnx --dict keys.txt [--cls cls.onnx] [--out marked.jpg]` | PaddleOCR: DB detection → orientation → CTC decode, text lines in source pixels (can draw a marked image) |
| `plate <image> --det det.onnx --rec rec.onnx --dict keys.txt [--direct \| --crop x,y,w,h]` | Plate OCR: restricted charset + positional rules + plate shape validation; `--direct` = the input *is* one cropped plate (no detection) |
| `face <yunet.onnx> <image> [--conf 0.6] [--iou 0.3] [--max-faces N] [--json out.json] [--out marked.jpg]` | YuNet face detection: box + 5 landmarks, coordinates in source pixels (can draw a marked image) |
| `features <model.onnx> <wav> [--out feats.f32]` | fbank → LFR → CMVN (Paraformer `speech` input), params from the model metadata |
| `asr <wav> --models <dir> [--tokens f] [--json] [--out text.txt]` | End-to-end recognition (fbank → encoder → CIF → decoder → text); `<dir>` holds `encoder.onnx` / `decoder.onnx` / `tokens.txt` |
| `gemmbench` / `convbench` / `flopsbench` / `parbench` | Micro-benchmarks for tuning |

Key `detect` flags: `--family auto|yolo|rfdetr`, `--conf`, `--iou` (YOLO), `--max-det`, `--classes`, `--layout`, `--class-agnostic`, `--background -1|none` and `--norm imagenet|255` (RF-DETR), `--labels` (one name per line, overrides the metadata `names`), `--json`, `--input <f32.bin> --shape 1,3,640,640`, `--iters N`, `--size N`.

Name table priority in `detect`: `--labels` file > model metadata `names` > COCO guess (when the head has 4+80 fields). The chosen source is printed on the `labels :` line.

Key `plate` flags: `--direct` (the input is already one plate — skip DB), `--crop x,y,w,h` (one box you already know), `--beam` / `--greedy`, `--no-structure` / `--loose`, `--score`, `--max`, `--limit` / `--det-conf` / `--box` / `--unclip` (DB side, ignored with `--direct`/`--crop`), `--iters N` (only with `--direct`/`--crop`, steady-state single-plate rec), `--json`.

> Timing note: `--iters` defaults to 1, i.e. real first-frame latency including JIT. For steady-state numbers use `--iters 40` — .NET tiered compilation promotes hot methods only after ~30 calls, otherwise results are 1.5–2× too slow.

`sft2onnx` (separate small tool: patch safetensors weights into an existing ONNX skeleton, **pure C#, no Python**):

| Command | Purpose |
|---|---|
| `sft2onnx patch <skeleton.onnx> <weights.safetensors> -o out.onnx [--verify]` | Replace only the initializer data fields; graph, metadata and unknown fields are carried over byte-for-byte |
| `sft2onnx extract <model.onnx> -o w.safetensors [--graph-out skeleton.onnx]` | Reverse direction: pull the weights out into safetensors and (optionally) write a graph-only skeleton |
| `sft2onnx list <model.onnx\|weights.safetensors>` | List skeleton initializers / weight-file tensors (auto-detected by extension) |

Flags: `--map <skeletonName>=<weightName>` (repeatable), `--strip a,b` (defaults to `module.,model.,_orig_mod.,backbone.`), `--strict`, `--external-threshold <GiB>` (default 1), `--data-name`, `--verify`.

> **Why a skeleton is required**: an ONNX file is a graph *plus* weights; safetensors only has weights. The architecture cannot be recovered from weights, so the graph comes from a skeleton — training produces `best.safetensors`, deployment turns it into a model with the skeleton shipped alongside.
>
> Measured on yolo11n (196 initializers, 10.2 MiB): **67 ms** to patch, and ONNXRuntime gives **max\|diff\| = 0.0** between the original and the patched model. F16 weights against an F32 skeleton are converted automatically (max\|diff\| 1.28, i.e. F16 precision at 2e-3 relative).
>
> ⚠️ Export the skeleton with **`fuse=False`** — Ultralytics defaults to `fuse=True`, which folds BatchNorm into Conv so `bn.*` weights have nothing to match. The tool detects unmatched `running_mean/running_var` and says so.
>
> **The two directions are not symmetric**: `ONNX → safetensors + skeleton` is lossless packing (graph *and* weights are both in the file; yolo11n splits into an 86 KB skeleton + 10.15 MiB of weights and patches back to a **byte-identical size with max\|diff\| = 0.0** in ONNXRuntime). The opposite direction (`safetensors + config → ONNX`) is the hard one, because the graph does not exist and needs architecture code. Note that an `extract`-ed skeleton **is not runnable** — its data is stripped; it only pairs with `patch`.

### Quantized inference (int8)

The two core QOperator ops are supported — `DynamicQuantizeLinear` and `MatMulInteger` (u8/i8 inputs,
scalar or per-column zero points) — and they match ONNXRuntime **bit-exactly** (regression test
`quant: quantized graph matches onnxruntime`, model + golden generated by `tools/scripts/make_quant_model.py`).

Three ideas for an "int8 weights + int8 MAC" fast path were measured; all of them lost:

| Idea | Measurement | Verdict |
|---|---|---|
| Keep int8 weights, dequantise per column block with `vpmovzxbw` and reuse the FMA kernel (4× less weight traffic) | `96×320×1280`: 10.88 ms vs 1.07 ms FP32 (**0.10×**, both single-threaded in the test process) | No — per-block dequantisation turns each GEMM into `96×320×64`, below the parallel threshold, 20 serial calls |
| Weight bandwidth is the bottleneck | With 16 threads the same FP32 GEMM runs at **496 GFLOPS** | No — bandwidth is not the limit; the earlier "73 GFLOPS" figure was single-threaded and is corrected |
| VNNI (`vpdpbusd`, 4× integer throughput) | `AvxVnni.IsSupported = False` on `net10.0` (the old finding); **`True` on `net11.0` on the same Zen4** | **Landed**: 256-bit VNNI kernel, 733 GFLOPS on 4 threads / 183 GFLOPS per core (= the ceiling); see [Optimizations & results](#optimizations--results) |

**What actually works** (implemented + covered by tests): without VNNI, `MatMulInteger` hands the integer
matrix product to the F32 GEMM — weights are widened once to `(w - w_zp)` floats and cached, activations are
widened per call, and the result is rounded back to int32. Every factor is a small integer and the running
sums stay far below 2^24, so the result is **bit-identical** to the integer dot product.
Both paths coexist and the runtime picks one:

| Runtime | What `MatMulInteger` uses | decoder, pure forward |
|---|---|---|
| `net11.0` with `AvxVnni.IsSupported` (Zen4 / Zen5 / Intel 12th-gen+) | **4-row × 24-column VNNI kernel** (256-bit `vpdpbusd`), 733 GFLOPS on 4 threads | **6.0 ms** |
| Anything else (`net8.0` / `net10.0`, older CPUs) | F32 GEMM widening (bit-equivalent) | 6.9 ms |

The table below is the **historical** (`.NET 10`, no VNNI) chain that took the encoder sub-graph from
10.9 s to 104 ms; kept for reference (`speech[1,96,560]`, 97 frames):

| Step | Time |
|---|---|
| Scalar int32 `MatMulInteger` | 10,871 ms |
| + F32-GEMM substitute | 578 ms (18.8×) |
| + SIMD `Cast` (170 × i32→f32, 33% → 0.8%) | 411 ms |
| + `Pow(x,2)` rewritten as `Mul` (12% → 0.1%) | 443 ms |
| + per-run scalar broadcast fast path (`EwInner` idea) | 171 ms |
| + vectorised 1-D `Conv` and `DynamicQuantizeLinear` min/max | **104 ms (104× overall)** |
| ONNXRuntime (same split graph) | 30.9 ms |

Later work, measured with **alternating A/B** on the 76-frame end-to-end fixture (`profile` sum-of-min,
each switch toggled through `ONNXSHARP_NO_FUSE` / `ONNXSHARP_NO_SHAPECACHE`, best of 5–6 runs):
encoder **66.5 → 43.8 ms (−33%)**, of which `DynamicQuantizeMatMul` (the `DQL → MatMulInteger → Cast →
Mul → Add` chain folded into one layer) is −14.1 ms, `FuseAttention` (40 self-attention blocks folded into
one layer each, killing 40 `Softmax`/80 `MatMul`/the `[B,H,T,T]` scores tensors) is −7.4 ms, and
`FuseLayerNorm` is −0.7 ms. The per-op shape/type inference cache measures **0** on this graph — it keeps
allocation flat but does not speed anything up. ORT needs 39 ms on the same encoder, so what is left is the
`DynamicQuantizeMatMul` GEMMs themselves (~74% of our time) — i.e. an int8 kernel, not more fusion.

**Follow-up (resolved)**: that int8 kernel was later implemented — the 4×24 VNNI block above, which took
`MatMulInteger` from 314 to 732 GFLOPS. On the current fixture the encoder is **30.3 ms vs ORT 24.1 ms**
(1.26×) and the decoder **6.0 ms vs 3.60 ms** (1.67×); the decoder gap halved. The remaining encoder gap is
F32-domain (its 160 `DynamicQuantizeMatMul` nodes still use the F32 path, because moving them to the integer
kernel measured *negative* — see [Optimizations & results](#optimizations--results)).

So quantized graphs are both correct and usable. The remaining gap is **not VNNI** — neither implementation
gets VNNI on this machine. ORT's own profiler shows it fuses `DQL + MatMulInteger` (and the manual LayerNorm)
into single kernels; we now do the same fusion (`DynamicQuantizeMatMul`), and what is left is that ORT's
`DynamicQuantizeMatMul` also picks an int8 kernel when the CPU has one (we get the F32-GEMM substitute).

### Speech (Paraformer, pure C#)

```csharp
var recognizer = new ParaformerRecognizer(new ParaformerOptions
{
    EncoderPath = @"sherpa\encoder.onnx",   // cut by split_before_control_flow.py
    DecoderPath = @"sherpa\decoder.onnx",   // cut by split_decoder.py
    TokensPath = @"sherpa\tokens.txt",
});

var result = recognizer.TranscribeWav("test.wav");   // 16 kHz mono
Console.WriteLine(result.Text);       // 今天气真不错我们出去走一走吧
```

CLI: `OnnxSharp.Cli asr test.wav --models sherpa --tokens sherpa\tokens.txt`.
CIF (`ContinuousIntegrateAndFire`) is hand-ported from the 80-node `Loop` body and matches ORT bit-exactly;
token decoding follows sherpa: argmax → skip `<blank>` → fold repeats → stop at `</s>`.

### Performance

**Everything below is measured on pure CPU with 4 threads** (`ONNXSHARP_THREADS=4`) — a deliberate
choice, not a machine limit: most servers have no GPU and this process never owns the whole box.
Forward pass only, batch=1, FP32, input all 0.5 (Ryzen 7 8845HS / Zen4 / Windows 11 / .NET 10).
All engines rotate within one session (also 4 threads), median shown (min in parentheses):

| Model | OnnxSharp | ONNXRuntime CPU | OpenCV 5 dnn | SimdYolo |
|---|---|---|---|---|
| yolo11n 640² | 29.2 (28.7) | 23.3 (23.2) | 45.8 (45.0) | 32.2 (29.3) |
| yolo11s 640² | 79.7 (77.1) | 64.6 (63.8) | 132.3 (126.2) | 78.7 (77.5) |
| rfdetr-small 512² | 198.0 (196.3) | 187.5 (187.2) | 315.2 (311.2) | cannot run |

| Model | ÷ ONNXRuntime | ÷ OpenCV | ÷ SimdYolo |
|---|---|---|---|
| yolo11n | **1.25×** | 0.64× | 0.91× |
| yolo11s | **1.23×** | 0.60× | 1.01× |
| rfdetr-small | **1.06×** | 0.63× | — |

Speech (sherpa Paraformer int8, Chinese, `huihui.wav` → 76 feature frames / 16 tokens, ms):

| Stage | OnnxSharp | ONNXRuntime | Ratio |
|---|---|---|---|
| encoder, pure forward | 30.3 | 24.1 | **1.26×** |
| decoder, pure forward | **6.0** | 3.60 | **1.67×** |
| end-to-end (min) | 48 (feat 4 / enc 37 / dec 6) | enc+dec 27.5 | — |

OCR (PP-OCRv6 tiny, **same det/cls/rec models and the same pixels**, against `Sdcb.SimdPaddleOCR 2.0.0`,
4 threads):

| Image (median) | OnnxSharp | SimdPaddleOCR | Ratio | §35 back then (min) |
|---|---|---:|---:|---:|
| 900×300, 2 lines | 40.5 | 33.0 | **1.23×** | 1.9× (34.8 / 18.2) |
| large, 3 lines | 53.3 | 47.7 | **1.12×** | 1.7× (59.7 / 34.3) |

> Switched from min to median: this machine has occasional 100 ms spikes (they pollute `min` too),
> so within one session the min ratios were 0.93× and 1.42× (self-contradictory) while the median
> ratios agreed (1.23× / 1.12×). Absolute values are high because that session carried ~30% background
> load. Quiet-window breakdown (min): **det 13.2 / cls 1.6 / rec 2.9 ms, end to end 24.2 ms** — it was
> det 17.9 / cls 1.9 / rec 4.4, end to end 34.3 before. `det` alone is no longer slower than their whole
> pipeline. See [bench.md](bench.md) §35 / §37.2.

Faces (YuNet, 640², 0.22 MB — too small for the four-engine table above):

| Stage | min | median |
|---|---:|---:|
| Pure forward (53 Conv layers) | 5.4~5.7 | 5.9~6.1 |
| Preprocess (stretch to 640² + BGR→CHW) | 1.7 | 2.8 |
| Decode (12 outputs + NMS) | ~0.5 | ~0.5 |
| **End to end** | **8.4** | **10.9** |

> 94% of forward is those 53 Conv layers, and `Conv_0` (3→16, 3×3, s2) alone is 15%.
> The int8 build runs but is **3.5× slower** (two extra f32 materialisations per layer, 20.3 vs 5.7 ms;
> 11% faster than the 22.95 in §36 after the pool/resize row split); all three optimisations tried were
> neutral or negative — [bench.md](bench.md) §36 / §37.4.

**Verdict**: ONNXRuntime is still the fastest CPU engine, and the gap now splits by compute domain —
**1.05~1.28× behind on F32 graphs**, 1.67× behind on the int8 decoder (that remainder is ORT's
`vpdpbusd` peak against our F32 `DynamicQuantizeMatMul`), **1.1~1.3× behind on OCR** (different
opponent); OpenCV is beaten across the board (0.59~0.70×) and SimdYolo is on par (0.97~1.14×). That
OpenCV day was in a bad state; cross-session comparisons only use ratios, see [bench.md](bench.md) §4.

> Excludes pre/post-processing; sensitive to background load (±8% between runs); absolute numbers are
> not comparable across sessions, only ratios measured within one session are. Everything, with
> reproduction commands, lives in [bench.md](bench.md) (this table is §28 + §34.5 + §37).

### Optimizations & results

What was actually done, what it bought, and what the data killed. Full derivations, per-round A/B
and repro commands are in [bench.md](bench.md) — the single source of truth.

> **Every number below is a 4-thread number** (`ONNXSHARP_THREADS=4`) on a machine that has 16.
> More threads would of course be faster, but that speed does not survive into real deployments.
> Kernel peaks are quoted per core or per 4 threads (`183 GFLOPS/core`, `733 GFLOPS/4 threads`)
> on purpose — no full-machine benchmark padding.

| Change | Measured |
|---|---|
| int8 weights routed through the F32 GEMM (weights expanded to `w − zp` and cached) instead of a scalar int32 kernel | Paraformer encoder sub-graph 10,871 → 578 ms (18.8×), result **bit-identical** to the integer dot product |
| Per-op kernel work: `Cast`, `Pow(x,2)`→`Mul`, 1-D `Conv`, `DynamicQuantizeLinear` min/max vectorisation, scalar-broadcast fast path | 578 → **104 ms** (104× cumulative; ORT on the same sub-graph 30.9) |
| `DQL + MatMulInteger + Cast + Mul + Add` fused into `DynamicQuantizeMatMul` | Removes u8/int32 materialisation and 3–5 separate memory passes |
| **64-byte alignment** for all tensor buffers (`Tensor.Alignment`, POH + offset pushed to the boundary) | yolo11n 29.9 → **28.4 ms**; 1×1 GEMM 124 → **138 GMAC/s** |
| int8 GEMM rewritten to a **4-row × 24-column** VNNI block (12 accumulators + 3 B + 1 A = all 16 ymm) with a vectorised epilogue | 76×320×1280 **314 → 732 GFLOPS** (183 per core = the 256-bit `vpdpbusd` ceiling); decoder 10.6 → **6.0 ms**, i.e. 2.97× → **1.67×** vs ORT |
| Resident intra-op thread pool (caller thread participates, spin-based task stealing) | Removes per-frame dispatch cost; narrow-m (16 rows) shapes no longer pay for it |
| View-based `Reshape` / `Transpose` (zero-copy) | After folding, 36 `Transpose` ops cost **0.5%** in total, `Reshape` 0.0% |
| Hand-written speech front-end (kaldi fbank / LFR / CMVN) | median \|diff\| **4.8e-7**, 54.9% bit-identical |

**Dead ends** (measured, not guessed — worth reading before optimising anything here):

| Idea | Measurement | Verdict |
|---|---|---|
| int8 weights + F32 math to cut weight bandwidth 4× | `96×320×1280`: 10.88 ms vs 1.07 ms FP32 (**0.10×**) | No — column-block dequantisation shrinks GEMMs below the parallel threshold |
| The bottleneck is weight bandwidth | 496 GFLOPS FP32 GEMM on 16 threads (peak 400–600) | No |
| i16 widening + `vpmaddwd` | **4× slower** than F32 | No |
| Scalar int32 kernel | **40× slower** (352–356 ms) | Kept only as the correctness oracle |
| Narrower MR (6 / 4) for m=16 | Ties or loses against MR=12 | No — small m is bound by fixed overhead |
| Fusing another 170 `Transpose` / `Concat` / `Reshape` nodes | Together they are **0.5%** after folding | Premise was wrong (raw-graph numbers) |
| Winograd for 3×3 | **0 hits** (every 3×3 exceeds `MaxSpatialPixels`) | The current implementation is scalar anyway |
| Direct conv to skip im2col packing | All packing is **5.7%** of the frame | Ceiling too low |
| Vectorising the 3-channel first layer along output width | YuNet `Conv_0` went **0.81 → 1.08 ms** (slower): each `vfmadd` needs a `vbroadcastss`, while the panel form reuses the broadcast across 4 rows × 16 columns | No. Code kept but off by default (`ONNXSHARP_FIRSTCONV`) |
| `DynamicQuantizeMatMul` on the integer kernel | Its cost *is* F32 GEMM (≈512 GFLOPS); moving it adds an `int32 → float` pass | Negative |
| Hand-written native int8 kernel / P-Invoke | — | **Not needed**: with `net11.0`, `AvxVnni` is available and the 256-bit path already hits the theoretical peak |

**Our weakest link (pointed at by data, not by feeling)**: against `Sdcb.SimdPaddleOCR` with identical
models and pixels, OCR is still **1.1~1.3×** behind (it was 1.7~2.1× back in §35). Most of the
"narrow channels + small image" fixed overhead identified there is gone — det 17.9 → 13.2 ms, rec
4.4 → 2.9 ms ([bench.md](bench.md) §37.2). What is left is det's Conv (6.5 ms, measured 87~195 GMAC/s)
and its elementwise chain (~2.5 ms, purely memory bound) — **no single item above 5%**, so this is no
longer the main line.

**One lesson**: every real win came from measuring first. §29 showed 3×3 packing waste is only 5.7%;
§30 measured the in-graph alignment rate (A 18% / B 8% / C 0%) and found 1.2 ms; §32 inferred from the
F32 peak that ORT was running an integer kernel; §34 followed that lead and halved the decoder gap.
Conversely, roughly **half of the "obviously optimisable here" intuitions were killed by data**.

### Verified models

| Model | How it is verified |
|---|---|
| yolo11n | End-to-end outputs match ONNXRuntime value-by-value (regression test); detections match SimdYolo within 0.005 px |
| yolov8n / yolo11s / yolov8s | Runs; included in benchmarks |
| rfdetr-small / rfdetr-medium | Runs; decoding matches official `rfdetr.export._runtime.decode.decode_detections` (41/41 boxes, 0.00 px) |
| Micro models | Single-operator graphs for kernel-level tuning |
| sherpa Paraformer encoder (int8) | The 5715-node encoder sub-graph (split before the first `Loop`) runs end-to-end (`speech[1,96,560]` → `[1,97,320]`); `DynamicQuantizeLinear` + `MatMulInteger` (s8 weights, per-column zero points) match ONNXRuntime **bit-exactly** (max\|diff\| = 0). Current encoder pure forward (76 frames, 4 threads): **30.3 ms** vs ORT 24.1 ms |
| **Paraformer end-to-end (Chinese)** | 4.54 s Chinese wav → 76 feature frames → 16 tokens → the reference sentence, character-for-character identical to ONNXRuntime; the whole chain is pure C#. **48 ms min / 65 ms median** warm (feat 4 / encoder 37 / decoder 6, 4 threads); ORT enc+dec 27.5 ms (front-end excluded) |
| **YuNet (face detection, fp32)** | The official static `face_detection_yunet` export (`[1,3,640,640]`, 106 nodes, 8 op types). **Matches OpenCV `cv2.FaceDetectorYN` value-by-value** on the same 640×640 input: box **0.00 px**, 5 landmarks **3e-5 px**, score 5e-7 (`YuNetTests`). Detects 1 face (score 0.9135) on 512×512 lena, with coordinates mapped back to the original image; preprocessing is "stretch + BGR + raw 0..255". `face` CLI prints boxes / JSON / a marked image. Steady state (640×640, **4 threads**): forward **5.4 ms min / 6.1 ms median**; end to end (stretch 1.7 + forward + decode 0.5) **8.4 ms min / 10.9 ms median**. The 53 Conv layers are 94% of forward and `Conv_0` (3→16, 3×3, s2) alone is 15%; the three optimisations tried were all neutral or negative ([bench.md](bench.md) §36) |
| YuNet int8 (`yunet_int8.onnx`) | ✅ supported: 53 `QLinearConv` + 16 `DequantizeLinear` + 5 `QuantizeLinear` + 6 `QLinearSigmoid` + 2 `QLinearAdd` + 4 `MaxPool` all match, runs end to end. vs fp32 on the same image: score 0.8944 vs 0.9135, box centre within ~1 px — but **3.5× slower** (20.3 vs 5.7 ms/frame, 4 threads; 11% faster than the 22.95 in §36 after the pool/resize row split), because each `QLinearConv` materialises f32 twice (RQ→DQ round trip) and a model this small cannot win that back. Root cause and the three attempted fixes in [bench.md](bench.md) §36 / §37.4 |
| **PP-OCRv6 tiny + plate restricted decoding** (`Ocr/Plate`) | Plate-specialised path: charset 6906 → ~70 classes + positional rules + shape validation. Measured (3200×1800): `浙AES8381` 97.6% NewEnergy, `新A3M15X` 62.9% Fuel; `--direct` on cropped plates `浙ABM770` **100.0%** / `浙ACP5872` **99.6%** (135×46 / 175×49). Against plain `ocr` on the same photo: plain OCR emits `H浙ACP5872` (a decal inside the box), the restricted decoder drops it; a full-size photo fed to `--direct` returns **0 plates** |
| SAM3 (vision / geometry / text encoders + decoder) | Box-prompted segmentation runs; decoder matches ONNXRuntime **value-by-value**; pedestrian box prompt top-1 **0.932** |

> Cross-engine numbers (OnnxSharp / ONNXRuntime CPU / OpenCV 5 dnn / SimdYolo) live in [bench.md](bench.md).
> **Sdcb.SimdOnnx cannot run YOLO** today: its `Softmax` only supports the last axis and YOLO's DFL uses
> `axis=1`, so `GraphCompiler.EmitSoftmax` throws `NotSupportedException` (yolo11n / v8n / v8s / 11s alike).

### Limitations

- **CPU first** — no CUDA / DirectML / OpenVINO backends; a Vulkan compute backend is planned as a separate optional package.
- **FP32 compute**, with support for the legacy QOperator pair `DynamicQuantizeLinear` + `MatMulInteger`
  (u8/i8 inputs, scalar or per-column zero points); `QuantizeLinear` / `DequantizeLinear` / `QLinearConv` (QDQ) are not implemented.
- **Speech recognition works (Paraformer)**: `Speech/Paraformer/ParaformerRecognizer` runs
  `wav → fbank/LFR/CMVN → encoder → CIF → decoder → text` and matches the full-model ONNXRuntime
  reference text character-by-character. **Prerequisite**: split the model first with
  `tools/scripts/split_*.py` into `encoder.onnx` / `decoder.onnx` (the engine does not execute the two
  `Loop` subgraphs). Only this one Chinese model is verified; no streaming, hotwords, punctuation or
  timestamps; other languages are untested.
- Dynamic dimensions are minimally supported (batch=1 focus); dynamic H/W needs `TargetWidth/TargetHeight`.
- WASM / IL2CPP are not targeted; NativeAOT is not validated (code stays AOT-friendly).
- **No image decoding in the library**; bring your own pixels. The CLI decodes JPEG/PNG/BMP through ImageFacade and can draw/save the result.
- `OnnxSharp.Vision` covers single-image, 3-channel object detection only — no segmentation, pose, OBB or keypoints.
- `OnnxSharp.Ocr.Plate` is **decoder-side only**: mainland-China plate shapes (province + issuing letter + serial, 7 or 8 chars, 学·警·挂·港·澳 suffix, 使·领 prefix). Plate colour is ignored, two-line yellow plates are not stitched, and heavy skew loses accuracy because detection boxes are axis-aligned; `ReadWhole` / `--direct` expect *one* plate filling the image.

### Layout

```
src/OnnxSharp/   Core (graph/runtime) · Onnx (protobuf reader) · Tensors (incl. Quantization) · Ops (incl. QuantOps) · Kernels (incl. Int8Gemm)
                 Threading · Vision / Ocr (PaddleOCR + Plate) / Sam3 / Speech (Paraformer) — optional facades · Imaging (shared preprocessing)
tools/OnnxSharp.Cli/   info / run / bench / profile / detect / ocr / plate / face / asr / sam3 / micro-benchmarks
tests/OnnxSharp.Tests/ zero-dependency test runner (124 cases)
bench/                 four-engine × six-model comparison scripts
```

### Docs

| File | Content |
|---|---|
| [plan.md](plan.md) | Design: architecture, milestones, operator list, performance targets |
| [misc.md](misc.md) | Decision log and pitfalls |
| [bench.md](bench.md) | **Optimisation history + performance source of truth**: methodology, commands, per-round A/B, dead ends (§1–§40, §40 covers the plate specialisation and yolo26) |
| [upstream.lock](upstream.lock) | Pinned OpenCV 5 dnn baseline |

### Acknowledgements

This engine was not invented from scratch — it re-implements, in a different scope, what two excellent projects had already worked out. **Thanks to the authors and contributors of SimdYolo and [OpenCV](https://opencv.org/) 5 `dnn`**; many of the "right answers" here came from them:

**SimdYolo — proof that the pure-C# path works**

- That managed code plus SIMD is fast enough for YOLO was its conclusion first; this project only widens the scope to general ONNX graphs.
- The letterbox **half-pixel-centre** sampling formula `(dst + 0.5) * src / dst - 0.5`, the pad value 114, and the decision to un-map boxes with the *actually rounded* scale rather than the theoretical ratio all follow its conventions — without them the same image would shift by 1–2 px between engines.
- The shape of decode + NMS (per-anchor max, same-class suppression, sort and truncate) was modelled on its implementation.
- So was the validation method: align boxes with the reference to sub-pixel precision first, optimise later. YOLO detections here match it box-for-box (max 0.005 px).

**OpenCV 5 `dnn` — the reference for semantics and speed**

- The **de-facto standard** for operator edge cases: bilinear behaviour and out-of-range handling in `Resize` / `GridSample` / `Pad` / `Slice`, the `INTER_LINEAR` half-pixel convention, `Conv` auto_pad semantics. When in doubt, its output wins.
- Its ONNX importer layout (operator registration, constant-folding timing, where shape inference sits) provided a skeleton to follow.
- **The performance baseline**: every optimisation round targeted "no slower than OpenCV 5 dnn on this machine", which is exactly what `upstream.lock` pins down. The scripts under `bench/` and the results in [bench.md](bench.md) are measured against it.

---

<p align="center">
  <a href="https://github.com/sevencat/OnnxSharp">GitHub 仓库</a> ｜
  <a href="https://www.nuget.org/packages/Sevencat.OnnxSharp">NuGet 包</a> ｜
  <a href="https://github.com/sevencat/OnnxSharp/issues">Issues</a> ｜
  <a href="https://github.com/sevencat/OnnxSharp/blob/master/bench.md">基准数据</a>
</p>

