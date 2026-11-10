# OnnxSharp 实施计划

> 纯 C# 的 ONNX 推理运行时，以 OpenCV 5 dnn 新版 graph 引擎为参考实现。
> 首版目标：跑通 YOLOv11 检测 与 RF-DETR。
>
> **当前状态：M0 + M0.5 + M1 + M2 + M3 + M7（OCR）+ M8（SAM3 / 共享图像预处理）已完成，M4 进行中**
> 已完成：引擎内核（共享缓冲槽 + temp 池 + 动态形状）、ONNX 全链路导入、72 个注册算子、yolo11n / rfdetr-small / rfdetr-medium / PP-OCRv6（det/rec/cls）/ SAM3（vision+geometry+decoder）端到端对齐
> 校验：`dotnet build` 0 警告 0 错误；`tests` 56/59 PASS（3 个用例需本地模型，缺失则 SKIP）
> **性能数字只看 `bench.md`**（本节 §9 只留目标与口径，不再重复实测表）
> 下一步：M9-01 ASR 基线取证（ORT/sherpa 跑同一 wav 落 fixture）；M4 收尾 —— GEMM 二维 cache blocking
> 最后更新：2026-11-10

---

## 0. 本轮修订要点

基于对 OpenCV 5.2 源码、两个目标模型的实测（python/onnx + ORT）以及 SimdYolo 源码的核对，本轮修正：

| # | 修订 | 依据 |
|---|---|---|
| 1 | 算子总数 **41 → 42**（原并集列表漏了 `Greater`，Batch E 少算 1） | 模型实测 `Greater×2` |
| 2 | **新增 M0.5**：`LayerInfo → Net` 反查、`BufferAllocator`、全局 temp 池、`prog` 可重建、测试工程 | 见 §5.3 / §5.7 |
| 3 | `GetMemoryShapes` 必须能读到**常量输入的值**（不是只有 shape） | `slice2_layer.cpp:363-376` |
| 4 | 缓冲复用从 M4 **提前到 M0.5**（实测 1:1 需 273MB / 1381MB） | §5.7 内存实测 |
| 5 | 动态形状按 OpenCV 语义实现：**输出不预 fit，layer 自己 fit** | `net_impl2.cpp:1891-1898` |
| 6 | M1 验收改为对齐 `onnx.shape_inference`（两模型形状 100% 可推） | 实测未知张量数 = 0 |
| 7 | golden 生成改为 **python 图改写追加 graph.output**（ORT 无「加中间输出」API） | §8.2 |
| 8 | 平台目标收窄：**不支持 WASM / IL2CPP（不在路线图），首版不验 NativeAOT**，仅保持 AOT 友好编码 | 本轮结论 |
| 9 | 性能目标重设（原「优化后 <60ms」过松，「首版 <150ms」过紧） | SimdYolo 22.19ms、ORT 26.4ms |
| 10 | 检测头的前后处理（YOLO 的 letterbox+NMS、RF-DETR 的拉伸+top-k）移出**引擎核心**，作为可选门面放 `src/OnnxSharp/Vision/`（`YoloDetector` / `RfDetrDetector`：一张图进、原图坐标的框出）；引擎本身仍只出原始张量（`[1,84,8400]` / `dets`+`labels`） | 与 §1.3 一致；2026-11-07 落地 |
| 11 | `ConstFolder` 折叠集合补 `Transpose`（可消灭 1 处数据相关 `Where`） | 依赖追踪实测 |
| 12 | 新增 **§13 上游同步机制**：后续更新/优化/bugfix 一律先跟 OpenCV 5 dnn 的 git | `misc.md §11`（基线 `5.x @ 43f8b1f5`） |
| 13 | **性能结果全部搬进 `bench.md`**（唯一事实源），`plan.md` §9 只保留目标与口径；四引擎 × 6 模型可一条命令复现 | 原 §9 是 i7-8700K 的手抄数字，机器/口径都变了 |
| 14 | M4 重排：**GEMM 已不是瓶颈**（真实模型 407 GFLOPS，机器峰值 753）；重点转向逐元素融合与 host 侧开销。**任何内核级优化必须用端到端交替 A/B 判定** | `bench.md` 结论 2/3（`profile` + `fma`；三次内核加速均无端到端收益） |
| 15 | **`profile` 必须带 `--warmup` 且只看每算子最小耗时**，否则 tier-0 代码与后台负载重尾会把均值抬高 20%~4 倍 | `bench.md`「测量方法上的坑」 |

---

## 1. 背景与目标

### 1.1 背景

OpenCV 5.2 的 `modules/dnn` 内含两套引擎：

| 引擎 | 特征 | 本项目是否参考 |
|---|---|---|
| legacy 引擎 | `LayerData` + `allocateLayers`/`forwardLayer` + `BackendWrapper`，4.x 延续 | ❌ 不参考 |
| **新 graph 引擎** | `Arg`/`ArgData` SSA 值表 + `LayerInfo`（只做 shape/type 推断）+ `Layer`（执行器）+ 线性 `prog` 列表 + 解释执行 | ✅ **唯一参考** |

新引擎目前只支持 CPU/CUDA（其它后端在 `net_impl_backend.cpp:330` 被显式拒绝），因此"精简 CPU 运行时 + 纯托管"与官方演进方向一致。

### 1.2 目标

1. 纯 C# 类库，**产品侧零原生依赖**（只依赖 BCL）。
2. 能加载并执行 **ONNX 文件**（只把 ONNX 当模型格式，不需要 ONNX Runtime 执行）。
3. 首版跑通 4 个模型：
   - `F:\ai\models\yolo\yolo11n.onnx`（1×3×640×640 → output0 [1,84,8400]，10.5 MB，opset 19）
   - `F:\ai\models\yolo\yolov11s.onnx`（37.1 MB，opset 17）
   - `F:\ai\rfdetr\onnx_model\rfdetr-small.onnx`（1×3×512×512 → dets[1,300,4] + labels[1,300,91]，118 MB，opset 17）
   - `F:\ai\rfdetr\onnx_model\rfdetr-medium.onnx`（1×3×576×576，124 MB）
4. 数值与 ONNX Runtime 对齐（逐层 MAE/余弦 + 端到端检测结果一致）。
5. 单一托管 DLL 分发，零原生依赖。

### 1.3 平台与部署策略（本轮收窄）

| 项 | 决策 |
|---|---|
| 目标框架 | **`net8.0;net10.0` 双目标，保留不退役（已定）**。net8 于 2026-11-10 EOL，仅作老宿主兼容目标；新特性以 net10 为基准，但不得使用 net8 缺失的 API（`Vector256`/`Fma` 两者都有）。 |
| 运行时形态 | 纯托管、纯 CPU。允许 `unsafe`/`Span`/`Vector256`/`Fma`（`AllowUnsafeBlocks=true`）。 |
| **NativeAOT** | **首版不验证、不写进验收**。但编码保持 AOT 友好：不用反射、不用 `dynamic`、不用 `Emit`、protobuf 手写解析、`Dictionary<string, T>` 用显式 comparer。为将来一句话开 AOT 留路。 |
| **WASM** | **不做**。不写 `browser-wasm` 目标，不为 4GB 线性内存做特殊设计。 |
| **IL2CPP / Unity** | **不支持，且不在路线图**。不提供 `netstandard2.1` 目标，不为 IL2CPP 做裁剪/反射限制改造。宿主必须是 **.NET 8/10 运行时**（桌面 / 服务端）。 |
| 图像编解码 | 交给调用方。`tools/OnnxSharp.Cli` 用 ImageFacade + `Sdcb.SimdCodecs.Jpeg/Png/Bmp`（格式包额外引入并在 `ImageHost` 注册），只服务 CLI，不进产品库。 |
| 图像预处理 / 检测后处理 | **`src/OnnxSharp/Vision/` 提供可选门面**：`ImagePreprocessor`（`Letterbox` 等比填边 / `Stretch` 直接拉伸）+ `YoloDetector`/`YoloDecoder` + `RfDetrDetector`/`RfDetrDecoder`。一张图进去、原图坐标的 `Detection` 出来。像素输入统一按 `(pixels, width, height, bgr, rowStride)` 约定：RGB/BGR 可切换，行跨度以字节计（`0` = `width * 3`），OpenCV `Mat`/相机缓冲可直接喂。它只依赖引擎的 `Net`，不参与图的执行，删掉不影响推理核心。 |

### 1.4 非目标（首版明确不做）

| 不做 | 原因 |
|---|---|
| **WASM / IL2CPP / Unity 宿主** | 部署形态限定为 .NET 8/10 桌面与服务端；不做 WASM 线性内存、不做 IL2CPP 裁剪适配 |
| CUDA / OpenCL / Vulkan / OpenVINO / CANN / WebNN / TimVX | 纯 CPU 托管 |
| int8 量化（`int8layers`） | 目标模型全 FP32 |
| FP16 / BF16 / FP8 | 目标模型全 FP32（`Tensor` 类型预留，kernel 后置） |
| **图级** blocked layout（NC8HW8 / `TransformLayout`） | 首版 NCHW。**允许 kernel 内部打包/padding**（这是 SimdYolo 的做法，对图透明），但不做图级布局改写；接口预留 `Layout` 字段（见 §12 Q7） |
| 图融合 pass（QDQ / attention / matmul→gemm / shared gemm / reshape-transpose / scale-softmax / transpose-matmul） | 纯优化 |
| `onnx_graph_simplifier` 的 2,440 行 pattern 规则 | 两模型都用原生 op，无 decomposed LayerNorm / Conv+BN |
| `If` / `Loop` / `Scan` 子图 | 两模型不含 |
| TFLite 导入 / tokenizer / LLM 相关（SDPA、Rotary、MatMulNBits、KVCache） | 非目标模型 |
| 训练 / 反向传播 / 自动微分 | — |
| 图像编解码（PNG/JPEG 等）、letterbox 的**像素级实现**、NMS 的**算法优化** | 交给调用方 / 由 `Vision/` 门面提供朴素实现（与 OpenCV dnn 一致：引擎只出 `[1,84,8400]` 原始张量）。`Vision/` 里不做 SIMD、不做并行以外的优化 |

> 非目标项均在 `Tensor`/`Graph` 层预留扩展位，但不写实现。

### 1.5 实现参考优先级（本轮新增）

| 优先级 | 参考 | 取什么 |
|---|---|---|
| **1（首选）** | `F:\code\SimdYolo\src` —— 已跑通 YOLO11n，CPU **22.19 ms**（快于 ORT 26.4 ms） | ONNX protobuf 读取、权重打包缓存、GEMM / Conv panel / Winograd kernel、张量池与生命周期扫描、常量折叠 + 形状计划。**能照搬就照搬。** |
| 2 | `F:\code\opencv\modules\dnn`（5.2 新引擎） | SimdYolo 没有的通用结构：`Arg` 表、`LayerInfo/Layer` 分离、`GetMemoryShapes` 契约、42 算子的完整语义与边界 |
| 3 | 其它 | 仅在上述两者都缺失时使用（如 `GridSample`、`LayerNormalization`） |

**冲突处理规则**：若 SimdYolo 的做法与「通用 ONNX 运行时」目标冲突（只支持单一 FP32 输入、把后处理塞进图、`float[]` 无 dtype 等），**不自行改造**，而是：

1. 在 `misc.md §2.5` 记录 SimdYolo 的原做法与依据；
2. 把偏离点列成 `§12` 待决项（Q8~Q10）；
3. 由用户拍板后再动代码。

> 已知需拍板的偏离点见 `misc.md §2.6`（Tensor 内存模型、池化/输出所有权、线程模型），其余偏离点已按建议定稿。

---

## 2. 总体架构

```
┌──────────────────────────────────────────────────────────────┐
│  Onnx/                                                        │
│   Proto/   →  手写 protobuf reader（移植 SimdYolo Proto）      │
│   OnnxImporter  →  LayerParams + Graph 组装（42 个 handler）   │
└───────────────────────────┬──────────────────────────────────┘
                            ↓ Graph（Arg 表 + 线性 prog）
┌──────────────────────────────────────────────────────────────┐
│  Graph/                                                       │
│   ArgTable（ArgData：name/kind/shape/type/tensor）             │
│   Graph（prog: List<LayerInfo>, inputs, outputs, subgraphs）    │
│   ※ prog 必须支持「常量折叠后删节点并重建」                     │
└───────────────────────────┬──────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│  Engine/                                                      │
│   Prepare:  ConstFold → ConstArgs → 重建 prog                  │
│   Finalize: BindExec（LayerInfo → Layer）→ AssignBuffers      │
│   Forward:  逐 op：取输入 Tensor（含常量值）→ Shape/Type 推断   │
│                    → fit 输出 → layer.Finalize → layer.Forward │
└───────────────────────────┬──────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│  Ops/  (42)          Kernels/                                 │
│   逐元素/搬运算子      conv3x3 / conv1x1 / depthwise / GEMM     │
│   归约/索引/形状类     maxpool / softmax / layernorm /          │
│                      nearest-resize / gridsample / topk        │
└──────────────────────────────────────────────────────────────┘
```

**设计铁律（对齐 OpenCV 新引擎）**

1. **值编号制**：所有张量都是 `Arg`（int）。`ArgData.kind ∈ {Const, Input, Output, Temp}`。
2. **推断与执行分离**：`LayerInfo` 只实现 `GetMemoryShapes/GetTypes`；`Layer` 只实现 `Finalize/Forward`。
3. **线性执行**：`prog` 是拓扑序线性表，解释循环顺序执行，不建调度图。
4. **缓冲区池化**：按 `Arg` 复用，但**必须经过冲突图分配**（`BufferAllocator`），不能 1:1。
5. **动态形状是例外**：默认静态，仅 `Where/Expand/ConstantOfShape` 标 `DynamicOutputShapes=true`；这类 op 的输出由 layer 自己 `fit`。

---

## 3. 目录结构

```
F:\code\OnnxSharp\
├─ plan.md                     本文件（目标 / 设计 / 里程碑）
├─ misc.md                     事实库：源码地图、实测数据、环境、坑
├─ bench.md                    **性能唯一事实源**：四引擎 × 6 模型 + 瓶颈结论
├─ upstream.lock               上游 OpenCV dnn 跟踪版本锁（机器可读）
├─ OnnxSharp.slnx
├─ bench\                      四引擎对照基准（结果抄进 bench.md）
│  ├─ run_all.py               编排 4 引擎 × 6 模型，写 results/latest.json
│  ├─ run_python_bench.py      ORT / cv2.dnn 单独跑
│  ├─ models.json              模型清单
│  └─ make_micro_models.py     生成定点测量用微模型
├─ src\
│  └─ OnnxSharp\               产品库（零第三方依赖）
│     ├─ Tensors\
│     │  ├─ Tensor.cs          多维张量（紧凑布局 + Fit 复用 + Layout）
│     │  ├─ TensorShape.cs     MatShape 等价物
│     │  ├─ TensorOps.cs       广播 / 逐元素（三条无广播快路径）
│     │  ├─ GemmKernel.cs      GEMM 微内核 + 面板打包                      [M4]
│     │  ├─ TransposeKernel.cs 块向量转置                                  [M4]
│     │  ├─ VectorMath.cs      共用 AVX2 工具
│     │  ├─ Layout.cs          Layout 枚举（仅 NCHW，接口预留）
│     │  └─ DType.cs
│     ├─ Core\
│     │  ├─ Arg.cs             Arg / ArgData / ArgKind + 槽号
│     │  ├─ Graph.cs           prog 线性表 + Version/Reset/RemoveAll
│     │  ├─ LayerInfo.cs       LayerInfo + Layer；Net 反查 + In() + GetLayouts
│     │  ├─ LayerParams.cs / DictValue.cs / LayerRegistry.cs
│     │  ├─ Interpreter.cs     解释循环 + 动态形状分支 + temp 池
│     │  ├─ NetImpl.cs         Arg 表 + 槽/池 + Declare/Require
│     │  ├─ ConstFolder.cs     常量折叠 / Shape 链                        [M3]
│     │  ├─ GraphFusion.cs     图级融合 pass                              [M4]
│     │  └─ Net.cs             门面（ReadModel / Forward / ForwardAll）
│     ├─ Threading\
│     │  └─ ParallelRange.cs   parallel_for_ 等价物（小任务直接串行）
│     ├─ Engine\
│     │  └─ BufferAllocator.cs 生命周期区间着色，共享缓冲槽
│     ├─ Onnx\
│     │  ├─ Proto.cs           protobuf 消息视图（零拷贝，移植 SimdYolo）
│     │  ├─ OnnxFile.cs        模型结构 + TensorProto 解码（含外部数据）
│     │  └─ OnnxImporter.cs    图导入：initializer/Constant→Const Arg、属性
│     ├─ Ops\                  42 个 LayerInfo+Layer 实现（Dnn/Attn/Broadcast/Index/Misc/Shape/Nary/Activation）
│     ├─ Kernels\
│     │  ├─ ConvKernels.cs     直接卷积 + MaxPool
│     │  ├─ PackedConv.cs      面板打包卷积                               [M4]
│     │  └─ Winograd.cs        Winograd F(2×2,3×3)                        [M4]
│     └─ Vision\               可选门面：图进、框出（不参与图执行，可整体删除）
│        ├─ ImagePreprocessor.cs  Letterbox（等比 + 填 114）与 Stretch（直接拉伸）+ 双线性 + 归一化
│        ├─ YoloDetector.cs       YOLO 一站式封装：Net + letterbox + 解码 + NMS
│        ├─ YoloDecoder.cs        检测头解码（ChannelsFirst/ChannelsLast/EndToEnd）+ 反 letterbox
│        ├─ RfDetrDetector.cs     RF-DETR 一站式封装：Net + 拉伸 + 解码（dets/labels 双输出）
│        ├─ RfDetrDecoder.cs      逐类 sigmoid + 背景剔除 + Q×C top-k + 归一化框还原
│        └─ Detection.cs          Detection / IoU / 贪心 NMS
├─ tools\
│  ├─ OnnxSharp.Cli\           info / run / detect / ocr / bench / profile / dump-graph / gemmbench / convbench
│  │  ├─ ImageHost.cs          ImageFacade 引擎 + 显式注册 JPEG/PNG/BMP 解码器（格式包额外引入）
│  │  └─（画框 / 保存都走 ImageFacade：DetectionDrawing + FacadeImage.Save）
│  └─ scripts\                 dump_shapes.py / gen_golden.py / extract_ocr_models.ps1
└─ tests\
   └─ OnnxSharp.Tests\         零依赖测试运行器 + 59 用例
```

---

## 4. 里程碑

### M0 — 骨架与图基础设施（已完成）

**任务**
- [x] `OnnxSharp.slnx` + `src/OnnxSharp`（`net8.0;net10.0`，`AllowUnsafeBlocks=true`）
- [x] `DType`：F32/F16/I64/I32/I8/U8/Bool（首版只用 F32/I64/Bool/I32）
- [x] `TensorShape`：`Total()` / 比较 / `With()` / `Dims`
- [x] `Tensor`：`Shape`、`DType`、`Strides`、`Buffer`、`Fit()`（复用）、`Clone()`、`AsSpan<T>()`
- [x] `Arg/ArgData/ArgKind/Graph`
- [x] `LayerParams` + `DictValue`（int/float/string/数组/tensor）
- [x] `LayerInfo`（`GetMemoryShapes/GetTypes/DynamicOutputShapes/Subgraphs`）
- [x] `Layer`（`Forward/PrepackWeights/Finalize` + 形状签名重 Finalize）
- [x] `NetImpl`：Arg 表（`GetArg/NewArg/NewConstArg`）、`NewGraph`、`SetInput/Forward`
- [x] `Interpreter`：推断形状 → 绑定缓冲 → finalize → forward 的最小循环
- [x] `ParallelRange.For(int total, Action<int,int>)`（基础版）
- [x] 广播 + 逐元素（`TensorOps`：Add/Sub/Mul/Div/Max/Min、Relu/Sigmoid/Sqrt/Exp/Neg/Abs）

**验收**：`tools/OnnxSharp.Cli` 手工建图 `Input[2,3] → Add(const[3]) → Relu` 输出 `1.5, 0, 2, 4.5, 0, 5` —— **已通过**；`dotnet build` 0 警告 0 错误。

### M0.5 — 引擎能力补全（**已完成 2026-11-07**）

这一批是 M1/M2/M3 的硬前置。

**任务**
- [x] **常量输入反查**：`LayerInfo.Net` 回填（等价 `getNetImpl`）+ `In(int)` / `IsConstInput(int)` 便捷方法
- [x] `Graph.Prog` 可重建：`Graph.Reset(ops)` / `RemoveAll(pred)` / `Version`（结构变更递增，驱动缓冲槽重算）
- [x] `BufferAllocator`（生命周期区间着色）：`Engine/BufferAllocator.cs`，`Analyze` + `Assign`，槽只增不减
- [x] **布局接口预留**：`Tensor.Layout`、`ArgData.Layout`、`LayerInfo.GetLayouts`（恒返回 NCHW），无 `TransformLayout`
- [x] **AOT 友好约定**：`LayerRegistry` 为显式 `Dictionary<string, Func<...>>` 注册，全库无反射 / `dynamic` / `Emit`
- [x] 全局 temp 池（`NetImpl.FitTemps`，对齐 `globalTemps`），取消每次 forward `new`
- [x] `Interpreter` 动态形状分支：此类 op **不预 fit**，输出 Tensor 交给 layer 自己 fit
- [x] `Interpreter` 重 finalize 判据补 `inpTypes`
- [x] `NetImpl` 拆出 `Declare`（新建）/ `Require`（存在或报错）/ `TryGetArg`
- [x] 建 `tests/OnnxSharp.Tests`：**零第三方依赖**的控制台运行器（xunit 未缓存且离线，后续可迁移）+ 16 个用例
- [x] `Tensor.Fit` 溢出路径检查（错误信息带上字节数）
- [ ] `tests/OnnxSharp.GoldenGen` → 推迟到 M1（依赖 ORT + python 改图）

**验收（已通过）**
- `dotnet build OnnxSharp.slnx`：0 警告 0 错误（net8.0 + net10.0）
- `dotnet run --project tests/OnnxSharp.Tests`：**16/16 PASS**
  - 常量 shape 驱动输出形状（模拟 `Reshape`/`Slice`）
  - 动态 op 自 Fit 输出（`Where/Expand` 路径）
  - temp 张量对象跨 forward 复用（`TempCount == 1`）
  - 共享槽数跨 forward 稳定；图追加算子后槽自动重算
  - 分配器：链式 4 个 Arg 用 2 槽、并发分支不共享、图输出不被覆盖、常量/输入无槽
- CLI 冒烟：`SMOKE TEST: PASS`

### M1 — ONNX 加载（dump 与参考对齐）— **已完成 2026-11-07**

**任务**
- [x] protobuf reader：`Onnx/Proto.cs`（移植 SimdYolo `Proto`，长度分隔字段全部用 `ReadOnlyMemory<byte>` 视图，零拷贝）
- [x] 对象模型：`Onnx/OnnxFile.cs`（ModelProto/GraphProto/NodeProto/AttributeProto/TensorProto/ValueInfoProto/TypeProto 的按需读取）
- [x] `TensorProto` 解码：`raw_data` / `float_data` / `int32_data` / `int64_data` / 外部数据（含**路径逃逸校验**），DType 保留（I64/Bool/I8/U8/F16 不再退化成 float）
- [x] CLI `info <model>`：结构统计（真实模型已验证）
- [x] `OnnxImporter`：initializer→Const Arg、`Constant` 节点→Const Arg、value_info→shape/dtype、节点→Layer
- [x] `LayerParams` 属性解析：`FLOAT/INT/STRING/TENSOR/FLOATS/INTS/STRINGS`，`type` 字段缺失时按字段存在性推断
- [x] **yolo11n 所需的 15 类算子 handler**：Conv / MaxPool / Resize / Softmax / MatMul / Add / Sub / Mul / Div / Sigmoid / Concat / Split / Reshape / Transpose / Slice（其余 27 类留给 M3）
- [x] CLI `dump-graph <model> [--type X] [--limit N]`：纯形状推断，不执行 kernel

**实测（2026-11-07）**

| 模型 | 大小 | 解析耗时 | opset | 节点 | initializer | 输入 | 输出 |
|---|---|---|---|---|---|---|---|
| `yolo11n.onnx` | 10.2 MB | 22 ms | 19 | 320 | 196 (10.1 MB) | `images:F32[1,3,640,640]` | `output0:F32[1,84,8400]` |
| `rfdetr-small.onnx` | 115.4 MB | 91 ms | 17 | 1539 | 354 (108.7 MB) | `input:F32[1,3,512,512]` | **`dets:F32[1,300,4]` + `labels:F32[1,300,91]`** |

- **RF-DETR 是双输出**，`Forward()` 单输出重载对它不适用，调用方需用 `Forward(name)` 或 `ForwardAll()`。
- RF-DETR 的 459 个 `Constant` 节点里有 **379 个 I64 张量** + 80 个 F32 —— 印证「DType 不能丢」。
- 两模型输入都是**静态形状**（RF-DETR 为 512²，非动态）。
- yolo11n 导入后：320 op / **10 个共享缓冲槽** / 51 ms（含形状推断）。

**验收（已通过）**
- 节点数、op 类型直方图、输入输出名与 python onnx 一致；
- **333 个张量的 shape/dtype 与 `onnx.shape_inference.infer_shapes()` 完全一致**（`OnnxShapeTests`，一个不多一个不少）。

### M2 — YOLOv11 跑通（15 个算子）— **已完成（2026-11-07）**

**任务**
- [x] Conv：`Kernels/ConvKernels.cs` 直接卷积（group / stride / dilation / pad / auto_pad，按输出通道并行）
- [x] MaxPool 5×5 s1 pad2 ceil=0
- [x] Resize nearest / asymmetric / floor（`scales` 走第 3 输入 + `sizes` 分支 + bilinear）
- [x] Softmax（任意 axis，数值稳定）
- [x] 逐元素：Add/Sub/Mul/Div（含广播）+ Sigmoid
- [x] 形状类：Reshape/Slice/Split/Concat/Transpose
- [x] MatMul（N-D batched + broadcast，含 numpy 的 1D 补 1 语义）
- [x] 与 ORT 端到端比对（`gen_golden.py` + `YoloGoldenTests`）
- [ ] 与 ORT **逐层**比对（定位误差来源用，M2 收尾再补）
- [x] （非引擎）decode+NMS：**`src/OnnxSharp/Vision/` 门面**（`YoloDetector` / `YoloDecoder` / `ImagePreprocessor` / `Detection`），CLI `detect` 支持「图片进、框出」；与 SimdYolo 输出逐框一致（最大偏差 0.005 px）
- [x] （非引擎）RF-DETR 同款门面：`RfDetrDetector` / `RfDetrDecoder`（拉伸预处理 + 逐类 sigmoid + 背景剔除 + Q×C top-k），CLI `detect` 按输出个数自动判族；与官方 `decode_detections` 逐框一致（41/41 框，坐标偏差 0.00 px）

**实测（2026-11-07）**

```
yolo11n.onnx 端到端（1×3×640×640，单线程朴素卷积）
forward 5525 ms   max|diff|=1.862E-003   mean|diff|=3.293E-006
```

**验收**：`yolo11n.onnx` 输出 `[1,84,8400]` 与 ORT 的**最大绝对误差 < 1e-4**（当前 1.9e-3，见下）；
`yolov11s.onnx` 同标准；单张 640² 端到端 **< 200 ms**（朴素 SIMD 版）。

> **误差与性能说明**
> - 1.9e-3 的绝对误差出现在量级 ~267 的通道上（相对误差 ~7e-6），根源是**累加顺序不同**（直接卷积 vs ORT 的 im2col+GEMM 分块）。要看「数值是否正确」应看相对误差，绝对误差 1e-4 的目标对 F32 累加其实过紧 —— M4 换成分块 GEMM 后误差会自然靠近 ORT。
> - 5.5 s 是**标量直接卷积**的成绩，只用于验证正确性；M4 的 im2col+GEMM / Winograd 才是性能路径。

### M3 — RF-DETR 跑通（补 27 个算子）— **已完成（2026-11-07）**

**任务**
- [x] LayerNormalization（axis=-1，eps 逐节点：1e-6×37 / 1e-5×12）
- [x] Gemm（transB=1, alpha=beta=1）
- [x] GridSample（bilinear / zeros / align_corners=0）
- [x] ReduceSum/ReduceMax（keepdims 0/1，axes 来自 const 输入）
- [x] TopK（axis=1，k 来自 const 输入）
- [x] Gather / GatherElements
- [x] Erf / Exp / Sqrt / Sin / Cos / Relu / Not / Where
- [x] 比较：Equal / Less / Greater / And
- [x] Cast（FLOAT / BOOL / INT64）
- [x] Tile / Expand / Unsqueeze / Shape / ConstantOfShape
- [x] **`ConstFolder` + Shape 链**：折叠集合含 `Transpose`（见 §5.5），折叠后重建 `prog`
- [x] **内存达标**：1:1 方案的 1381 MB 由 `BufferAllocator`（生命周期着色）+ 常量失效释放压到可接受范围（`misc.md §3.1.1` 有 1:1 的基准数）

**验收**：`rfdetr-small/medium` 的 `dets` / `labels` 均已与 ORT golden 对齐，绝对误差在 1e-3 量级、相对误差 ~7e-6（与 yolo11n 同源：F32 累加顺序不同，不是实现错误）；测试见 `YoloGoldenTests`（29/29 PASS）。
单张 512² 实测 **275 ms（small）/ 349 ms（medium）**，优于 1500 ms 的朴素版目标，见 `bench.md`。

### M4 — 性能

**已完成（2026-11-07）**

- [x] GEMM：`GemmKernel.Multiply` 4 行×16 列 AVX2/FMA 微内核 + `kPanel` 分块 + B 面板打包（6 行版因 12 累加器溢出 YMM，实测慢 25%，已弃用）
- [x] Conv：**panel 打包（16 空间位置）代替整图 im2col**（`PackedConvKernels`，`SliceDepth=128` 保 L1）
- [x] Winograd F(2×2,3×3)：8-tile 向量化（`Avx2.PermuteVar8x32` / `Permute2x128` / `UnpackLow/High`）
- [x] Depthwise 专用 kernel、MaxPool 滑窗 AVX2
- [x] 逐元素/Softmax/LayerNorm 的 `Span<float>` + 无广播快速路径（`ApplyContiguous` / `ApplyScalar` / `ApplyBroadcast` 三条快路径 + `ParallelRange`）
- [x] 多线程：`ParallelRange` 按输出面板切分；**小任务（< 4M 乘加）直接串行**，避开 `Parallel.For` 的 ~7 µs 固定开销
- [x] 性能记录体系：`bench/`（四引擎 × 6 模型）+ `bench.md`（唯一事实源）+ `gemmbench` / `convbench`（按时间预热）+ `fma`（机器算力上限）+ `profile`（带预热、每算子最小耗时）+ `bench` 的 `alloc`/`gc`/`gc pause` 诊断行

**未完成 / 重新排序**

1. [x] **MatMul 合并 batch 维**：B 不广播时把 `(batch, m)` 拼成一次 GEMM。修掉 decoder q/k/v 投影「300 个 batch × m=1、权重被打包 300 遍」的浪费，单算子 4.06 → 0.13 ms，rfdetr-small -24%、medium -12%
2. [x] **逐元素融合（第二收益点）** —— 2026-11-07 完成：`Core/GraphFusion.FuseElementWiseChains` 把「中间结果只有 1 个消费者、不是图输出、链上同形」的逐元素链编译成 `Ops/FusedEltwiseLayer`（寄存器机 `EltwiseStep[]`，中间值留在 YMM），rfdetr-small 12 条 GELU 链、节点 857 → 801，端到端 **small -7.4% / medium -9.1%**（min，交替 A/B），4 模型输出 `max|diff| = 0`。
   - 关键教训：**通用解释型内核在 SIMD 上是反模式**。`Vector256[]` 用运行期下标访问会被 JIT 打回内存 + 每步一次 `switch`，实测比不融合还慢（1.65 vs 1.50 ms/层）。给 GELU 写手写内核后 0.30 ms/层。
   - 短链要设门槛：`chain.Count < 4 && 元素数 < 1<<18` 不融合（否则 yolo median 反而 +4%）。
   - 已贴带宽墙：融合后 42 GB/s，与拆分时一致，**这条线到头**。
3. [ ] **静态形状图缓存逐算子的形状/类型推断**：`Interpreter.Run` 每 forward 对 **2064 个算子**重算 `GetMemoryShapes`/`GetTypes`/`BindArgTensor`/`FitTemps`，实测每 forward 分配 **113 MB**。代价目前很小（GC 暂停 0.7 ms/iter、无降频），所以排在逐元素融合之后，但属于纯收益
4. [ ] **GEMM 多线程扩展性**：16 线程跑出机器峰值（753 GFLOPS）的 40%~46%：
   - [ ] 二维 cache blocking（m 块 × n 块都分块），让每个线程的 B 面板只读一遍
   - [ ] batched attention 在**一个** `ParallelRange` 区域里按 `(batch, 行块)` 展平调度（`257×64×257 ×24` 只有 156 GFLOPS，是最差一组）
   - [ ] `m·n·k < 4M` 的形状按 batch 并行（`300×32×300 ×8` 现在是串行，仅 60 GFLOPS）
5. [ ] **Transpose**：113 个（5.6%），已是 8×8 向量分块 + 同布局 memcpy 快路径，空间有限
6. [ ] **s 档大 GEMM**：对照 SimdYolo `CpuPackedGemm` / OpenCV `fastConv` 的权重 block 布局，调 `Panel / RowBlock / SliceDepth`
7. [ ] Conv+SiLU 融合、Winograd 覆盖 `5×5`（DETR 上 Conv 仅 3.4%，收益上限低）

> **已验证无效、不要再做**（详见 `bench.md` 结论 3）：
> - 把整个 B 预打包成 `[列块][k][16]` 供所有行块复用：304 → 234 GFLOPS（行块 40 行已把打包摊薄到 5%）
> - 微内核 k 方向 2 步展开：单线程 136 → 135 GFLOPS（JIT 已展开）
> - 行块高度 `mb` 扫参（16/32/64/128）：微基准 250~340 GFLOPS 波动，端到端无差异
> - 按流量改切分方向（行切→列切）：微基准 `257,1536,384` 343 → 499 GFLOPS，端到端交替 A/B 三轮完全相同

- 目标见 §9，实测见 `bench.md`

### M5 — 工程化

- [ ] 多线程可配置（线程数、关闭并行）
- [ ] 内存上限/诊断、`dump-graph`/`bench` 完善
- [ ] NuGet 打包
- [ ] **上游同步流程落地**（§13 / `misc.md §11`）：命令片段写进 README，`misc.md §11.6` 记录表保持更新
- [ ] **AOT 友好性自检**（不做验收）：全库无反射/Emit/`dynamic`；如需，可一条命令试 `PublishAot`
- [ ] （可选）net8 目标退役评估（EOL 后）

### M6 — 可选扩展

- [ ] `yolov11s-seg/pose/obb`（需 ConvTranspose、Sigmoid 后处理变体）
- [ ] FP16 / int8
- [ ] 图融合（conv+act、matmul 系列）

### M7 — OCR（PaddleOCR / PP-OCRv6）— **已完成（2026-11-10）**

引入方式：模型来自 `Sdcb.SimdPaddleOCR.Models.ChineseV6Tiny` / `TextLineOrientation`
（**只有 ONNX 流与字典、没有推理代码**），推理走本引擎 —— 与「零第三方依赖」不冲突。

**引擎侧**

- [x] **动态维特化**：`OnnxFile.WithInputShapes` + `Net.ReadModel(path, 输入形状)`（§5.6）
- [x] `HardSigmoid`（alpha/beta 属性两档）
- [x] `BatchNormalization`（推理模式，每通道一次仿射）
- [x] `Identity`（rec 图里 58 个）
- [x] 复用已有实现：`GlobalAveragePool` / `ReduceMean` / `ConvTranspose`(2×2 s2) / `AveragePool` / `Squeeze`
- [x] 修 `PackedConv.Multiply` 在**无 bias** 时的空指针（`Conv+BN` 结构）
- [x] 与 ORT 对齐：det（1×3×960×960）`max|diff|` **1.4e-7**；rec（1×3×48×320）**7.6e-6**

**门面侧（`src/OnnxSharp/Ocr/`）**

- [x] `Ocr/PaddleOCR/`（全部 OCR 代码都在这一层，加别的 OCR 引擎就再开一个 `Ocr/<Engine>/`）：
      `PaddleOcrModel`（文件/流/内存字节三种来源）、`PaddleOcrPipeline`、`DbPostprocess`、`PaddleOcrOptions`、
      `OcrLine` / `OcrPixelFormat`（结果与像素排列）、像素→NCHW（裁剪 + 双线性 + 归一化一次完成）、CTC 贪心解码
- [x] 检测按 32 对齐、识别按 16 分桶，按形状缓存特化图（det 2 份 / rec 8 份）
- [x] 可选方向分类（PP-LCNet，倒置概率 > 0.9 时旋转 180°）

**工具与验证**

- [x] CLI `ocr`：直接吃 JPG/PNG/BMP（ImageFacade + `Sdcb.SimdCodecs.Jpeg/Png/Bmp`），`--out` 画框存图，`--json` 出结果
- [x] `tools/scripts/extract_ocr_models.ps1`：从 nuget 模型包解出 det/rec/dict/cls
- [x] 回归：`OcrTests` 8 个用例（算子 / DB / CTC / 归一化 / 动态图特化 / 端到端链路）
- [x] 实测：900×300 合成图两行文字全对；`F:\ai\20260828093055568_121.jpg`（3200×1800 车牌，limit 1600）
      识别 `浙A ES8381` 0.883 / `浙A：:3M15X` 0.802，端到端约 0.85~1.2 s（含 det+rec 前向）

**明确不做**

- [ ] 语音识别（sherpa-onnx）：模型是 int8 + `Loop/NonZero/GatherND`，超出静态 FP32 图模型；README「库用法 6」给接法
- [ ] 表格识别 / 版面分析 / SAST 等其它 OCR 结构
- [ ] 多边形（pyclipper）检测框：当前是轴对齐框 + 面积/周长外扩

### M8 — SAM3（框提示）与共享图像预处理 — **已完成（2026-11-10）**

**引擎侧**

- [x] `IsNaN`（softmax 后的 NaN 保护，40 处）、`GreaterOrEqual`
- [x] `InstanceNormalization`（mask decoder 的 pixel_decoder）
- [x] `Einsum`：只实现 `bqc,bchw->bqhw`（mask embedder 的逐批矩阵乘），其它方程报错
- [x] `RoiAlign`：avg/max、`spatial_scale`、`half_pixel` / `output_half_pixel`、`sampling_ratio=0` 自适应
- [x] 修三个真 bug：`Clip` 的可选 min/max 占位被当成 0、`FusedEltwise` 放行广播形状、`Flatten` 的 `axis == rank`
- [x] 调试能力：`run --debug` 逐算子检 NaN（`Net.ForwardAll(detectNaN)`），`--input` 支持 I64/Bool

**FP16 → FP32 工具**

- [x] `tools/scripts/onnx_fp16_to_f32.py`：改 F16 initializer / `Constant`·`ConstantOfShape` 的 value、
      剥掉 FP16 边界的 `Cast`、修 `value_info` 类型、每个模型写**独立**外部权重文件

**门面侧（`src/OnnxSharp/Sam3/`）**

- [x] `Sam3Segmenter`：vision → geometry（归一化 cxcywh + label=1）→ decoder，按框数/提示长度缓存特化图
- [x] `Sam3Nms`：框级贪心 NMS（默认 IoU 0.5）+ `MaxItems`（实测把同一人的 79 条降到 5 条）
- [x] 掩码是 logits → sigmoid 成概率；源图坐标用 `MaskScaleX/Y`

**共享图像预处理（`src/OnnxSharp/Imaging/`）**

- [x] `ImagePixels`（4 种像素排列 + rowStride）、`ImageGeometry`（Contain / Stretch）、`AxisMap`（双线性 / 最近邻）
- [x] `ImageTensor.ToChw`：唯一的「裁剪 + 缩放 + 归一化 + 填边 + 可选 180°」实现
- [x] Vision（YOLO letterbox 114 / RF-DETR stretch）、OCR（文本行 48 高 + 右侧补 0）、SAM3（白边左上）全部改用它们
- [x] 回归：YOLO persons.jpg 的检测结果与重构前**逐位一致**（0.7537 [1080.5,732.2,127.9x325.6] …）

**实测**

- [x] decoder 与 ORT 逐值一致（`pred_logits -4.8116~2.2649`、`presence 9.2011`）
- [x] 1568×1068 + 行人框提示：top-1 **0.932**，框与提示框几乎重合；整链 **30 s**（vision 26 s 主导）
- [x] 测试：`ImagingTests` 6 例 + `Sam3Tests` 4 例；全套 66/69 PASS

**明确不做**

- [ ] 文本提示：需要纯托管 BPE tokenizer（`tokenizer.json`），当前只支持框提示
- [ ] vision 特征复用：现在每次 `Segment` 都重跑 vision（26 s），拆出 `EncodeImage` 可省 80%+ 时间
- [ ] int8 / FP16 计算：见 `misc.md §6.11`（Zen4 无 AVX512-FP16，FP16 计算是负收益）

### M9 — 纯托管 ASR（sherpa 的 Paraformer int8）— **计划中（2026-11-10 立项）**

> 目标：**不依赖 ONNX Runtime / sherpa 原生库**，用本引擎跑 `F:\ai\sherpa\model.int8.onnx`
> （Paraformer，中文），输入 16 kHz WAV → 输出文字，与 sherpa-onnx **逐字一致**。

**模型实测（立项依据）**

| 项 | 值 |
|---|---|
| 图 | opset 14，**7881 节点**，1197 权重（753 F32 + **444 U8**） |
| 输入 | `speech` F32 `[B, T, 560]`（**80 维 fbank × LFR 7**）、`speech_lengths` I32 `[B]` |
| 输出 | `logits` F32 `[B, T', 8359]`、`token_num` I32 |
| int8 | `MatMulInteger` ×222 + `DynamicQuantizeLinear` ×211 |
| 控制流 | `Loop` ×2（CIF 80 节点 / 取词 106 节点，含 `If`/`NonZero`/`ScatterND`/`SequenceInsert`）、`SequenceEmpty`/`ConcatFromSequence` ×2 |
| 其它缺口 | `Pad`(53，全 constant)、`LogSoftmax`(1)；`GatherND`/`Range`/`Pow`/`Sqrt`/`ReduceMean`/`GreaterOrEqual` 已有 |

**路线选择：B′（推荐）—— int8 保留 + 只切控制流（CIF 手写）**

> 为什么不反量化：ASR 是「短序列 + 大权重」的负载，**权重带宽**是第一约束。实测该模型
> int8 权重 82 MB、反量化后 260 MB（3.2×）；再加上 Zen4 的 VNNI（`vpdpbusd` 一条指令 64 个 int8 MAC，
> 是 AVX2 FP32 FMA 的 8 倍、AVX-512 FP32 的 4 倍），int8 是这条路线的**性能前提**，不能丢。

| 路线 | 做法 | 工作量 | 结论 |
|---|---|---|---|
| **B′** | **保留 int8 图**：引擎实现 `DynamicQuantizeLinear` + `MatMulInteger`（VNNI 内核，AVX2 回退）；只把两个 `Loop` 那一段（CIF/取词）切出来**用 C# 重写** | **3~4 周** | ✅ 推荐：既留 int8 的性能，又避开 `Loop/If/Sequence` |
| A | 离线反量化成 F32 再跑 | 2~3 周 | 链路最容易验证，但性能只有 sherpa 的 1/3~1/10；**只作为 B′ 的第一步（跑通用），不作为终态** |
| C | 引擎实现通用 `Loop`/`If`/`Sequence`，图原样跑 | 4~6 周，易错 | ❌ 为一个模型写控制流执行器，收益低 |

> **实施顺序**：先用 A 的最小子集（M9-03'）把「encoder → CIF → decoder → 文本」整链跑通（正确性基线），
> 再原地换上 B′ 的 int8 内核（同样输入、同样输出对拍）——两者产物完全等价，可以 A/B 切换，
> 这样「性能」和「正确性」两个风险不会绑在一起。


**任务与验收**

| ID | 任务 | 估时 | 验收 |
|---|---|---|---|
| M9-01 | **基线取证**：ORT 跑 int8 图 + sherpa 跑同一 wav，落 `tests/Fixtures/asr/` | 0.5 天 | **部分完成**：`tools/scripts/sherpa_encoder_probe.py` 已能录 encoder 段的输入/金标准；正式 fixture（含 fbank、logits、文本、sha256）仍缺 |
| M9-02 | **fbank 前端（纯 C#）**：25 ms/10 ms、povey 窗、preemph 0.97、nfft 512、mel 80、20~8000 Hz、log、**LFR(7,6)**、CMVN | 2~4 天 | 与 kaldi-native-fbank 逐值 `max|diff| < 1e-5`；`speech` 与 ORT 输入对齐 |
| M9-03 | **（先跑通用）int8 → F32 图改写**（`tools/scripts/onnx_int8_to_f32.py`）：U8 权重按 `(W-zp)*scale` 还原、`MatMulInteger→MatMul`、清掉配套 `DynamicQuantizeLinear` | 1~2 天 | 改写后 ORT 输出与原模型一致（`max|diff| < 1e-3`）；**只用于打通链路与做基线，不作终态** |
| M9-04 | **补小算子**：`Pad`(constant) ✅、`LogSoftmax`（未做） | 1~2 天 | **完成一半**：`Pad` 三种 mode + 负 pads 已实现并有单测；顺带补了 1-D `Conv`（FSMN）与 `Range`/`Reshape`/`ConstantOfShape` 的**运行期形状** |
| M9-05 | **图切分**：`tools/scripts/split_before_control_flow.py` —— 在第一个 `Loop`/`If`/`Scan` 前切开，自动拓扑排序（sherpa 的导出不是拓扑序），跳过 `Sequence*` 边界值，可显式追加输出张量 | 0.5~1 天 | ✅ `model.int8.onnx` 7881 → 5715 节点，encoder 段被引擎直接加载并跑通 |
| M9-06 | **CIF 用 C# 重写**：`cif_conv1d` + integrate-and-excite（alpha 累加、阈值 1.0、取整、remainder）→ acoustic embeddings | 2~4 天 | token 数与 ORT 相同；embeddings `max|diff| < 1e-4` |
| M9-07 | **decoder 跑通**：`logits_length = token_num` 特化（必要时按 8 的倍数分桶） | 1~2 天 | logits 与 ORT `max|diff| < 1e-3` |
| M9-08 | **token 解码**：argmax + `tokens.txt` 去词/去重 + 标点 | 1 天 | 文本与 sherpa 逐字一致 |
| M9-09 | **门面 + CLI**：`src/OnnxSharp/Speech/ParaformerRecognizer`（`WavReader` + 上述链路），CLI `asr <wav> --models <dir> [--json]` | 2 天 | 一条命令出文本；`tts-zh.wav` 端到端 < 3 s（FP32/16 线程） |
| M9-10 | **回归 + 文档**：fixture 对拍用例（缺模型 SKIP）、README「库用法 9」、坑清单 | 1~2 天 | `dotnet run --project tests` 全绿；文档含复现命令 |
| M9-11 | **int8 GEMM 内核**：U8 权重预打包 + `vpdpbusd`(AVX512-VNNI) 主路径 + `vpmaddubsw`/`vpmaddwd`(AVX2) 回退；A 的 zero-point 用「行和 × zp」校正，避免逐元素减 | 4~6 天 | 同形状单算子吞吐 ≥ **3×** 现有 FP32 内核；与 ORT `MatMulInteger` 输出 `max|diff| ≤ 1`（int32 累加应逐位一致） |
| M9-12 | **两个 int8 算子 + 融合**：`DynamicQuantizeLinear`（per-tensor min/max → scale/zp → u8）、`MatMulInteger`（u8×u8 + zp → i32）；再把 `DQL → MatMulInteger` 融合成一层（省一次 u8 写回和一遍 min/max 扫描） | 3~4 天 | encoder 段与 ORT int8 逐值对齐（`max|diff| < 1e-3`）；融合后不再物化 u8 激活 |
| M9-13 | **性能门槛与三边 A/B**：同一 wav 跑「FP32 通道 / int8 通道 / sherpa」 | 1~2 天 | **VNNI 机器**：≤ 3× sherpa（ORT 的 MLAS 用到 512 位 VNNI，我们只有 256 位）；**无 VNNI**：int8 通道 ≤ FP32 通道 × 0.7；达不到先调 packing 粒度与并行 |

**风险与对策**

| # | 风险 | 对策 |
|---|---|---|
| R1 | 反量化折算错（per-channel scale、zero-point） | M9-03 逐张量与 ORT 对拍，先比最大/均值和逐元素 |
| R2 | CIF 细节（alpha 累加、remainder、ceil、阈值）难对齐 | 把 Loop body 的 80 个节点逐条翻译；先对齐 token 数再对齐 embeddings |
| R3 | decoder 动态长度 | 特化 `logits_length`（必要时按桶缓存，复用 OCR 的「按形状缓存特化图」经验） |
| R4 | 前端 fbank 与 sherpa 差 1e-5 级以上 | 先对齐分帧/窗/preemph/CMVN 四个中间量，再对最终特征 |
| R5 | 性能只有 sherpa 的 1/3~1/10（FP32 通道） | 这是预期结果，**所以终态走 int8**（M9-11/12/13）；FP32 只用于打通链路 |
| R6 | int8 内核正确性（zero-point、per-tensor scale、饱和） | 先写标量参考 + 单算子对拍，再上 SIMD；ORT 的 `MatMulInteger` 当 oracle（int32 累加应逐位一致） |
| R7 | AVX2 回退路径慢（无 VNNI 的机器） | 只承诺 VNNI 机器快；AVX2 走 `vpmaddubsw`，目标 ≥1.5× FP32 |
| R8 | `DQL→MatMulInteger` 融合 pass 破坏图 | 仅在「DQL 只有一个消费者，且 scale/zp 只服务该 MatMulInteger」时融合，并留开关可关 |

**总验收**：`asr` 对 3 段中文音频（含 `tts-zh.wav`）输出与 sherpa 逐字一致；
端到端 ≤ **2× sherpa**（同一台机器、16 线程），且 FP32 通道与 int8 通道输出文本一致。

### M10 — 原生 int8 量化推理（引擎能力）— **计划中（2026-11-10 立项）**

> 目标：引擎**原生**执行量化 ONNX（不需要离线反量化），把 int8 变成和 FP32 并列的一等公民。
> 第一个消费者是 M9 的 Paraformer（`MatMulInteger` ×222 + `DynamicQuantizeLinear` ×211），
> 但能力本身是通用的：量化 YOLO/MobileNet、QDQ 导出、QLinearConv 都应该能直接跑。

**为什么要做（实测依据）**

| 事实 | 数字 |
|---|---|
| 量化模型的权重带宽 | int8 **82 MB** vs 反量化后 **260 MB**（ASR 是权重带宽受限的负载） |
| Zen4 指令吞吐 | `vpdpbusd` 一条 **64 个 int8 MAC**；AVX2 FP32 FMA 8 个、AVX-512 FP32 16 个 |
| 现状 | 引擎只有 F32/F16(存)/I32/I64/Bool 的**搬运**能力，没有 int8 计算；量化模型导入时直接报 `MatMulInteger` 未实现 |

**范围（分两批）**

| 批次 | 内容 | 说明 |
|---|---|---|
| 第一批（本里程碑） | `MatMulInteger`、`DynamicQuantizeLinear`、`QLinearMatMul`、`ConvInteger`、`QLinearConv`、`QuantizeLinear`、`DequantizeLinear`（支持 per-tensor / per-axis） | 覆盖 ORT 旧式量化（QOperator）与 QDQ 两种主流导出 |
| 第二批（后续） | `QLinearAdd`/`QLinearMul`/`QLinearGlobalAveragePool`/`QLinearSigmoid`…、QDQ 折叠成 FP32 的「精度等价」通道 | 按实际模型需求加，不预先实现 |

**任务与验收**

| ID | 任务 | 估时 | 验收 |
|---|---|---|---|
| M10-01 | **int8 类型与工具**：`Tensor` 上 `Span<sbyte>/Span<byte>/Span<int>` 访问、饱和转换（f32↔u8 带 zp/scale）、量化参数常量读取（`OpUtil.QuantParams`） | 0.5~1 天 | 单元测试：饱和边界、zp 偏移、per-axis 取值 |
| M10-02 | **int8 GEMM 主路径（VNNI）**：权重 U8/S8 预打包（列块）、`vpdpbusd`（`AvxVnni`，256 位；512 位向量拆两次）累加、A 的 zero-point 用「行和 × zp」校正、epilogue per-column 反量化 | 3~4 天 | 与 ORT `MatMulInteger` **int32 逐位一致**；VNNI 机器吞吐 ≥ **2×** 现有 FP32 GEMM |
| M10-02b | **（已实测，结论：不可行）int8 权重 + F32 计算**：权重保持 u8，按列块 `k×64` 用 `vpmovzxbw` 解包成 F32 再调 GEMM。**实测 ASR 形状反而慢 5~10×**（`96×320×1280`：10.88 ms vs FP32 1.07 ms），原因是逐列块重复打包 + 小 GEMM 固定开销，而不是解包本身 | 2~3 天（已完成，结论否定） | ~~≥1.5×~~ 实测 0.10~0.19×；代码保留在 `Kernels/Int8DequantGemm.cs`，等 M10-02c 的「融合解包进微内核」再评估 |
| M10-02c | **小 GEMM 效率（新发现的主线）**：ASR 形状下 FP32 GEMM 只有 **73 GFLOPS**（机器峰值 400+），瓶颈是 row-block 打包与 `Parallel.For` 固定开销。按形状自适应（m≤128 时用细粒度列切分 / 合并 batch 维 / 免打包路径） | 2~3 天 | `96×320×1280` ≥ **150 GFLOPS**；ASR encoder 端到端 ≥1.5× |

> **AVX-512 子集取舍（2026-11-10 本机实测：F/BW/CD/DQ/VBMI 全 True，VNNI False）**
>
> | 子集 | 给什么 | 我们要不要 | 用在哪 |
> |---|---|---|---|
> | `Avx512F` | 512 位地基、`vpbroadcastd`（广播激活 dword）、掩码、`vpermd` | **要** | 所有 512 位内核的地基 |
> | `Avx512BW` | `vpmaddubsw` / `vpmaddwd`（`MultiplyAddAdjacent`，实测 3×5×4=60）、`vpmovzxbw`（`ConvertToVector512Int16`）、字节掩码 | **要（核心）** | int8 GEMM 的全部原语 + u8→i16→f32 解包 + 尾巴掩码 |
> | `Avx512Vbmi` | `vpermb` / `vpermi2b`（`PermuteVar64x8`，实测可用） | 可选 | int8 权重重排、NHWC↔NCHW 字节 deinterleave（都是加载期一次性） |
> | `Avx512DQ` | `vpmullq` 等 64 位整数运算 | 暂不需要 | 只对 int64 索引算子的边角有用 |
> | `Avx512CD` | `vpconflictd`（`DetectConflicts`） | 暂不需要 | `ScatterND`/去重/NMS 这类索引算子；与 GEMM 无关 |
> | ~~`Avx512Vnni`~~ | `vpdpbusd`（1 条指令 4× 吞吐） | **.NET 没有** | 只有在别处用汇编/native 时才拿得到 |
| M10-03 | **`MatMulInteger`** 算子（u8×u8 / s8×s8、`a_zero_point` / `b_zero_point` 标量或 per-column） | 1 天 | ✅ 已实现（`Ops/QuantOps.cs`）；对拍 ORT `max|diff| = 0`（u8 与 s8+per-column 两组微模型） |
| M10-04 | **`QLinearMatMul` / `ConvInteger` / `QLinearConv`**：int8 卷积走 im2col + int8 GEMM，int32 bias，输出按 `y_scale/y_zero_point` 重量化 | 2~3 天 | 对拍 ORT，`max|diff| ≤ 1`（u8 输出） |
| M10-05 | **`QuantizeLinear` / `DequantizeLinear`**（opset 10/13/19 的 axis 语义） | 1~2 天 | 与 ORT 逐值一致；per-axis 覆盖 |
| M10-06 | **`DynamicQuantizeLinear`**（per-tensor min/max → scale/zp） | 0.5 天 | ✅ 已实现；与 ORT 的 scale/zp/量化值逐位一致 |
| M10-07 | **图级优化**：① `DQL → MatMulInteger` 融合成一层（省一次 u8 物化 + 一遍 min/max 扫描）；② QDQ 对（`QuantizeLinear→DequantizeLinear` 相邻且无中间消费者）直接消掉；③ 量化参数常量预提取 | 2~3 天 | 融合后 ASR encoder 少 ~200 次全量扫描；数值不变 |
| M10-08 | **对拍与回归**：单算子 oracle（ORT）+ 端到端量化模型（ASR Paraformer、一个量化 YOLO） | 2~3 天 | **部分完成**：`quant: quantized graph matches onnxruntime` 覆盖 DQL+MatMulInteger+Cast+Mul（bit-exact）；ASR encoder 段有 `probe_tensors.py` 逐层对拍（中间层 1e-6~3.6e-5，主输出 mean 8e-4）；量化 YOLO 未测 |
| M10-09 | **性能基准**：CLI `int8bench`（GEMM/Conv 形状扫参）+ `bench.md` 增补 | 1~2 天 | 表格含 VNNI/AVX2 两档，标注 CPU 能力 |
| M10-10 | **文档**：README「原生 int8」一节、`misc.md` 踩坑（zp 校正、饱和、per-axis） | 1 天 | 含复现命令与精度/性能对照 |

**合计：约 3~4 周**（M10-02/02b 是主项；M10-03~06 可以并行）。

**内核分档（运行时探测，`Int8KernelTier`）**

> ⚠️ **实测约束（2026-11-10，本机 Ryzen 7 8845HS / Zen4，.NET 10.0.101）**：
> ① `System.Runtime.Intrinsics.X86` **没有 `Avx512Vnni`**（写一行就 `CS0103`；ref 程序集字节里也搜不到）；
> ② `AvxVnni`（Intel 专属 AVX-VNNI，256 位）在本机 **`False`**，但它对应的 512 位 `Avx512BW.MultiplyAddAdjacent`
> **可用**（实测 `3×5×4 = 60` 语义正确）。
>
> **指令级结论**（每 32/64 个 MAC 需要的指令数）：
>
> | 路径 | 指令 | 每指令 MAC | vs FP32 |
> |---|---|---|---|
> | `vpdpbusd`（VNNI，Intel 12 代+） | 1 | 32 / 64 | **4×** |
> | `vpmaddubsw`+`vpmaddwd` 单趟 | 2 | 32 / 64 | 2×，但 **i16 会饱和**（255×128×2 &gt; 32767） |
> | 同上 + 7 位拆分（安全） | ~8 | 32 / 64 | **0.5×**（比 FP32 慢） |
> | 标量 | 1/元素 | ~1 | 0.1× |
>
> 因此：**能安全跑的 int8 内核在 Zen4 / 老 Xeon 上拿不到加速**；
> 主路径改成 **M10-02b「int8 权重存储 + F32 计算」**（任何 CPU 都有带宽收益），
> VNNI 路径（`vpdpbusd`）留给 Intel 12 代+。

| 档位 | 触发条件（.NET 8/10 都有对应 intrinsic） | 实现 | 目标倍率（vs 我们的 FP32 GEMM） | 典型 CPU |
|---|---|---|---|---|
| `Vnni256` | `AvxVnni.IsSupported`（**仅 Intel 12 代+**） | `vpdpbusd` ymm | **2~3×**（≈2× vs AVX-512 FP32、≈4× vs AVX2 FP32） | Alder Lake / Raptor Lake / Meteor Lake 等客户端 |
| `Avx512Bw` | `Avx512BW.IsSupported`（有 512 位但无 VNNI） | `vpmaddubsw`+`vpmaddwd` zmm（**7 位拆分，防饱和**） | **2~4×** | Skylake-SP / Cascade Lake / **Zen4 / Zen5** |
| `Avx2` | `Avx2.IsSupported` | `vpmaddubsw`+`vpmaddwd` ymm（同上） | **1.5~2×** | Haswell~Coffee Lake、Zen1~Zen3 |
| `Neon` | ARM64 `AdvSimd` | `vmull`/`vpaddl`（无 `sdot` 时等价实现） | 1.5~2× | Apple M / 骁龙 / 鲲鹏 |
| `Scalar` | 其它 | 纯标量（同时作为测试 oracle） | ~1×（**不劣化**：权重小 3×，带宽仍是收益） | 任意 |

> **没有 VNNI 不等于白做**：ASR/LLM 这类负载首先是权重带宽受限，int8 权重 82 MB vs FP32 260 MB，
> 即使算力只有 1.5×，端到端通常仍能拿到 1.5~2.5×；只有在「算力受限 + 无任何 int8 SIMD」的机器上才可能打平。
> 这类机器可以直接切回 M9-03 的 FP32 反量化通道（保留开关）。

**关键设计决定**

1. **只量化「量化算子」**：int8 计算只发生在 `MatMulInteger`/`QLinearConv` 等内部，其余算子继续 F32/I32 ——
   不做全局「整图 int8」执行器（那是另一套图编译器的工作量）。
2. **zero-point 走权重预减**：`sum((a-za)(b-zb)) = sum(a·b') - za·sum(b')`，`b' = b - zb` 在打包时一次算好；
   这样主循环只剩 `vpdpbusd` + int32 累加，A 侧不需要逐元素减。
3. **epilogue 反量化**：int32 累加 → `* (a_scale*b_scale)` → `+ bias` → 可选重量化，全部在寄存器块内完成，
   不物化中间 F32 大张量。
4. **打包缓存**：与现有 `PackedConv` 一样按权重张量实例缓存，权重只打包一次（量化模型权重常驻内存也小）。
5. **AVX2 回退必须正确**：`vpmaddubsw` 会饱和（15 位），需要成对处理 + `vpmaddwd` 扩展到 int32；
   这是最容易写错的地方，标量参考实现要保留并参与测试。

**风险**

| # | 风险 | 对策 |
|---|---|---|
| R1 | `vpdpbusd` 的 u8×s8 语义与权重符号（权重保留 s8 还是 u8）搞错 | 统一成「A=u8、B=s8（已减 zp）」；标量参考 + ORT 逐位对拍 |
| R2 | `vpmaddubsw` 饱和导致 AVX2 路径结果错 | 成对操作（`maddubs` 后立即 `maddwd`），并用随机 ±127 边界数据测 |
| R3 | per-axis scale 的 epilogue 写错（列序 vs 行序） | 单独用例：per-column 与 per-tensor 结果在 scale 全相同时必须一致 |
| R4 | 融合 pass 改坏图 | 融合有开关；融合前后 ASR/YOLO 输出逐值对比 |
| R5 | 无 VNNI 的机器收益低（老 Xeon / AVX2 平台） | 分档内核（`Avx512Bw` / `Avx2` / `Neon` / 标量）+ 权重小 3× 的带宽红利；仍不达标就切回 FP32 反量化通道 |
| R6 | 能力探测与实现不一致（探测到但路径写错）导致 `IllegalInstruction` | 每档都用**同一组随机数据**与标量参考逐位比对；探测结果在 `int8bench --info` 打印并进测试断言 |
| R7 | ARM 无 `sdot` 时只用 `vmull/vpaddl`，收益有限 | 先用等价实现保正确；`AdvSimd` 的 dot-product intrinsic 可用时再切快路径 |

**对 M9 的影响**：M9-11/12/13 直接复用 M10-01~07，只保留「ASR 特有」的 CIF/前端/解码任务。
建议顺序：**先做 M10-01/02/03/06（一周）**，M9 的 encoder 段就能跑；再补 M10-04/05/07 收尾通用性。

---

## 5. 核心设计

### 5.1 Tensor

```csharp
public sealed class Tensor
{
    public TensorShape Shape;
    public DType DType;
    public Memory<byte> Storage;   // 池化内存（可能被多个 Arg 复用）
    public int[]? Strides;         // null => 紧凑
    public int ByteOffset;
    public bool IsContiguous => Strides is null;
}
```

要点：
- **只做"紧凑 + 别名"两种形态**（首版不做任意 stride 慢路径）。
- `Fit(shape, dtype)`：池化内存复用，与 OpenCV `Mat::fit` 语义一致（尺寸够就复用，不够才扩）。
- **别名安全规则（本轮明确）**：`Reshape`/`Shape` 这类零拷贝别名，必须让 `BufferAllocator` 把源 Arg 与目标 Arg 绑定到**同一 buffer**，且目标 Arg 的生命周期包含在源 Arg 之内；否则退化为拷贝。禁止两个生命周期重叠的 Arg 共享同一 buffer（坑清单 #16）。
- 不支持"通道分块布局"，首版 NCHW 直通。

### 5.2 Arg / ArgData

```csharp
public enum ArgKind { Empty = 0, Const = 1, Input = 2, Output = 3, Temp = 4 }

public struct Arg { public int Idx; }          // 0 = 空

public sealed class ArgData
{
    public string Name = "";
    public ArgKind Kind;
    public TensorShape Shape;
    public DType Type;
    public Tensor? Const;      // Kind==Const 时有效
    public int BufferIndex = -1; // 指向 BufferAllocator
}
```

`NetImpl.Args` 是 `List<ArgData>`，`Arg.Idx` 即下标；`ArgNames: Dictionary<string,int>` 负责符号名 → Arg。

### 5.3 LayerInfo / Layer（本轮关键修订）

```csharp
public class LayerInfo
{
    public string Name = "", Type = "";
    public List<Tensor> Blobs = new();      // 权重
    public List<Arg> Inputs = new(), Outputs = new();

    // ★ 新增：等价 OpenCV getNetImpl(this)——Shape 类算子必须能读常量输入的值
    internal NetImpl? Net;

    public virtual void GetMemoryShapes(
        IReadOnlyList<TensorShape> inShapes, int requiredOutputs,
        List<TensorShape> outShapes, List<TensorShape> tempShapes);

    public virtual void GetTypes(...);
    public virtual bool DynamicOutputShapes() => false;
    public virtual IReadOnlyList<Graph>? Subgraphs() => null;
}
```

**为什么必须有 `Net` 反查**：OpenCV 的 `Slice2Layer::getMemoryShapes` 是这样拿参数的：

```cpp
// modules/dnn/src/layers/slice2_layer.cpp:363-376
Net::Impl* netimpl_ = getNetImpl(this);
Mat startsTensor = netimpl_->argTensor(this->inputs[1]);
Mat endsTensor = netimpl_->argTensor(this->inputs[2]);
```

而 `constArgs()` 只把 Conv/Deconv/BN/PReLU 的常量输入搬进 `Blobs`（`graph_const_args.cpp:74-100`），**不会**帮 `Reshape/Slice/Split/Expand/TopK/Reduce/ConstantOfShape` 搬。因此：

- 方案 A（推荐，贴近 OpenCV）：`LayerInfo` 持 `NetImpl`，提供 `Tensor In(int i)` 便捷方法；`GetMemoryShapes` 内部对 `Inputs[1..]` 用 `isConstArg` 判断后读常量值。
- 方案 B（更省事）：`GetMemoryShapes` 形参改为 `IReadOnlyList<Tensor>`（形状从 `Tensor.Shape` 取）。代价是与 OpenCV 签名不完全一致，但纯 CPU 无副作用（CUDA 时 OpenCV 才需要 header-only 限制）。

首版让 `Layer : LayerInfo`，一个对象兼任两者（与 OpenCV 中 CPU 路径一致，省一层映射）；`LayerFactory` 负责 `Type → 构造函数` 注册。注意 OpenCV 把 packed weight 放在 exec 上并用 `weightEpoch` 失效（`net_impl2.cpp:1905-1910`）——合并后 `PrepackWeights` 需要自己的失效信号。

### 5.4 推理流水线

```
ReadModel(path)
  → OnnxImporter → Graph（不执行、不做 shape 推断）

SetInput(name, tensor)

Forward(outputNames)
  → Prepare():   ConstFold() → ConstArgs() → 重建 prog        // 一次性
  → Finalize():  BindExec() → AssignBuffers()                 // 一次性
  → Interpret(mainGraph):
        for op in prog:
            inTensors = op.Inputs.Select(argTensor)           // 含 Const 值
            if !op.DynamicOutputShapes():
                op.GetMemoryShapes(inShapes, ...) → outShapes/tempShapes
                op.GetTypes(...) → outTypes
                Fit 输出到（Allocator 分配的）缓冲；temp 取全局 temp 池
            else:
                outTensors = op.Outputs.Select(argTensor)      // ★ 不 fit、不检查 Shape
                temps = 全局 temp 池视图
            if (!layer.FinalizedOnce || 输入 shape/type 签名变化) layer.Finalize(...)
            if !op.DynamicOutputShapes(): layer.Forward(in, out, temps)
            else:                          layer.Forward(in, out, temps)  // layer 内部自行 out.Fit()
```

**动态形状的确切语义（对齐 `net_impl2.cpp:1891-1898`）**：

```cpp
// modules/dnn/src/net_impl2.cpp:1891-1898
} else {
    outMats.resize(noutputs);
    for (i = 0; i < noutputs; i++) {
        Arg out = outputs[i];
        outMats[i] = argTensor(out);
    }
    tempMats = scratchBufs;
}
```

即：动态 op 的 `outMats` 就是 `ArgData.Runtime` 本身（可能还是空的），**由 layer 在自己的 `Forward` 里先 `fit()` 再写**。C# 侧因为 `Tensor` 是引用类型，天然可行；但 `Interpreter` 必须显式分支，不能走 `BindArgTensor`。

> 前提：`ConstFolder` 做得够全时，动态 op 会只剩极少数（见 §5.5）。

### 5.5 常量折叠与 Shape 链

RF-DETR 有 459 个 `Constant`、25 个 `Shape`、122 个 `Reshape`。`ConstFolder` 必须能把 **纯 shape 子图** 折叠成常量：

- 参与折叠的算子：`Shape / Gather / Slice / Concat / Unsqueeze / Squeeze / Cast / Reshape / Expand / ConstantOfShape / Range / Mul / Add / Div / Sub / Equal / Less / Greater / And / Not / Where / ReduceSum / ReduceMax`，**并新增 `Transpose`**（且所有输入均为 const）。
- 实测（本轮重跑依赖追踪）：
  - `/transformer/Where` ← `Reshape`（数据相关）
  - `/transformer/Where_1` ← `Transpose`（数据相关，**把 `Transpose` 纳入折叠集合即可消灭**）
  - 其余 `Where`/`Expand`/`ConstantOfShape` 上游为 `Equal/Constant/Cast/Slice/Unsqueeze`，全部可折叠
- **结论**：实际数据相关 `Where` 是 **2 处**（原文档写 1 处）；纳入 `Transpose` 后预期降到 1 处。
- 残留部分保留 `DynamicOutputShapes = true` 作为兜底。
- 折叠时机：`Forward` 前一次性；折叠后 **必须重建 `prog`**（删除被折叠节点 + 更新 graph 偏移），并释放失效的常量张量（`useCount → 0` 时释放，对齐 `graph_const_args.cpp:29-35`）。

### 5.6 动态形状（只做最小可用）

- 标记为动态的 3 个算子：`Where`、`Expand`、`ConstantOfShape`。
- 处理方式：**layer 自己 `fit` 输出**（§5.4）。
- 不做通用符号维度、不做动态 batch。
- **输入维特化（2026-11-10 新增）**：ONNX 里 `dim_param` / 缺失 `dim_value` 都解析成 -1。
  图输入带 -1 时，`Shape → Slice → Concat → Reshape` 链在导入期折不出常量，`Reshape/Resize` 的目标形状
  永远是「运行时才知道」，整图卡死。因此提供
  `Net.ReadModel(path, 输入形状)` / `Net.ReadModel(path, Dictionary<string,int[]>)`
  → `OnnxFile.WithInputShapes(...)`：把输入维改写成具体值（**其余节点与权重共享，不重新解析**），
  再跑常量折叠与形状传播。语义是「为这个尺寸特化一份图」——一个 `(H,W)` 一份 `Net`，
  调用方按形状缓存。PaddleOCR 的 det / rec 就是靠它跑起来的（`Ocr/PaddleOCR/PaddleOcrPipeline.cs`）。

### 5.7 内存池（本轮上调优先级）

- 实测（onnx 形状推断累加全部节点输出，即 1:1 方案的常驻量）：

| 模型 | 1:1 激活合计 | 最大单张量 |
|---|---|---|
| yolo11n | **273.6 MB** | 6.25 MB |
| rfdetr-small | **1381.0 MB** | 24.19 MB (`[1,6,1028,1028]`) |

- 因此 `BufferAllocator` **不是 M4 优化，是 M0.5 前置**：
  - `useCount` + lifetime 区间（首个写 → 最后读）→ 冲突图着色 → `ArgBufferIndex[argIdx]`。
  - 同一 buffer 的 Arg 只需 `Fit` 到最大需求，不缩容（与 OpenCV 一致）。
  - 常量 Arg 在 `useCount==0` 时释放。
- **全局 temp 池**：`List<Tensor> GlobalTemps`，按 op 需求 `Fit` 复用（对齐 `allocateLayerOutputs` 的 `globalTemps`）。
- `inputs`/`outputs` 由 NetImpl 持有，跨 forward 保留；`Reset()` 释放。
- 目标：RF-DETR 峰值 < 400 MB。

### 5.8 线程

```csharp
public static class ParallelRange
{
    public static void For(int total, Action<int,int> body); // [begin,end)
}
```
- 默认 `Environment.ProcessorCount`，可按任务数切分；小任务（total<阈值）直接串行。
- 语义对齐 `cv::parallel_for_`：调用方（引擎）拥有并行度，kernel 不自行开线程。
- **实测约束（2026-11-07）**：`Parallel.For` 固定开销约 **7 µs**，阈值取 `GEMM ≥ 4M 乘加` / `逐元素 ≥ 64K 元素`。
- **已知缺陷**：batched op 是按 batch 串行调用 `Multiply` 的，24 个 batch 就是 24 个连续并行屏障（`257×64×257 ×24` 只有 156 GFLOPS）。改法是把 `(batch, 行块)` 展平到**一个** `ParallelRange` 里调度。

### 5.9 SIMD 抽象

- 统一用 `Vector<float>`（`System.Numerics`）→ 自动映射 AVX2/AVX512/NEON。
- 热点下沉到 `Vector256` + `Fma.MultiplyAdd`（注意：`Vector256`/`Fma` 在 net8/net10 都直接可用）。
- 优先级：GEMM 微内核 > Conv panel 打包 > depthwise > softmax/layernorm > 逐元素。

---

## 6. Kernel 需求（实测定稿）

只有 **9 类数值 kernel**，且变体极少：

| Kernel | 需要的变体（来自实测） | 备注 / 可复用资产 |
|---|---|---|
| Conv 3×3 | stride 1 / stride 2，padding 1，无 dilation | YOLO 主力；RF-DETR 有 6 个 3×3 s1 (Cin=128)。参考 SimdYolo `CpuConvolution`（16 空间位置 panel，不物化整图 im2col） |
| Conv 1×1 | stride 1 | 等价 GEMM |
| Conv 16×16 | stride 16 | 仅 RF-DETR patch-embed |
| Depthwise 3×3 | stride 1 | 仅 YOLO（7 处） |
| GEMM / MatMul | 2D、4D batched + broadcast、transB、3D `[1,T,C]×[C,C]` | YOLO 2 + RF-DETR 140（MLP/投影/attention 都有，形状见 `misc.md §3.10`）。微内核已定稿：4 行×16 列 / AVX2+FMA；B 不广播时 `(batch, m)` 合并成一次 GEMM；剩余问题是并行扩展性与「独立内核快 ≠ 端到端快」 |
| MaxPool | 5×5 s1 pad2，ceil_mode=0 | 仅 YOLO SPPF（3 处） |
| Softmax | axis=-1（18 处）、axis=1（1 处） | |
| LayerNorm | axis=-1，eps=1e-6 / 1e-5 | 仅 RF-DETR（49 处） |
| Resize | nearest + asymmetric + floor | 仅 YOLO（2 处） |
| GridSample | bilinear / zeros / align_corners=0 | 仅 RF-DETR（3 处） |
| TopK | axis=1 | 仅 RF-DETR（1 处） |

**不需要**：bilinear/cubic Resize、NonMaxSuppression（op）、DetectionOutput、PriorBox、RNN/LSTM/GRU、Einsum、SDPA、Rotary、MatMulNBits。
**OCR 追加（2026-11-10）**：`ConvTranspose`（2×2/s2）、`GlobalAveragePool`、`ReduceMean`、`HardSigmoid`、
`BatchNormalization`（推理模式）、`AveragePool`、`Identity`、`Squeeze` —— 都已实现；BN 在 YOLO/RF-DETR 里仍走折叠，
OCR 的 rec 图里挂在上激活之后，直接按算子实现（每通道一次仿射）。

---

## 7. 算子清单（72 个）

> 早期按「目标模型所需」分批（Batch A~E，42 个）；后续补的算子在 Batch F/G 里，总数已经是
> `LayerRegistry` 里注册的 67 个（另有 `Constant` 导入期内联，`Silu` / `FusedEltwise` 是融合产物）。

**Batch A — 元数据/形状类（无 kernel，最先做）** (9)
`Constant` `Shape` `ConstantOfShape` `Expand` `Tile` `Unsqueeze` `Reshape` `Slice` `Split`

**Batch B — 搬运/类型** (3)
`Concat` `Transpose` `Cast`

**Batch C — 逐元素/比较** (17)
`Add` `Sub` `Mul` `Div` `Sqrt` `Exp` `Erf` `Sin` `Cos` `Sigmoid` `Relu` `Not` `Equal` `Less` `Greater` `And` `Where`

**Batch D — 归约/索引/选择** (5)
`ReduceSum` `ReduceMax` `TopK` `Gather` `GatherElements`

**Batch E — 重算子** (8)
`Conv` `MatMul` `Gemm` `MaxPool` `Softmax` `LayerNormalization` `Resize` `GridSample`

**Batch F — OCR（PP-OCRv6，2026-11-10）** (7)
`HardSigmoid` `BatchNormalization`（推理模式） `Identity` `AveragePool` `GlobalAveragePool` `ReduceMean` `ConvTranspose`(2×2/s2)

**Batch H — SAM3（2026-11-10）** (5)
`IsNaN` `GreaterOrEqual` `InstanceNormalization` `Einsum`(bqc,bchw-&gt;bqhw) `RoiAlign`(avg/max, output_half_pixel)

**Batch G — 其它导出带出来的** (18)
`Abs` `Neg` `Log` `Tanh` `Clip` `Pow` `Max` `Min` `Mod` `Ceil` `Floor` `Round` `Sign` `Reciprocal`
`Flatten` `Squeeze` `Range` `GatherND` `GlobalMaxPool`

> 42 = 9+3+17+5+8 是首版的口径；现在 67 = 上面 A~G 去重后的注册数。
> 实现顺序：A → B → C → D → E（先让图能"流动"，再补算力），F/G 由 OCR 与其它导出驱动。

---

## 8. 测试与对齐策略

### 8.1 三层验证

| 层 | 方法 | 通过标准 |
|---|---|---|
| 加载层 | `dump-graph` vs python onnx 导出（节点序、类型、输入输出名）+ **逐 Arg shape/dtype vs `onnx.shape_inference`** | 完全一致 |
| 算子层 | 每个算子单独建图，与 ORT 同算子对比 | MAE < 1e-6（逐元素）/ 1e-5（卷积） |
| 端到端 | 4 个模型整体输出对比 | YOLO: MAE < 1e-4；RF-DETR: dets MAE < 1e-3、labels 全等 |

> **不再依赖 OpenCV 的 `dumpToPbtxt` 作为 oracle**：本机未编译 OpenCV，且 importer 会改图（常量转 blob、融合），逐字比对不可行。实测 `onnx.shape_inference.infer_shapes(check_type=False, strict_mode=False)` 对 `yolo11n` 与 `rfdetr-small` **全部张量（含 1539 节点）均能推出形状，未知数 = 0**，可直接当形状 oracle。

### 8.2 golden 生成（**改法**）

`tests/OnnxSharp.GoldenGen`（引用 `Microsoft.ML.OnnxRuntime` 1.31.1）：

- ⚠️ ORT **没有**「通过 SessionOptions 加中间输出」的 API；`session.Run(inputs, outputNames)` 只能取模型 `graph.output` 里已声明的名字。
- 正确做法：**python 侧做图改写**（`onnx.helper`），把需要的中间 node output 追加进 `graph.output`（并补 `value_info`），另存 `*_probe.onnx`，C# 侧再对 probe 模型跑 ORT 取中间结果。

```python
# tools/scripts/make_probe.py  （示意）
import onnx, sys
m = onnx.load(sys.argv[1]); g = m.graph
wanted = [n.output[0] for n in g.node if n.op_type in {"Conv", "Softmax", "MatMul", "LayerNormalization"}]
existing = {o.name for o in g.output}
for name in wanted:
    if name not in existing:
        g.output.append(onnx.ValueInfoProto(name=name))
onnx.save(m, sys.argv[2])
```

- 输入用**确定性伪随机**（固定 seed）或真实图片 letterbox 后的 NCHW。
- 输出写 `.bin`（原始字节）+ `.json`（name/dtype/shape）。

### 8.3 回归用例集

- 4 个目标模型（必须）
- 每算子最小用例（约 65 个）
- 边界：`Split` 不等分、`Slice` 负索引/step、`Reshape` 含 -1/0（allowzero）、广播 1×N、`Softmax` 非末轴、`LayerNorm` 双 eps、`GridSample` 越界零填充、`TopK` 并列值

---

## 9. 性能目标（**重设**）

> **实测数据一律看 `bench.md`**（唯一事实源，含四引擎 × 6 模型的 median/min 双指标与复现命令）。
> 本节只放**目标**与**口径约定**，避免两处数字打架。

目标（本机 Ryzen 7 8845HS，8C/16T，FP32 batch=1，16 线程）：

| 模型 | 当前实测（median，2026-11-07 第七轮） | 同期 OpenCV dnn | 优化后目标 |
|---|---|---|---|
| yolo11n 640² | 36.7 ms | 28.6 | **< 30 ms** |
| yolo11s 640² | 85.1 ms | 73.1 | < 75 ms |
| yolov8n 640² | 35.8 ms | 30.9 | < 30 ms |
| yolov8s 640² | 92.4 ms | 77.4 | < 77 ms |
| rfdetr-small 512² | 231.5 ms | 191.7 | **< 192 ms**（追平 OpenCV dnn） |
| rfdetr-medium 576² | 299.4 ms | 243.1 | < 244 ms |

> 目标按「同期同会话的 OpenCV dnn 实测」设定（每轮随对方实测平移）。
> 第五轮曾把 rfdetr 目标设在 205 / 305，那一轮的 OpenCV 数字（204.9 / 304.4）本身失常，
> 第七轮对方跑出 191.7 / 243.1 —— 目标已随之收紧。**改目标前先确认对方读数在历史区间内。**
> `yolov8s < 55 ms` 那档不再作为近期目标：它要求追平 SimdYolo（56.7），属于 `s` 档大 GEMM 打包的独立课题。

> 口径说明：
> - 原「首版 < 150 ms」对标标量 kernel 偏乐观，已放弃该阶段口径；现直接对标 OpenCV 5 dnn / SimdYolo 实测值。
> - **口径更正（2026-11-07）**：上一版写过「SimdYolo 的 22 ms 依赖 channel 对齐」——**这是错的，已撤回**。核对源码：`CpuPointwise24` 的 24 是**空间列面板**（N=H·W），`CpuPointwiseAligned` 的 `n % 16 == 0` 也是 H·W 的缓存行对齐，`CpuConvolution` 按 16 个**空间位置**打包，全程 **NCHW、无通道 padding**。即「不做图级 block layout 就到不了 22ms」不成立，靠 M4 的 kernel 内打包即可。
> - 图级 blocked layout 的收益需实测，经验区间 5~15%（FP32 + 已打包 GEMM 时更小），代价见 §12 Q7。
> - 不含图像预处理与 decode/NMS。
> - **任何性能数字必须标注预热方式**：.NET 分层编译下按「迭代次数」预热会低估 4 倍以上，`gemmbench`/`convbench` 已改成按时间预热，详见 `bench.md`「测量方法上的坑」。

### 9.1 瓶颈定位（2026-11-07，结论供 M4 使用）

| 结论 | 依据 |
|---|---|
| 机器上限：纯 FMA 单线程 **91 GFLOPS**、16 线程 **753 GFLOPS**（SMT）；GEMM 最好约 500 GFLOPS | `fma --threads 1,16` + `gemmbench` |
| 16 线程 GEMM 只有峰值的 **40%~46%**：① 26~33 个行块分 16 线程要跑两波；② 每行块重读整个 B，DRAM 带宽受限 | `gemmbench --threads 1,2,4,8,16` + 手工估算 |
| RF-DETR 的 140 个 MatMul **不全是 batched attention**：MLP 的 `[257,384]×[384,1536]` 就是 MatMul 节点，全部合计 **54.1 GFLOP**（主干 MLP/投影带前置 batch 4）。卷积只影响 3.4% | 形状统计 + `profile` |
| MatMul **已经跑满**：133 ms / 54.1 GFLOP = **407 GFLOPS**，不低于同形状独立基准（386）。GEMM 不再是可挖的点 | `profile` + `gemmbench` |
| decoder q/k/v 投影运行时是 `[300,1,256]×[256,256]`（300 个 batch × m=1），按 batch 逐个调内核会把权重重打包 300 遍，**已修为合并 batch 维**（4.06 → 0.13 ms/算子） | `profile` 逐算子 + 端到端 A/B（-24%） |
| 逐元素算子已向量化+并行，实测 **56 GB/s**，是单线程 memcpy 上限（73 GB/s）的 77%，**没有带宽余量**，只能靠**融合减少遍数** | 微模型 `add-1024x1024` + numpy memcpy 对照 |
| 逐元素融合已落地并**贴到带宽墙**：GELU 五节点链 1.50 → 0.30 ms/层，但融合前后都是 **42 GB/s**（63 MB / 1.50 ms vs 12.6 MB / 0.30 ms），说明省下来的纯粹是趟数。rfdetr-small/medium 端到端 -7.4% / -9.1%，输出逐位不变 | `bench.md` 结论 2 + 交替 A/B |
| 融合内核的形态决定成败：**通用 `EltwiseStep[]` 解释器比不融合还慢**（运行期下标的 `Vector256[]` 被 JIT 打回内存 + 每步 `switch`），固定形状必须手写内核。短链（<4 节点、<256K 元素）融合是负收益 | `profile` 逐算子 1.65 vs 1.50 vs 0.30 ms |
| YOLO 侧差距集中在 `s` 档（层宽 128~256），是**大 GEMM 分块/打包**，不是单个卷积 | `bench.md` 相对倍率表 |
| **融合后的瓶颈分布**（rfdetr-small，by-type，sum-of-min 211 ms）：`MatMul 64.3%`、`Add 6.6%`、`Transpose 6.1%`、`Conv 3.6%`、`Softmax 3.4%`、`Mul 2.3%`、`LayerNorm 2.3%`、**`GridSample 2.3%`**、`FusedEltwise 1.8%`。MatMul 占比**升**了，因为别的部分变快了，而它本身已 407 GFLOPS —— 只剩 3~7% 的碎块可挖 | `profile --top 1` 的 by-type 段 |
| ~~`GridSample` 循环顺序 bug~~ **已修复**：原实现把通道 `c` 放最内层而输出是 NCHW（写步长 `outH*outW`，每元素换缓存行）。`c` 外提 + 采样坐标预算后 1.6 → 0.92 ms/个（**-48%**），端到端 rfdetr-small -1.4% / medium -1.1%（min），逐值 diff=0 | `profile` by-type 前后对比 + 6 轮交替 A/B |
| 小改动（<2%）在本机**只能用 min 或逐算子 profile 判定**：同代码重跑 6 次，yolo11n median 抖 49.2~53.5、yolov8s 96.9~106.9（±8%），后台负载会把信号整个淹没 | `ab_fuse.ps1` 6 轮交替实测 |
| **「追平对手」不能只看一轮比值**：第五轮记的 rfdetr vs OpenCV = 1.12/1.06（"已追平"），第七轮对方同代码跑出 191.7/243.1（medium 快 20%），说明那是对方发挥失常；真实差距一直是 **1.21~1.23×**。我们自己两轮几乎没变（231.5/299.4 vs 230.4/321.5） | `bench.md` 4.1 + 第七轮复测 |
| **跑全表前先查满载进程**：第七轮 `SC2_x64` 占满一个核，导致两轮全表作废（ORT 的 yolov8s 跑出 243/194 两次）。`(Get-Process X).CPU` 隔 4 秒的增量 ≈ 4 就说明它占满一核 | 第七轮实测 |
| **独立内核吞吐无法预测端到端**：B 预打包、`mb` 扫参、列切分（+45%）三次都出现「微基准变快、端到端无变化」；且统计 FLOPs 时漏掉前置 batch 会凭空造出「慢 3 倍」的假象。只能用同会话交替 A/B 判定 | `bench.md` 结论 3 |
| GC 与降频都已排除：GC 暂停 0.7 ms/iter；同一形状连续跑 3000 次不衰减（383 vs 386 GFLOPS）。**profile 与 bench 的唯一差别只是口径**：各算子均值之和 251 ms ≈ 端到端中位数，min 之和 215 ms 是下界 | `bench` 新增的 `alloc`/`gc`/`gc pause` 三行输出 |

---

## 10. 风险与对策

| # | 风险 | 影响 | 对策 |
|---|---|---|---|
| R1 | **`GetMemoryShapes` 拿不到常量值** | 42 个里约 8 个算子无法实现 | M0.5 先做 `LayerInfo → Net` 反查（§5.3） |
| R2 | **内存爆炸（1:1 需 1.4 GB）** | RF-DETR 跑不起来 | `BufferAllocator` 提到 M0.5；M3 验收含峰值内存 < 400 MB |
| R3 | 4D batched MatMul + broadcast 做错 | YOLO C2PSA / RF-DETR 全崩 | M0.5 写通用广播迭代器；算子层先做单测 |
| R4 | Shape 链折叠不完整（122 个 Reshape） | 形状推断失败 | 折叠集合含 `Transpose`；失败时回退 `DynamicOutputShapes` |
| R5 | opset 17/19 参数走 **input 而非 attribute** | 解析出空属性 | `LayerParams` 统一"先看 const 输入，再看 attribute"；补 opset 18+ `Split.num_outputs` |
| R6 | GEMM 性能不达预期 | 端到端慢 3~5× | 微内核已不是主要矛盾（单算子 300~500 GFLOPS），真实风险是**多线程只跑出机器峰值的 40%~46%**，以及对「独立内核快 ≠ 端到端快」还没解释清楚（`bench.md` 结论 3）；对策是二维 cache blocking + batched 展平调度 |
| R10 | 逐元素算子按遍数堆叠（DETR 占 ~20%） | Transformer 模型永远追不上 OpenCV | 实测 56 GB/s，是单线程 memcpy 上限（73 GB/s）的 77%，**没有带宽余量**，只能靠**图级融合**减少遍数；M4 先做同形状二元链原地融合 |
| R7 | 数值细节差异（`Resize` floor、`LayerNorm` eps、`GridSample` 边界） | MAE 超标 | 逐算子与 ORT 对齐；`misc.md` 记录每个坑 |
| R8 | protobuf reader 出错（118 MB 模型） | 加载失败 | 直接移植 SimdYolo `Proto` 并在 4 个真实模型上跑 |
| R9 | 别名冲突（零拷贝 Reshape 被下游覆盖） | 结果随机错 | 别名 Arg 必须由 Allocator 绑同一 buffer 且生命周期包含；否则退化为拷贝 |

---

## 11. 时间表（单人全职估算）

| 里程碑 | 内容 | 估时 |
|---|---|---|
| M0 | 骨架 + 图 + 广播逐元素 | ~~2~3 周~~ **已完成** |
| **M0.5** | 引擎能力补全（反查/Allocator/temp 池/动态形状/测试工程） | ~~1 周~~ **已完成（2026-11-07）** |
| M1 | ONNX 加载 + dump/形状对齐（protobuf 现成） | ~~1 ~ 2 周~~ **已完成（2026-11-07）** |
| M2 | YOLOv11 跑通 | ~~3 ~ 4 周~~ **已完成（2026-11-07）** |
| M3 | RF-DETR 跑通（含内存达标） | ~~3 ~ 4 周~~ **已完成（2026-11-07）** |
| M4 | 性能优化（微内核已定稿，剩 GEMM 并行扩展性 + 融合 + Transpose） | **进行中**，剩余约 1.5 ~ 2 周 |
| M5 | 工程化 + 打包 | 1 周 |
| **合计（剩余）** | | **约 2.5 ~ 3 周** |

---

## 12. 待确认事项

| # | 问题 | 当前默认 |
|---|---|---|
| Q1 | 目标框架 | **已定：保留 `net8.0;net10.0` 双目标，不退役**。net8 于 2026-11-10 EOL，仅作老宿主兼容；新代码以 net10 为基准，且不得使用 net8 缺失的 API |
| Q2 | protobuf：手写 vs `Google.Protobuf` | **手写**（直接移植 SimdYolo `Proto`，加 DType 维度） |
| Q3 | 是否需要 OpenCV 作 oracle | 不需要。**ORT（算子/端到端）+ onnx.shape_inference（形状）** 双 oracle |
| Q4 | 是否纳入 `yolov11s-seg/pose/obb` | 不纳入（M6 再议） |
| Q5 | 图像预处理辅助 | **已定（2026-11-07）**：`src/OnnxSharp/Vision/` 提供可选门面（letterbox / stretch + decode + NMS / top-k + `Detection`，覆盖 YOLO 与 RF-DETR 两种检测头），不依赖任何第三方包、不参与图执行、可整体删除；图像**编解码**仍只在 `tools/` |
| Q6 | NativeAOT / WASM / IL2CPP | **AOT**：首版不验证，仅保持 AOT 友好编码约定。**WASM / IL2CPP / Unity：不支持，不在路线图** |
| Q7 | 是否允许 channel 对齐 | **分两层**：①**kernel 内部**打包 / padding / 缓存行对齐任务切分 → **允许，M4 必做**（对图透明、回退成本低）；②**图级** blocked layout（NC8HW8 + `TransformLayout`）→ **不实现**，仅在 `Tensor.Layout`、`ArgData.Layout`、`LayerInfo.GetLayouts()`（恒返回 NCHW）预留接口，成本 < 1 天。将来真要做需 +3~5 周，收益 5~15% 且需实测 |

### 12.1 待用户拍板的 SimdYolo 偏离点（影响功能）

> 依据 `misc.md §2.6`。**未拍板前按"暂用 OnnxSharp 现方案"实现，不阻塞 M0.5/M1。**

| # | 议题 | SimdYolo 做法 | 若跟它的后果 | 建议 |
|---|---|---|---|---|
| Q8 | `Tensor` 内存模型 | `float[]` + `int[] Shape`，无 dtype、无 `Fit` | 无法承载 `I64/Bool/I32` → **RF-DETR 跑不了**（`Cast/TopK/Gather/Reduce*` 处处需要 int64） | 保留 `byte[]`+`DType`，kernel 用 `Span<float>` 适配 |
| Q9 | 缓冲池与输出所有权 | 精确长度池（≤256 MiB）+ 图边界存活扫描 `RetainOnly` + `DetachOutputs`（公开输出**永久脱离池**，同 buffer 多别名则克隆） | 输出不能跨帧复用（每次 forward 返回新数组）；与 OpenCV `assignBuffers`+`fit` 的"输出缓冲常驻"语义不同 | 首版跟 SimdYolo（更简单且已验证）；`Forward()` 语义定为"调用方拥有的独立结果" |
| Q10 | 线程模型 | 专用 `CpuExecutor`（自有工作线程，调用线程参与计算，禁止嵌套扩张）+ `CpuScheduling` | 要自建执行器（约 400 行），但能避免 `ThreadPool` 饥饿与嵌套超订；`Parallel.For` 版实现简单但高峰时抖动 | 首版 `ParallelRange`（`Parallel.For`），M4 视实测决定是否换 `CpuExecutor` |
| Q11 | 输入数量/类型约束 | `OnnxGraph.Load` 强制"恰好 1 个 FP32 输入"，opset 限 11~21 | 目标 4 模型都满足；但限制了通用性（多输入模型直接拒载） | 放开为"任意数量/类型"，opset 未知版本给警告而非异常（成本低） |

---

## 13. 维护：跟进上游 OpenCV 5 dnn

> 后续的算子更新、性能优化、bug 修复，**一律先看 OpenCV 5.x `modules/dnn` 的提交**，再决定是否移植。
> 详细命令、路径映射、分类规则、跟踪记录见 **`misc.md §11`**。

### 13.1 基线

| 项 | 值 |
|---|---|
| 参考仓 / 分支 | `F:\code\opencv` / `5.x` |
| 基线提交 | `43f8b1f5`（2026-11-06） |
| 语义基准 | 42 个算子的行为、shape 推断、属性解析以该提交为准 |

> **版本台账在 `misc.md §11.1`，那是唯一事实源。** 本节只放一个便于速览的当前值；推进版本时**必须同时改两处**（先改 `misc.md §11.1` 的推进表，再回来改这里）。
>
> 该仓库只在**只读**方式下使用：`fetch` + `log` + `diff`，不修改、不提交、不切分支。

### 13.2 触发时机

| 时机 | 动作 |
|---|---|
| 每个里程碑**开始前** | `git fetch` → 过滤 dnn 提交 → 按 `misc.md §11.5` 分类 |
| 每个里程碑**结束时** | 把结论写进 `misc.md §11.6`；采纳项必须补回归用例 |
| 遇到数值/形状疑难 | 先在上游 `git log -i --grep=<算子>` 搜同类修复，再自己调 |
| 基线提交变更时 | 先追加 `misc.md §11.1` 的**版本推进记录**，再回来更新 §13.1 与 `misc.md §11.6` |

### 13.3 采纳判据（与 §1.5 参考优先级一致）

1. **bugfix（数值/形状）** → 无条件采纳。
2. **conformance 测试** → 抄成我们的用例。
3. **性能优化** → 先看 SimdYolo 是否已有等效/更好的实现；**若与 SimdYolo 冲突，交用户拍板**。
4. **新增算子 / 新后端 / int8-fp16 / 融合 pass** → 只记录，不动首版范围。

### 13.4 里程碑任务

- [ ] M0.5 结束时完成首次全量扫描，并把结果写入 `misc.md §11.6`；`§11.1` 台账已建立（`43f8b1f5`，2026-11-07）
- [ ] M2/M3 结束前各执行一次同步，把 `Resize`、`GridSample`、`LayerNormalization`、广播规则的上游修复全部吸收
- [ ] M5 把同步流程写进 README（含命令片段）
