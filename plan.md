# OnnxSharp 实施计划

> 纯 C# 的 ONNX 推理运行时，以 OpenCV 5 dnn 新版 graph 引擎为参考实现。
> 首版目标：跑通 YOLOv11 检测 与 RF-DETR。
>
> **当前状态：M0 骨架可跑**（`dotnet build` 0 警告；`SmokeTest` PASS）
> 最后更新：2026-11-07（第二轮评审修订）

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
| 10 | YOLO decode+NMS 移出引擎（属调用方后处理） | 与 §1.3 一致 |
| 11 | `ConstFolder` 折叠集合补 `Transpose`（可消灭 1 处数据相关 `Where`） | 依赖追踪实测 |
| 12 | 新增 **§13 上游同步机制**：后续更新/优化/bugfix 一律先跟 OpenCV 5 dnn 的 git | `misc.md §11`（基线 `5.x @ 43f8b1f5`） |

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
| 图像编解码/预处理 | 交给调用方；仅在 `tools/` 提供最小 letterbox 参考实现。 |

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
| 图像编解码、letterbox、**YOLO decode+NMS 后处理** | 交给调用方（与 OpenCV dnn 一致：只出 `[1,84,8400]` 原始张量） |

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
├─ plan.md                     本文件
├─ misc.md                     杂项：实测数据、源码地图、环境、坑
├─ OnnxSharp.slnx
├─ src\
│  └─ OnnxSharp\               产品库（零第三方依赖）
│     ├─ Tensors\
│     │  ├─ Tensor.cs          多维张量（紧凑布局 + Fit 复用）
│     │  ├─ TensorShape.cs     MatShape 等价物
│     │  ├─ TensorOps.cs       广播 / 逐元素运算
│     │  ├─ DType.cs
│     │  └─ BufferPool.cs      （M0.5）ArrayPool 封装
│     ├─ Core\
│     │  ├─ Arg.cs             Arg / ArgData / ArgKind        [完成]
│     │  ├─ Graph.cs           prog 线性表（M0.5 改为可重建）  [完成→改]
│     │  ├─ LayerInfo.cs       LayerInfo + Layer（执行器）     [完成]
│     │  ├─ LayerParams.cs     [完成]
│     │  ├─ DictValue.cs       [完成]
│     │  ├─ LayerRegistry.cs   [完成]
│     │  ├─ Interpreter.cs     [完成→M0.5 补动态形状/temp 池]
│     │  ├─ NetImpl.cs         [完成→M0.5 补 BufferAllocator]
│     │  └─ Net.cs             门面（ReadModel 在 M1）
│     ├─ Threading\
│     │  └─ ParallelRange.cs   parallel_for_ 等价物             [完成]
│     ├─ Engine\
│     │  ├─ ConstFolder.cs     （M3）常量折叠 / Shape 链
│     │  └─ BufferAllocator.cs （M0.5）冲突图缓冲复用
│     ├─ Onnx\
│     │  ├─ Proto\             protobuf 读取（M1，移植 SimdYolo Proto）
│     │  ├─ OnnxImporter.cs    通用骨架：graph/value_info/initializer/属性
│     │  └─ Handlers\          按算子分文件的 handler
│     ├─ Ops\                  42 个 LayerInfo+Layer 实现
│     └─ Kernels\
│        ├─ ConvKernels.cs
│        ├─ GemmKernels.cs
│        ├─ PoolKernels.cs
│        ├─ ActKernels.cs
│        └─ Simd\              Vector<T>/Vector256 抽象
├─ tools\
│  └─ OnnxSharp.Cli\           run / dump-graph / bench（+ letterbox 参考实现）
└─ tests\
   ├─ OnnxSharp.Tests\         单元 + 端到端（M0.5 建立）
   └─ OnnxSharp.GoldenGen\     引用 Microsoft.ML.OnnxRuntime 生成 golden（不进产品）
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

### M0.5 — 引擎能力补全（**新增，M1 之前必须完成**）

这一批是 M1/M2/M3 的硬前置，做完之前不要动 importer。

**任务**
- [ ] **常量输入反查**：`LayerInfo` 增加 `NetImpl` 回填（等价 `getNetImpl`），或把 `GetMemoryShapes` 形参改为 `IReadOnlyList<Tensor>`，使 Shape 类算子能读到常量**值**（见 §5.3）
- [ ] `Graph.Prog` 可重建 + `UpdateGraphOfs`（常量折叠后要删节点）
- [ ] `BufferAllocator`（冲突图）：`useCount` / lifetime 区间 → 复用同一 buffer（见 §5.7）；**优先照搬 SimdYolo `OnnxExecutionPlan` 的 `remaining`+`Live` 存活集合与 `CpuTensorPool` 的边界扫描**（`misc.md §2.5`）
- [ ] **布局接口预留（Q7 层 2 的代价最小化）**：
  - `Tensor.Layout`（`Layout.NCHW` 唯一取值）
  - `ArgData.Layout`
  - `LayerInfo.GetLayouts(inLayouts, outLayouts, ...)`，全实现恒返回 NCHW
  - 不写 `TransformLayout` 算子、不写 `useBlockLayout()` pass
- [ ] **AOT 友好约定落地检查**：全库无 `dynamic`/`Emit`/反射式注册（`LayerRegistry` 用显式 switch 或 `Dictionary<string, Func<...>>`）
- [ ] 全局 temp 池（对齐 `globalTemps`），取消 `Interpreter` 每次 forward `new` 临时张量
- [ ] `Interpreter` 补 `DynamicOutputShapes()` 分支：此类 op **不预 fit**，输出 Tensor 交给 layer 自己 fit（见 §5.4）
- [ ] `Interpreter` 重 finalize 判据补 `inpTypes`（现只比 shapes）
- [ ] `NetImpl.GetArg` 拆成 `Declare`（新建）/ `Require`（存在或报错），避免 importer 拼错名字被静默吞掉
- [ ] 建 `tests/OnnxSharp.Tests`（xunit）与 `tests/OnnxSharp.GoldenGen`
- [ ] `Tensor.Fit` 的 `Count/Total` 溢出路径检查（`long` → `int` 截断）

**验收**：手工建图跑「常量 shape 驱动 Reshape/Slice」；`BufferAllocator` 单测证明 4 个不重叠 Arg 共用 1 块缓冲；连续两次 `Forward` 的 GC 分配量 < 1 MB。

### M1 — ONNX 加载（dump 与参考对齐）

**任务**
- [ ] protobuf reader：**移植 SimdYolo `OnnxReader.cs` 的 `Proto`**（约 139 行，已覆盖 varint / fixed32 / fixed64 / length-delimited / packed / 未知字段跳过 / `ParseView` 零拷贝），**加 DType 维度**（SimdYolo 全转 float，会丢 I64）
- [ ] 对象模型：`ModelProto/GraphProto/NodeProto/AttributeProto/TensorProto/ValueInfoProto/TypeProto`（只保留用到的字段）
- [ ] `OnnxImporter` 骨架：initializer→Const Arg、value_info→shape/type、`Constant` 节点、外部数据（`external_data`，含路径逃逸校验）
- [ ] `LayerParams` 属性解析：`ints/floats/strings/tensor`；**统一"先看 const 输入，再看 attribute"**
- [ ] **42 个算子的 handler**（分 5 批，见 §7）
- [ ] CLI `dump-graph`：输出节点序/类型/输入名/推断 shape

**验收**（改法，见 §8.1）：节点数、节点类型序列、输入输出名集合与 python onnx 一致；**每个 Arg 的 shape/dtype 与 `onnx.shape_inference.infer_shapes()` 完全一致**（实测两模型均 100% 可推，未知张量 = 0）。不需要任何 kernel。

### M2 — YOLOv11 跑通（15 个算子）

**任务**
- [ ] Conv：3×3 s1/s2、1×1、3×3 depthwise（先标量，后移植 SimdYolo 的 panel 打包 + packed GEMM）
- [ ] MaxPool 5×5 s1 pad2 ceil=0
- [ ] Resize nearest / asymmetric / floor（`scales` 走第 3 输入）
- [ ] Softmax（axis=-1 与 axis=1）
- [ ] 逐元素：Add/Sub/Mul/Div（含广播）+ Sigmoid
- [ ] 形状类：Reshape/Slice/Split/Concat/Transpose
- [ ] MatMul（4D batched + broadcast）+ GEMM
- [ ] 与 ORT 逐层 + 端到端比对
- [ ] （非引擎）CLI 里提供 decode+NMS 的**调用示例**，不写进产品库

**验收**：`yolo11n.onnx` 输出 `[1,84,8400]` 与 ORT 的**最大绝对误差 < 1e-4**；`yolov11s.onnx` 同标准；单张 640² 端到端 **< 200 ms**（朴素 SIMD 版）。

### M3 — RF-DETR 跑通（补 27 个算子）

**任务**
- [ ] LayerNormalization（axis=-1，eps 逐节点：1e-6×37 / 1e-5×12）
- [ ] Gemm（transB=1, alpha=beta=1）
- [ ] GridSample（bilinear / zeros / align_corners=0）
- [ ] ReduceSum/ReduceMax（keepdims 0/1，axes 来自 const 输入）
- [ ] TopK（axis=1，k 来自 const 输入）
- [ ] Gather / GatherElements
- [ ] Erf / Exp / Sqrt / Sin / Cos / Relu / Not / Where
- [ ] 比较：Equal / Less / Greater / And
- [ ] Cast（FLOAT / BOOL / INT64）
- [ ] Tile / Expand / Unsqueeze / Shape / ConstantOfShape
- [ ] **`ConstFolder` + Shape 链**：折叠集合含 `Transpose`（见 §5.5），折叠后重建 `prog`
- [ ] **内存达标**：把 1:1 的 1381 MB 压到 **< 400 MB**（靠 `BufferAllocator` + 常量失效释放）

**验收**：`rfdetr-small/medium` 的 `dets` MAE < 1e-3、`labels` 逐元素一致；峰值内存 < 400 MB；单张 512² < 1500 ms（朴素版）。

### M4 — 性能

- [ ] GEMM：移植 SimdYolo `CpuGemm`/`CpuPointwise24`（16/24 列 packed + AVX2/FMA 微内核 + batch offsets）
- [ ] Conv：**panel 打包（16 空间位置）代替整图 im2col**（SimdYolo `CpuConvolution`）
- [ ] Winograd F(2×3)/F(4×4)（SimdYolo `CpuWinograd4`/`CpuWinogradPanels` 可直接搬，**不必留到 M6**）
- [ ] Depthwise 专用 kernel
- [ ] 逐元素/softmax/layernorm 的 `Span<float>` + 无广播快速路径
- [ ] 多线程：`ParallelRange` 按输出列面板切分，对齐 `ParallelOptions`
- [ ] **kernel 内部打包/对齐（必做）**：packed panel + 缓存行对齐的任务切分（SimdYolo 路线，对图透明）
- [ ] （可选，M6）**图级** blocked layout（NC8HW8）：仅当实测证明"内核打包"仍是瓶颈时才做，见 §12 Q7
- 目标见 §9

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
| GEMM / MatMul | 2D、4D batched + broadcast、transB、3D `[1,T,C]×[C,C]` | YOLO 2 + RF-DETR 140。参考 SimdYolo `CpuGemm`/`CpuPointwise24` |
| MaxPool | 5×5 s1 pad2，ceil_mode=0 | 仅 YOLO SPPF（3 处） |
| Softmax | axis=-1（18 处）、axis=1（1 处） | |
| LayerNorm | axis=-1，eps=1e-6 / 1e-5 | 仅 RF-DETR（49 处） |
| Resize | nearest + asymmetric + floor | 仅 YOLO（2 处） |
| GridSample | bilinear / zeros / align_corners=0 | 仅 RF-DETR（3 处） |
| TopK | axis=1 | 仅 RF-DETR（1 处） |

**不需要**：bilinear/cubic Resize、AvgPool、GlobalPool、ConvTranspose、NonMaxSuppression（op）、DetectionOutput、PriorBox、RNN/LSTM/GRU、Einsum、SDPA、Rotary、MatMulNBits、BN（已折叠进 Conv）。

---

## 7. 算子清单（42 个，分批）

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

> 42 = 9+3+17+5+8。实现顺序：A → B → C → D → E（先让图能"流动"，再补算力）。
> 修正记录：原并集列表漏了 `Greater`（模型实测 ×2），Batch E 原标 (7) 实为 8 个。

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

实测基准（本机 i7-8700K，12 逻辑核，FP32 batch=1）：

| 模型 | ORT 1.31 CPU（实测） | SimdYolo 托管 CPU（官方报告） | 首版目标（朴素 SIMD） | 优化后目标 |
|---|---|---|---|---|
| yolo11n 640² | 26.4 ms | **22.19 ms** | < 200 ms | **< 30 ms** |
| yolov11s 640² | 待测 | — | < 450 ms | < 80 ms |
| rfdetr-small 512² | 186.6 ms | — | < 1500 ms | **< 450 ms** |
| rfdetr-medium 576² | 待测 | — | < 1800 ms | < 550 ms |

> 口径说明：
> - 原计划「首版 < 150ms」对标标量 kernel 偏乐观；「优化后 < 60ms」相对 SimdYolo 实测偏松（自缚 3 倍）。
> - **口径更正（2026-11-07）**：上一版写过「SimdYolo 的 22ms 依赖 channel 对齐」——**这是错的，已撤回**。核对源码：`CpuPointwise24` 的 24 是**空间列面板**（N=H·W），`CpuPointwiseAligned` 的 `n % 16 == 0` 也是 H·W 的缓存行对齐，`CpuConvolution` 按 16 个**空间位置**打包，全程 **NCHW、无通道 padding**。即「不做图级 block layout 就到不了 22ms」不成立，靠 M4 的 kernel 内打包即可。
> - 图级 blocked layout 的收益需实测，经验区间 5~15%（FP32 + 已打包 GEMM 时更小），代价见 §12 Q7。
> - 不含图像预处理与 decode/NMS。

---

## 10. 风险与对策

| # | 风险 | 影响 | 对策 |
|---|---|---|---|
| R1 | **`GetMemoryShapes` 拿不到常量值** | 42 个里约 8 个算子无法实现 | M0.5 先做 `LayerInfo → Net` 反查（§5.3） |
| R2 | **内存爆炸（1:1 需 1.4 GB）** | RF-DETR 跑不起来 | `BufferAllocator` 提到 M0.5；M3 验收含峰值内存 < 400 MB |
| R3 | 4D batched MatMul + broadcast 做错 | YOLO C2PSA / RF-DETR 全崩 | M0.5 写通用广播迭代器；算子层先做单测 |
| R4 | Shape 链折叠不完整（122 个 Reshape） | 形状推断失败 | 折叠集合含 `Transpose`；失败时回退 `DynamicOutputShapes` |
| R5 | opset 17/19 参数走 **input 而非 attribute** | 解析出空属性 | `LayerParams` 统一"先看 const 输入，再看 attribute"；补 opset 18+ `Split.num_outputs` |
| R6 | GEMM 性能不达预期 | 端到端慢 3~5× | 直接移植 SimdYolo packed GEMM，而不是自己调 |
| R7 | 数值细节差异（`Resize` floor、`LayerNorm` eps、`GridSample` 边界） | MAE 超标 | 逐算子与 ORT 对齐；`misc.md` 记录每个坑 |
| R8 | protobuf reader 出错（118 MB 模型） | 加载失败 | 直接移植 SimdYolo `Proto` 并在 4 个真实模型上跑 |
| R9 | 别名冲突（零拷贝 Reshape 被下游覆盖） | 结果随机错 | 别名 Arg 必须由 Allocator 绑同一 buffer 且生命周期包含；否则退化为拷贝 |

---

## 11. 时间表（单人全职估算）

| 里程碑 | 内容 | 估时 |
|---|---|---|
| M0 | 骨架 + 图 + 广播逐元素 | ~~2~3 周~~ **已完成** |
| **M0.5** | 引擎能力补全（反查/Allocator/temp 池/动态形状/测试工程） | **1 周** |
| M1 | ONNX 加载 + dump/形状对齐（protobuf 现成） | 1 ~ 2 周 |
| M2 | YOLOv11 跑通 | 3 ~ 4 周 |
| M3 | RF-DETR 跑通（含内存达标） | 3 ~ 4 周 |
| M4 | 性能优化（移植 SimdYolo kernel） | 3 ~ 4 周 |
| M5 | 工程化 + 打包 | 1 周 |
| **合计（剩余）** | | **约 12 ~ 16 周** |

---

## 12. 待确认事项

| # | 问题 | 当前默认 |
|---|---|---|
| Q1 | 目标框架 | **已定：保留 `net8.0;net10.0` 双目标，不退役**。net8 于 2026-11-10 EOL，仅作老宿主兼容；新代码以 net10 为基准，且不得使用 net8 缺失的 API |
| Q2 | protobuf：手写 vs `Google.Protobuf` | **手写**（直接移植 SimdYolo `Proto`，加 DType 维度） |
| Q3 | 是否需要 OpenCV 作 oracle | 不需要。**ORT（算子/端到端）+ onnx.shape_inference（形状）** 双 oracle |
| Q4 | 是否纳入 `yolov11s-seg/pose/obb` | 不纳入（M6 再议） |
| Q5 | 图像预处理辅助 | 仅 `tools/` 内最小 letterbox，产品库不含 |
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
