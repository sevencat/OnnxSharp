# OnnxSharp 杂项

> 与 `plan.md` 分离的“事实库”：实测数据、源码地图、环境、坑、脚本、备忘。
> 本文件允许频繁追加，不追求结构性。

---

## 1. 开发环境（实测）

| 项 | 值 |
|---|---|
| .NET SDK | `7.0.410`, `8.0.416`, `10.0.101`, `10.0.402` |
| .NET 运行时 | `Microsoft.NETCore.App 7.0.20 / 8.0.22 / 10.0.1 / 10.0.13` |
| Python | `3.11.17` |
| onnx | `1.23.2` |
| onnxruntime | `1.31.1`（用作 golden oracle） |
| cv2 | **未安装**（本项目不以 OpenCV 为 oracle，不需要） |
| OpenCV 参考源码 | `F:\code\opencv`（**5.2.0-dev**） |
| 本项目路径 | `F:\code\OnnxSharp` |

> 目标框架：`net8.0;net10.0`（SDK 8 与 10 均已安装，双目标可直接 build）。
>
> ⚠️ **net8 的支持结束日期为 2026-11-10**（今天 2026-11-07）。**已决定：保留 `net8.0;net10.0` 双目标，不退役**（`plan.md` Q1）；net8 仅作老宿主兼容目标，新代码以 net10 为基准，且不得使用 net8 缺失的 API（`Vector256`/`Fma` 两者都可用）。
>
> 平台决策（2026-11-07 明确）：
> - **WASM：不做**。
> - **IL2CPP / Unity：不支持，且不在路线图**（不提供 `netstandard2.1` 目标，不做 IL2CPP 裁剪/反射限制适配；宿主必须是 .NET 8/10 运行时）。
> - **NativeAOT：首版不验证**，但代码保持 AOT 友好（无反射 / 无 `dynamic` / 无 `Emit` / protobuf 手写 / 同步 `Span` 解析）。

---

## 2. 参考源码地图（OpenCV 5.2 `modules/dnn` + SimdYolo）

### 2.1 规模实测（行数）

| 路径 | 行数 | 本项目 |
|---|---|---|
| `src/*.cpp,hpp`（引擎 + 后端胶水） | 28,162 | 取引擎部分 |
| `src/layers/` | 59,551 | 取 41 op 子集 |
| `src/layers/cpu_kernels/` | 17,341 | 取 9 类 kernel |
| `src/onnx/` | 6,726 | 取加载部分（跳 simplifier） |
| `src/int8layers/` | 13,713 | ❌ |
| `src/cuda/`（不含 cuda4dnn 子目录） | 10,250 | ❌ |
| `src/tokenizer`+`ocl4dnn`+`opencl`+`webnn`+`vkcom` | ≈ 17,346 | ❌ |
| `src/tflite/` | 1,959 | ❌ |
| `include/` | 5,419 | 取公开 API 语义 |
| **合计（src+include）** | **≈ 160,000** | 目标 ≈ 20~25k 行 C# |

其它计数：
- `CV_DNN_REGISTER_LAYER_CLASS` 注册项 = **226**
- `cv::parallel_for_` 调用 = **199**
- `CV_SIMD` 使用点 = **253**；`v_float32` 类 = **352**
- `cv::hal::` = **0**（dnn 不走 core HAL 分发）
- `cv::resize(` = **0**（`resize2_layer.cpp` 自实现，27 处 interpolation 分支）
- `AutoBuffer` = 57，`Mutex` = 29，`getTickCount/TickMeter` = 7，`FileStorage` = 14，`RNG` = 1

### 2.2 必读文件（按优先级）

| 优先级 | 文件 | 行数 | 看什么 |
|---|---|---|---|
| ★★★ | `src/net_impl.hpp` | 621 | `Arg/ArgData/Net::Impl` 字段；`newGraph/getArg/newConstArg` 声明在 450~520 行 |
| ★★★ | `src/net_impl2.cpp` | 3,104 | `prepareForInference`(582)、`finalize`(795)、`allocateLayerOutputs`(855)、**`forwardGraph`(1750)** 解释循环 |
| ★★★ | `src/graph_const_fold.cpp` + `src/graph_const_args.cpp` | 211+120 | 常量折叠 / 常量转 Arg（RF-DETR 必需） |
| ★★★ | `src/layers/conv2_layer.cpp` + `convolution_layer.cpp` + `cpu_kernels/convolution.cpp` | 1384+1329+2384 | 卷积 shape 推断 + kernel 选择 |
| ★★ | `src/layers/layers_common.hpp/.cpp` | 161+462 | 布局/类型枚举、公共工具 |
| ★★ | `src/layers/nary_eltwise_layers.cpp` | 1851 | 17 个逐元素/比较算子的广播实现 |
| ★★★ | `src/layers/slice2_layer.cpp` | — | **Shape 类算子如何反查常量值**：`getNetImpl(this)` + `argTensor(inputs[i])`（363~376 行）——C# 侧 `LayerInfo` 必须能拿到 Net 的直接依据 |
| ★★ | `src/layers/reshape2_layer.cpp` / `split2_layer.cpp` / `expand2_layer.cpp` / `topk2_layer.cpp` | — | 同上：全部在 `getMemoryShapes` 里读常量输入的值，不靠 `Blobs`（`constArgs()` 只搬 Conv/Deconv/BN/PReLU 的权重） |
| ★★ | `src/layers/cpu_kernels/fast_gemm.*` | 1088+274+1315+746 | GEMM 微内核（C# `Vector256` tiling 蓝本） |
| ★★ | `src/onnx/onnx_importer2.cpp` | 3,826 | 122 个 handler 的属性映射语义（照抄） |
| ★★ | `src/onnx/opencv-onnx.proto` | 978 | 手写 reader 的字段号来源 |
| ★ | `src/layers/cpu_kernels/conv2_depthwise.*` | 105+436 | depthwise kernel |
| ★ | `src/layers/cpu_kernels/activation_kernels.*` / `softmax.cpp` | 19/40 | 小 kernel |
| ★ | `src/graph_buffer_allocator.cpp` | 432 | 缓冲复用策略（M4 再看） |
| ★ | `src/net_impl.cpp` 内 `dumpToPbtxt` | — | dump 文本格式参考 |
| — | `src/graph_block_layout.cpp` | 369 | ❌ 首版不做 |

### 2.3 明确不看的部分

- legacy 引擎：`allocateLayers`(624)、`forwardLayer`(702)、`legacy_backend.*`、`LayerData`、`BackendWrapper`
- 全部 `graph_fusion_*.cpp`（QDQ/attention/matmul_to_gemm/shared_gemm/reshape_transpose/scale_softmax/transpose_matmul）≈ 4,900 行
- `op_cuda.cpp` / `op_inf_engine.cpp` / `op_vkcom.cpp` / `op_webnn.cpp` / `op_cann.cpp` / `net_openvino.cpp` / `net_cann.cpp` / `ie_ngraph.cpp` / `plugin_*`
- `HAVE_ONNXRUNTIME` 段：`net_impl2.cpp` 55~258、`net_impl_backend.cpp` 28~139、`collectOrtProfileData`（`net_impl.cpp` 2771~2900）
- `onnx_graph_simplifier.cpp`（2,440 行 pattern 规则）

### 2.4 关键代码位置速查

| 概念 | 位置 |
|---|---|
| SSA 值表 + 新引擎 API | `modules/dnn/src/net_impl.hpp:450-520` |
| 推理前置流水线 | `modules/dnn/src/net_impl2.cpp:592-601` |
| 执行器绑定 + 缓冲分配 | `modules/dnn/src/net_impl2.cpp:840-852` |
| 解释执行主循环 | `modules/dnn/src/net_impl2.cpp:1835-1932` |
| 输出分配（`getMemoryShapes`→`fit`） | `modules/dnn/src/net_impl2.cpp:855-915` |
| 新引擎仅支持 CPU/CUDA | `modules/dnn/src/net_impl_backend.cpp:330` |
| 算子注册表（226 条） | `modules/dnn/src/init.cpp:102-334` |
| importer 建图 `addLayer` | `modules/dnn/src/onnx/onnx_importer2.cpp:1299-1315` |
| dispatch 表（132 条） | `modules/dnn/src/onnx/onnx_importer2.cpp:3595-3728` |
| DNN HAL 钩子（4 个） | `modules/dnn/src/hal_replacement.hpp` |

---

### 2.5 SimdYolo 源码地图（`F:\code\SimdYolo`）

> **定位：第一实现参考。** 它已经跑通 YOLO11n，CPU **22.19 ms**（快于 ORT 26.4 ms），且是纯托管 C#。凡与本项目重合的部分，优先照搬其做法。
> 性能基线出处：`artifacts/runtime-comparison-0.11.2.md`、`artifacts/cpu-runtime-comparison.md`。

#### 2.5.1 顶层结构

| 路径 | 内容 |
|---|---|
| `src/SimdYolo/` | 核心库（约 90 个 .cs，见 2.5.2） |
| `src/SimdYolo.Cli/` | 命令行 |
| `src/SimdYolo.Training/` | 训练相关（**本项目不用**） |
| `docs/` | `inference.md` / `model-workflows.md` / `startup.md` / `training.md` |
| `artifacts/` | 十余份性能对比报告（CPU/GPU/ORT/PyTorch、启动、优化历程）→ §9 基线取值来源 |
| `tools/` / `tests/` / `models/` / `shaders/` | 脚本、测试、模型、Vulkan shader |
| `README.md` / `VALIDATION.md` | 能力与验证说明 |

#### 2.5.2 文件地图（行数为实测，关系列：①直接移植 ②改造移植 ③仅参考 ④不用）

**A. ONNX 加载与执行计划**

| 文件 | 行 | 作用 | 关系 |
|---|---|---|---|
| `OnnxReader.cs` | 260 | `Proto`（varint/fixed32/64/length-delimited/packed/未知字段/`ParseView` 零拷贝）、`OnnxNode` 属性读取、`OnnxGraph.Load`、`ReadTensor`（含 `external_data` 与路径逃逸校验） | ① **加 DType 维度** |
| `TensorStorageReader.cs` | 119 | 小端权重解码：F32/F16/BF16/I32/I64/I8/U8 | ① |
| `OnnxMetadataReader.cs` | 112 | 读输出 shape / 元数据 | ② 可选 |
| `Tensor.cs` | 134 | `float[]` + `int[] Shape`、`Reshape` 共享数组、`Temporary` 走池 | ③ 无 dtype / 无 `Fit` |
| `OnnxExecutionPlan.cs` | 78 | 名称→整数槽位、`remaining[]` 引用计数、逐节点 `Live` 存活集合、`Roots` | ② **= Arg 表 + 生命周期扫描** |
| `OnnxOptimization.cs` | 214 | 常量折叠 + `ShapeExecutionPlan`（固定输入形状下缓存常量结果与稳定 shape，不缓存数据输出） | ② **= `ConstFolder`** |
| `OnnxModel.cs` | 475 | 单个大 `switch` 逐 op 执行 + 图内融合伪 op（`_Dfl`/`_ConcatConv`/`_SigmoidConcat`/`_OnnxEndToEnd`） | ③ 执行器架构**不照搬** |
| `OnnxReduction.cs`(57) `OnnxHead.cs`(136) `OnnxConcatConv.cs`(59) `OnnxCpuZeroAdd.cs`(59) `OnnxSigmoidConcat.cs`(87) `OnnxEndToEnd.cs`(166) `OnnxYolo26Head.cs`(167) | 731 | YOLO 专用图改写与头融合 | ④ 后处理不进引擎 |

**B. 线程与内存**

| 文件 | 行 | 作用 | 关系 |
|---|---|---|---|
| `CpuTensorPool.cs` | 88 | 精确长度栈池（保留上限 256 MiB）+ `RetainOnly` 图边界存活扫描 + `DetachOutputs`（公开输出永久离池、重复别名克隆） | ① **强烈建议**（可替代 `BufferAllocator`） |
| `CpuExecutor.cs` | 338 | 后独自有工作线程，调用线程也参与，线程预算由后端控制 | ② Q10 候选 |
| `CpuScheduling.cs` | 64 | 在专用执行器与 `Parallel.For` 间选择，嵌套调度降为串行 | ② Q10 候选 |

**C. GEMM / Conv / 逐元素 kernel**

| 文件 | 行 | 作用 | 关系 |
|---|---|---|---|
| `CpuGemm.cs` | 280 | packed 16/24 列 AVX+FMA 微内核、batch offsets、常量权重可缓存 | ① |
| `CpuGemmPortable.cs` | 164 | 无 AVX 回退 | ① |
| `CpuPackedGemm.cs` | 150 | 打包 GEMM | ① |
| `CpuPointwise24.cs` | 260 | 4 行 × **24 列（空间）** 微内核，一次权重广播供 3 个向量 | ① |
| `CpuPointwiseAligned.cs` / `CpuPointwiseAlignedSmall.cs` | 97 / 115 | 按输出物理地址切 48 列（回退 16 列）任务，缓存行不共享；要求 `H·W % 16 == 0` | ② 需 flat 输出缓冲 |
| `CpuConvolution.cs` | 210 | **16 空间位置 panel** 卷积，融合 bias+SiLU，不物化整图 im2col | ① **替代计划的 im2col** |
| `CpuConvolutionPack.cs` | 241 | 3×3 s2 的偶/奇采样复用打包 | ① 仅 stride 2 |
| `CpuWinograd.cs` / `Winograd24.cs` / `Winograd4.cs` / `Winograd424.cs` / `WinogradPanels.cs` | 203/93/344/89/715 | F(2×3)、F(4×4)、4×2×4 及线程内面板 | ① M4 |
| `CpuDepthwisePadded.cs` | 137 | depthwise 3×3 | ① |
| `CpuSpatial.cs` | 396 | nearest resize、depthwise、pool、特定转置卷积 | ② 按需 |
| `CpuTranspose.cs` | 101 | 转置（RF-DETR 有 113 个 `Transpose`） | ① |
| `CpuActivation.cs` | 110 | SIMD `exp` / `sigmoid` / SiLU | ① |
| `CpuEpilogue.cs` | 32 | 偏置 + 激活融合 | ① |
| `CpuWeightBatch.cs` | 51 | `ConditionalWeakTable` 以权重数组为弱键缓存打包结果 | ① 可改为以 `byte[] Buffer` 为键 |
| `Compute.cs` | 653 | `IComputeBackend` 契约与后端派发 | ③ |
| `CpuAverageTwo.cs` | 66 | 两路平均 | ④ |
| `InferencePrecision.cs` | 174 | 卷积/矩阵乘操作数精度选择（FP16 等） | ④ 首版不做 |
| `KernelFusion.cs` | 22 | 重参数化卷积分支合并 | ④ |

**D. 后处理（本项目不进引擎，仅作 `tools/` 参考）**

`CpuDfl.cs`(59) `CpuHead.cs`(80) `CpuEndToEnd.cs`(213) `YoloDetector.cs`(522) `VisionDecoder.cs`(95) `CpuMaskDecoder.cs`(91) `MaskConverter.cs`(27) `CpuPose.cs`(60)

**E. 不用**

`MetalBackend.cs`(287) `Vulkan*.cs`(≈2,000) `DeviceExecution.cs`(188) —— GPU 后端；
`PtModel.cs`(498) `PickleReader.cs`(324) `ModelExporter.cs`(227) `SimdYolo.Training/*` —— `.pt` 加载与训练。

#### 2.5.3 算子覆盖对比（关键结论）

`OnnxModel` 的 switch 支持 **52** 个 ONNX 算子（含 `ConvTranspose/BatchNormalization/LeakyRelu/Tanh/Pow/Mod/AveragePool/GlobalAveragePool/Flatten/Squeeze/ReduceMean/Range/Identity` 等本项目不需要的）。

与本项目 42 个算子的关系：

| 分类 | 数量 | 说明 |
|---|---|---|
| **可直接参考实现** | **35 / 42** | 覆盖 YOLO 需要的全部 15 个算子（100%） |
| **必须自己写** | **7** | `And` `Not` `Erf` `Sin` `Cos` `GridSample` `LayerNormalization` |

> 这 7 个**全部只出现在 RF-DETR**。也就是说：SimdYolo 已经把 M2（YOLO）的路铺完了，M3（RF-DETR）的增量是真增量——`GridSample` 最重，`LayerNormalization` 次之，其余是小算子。

#### 2.5.4 术语辨析

`CpuPointwise24` 的 24、`CpuPointwiseAligned` 的 16、`CpuConvolution` 的 16 都是**空间（H·W）方向的列面板**，不是通道对齐；SimdYolo 全程 NCHW、无通道 padding。详见 §6.4 坑 #28。

---

### 2.6 SimdYolo 偏离点（需用户拍板）

> 完整决策表见 `plan.md §12.1`（Q8~Q11）。这里只记 SimdYolo 的原始做法与出处。

| # | 议题 | SimdYolo 原做法 | 出处 |
|---|---|---|---|
| Q8 | `Tensor` 内存模型 | `float[]` + `int[] Shape`，无 dtype | `Tensor.cs` |
| Q9 | 池化与输出所有权 | 精确长度池 + 边界扫描 + 输出永久离池 | `CpuTensorPool.cs`、`OnnxExecutionPlan.cs:45-55` |
| Q10 | 线程模型 | 专用 `CpuExecutor`，调用线程参与、禁止嵌套扩张 | `CpuExecutor.cs`、`CpuScheduling.cs` |
| Q11 | 输入/opset 约束 | 强制恰好 1 个 FP32 输入，opset 11~21 | `OnnxReader.cs:204`（`inputs.Length != 1` 抛异常）、`:200` |

---

## 3. 目标模型实测数据

### 3.1 基本属性

| 模型 | ir | opset | nodes | initializers | value_info | 输入 | 输出 |
|---|---|---|---|---|---|---|---|
| `yolo11n.onnx` | 10 | 19 | 320 | 196 | 332 | `images` [1,3,640,640] | `output0` [1,84,8400] |
| `yolov11s.onnx` | 9 | 17 | 320 | 196 | 332 | `images` [1,3,640,640] | `output0` [1,84,8400] |
| `rfdetr-small.onnx` | 8 | 17 | 1539 | 354 | **0** | `input` [1,3,512,512] | `dets` [1,300,4] + `labels` [1,300,91] |
| `rfdetr-medium.onnx` | 8 | 17 | 1654 | 380 | **0** | `input` [1,3,576,576] | 同上 |

> `value_info = 0` 说明 RF-DETR 未带中间张量形状 → 全靠引擎 `GetMemoryShapes` 推断；`ConstFolder` 必须自己会算 shape。

### 3.1.1 文件大小与实测基线（2026-11-07 复核）

| 模型 | 文件大小 | ORT 1.31.1 CPU 实测延迟 | 1:1 逐 Arg 激活内存合计 | 最大单张量 |
|---|---|---|---|---|
| `yolo11n.onnx` | **10,469 KB** | **26.4 ms** | **273.6 MB** | 6.25 MB |
| `yolov11s.onnx` | **37,139 KB** | 待测 | 待测 | — |
| `rfdetr-small.onnx` | **118,181 KB** | **186.6 ms** | **1381.0 MB** | 24.19 MB `[1,6,1028,1028]` |
| `rfdetr-medium.onnx` | **124,788 KB** | 待测 | 待测 | — |

- ❌ **旧文档错误**：本节曾写「`rfdetr-medium.onnx` 30MB+」，实际 **124 MB**（small 118 MB）。加载策略、AOT 体积、权重常驻内存都要按这个量级估算。
- 内存口径：用 `onnx.shape_inference` 推出全部节点输出形状后累加 `4 × 元素数`，即「每个 Arg 各占一块独立缓冲」的常驻量。**由此确认 `BufferAllocator` 必须提前**（RF-DETR 1.38 GB 不可接受），目标压到 < 400 MB。
- ORT 测试条件：`CPUExecutionProvider`，默认线程，预热 3 次取 10 次平均，batch=1 FP32。
- 另一台同型机器上 SimdYolo 官方报告 yolo11n ONNX CPU = **22.19 ms**（i7-8700K，12 线程上限），比 ORT 略快 → 托管实现是有希望追平 ORT 的。

### 3.1.2 形状可推断性（重要，可当 oracle）

对 `yolo11n` 与 `rfdetr-small` 跑：

```python
onnx.shape_inference.infer_shapes(m, strict_mode=False, check_type=False)
```

结果：**所有节点输出都推出了形状，未推断出的张量数 = 0**（含 rfdetr-small 的 1539 个节点、459 个 Constant、122 个 Reshape）。

→ 结论：**M1 的验收不必依赖编译 OpenCV 生成 `dumpToPbtxt`**，直接用 `onnx.shape_inference` 做逐张量形状 oracle 即可（`plan.md` §8.1）。

### 3.2 YOLOv11 算子分布（15 种）

```
Conv 88 | Mul 79 | Sigmoid 78 | Concat 23 | Add 16 | Split 11 | Reshape 8
MaxPool 3 | Transpose 3 | MatMul 2 | Softmax 2 | Resize 2 | Slice 2 | Sub 2 | Div 1
```

- `Sigmoid + Mul` = **SiLU 的分解形式** → 不需要 activation kernel
- `MatMul(2) + Softmax(2)` = **C2PSA 的 4D attention** → MatMul 必须支持 batch + broadcast
- 无 `BatchNormalization`（导出时已折叠进 Conv）

### 3.3 YOLOv11 Conv 变体

| 形态 | 数量 |
|---|---|
| 3×3 s1 | 31 |
| 3×3 s2 | 7 |
| 1×1 s1 | 42 |
| 3×3 depthwise s1 | 7 |

无 dilation≠1、无 group≠depthwise、无 5×5。

### 3.4 YOLOv11 其它关键属性

- `Resize`：**nearest / asymmetric / floor**（2 处）→ 不需要 bilinear/cubic
- `MaxPool`：**5×5 s1, ceil_mode=0**（3 处，SPPF）
- `Softmax`：`axis=-1` ×1、`axis=1` ×1

### 3.5 RF-DETR 算子分布（40 种，small）

```
Constant 459 | Add 169 | MatMul 140 | Reshape 122 | Mul 114 | Transpose 113 | Slice 62
LayerNormalization 49 | Sqrt 45 | Cast 36 | Div 35 | Shape 25 | Concat 24 | Unsqueeze 19
Softmax 18 | Erf 12 | Conv 9 | Expand 9 | Where 8 | Sigmoid 8 | Relu 8 | ConstantOfShape 7
Equal 6 | ReduceSum 4 | Gather 4 | Sin 4 | Cos 4 | Not 3 | Exp 3 | Gemm 3 | Sub 3
GridSample 3 | Tile 2 | Greater 2 | GatherElements 2 | Split 1 | Less 1 | And 1
ReduceMax 1 | TopK 1
```

（medium 同构，数量略增：Add 184、MatMul 151、Reshape 137、Transpose 125…）

### 3.6 RF-DETR Conv 变体（仅 9 个）

| 形态 | 数量 |
|---|---|
| 3×3 s1 (Cin=128) | 6 |
| **16×16 s16 (Cin=3)** patch-embed | 1 |
| 1×1 s1 (Cin=1536 / 640) | 2 |

无 depthwise、无 pooling。

### 3.7 RF-DETR 其它关键属性（实测）

| 项 | 值 | 数量 |
|---|---|---|
| `LayerNormalization` | `axis=-1`, `eps≈1e-6` | 37 |
| `LayerNormalization` | `axis=-1`, `eps≈1e-5` | 12 |
| `Gemm` | `transB=1, alpha=1.0, beta=1.0` | 3 |
| `GridSample` | `bilinear / zeros / align_corners=0` | 3 |
| `ReduceSum` | `keepdims=0` ×3、`keepdims=1` ×1 | 4 |
| `ReduceMax` | `keepdims=0` | 1 |
| `TopK` | `axis=1` | 1 |
| `Cast` | `to=FLOAT(1)` ×33、`to=BOOL(9)` ×2、`to=INT64(7)` ×1 | 36 |
| `Softmax` | `axis=-1` | 18 |
| `Erf` | — | 12（GELU 分解） |
| `Sin` / `Cos` | — | 4（正弦位置编码） |

### 3.8 形状链可折叠性（实测追踪）

追踪 `Where/Expand/ConstantOfShape` 的上游算子类型：

```
Constant 19 | ConstantOfShape 7 | Equal 6 | Where 6 | init 3 | Unsqueeze 3 | Cast 2 | Slice 1 | Reshape 1
数据相关上游：2 处（见下表修正）
```

**逐个数据相关节点的上游复查（2026-11-07）**

| 节点 | 上游 | 是否可折叠 |
|---|---|---|
| `/transformer/Where` | `Cast`, const, **`Reshape`** | ❌ 数据相关 |
| `/transformer/Where_1` | `Cast`, const, **`Transpose`** | ❌ 数据相关（**把 `Transpose` 纳入折叠集合即可消除**） |
| `/transformer/Where_2..5` | `Equal`, `ConstantOfShape`, const | ✅ |
| `/transformer/Expand` / `Expand_1` | const, const | ✅ |
| `/transformer/Expand_2..5` | `Unsqueeze`, `Where` | 随上游 |
| `/backbone/.../Where` / `Expand` | `Equal`/`Slice`/`ConstantOfShape`/`Where` | 随上游 |
| 全部 `ConstantOfShape` | const | ✅ |

**结论（修正）**
- 绝大多数 shape 链可被常量折叠消除 → `ConstFolder` 是必做项
- **实际数据相关 `Where` 是 2 处，不是 1 处**（原文档漏了 `Where ← Reshape`）
- 折叠集合里加上 **`Transpose`**（对常量做转置）后，预期只剩 1 处动态
- 剩余部分由 `Where / Expand / ConstantOfShape` 的 `DynamicOutputShapes` 兜底（语义见 `plan.md` §5.4：输出不预 `fit`，layer 自己 `fit`）

### 3.9 42 算子并集（**已修正，原列表漏 `Greater`**）

```
Add And Cast Concat Constant ConstantOfShape Conv Cos Div Equal Erf Exp Expand Gather
GatherElements Gemm Greater GridSample LayerNormalization Less MatMul MaxPool Mul Not
ReduceMax ReduceSum Relu Reshape Resize Shape Sigmoid Sin Slice Softmax Split Sqrt Sub
Tile TopK Transpose Unsqueeze Where
```

> 口径：`rfdetr-small` 40 种 + `yolo11n` 独有的 `MaxPool`、`Resize` = **42**。
> 原文档此列表数出 41，是因为漏写了 `Greater`。`rfdetr-small` 实测有 2 个 `Greater`，均在 transformer 的 mask 构造里：
> - `/transformer/Greater`：`Reshape × Constant → Greater → And`
> - `/transformer/Greater_1`：`ReduceSum × Constant → Greater → Not`
>
> 两者都属于 §3.8 的 shape/mask 链（可被常量折叠吃掉），但**必须实现**。因此 `plan.md` §7 的分批数应为 A9 + B3 + C17 + D5 + **E8** = 42。

---

## 4. ONNX protobuf 手写 reader 备忘

### 4.1 需要实现的 message 与字段号

| message | 字段（号=类型） |
|---|---|
| `ModelProto` | `ir_version`=1 varint、`opset_import`=8 msg、`graph`=7 msg |
| `OperatorSetIdProto` | `domain`=1 string、`version`=2 varint |
| `GraphProto` | `node`=1 msg、`name`=2 string、`initializer`=5 msg、`input`=11 msg、`output`=12 msg、`value_info`=13 msg |
| `NodeProto` | `input`=1 string、`output`=2 string、`name`=3 string、`op_type`=4 string、`attribute`=5 msg、`domain`=7 string |
| `AttributeProto` | `name`=1 string、`f`=2 fixed32、`i`=3 varint、`s`=4 bytes、`t`=5 msg、`g`=6 msg、`floats`=7 packed fixed32、`ints`=8 packed varint、`strings`=9 bytes、`tensors`=10 msg、`graphs`=11 msg、`type`=20 varint |
| `ValueInfoProto` | `name`=1 string、`type`=2 msg |
| `TypeProto` | `tensor_type`=1 msg |
| `TypeProto.Tensor` | `elem_type`=1 varint、`shape`=2 msg |
| `TensorShapeProto` | `dim`=1 msg |
| `TensorShapeProto.Dimension` | `dim_value`=1 varint、`dim_param`=2 string |
| `TensorProto` | `dims`=1 packed varint、`data_type`=2 varint、`float_data`=4 packed fixed32、`int32_data`=5 packed varint、`int64_data`=7 packed varint、`name`=8 string、`raw_data`=9 bytes、`double_data`=10 packed fixed64、`uint64_data`=11 packed varint、`external_data`=13 msg、`data_location`=14 varint |
| `StringStringEntryProto` | `key`=1 string、`value`=2 string |

### 4.2 wire type

| # | 类型 | 读取方式 |
|---|---|---|
| 0 | varint | LEB128，最多 10 字节 |
| 1 | 64-bit | 8 字节（double / fixed64） |
| 2 | length-delimited | varint 长度 + n 字节（string / bytes / 嵌套 message / packed） |
| 5 | 32-bit | 4 字节（float / fixed32） |

- 未知字段**必须能跳过**（不在上表中的字段按 wire type 跳过）。
- `packed=true` 的 repeated 数值字段：长度分隔的一段连续 varint / fixed32。
- proto2 的 `oneof`（`Dimension`、`TypeProto.value`）在 wire 层只是不同字段号，按实际出现的号判断。

### 4.3 数据落位规则

- `data_location == 1`（EXTERNAL）→ 按 `external_data` 的 `location/offset/length` 读外部文件（4 个目标模型都没有，但要留口）。
- 有 `raw_data` 优先用它（小端，按 `data_type` 解释）；否则按 `float_data / int32_data / int64_data`。
- `BOOL` 在 `raw_data` 里是**每元素 1 字节**；在 `int32_data` 里是每个 int32 一个元素。

---

## 5. C++ → C# 对照

| OpenCV / C++ | C# |
|---|---|
| `cv::Mat`（dims/size/step/data + `fit()`） | `Tensor`（`TensorShape`+`Strides`+`Memory<byte>`+`Fit()`） |
| `MatShape`（`std::vector<int>`） | `TensorShape`（inline ≤8 维 + `Span<int>`） |
| `cv::dnn::Arg` | `readonly struct Arg { int Idx; }` |
| `ArgData` | `sealed class ArgData` |
| `Ptr<T>` | 普通引用 + 可空标注 |
| `LayerParams`（`Dict` + `DictValue`） | `LayerParams`（`Dictionary<string,DictValue>`） |
| `DictValue`（多态 int/real/string/array） | `readonly struct DictValue`（tag + 数组） |
| `LayerInfo` / `Layer` | 同名类（`Layer : LayerInfo`） |
| `LayerFactory::createLayerInstance` | `LayerRegistry.Create(type, params)` |
| `cv::parallel_for_(Range, fn)` | `ParallelRange.For(total, (b,e)=>{})` |
| `AutoBuffer<T>` | `ArrayPool<T>.Shared` |
| `CV_SIMD`（`v_float32`） | `Vector<float>` / `Vector256<float>` |
| `CV_32F / CV_64S / CV_Bool` | `DType.F32 / I64 / Bool` |
| `CV_Assert` / `CV_CheckEQ` | `ThrowHelper`（显式检查，不走 `Debug.Assert`） |
| `CV_LOG_*` | `ILogger` 抽象 / `Trace` |

---

## 6. 坑清单（逐步补充）

### 6.1 ONNX opset 17/19 相关

1. **参数走 input 而非 attribute**：`Slice`、`Split`、`Resize`、`ReduceSum`、`Expand`、`Tile`、`TopK`(k)、`Pad`、`Clip` 的参数是张量输入（多来自 initializer）→ `LayerParams` 必须支持“从 const 输入取属性”。
2. **`Split`**：opset 13+ 的 `split` 是可选 input；缺省时按输出个数均分，除不尽时**最后一组更小**。
3. **`Slice`**：`starts/ends/axes/steps` 均可为负；`ends` 可能是 `INT64_MAX`（表示到末尾）。
4. **`Resize`**：`coordinate_transformation_mode=asymmetric` + `nearest_mode=floor` → `src = floor(dst * scale)`；与常见“四舍五入”实现不同，必须照 floor。
5. **`Reshape`**：`allowzero=0` 时 `0` 表示“复制输入对应维”，`-1` 表示推断；`allowzero=1` 时 `0` 是真正的 0。两个目标模型都带 `allowzero` 属性。
6. **`ReduceSum`**：opset 13+ 的 `axes` 是输入；`noop_with_empty_axes=1` 且 axes 为空 → 不归约（identity）；`=0` → 归约所有轴。
7. **`TopK`**：`k` 是 int64 输入张量；输出两个（values + indices）。
8. **`ConstantOfShape`**：`value` 属性是 TensorProto（取第 0 个元素填充）。
9. **`Cast` 到 BOOL(9)**：非零即 true；BOOL → float 时 true→1.0。

### 6.2 数值 / 形状

10. **`LayerNormalization` 两种 eps 同模型共存**（1e-6 / 1e-5）→ eps 必须逐节点取值。
11. **`Erf` 需要自己实现**：.NET BCL **没有** `Math.Erf`。用 Abramowitz-Stegun 7.1.26 近似（最大误差 ≈1.5e-7），或高精度双精度实现后转 float。GELU 的精度直接依赖它。
12. **`GridSample`**：`align_corners=0` 的坐标映射为 `((x+1)*W-1)/2`；`padding_mode=zeros` 时**四个邻点任一越界都按 0 处理**（不是 clamp）。
13. **MatMul 的 batch 广播**：ONNX 是前导维广播（numpy 规则），不是 2D。YOLO 的 `[1,h,T,64]×[1,h,64,T]`、RF-DETR 的 `[1,T,C]×[C,C]` 都要支持。
14. **广播右对齐**：逐元素算子要支持 标量 / `[C]` / `[1,1,H,W]` 混搭。
15. **`Softmax` 非末轴**（YOLO 有 `axis=1`）：不能假设最后一维。

### 6.3 工程

16. **`Tensor.Fit` 的别名风险**：缓冲复用后输出可能覆盖仍被下游引用的输入。首版**禁止 in-place**，靠 `BufferAllocator` 的 `useCount` + 生命周期区间决定复用时机（**M0.5 就要做，不是 M4**，见 §6.4 #20）。
17. **浮点累加顺序**影响 GEMM 的 MAE；卷积对齐 1e-4 即可，不追求 bit-exact。
18. **大文件（已修正）**：`rfdetr-medium.onnx` 实测 **124 MB**、`rfdetr-small.onnx` **118 MB**（旧文档写 30MB+ 是错的）。用 `File.ReadAllBytes` + `ReadOnlyMemory<byte>` 切片（或 `MemoryMappedFile`），不要每层从 `Stream` 读。注意 `Proto.ParseView` 借用模型数组时，**节点/属性这类长寿对象要复制成独立小数组**，否则整个 118MB 文件被图长期钉住（SimdYolo 已踩过，`OnnxReader.cs:206` 用 `Proto.Parse` 而非 `ParseView`）。
19. **`Span` 与 async 不要混用**：protobuf reader 全程同步 `Span` 实现。

### 6.4 本轮新增坑（2026-11-07 复核）

20. **内存：逐 Arg 1:1 缓冲完全不可行**。实测 `yolo11n` 273.6 MB、`rfdetr-small` **1381.0 MB**（`plan.md` §5.7）。必须在 M1 之前实现 `BufferAllocator`（冲突图）+ 全局 temp 池；否则 RF-DETR 直接 OOM 或疯狂 GC。
21. **`GetMemoryShapes` 只有形状，读不到常量值**。`Reshape/Slice/Split/Expand/TopK/Reduce*/ConstantOfShape` 全部需要读常量**值**才能推断（OpenCV 用 `getNetImpl(this)->argTensor(...)`，`slice2_layer.cpp:363-376`）。C# 侧 `LayerInfo` 必须持 `NetImpl` 回填，或把 `GetMemoryShapes` 形参改成张量。**这是 M1 的硬前置。**
22. **OP 的动态形状语义**：动态 op 的输出**不要**预先 `Fit`，直接把 `ArgData.Runtime` 交给 layer，由 layer 自己 `fit`（`net_impl2.cpp:1891-1898`）。当前 `Interpreter` 完全没走这条分支，M3 会直接崩。
23. **ORT 没有「加中间输出」的 API**。`SessionOptions` 无相关开关；`session.Run(inputs, outputNames)` 只能取模型 `graph.output` 里已声明的名字。要逐层比对，必须 python 侧用 `onnx` 把中间 node output 追加进 `graph.output` 另存 `*_probe.onnx`（见 §7.1）。
24. **`Split` 在 opset 18+ 有 `num_outputs` 属性**。原坑 #2 只写了「opset 13+ `split` 是可选 input」，漏了 18+ 的 `num_outputs` 路径（`Split-18` 允许既不传 `split` 也不传 `num_outputs`，按输出个数均分）。`yolo11n`（opset 19）实测仍走第 2 输入，但 parser 要两条路都支持。
25. **`Resize` 的 `axes` / `antialias`**：opset 18+ 新增，默认空/0。目标模型没有，但属性表要能安全忽略（未知属性不能报错）。
26. **零拷贝 `Reshape` 与缓冲分配冲突**。视图共享 `Buffer` 时，per-Arg 各自 `Fit` 会互相覆盖。规则：别名 Arg 必须由 `BufferAllocator` 绑到**同一 buffer**，且生命周期包含于源 Arg；否则退化为真实拷贝。
27. **`net8` 目标将于 2026-11-10 EOL**。双目标构建仍可，但 CI/文档不要承诺 net8 支持；`Vector256`/`Fma` 在 net8 与 net10 都可用，不构成障碍。**本项目已决定保留双目标不退役**（`plan.md` Q1）。
28. **别把「kernel 列面板」当成「通道对齐」**（我曾在此判断上出错，已撤回）。核对 SimdYolo 源码：
    - `CpuPointwise24`：4 行 × **24 列** AVX 微内核，列 = 空间位置 `N = H·W`；
    - `CpuPointwiseAligned`：按输出数组物理地址分配 48 列任务，条件 `n % 16 == 0`（n 同样是 `H·W`），目的是缓存行不共享；
    - `CpuConvolution`：一次物化 **16 个空间位置**的 panel；
    - `CpuWinogradPanels`：线程私有变换面板。
    
    以上**都不改张量布局**，全程 NCHW、无通道 padding。所以 SimdYolo 的 22ms 与图级 blocked layout（OpenCV NC8HW8）无关；本项目靠 M4 的 kernel 内打包即可复现，不必引入 `TransformLayout`。
29. **图级 blocked layout 的真实代价**（若将来真要做）：`graph_block_layout.cpp` 369 行等价物（`useBlockLayout()` + `getLayouts()` + 边界插 `TransformLayout` + `fuseTransformLayoutChains` + 重跑 `assignBuffers`）；每个参与算子要两套路径；padding 零通道会污染沿 C 的归约（`Softmax(axis=1)`、`LayerNorm(axis=-1)`、`Concat`、`MatMul` 的 K 维）→ 必须在这些点退出 BLOCK，否则数值错。RF-DETR 几乎全是这类算子，收益接近 0。

30. **`Resize` 的步长是 `1/scale`，不是 `in/out`**（上游 `490dc835` 修复，`resize2_layer.cpp`）：

    ```cpp
    // 上游注释：The step is 1/scale, not in/out: they differ whenever in*scale is not integral.
    if (sizes.empty() && !alignCorners && onnxScaleH > 0 && onnxScaleW > 0 &&
        coordTransModeE != CoordTransMode::TF_CROP_AND_RESIZE)
    {
        scaleHeight = 1.f / onnxScaleH;
        scaleWidth  = 1.f / onnxScaleW;
    }
    ```

    yolo11n 的 2 处 `Resize` 都是 2× 上采样（`in*scale` 为整数），两种写法结果相同 → 当前不受影响，但**实现必须按 `1/scale`**，否则换成非整数倍率（如 640→400）会错。另外上游 importer 会把常量 `scales` 折进 `zoom_factor_*`，我们的 handler 也应把常量 `scales` 搬进 `LayerParams`，避免在 shape 推断时反复读输入。`align_corners=1` 时保持 shape 推导的 `(in-1)/(out-1)`；`sizes` 与 `tf_crop_and_resize` 不使用该 scale。

---

## 7. 工具脚本

### 7.1 golden 生成（ORT 作 oracle，**方法已改**）

⚠️ **旧写法错误**：ORT **没有**「用 `SessionOptions` 加中间输出」的 API。`session.Run(inputs, outputNames)` 要求名字已在模型 `graph.output` 里声明，否则直接报错。

**正确流程：先在 python 侧改图，再在 C# 侧跑 ORT。**

`tools/scripts/make_probe.py`（把关心的中间节点提升成图输出）：

```python
import onnx, sys
m = onnx.load(sys.argv[1]); g = m.graph
WANTED = {"Conv", "MatMul", "Gemm", "Softmax", "LayerNormalization", "Resize", "GridSample"}
existing = {o.name for o in g.output}
for n in g.node:
    if n.op_type in WANTED:
        for out in n.output:
            if out and out not in existing:
                g.output.append(onnx.ValueInfoProto(name=out))
onnx.save(m, sys.argv[2])
```

`tests/OnnxSharp.GoldenGen/Program.cs` 骨架：

```csharp
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

// probeModelPath 由 make_probe.py 生成，graph.output 已含中间张量
var session = new InferenceSession(probeModelPath);
var inputMeta = session.InputMetadata;          // name -> NodeMetadata(dims, type)
var input = new DenseTensor<float>(new[] { 1, 3, 640, 640 });
// 确定性填充：固定 seed 的伪随机 或 真实图 letterbox 后的 NCHW
var inputs = new[] { NamedOnnxValue.CreateFromTensor(inputMeta.First().Key, input) };

using var results = session.Run(inputs);
foreach (var r in results)   // 现在包含中间层
{
    File.WriteAllBytes($"{outDir}/{Sanitize(r.Name)}.bin", ToBytes(r.AsTensor<float>()));
    // 同时把 name/dtype/shape 写进 index.json
}
```

注意：
- 中间张量名带 `/` 和 `.`，写文件名前必须 `Sanitize`。
- 探针模型体积会因额外输出而**不增加**（权重不变），但 ORT 会保留中间结果，内存上升。
- `TopK` 的 indices 是 I64，`Cast` 到 BOOL 的是 bool，不能一律 `AsTensor<float>()`。

### 7.2 模型体检（Python，已在本机可用）

```powershell
# 算子清单 + 属性
python C:\Users\king\AppData\Local\Temp\onnx_scan.py  F:\ai\models\yolo\yolo11n.onnx
python C:\Users\king\AppData\Local\Temp\onnx_scan.py  F:\ai\rfdetr\onnx_model\rfdetr-small.onnx

# Conv / Resize / LayerNorm 等变体分布
python C:\Users\king\AppData\Local\Temp\onnx_detail.py F:\ai\models\yolo\yolo11n.onnx

# shape 链可折叠性
python C:\Users\king\AppData\Local\Temp\onnx_datapath.py F:\ai\rfdetr\onnx_model\rfdetr-small.onnx
```

> 以上脚本为临时脚本（在 `%TEMP%`）。若要长期使用，可复制到 `tools/scripts/` 并纳入版本管理。

### 7.2.1 形状 oracle 与内存估算（本轮新增，可直接用）

```python
import onnx
from onnx import shape_inference
m = onnx.load(path)
inf = shape_inference.infer_shapes(m, strict_mode=False, check_type=False)
shapes = {vi.name: [d.dim_value for d in vi.type.tensor_type.shape.dim]
          for vi in list(inf.graph.value_info) + list(inf.graph.output)}
total = sum(4 * __import__("math").prod(shapes[o]) for n in m.graph.node for o in n.output
            if o in shapes and all(x > 0 for x in shapes[o]))
unknown = sum(1 for n in m.graph.node for o in n.output if o not in shapes)
print("1:1 激活内存 %.1f MB, 未推断张量 %d" % (total / 1048576, unknown))
```

实测：`yolo11n` 273.6 MB / 未知 0；`rfdetr-small` 1381.0 MB / 未知 0。
→ `shapes` 字典即可作为 M1 的逐张量形状验收基准。

### 7.3 常用命令

```powershell
cd F:\code\OnnxSharp
dotnet build
dotnet run --project tools/OnnxSharp.Cli -- dump-graph F:\ai\models\yolo\yolo11n.onnx
dotnet run --project tools/OnnxSharp.Cli -- run F:\ai\models\yolo\yolo11n.onnx --input test.bin
dotnet run --project tools/OnnxSharp.Cli -- bench F:\ai\rfdetr\onnx_model\rfdetr-small.onnx --iters 20
```

---

## 8. 许可与合规

- OpenCV 5 的 dnn 文件头为 OpenCV 经典 **3-clause BSD**（`// This file is part of OpenCV project.` + 顶层 `LICENSE`）。照抄/翻译算法需在源码中保留原始版权声明与许可文本。
- ONNX proto 定义（`opencv-onnx.proto` / `onnx.proto`）为 **Apache-2.0**（文件内 SPDX 标注）。
- 本项目 `OnnxSharp` 建议同样采用 Apache-2.0，并在 `NOTICE` 中列出 "Portions derived from OpenCV (BSD-3-Clause)" 与 "ONNX protobuf schema (Apache-2.0)"。

---

## 9. 当前代码结构（实测，M0）

```
F:\code\OnnxSharp\
├─ plan.md / misc.md / .gitignore
├─ OnnxSharp.sln
├─ src\OnnxSharp\OnnxSharp.csproj        net8.0;net10.0, AllowUnsafeBlocks, Nullable
│  ├─ Tensors\DType.cs                   元素类型 + SizeOf/IsInteger/IsFloat
│  ├─ Tensors\TensorShape.cs             形状（值类型，params int[]）
│  ├─ Tensors\Tensor.cs                  张量（Fit 复用 / AsSpan<T> / Clone）
│  ├─ Tensors\TensorOps.cs               广播 + 逐元素（F32）
│  ├─ Core\Arg.cs                        Arg / ArgKind / ArgData（含运行期 Tensor 槽）
│  ├─ Core\Graph.cs                      prog 线性表 + Inputs/Outputs
│  ├─ Core\LayerInfo.cs                  LayerInfo + Layer（执行器）
│  ├─ Core\LayerParams.cs                参数集合
│  ├─ Core\DictValue.cs                  属性值（int/float/string/数组/tensor）
│  ├─ Core\LayerRegistry.cs              算子工厂表 + EnsureBuiltins
│  ├─ Core\Interpreter.cs                解释循环
│  ├─ Core\NetImpl.cs                    Arg 表 + 图管理 + Forward
│  ├─ Core\Net.cs                        门面（ReadModel 待 M1）
│  ├─ Core\OnnxSharpException.cs
│  ├─ Threading\ParallelRange.cs         parallel_for_ 等价物
│  ├─ Ops\NaryEltwiseLayer.cs            Add/Sub/Mul/Div/Max/Min
│  └─ Ops\ActivationLayer.cs             Relu/Sigmoid/Sqrt/Exp/Neg/Abs
└─ tools\OnnxSharp.Cli\                  冒烟测试（当前：手工建图 Add+Relu）
```

命名空间约定：`OnnxSharp.Tensors` / `OnnxSharp.Core` / `OnnxSharp.Ops` / `OnnxSharp.Threading`。
（注意：**不要**用 `OnnxSharp.Graph` 作命名空间，会与 `Graph` 类型名冲突。）

**M0.5 必须改动的现有文件（复核结论）**

| 文件 | 现状问题 | 改法 |
|---|---|---|
| `Core\LayerInfo.cs` | `GetMemoryShapes` 只有 shapes，读不到常量值 | 加 `internal NetImpl? Net` 回填 + `Tensor In(int)`；或形参改张量 |
| `Core\Interpreter.cs` | ① 无 `DynamicOutputShapes()` 分支；② 每次 forward `Tensor.Allocate` 临时张量；③ 重 finalize 只比 shapes 不比 types | 补动态分支（不预 fit）、改用全局 temp 池、判据加 `inpTypes` |
| `Core\NetImpl.cs` | `GetArg(name)` 对未知名字静默新建 Temp Arg（拼错名字不报错） | 拆 `Declare` / `Require`；补 `BufferAllocator` 挂载点 |
| `Core\Graph.cs` | `Prog` 无法删除节点 / 重建 | 支持折叠后重建 + `UpdateGraphOfs` |
| `Tensors\Tensor.cs` | `Strides` 永远为 null，与「视图」目标不符；`Count`(long)→`ByteLength`(int) 有溢出路径 | 明确首版只做别名 + 生命周期约束；加溢出检查 |

冒烟测试输出：

```
graph      : Graph(main, 2 ops)
output     : Tensor(F32 [2,3])
values     : 1.5, 0, 2, 4.5, 0, 5
SMOKE TEST: PASS
```

---

## 10. 变更记录

| 日期 | 内容 |
|---|---|
| 2026-11-07 | 建立文档；写入 OpenCV 5.2 源码实测规模、4 个目标模型算子/属性实测、protobuf 字段号表、坑清单 |
| 2026-11-07 | M0 骨架落地：解决方案 + 类库（双目标）+ 张量/Arg/图/参数/注册表/解释器 + CLI 冒烟测试通过 |
| 2026-11-07 | **第二轮复核修订**：① 模型实际大小（small 118MB / medium 124MB，原写 30MB+）；② 新增 §3.1.1 ORT 延迟与 1:1 内存实测（273.6 MB / 1381.0 MB）；③ 新增 §3.1.2 形状可推断性（未知 = 0，可作 oracle）；④ 数据相关 `Where` 由 1 处修正为 2 处并给出上游表；⑤ §7.1 golden 生成改为 python 改图（ORT 无「加中间输出」API）；⑥ 新增 §6.4 坑 #20~#27（内存、常量反查、动态形状语义、ORT 探测、Split num_outputs、Resize axes、零拷贝别名、net8 EOL）；⑦ §9 补 M0.5 待改文件清单；⑧ 平台决策：WASM/IL2CPP 不支持且不在路线图、首版不验 AOT |
| 2026-11-07 | **决策落定**：① `net8.0;net10.0` 双目标**保留不退役**；② 允许 **kernel 内部**打包/padding/缓存行对齐（对图透明），**不做图级 blocked layout**，仅预留 `Layout` 接口；③ 撤回错误判断「SimdYolo 22ms 依赖 channel 对齐」，新增坑 #28/#29 |
| 2026-11-07 | 新增 **§2.5 SimdYolo 源码地图**（约 90 个 .cs 按 5 组分类 + 行数 + 移植关系）与 **§2.6 偏离点**；关键结论：**42 个算子中 35 个可直接参考 SimdYolo，需自写的 7 个（`And/Not/Erf/Sin/Cos/GridSample/LayerNormalization`）全部只出现在 RF-DETR**；`plan.md` 增 §1.5 参考优先级与 §12.1 待拍板 Q8~Q11，M0.5 增布局接口预留任务 |
| 2026-11-07 | 新增 **§11 上游 OpenCV 5 dnn 变更跟踪**：基线 `5.x @ 43f8b1f5`（2026-11-06）、实测变更规模（6 个月 333 条 / 30 天 99 条）、只读 fetch 命令集、路径→本项目模块映射表、改动分类处理规则、跟踪记录模板，并完成首批 4 条评估（含 `resize2_layer` 的 `step = 1/scale` 修复） |
| 2026-11-07 | §11.1 升级为**跟踪版本台账（单一事实源）**：当前跟踪提交 + 提交日期/主题 + 校验命令（`rev-parse` / `log 43f8b1f5..origin/5.x`）+ **版本推进记录表（只增不改）**；`plan.md §13.1` 改为引用该台账 |

---

## 11. 上游 OpenCV 5 dnn 变更跟踪

> **长期机制**：后续的算子更新、性能优化、bug 修复，一律先看 OpenCV 5.x `modules/dnn` 的提交，再决定是否移植。
> 参考库是**只读**的：只 `fetch` 与 `diff`，不在里面改代码、不切分支。

### 11.1 跟踪版本台账（**单一事实源**）

本项目「跟到了哪个版本」**只记在这一节**，其它文档一律引用本节，不重复写 commit 号。

**当前跟踪版本**

| 项 | 值 |
|---|---|
| 参考仓 | `F:\code\opencv` |
| 分支 | `5.x` |
| **跟踪提交** | `43f8b1f5dacd967e5f632dd4a45109f182d41dce`（简写 `43f8b1f5`） |
| 提交日期 | 2026-11-06 |
| 提交主题 | `Merge pull request #30701 ... bump_release_to_5.1_...` |
| 记录日期 | 2026-11-07 |
| 语义基准 | 42 个算子的行为、shape 推断规则、属性解析以该提交为准；之后的每个提交逐条判定是否采纳 |

**校验命令（一秒确认是否落后）**

```powershell
# 1) 记录值 vs 实际 HEAD
git -C F:\code\opencv rev-parse --short HEAD

# 2) 拉远端引用后，统计从「跟踪提交」到现在多了多少 dnn 提交
git -C F:\code\opencv fetch origin 5.x
git -C F:\code\opencv --no-pager log --oneline 43f8b1f5..origin/5.x -- modules/dnn |
    Measure-Object -Line

# 3) 只看其中影响我们路径的部分
git -C F:\code\opencv --no-pager log --oneline 43f8b1f5..origin/5.x -- `
    modules/dnn/src/layers modules/dnn/src/onnx modules/dnn/src/layers/cpu_kernels `
    modules/dnn/src/net_impl2.cpp modules/dnn/src/graph_const_fold.cpp
```

**版本推进记录（只增不改，每次同步追加一行）**

| 日期 | 从 | 到 | dnn 提交数 | 已采纳 | 备注 |
|---|---|---|---|---|---|
| 2026-11-07 | — | `43f8b1f5` | — | — | 建立基线；首批评估 4 条，见 §11.6 |

> 推进规则：
> 1. 一次同步**只推进一次**，`到` 列写新的跟踪提交；
> 2. `已采纳` 填该批次在 §11.6 中标记为「采纳」的条目数；
> 3. 推进后需同步更新：本表 → `plan.md §13.1` 的基线行 → §11.6 记录表；
> 4. 提交号一旦记入推进表就**不再修改**（历史可追溯），只允许追加新行。

### 11.2 变更规模（实测，用于定同步节奏）

| 范围 | 提交数 |
|---|---|
| `modules/dnn` 近 6 个月 | **333** |
| `modules/dnn` 近 30 天 | **99** |
| 其中 `src/onnx`（6 个月） | 62 |
| 其中 `src/layers`（6 个月） | 188 |

> 平均每天 3 条以上，**必须按路径 + 关键词过滤**，不可能逐条跟。

### 11.3 同步命令（只读）

```powershell
# 0) 记录当前基线（写进 §11.1 / 提交说明）
git -C F:\code\opencv log -1 --format="%h %ad %s" --date=short

# 1) 只更新远端引用，不动工作区
git -C F:\code\opencv fetch origin 5.x

# 2) 只看 dnn 的全部提交
git -C F:\code\opencv --no-pager log --oneline --date=short `
    --format="%h %ad %s" origin/5.x -- modules/dnn

# 3) 只看会影响我们的路径
git -C F:\code\opencv --no-pager log --oneline origin/5.x -- `
    modules/dnn/src/layers modules/dnn/src/onnx modules/dnn/src/layers/cpu_kernels `
    modules/dnn/src/net_impl2.cpp modules/dnn/src/graph_const_fold.cpp `
    modules/dnn/src/graph_const_args.cpp modules/dnn/src/graph_buffer_allocator.cpp

# 4) 按关键词过滤（算子级）
git -C F:\code\opencv --no-pager log --oneline -i --grep="resize" origin/5.x -- modules/dnn
git -C F:\code\opencv --no-pager log --oneline -i --grep="layernorm\|gridsample\|topk" origin/5.x -- modules/dnn

# 5) 看某条提交改了什么
git -C F:\code\opencv --no-pager show --stat <hash>
git -C F:\code\opencv --no-pager diff <hash>^1 <hash> -- modules/dnn/src/layers/xxx.cpp
```

### 11.4 路径 → 本项目模块映射

| OpenCV 路径 | 本项目 | 关注点 |
|---|---|---|
| `src/net_impl2.cpp` | `Core/Interpreter.cs`、`Core/NetImpl.cs` | 执行循环、`allocateLayerOutputs`、`finalize` 重入条件、动态形状 |
| `src/net_impl.hpp` | `Core/Arg.cs`、`Graph.cs`、`LayerInfo.cs` | `Arg/ArgData` 字段、`getNetImpl` 反查 |
| `src/graph_const_fold.cpp` / `graph_const_args.cpp` | `Engine/ConstFolder.cs` | 折叠算子集合、常量释放时机 |
| `src/graph_buffer_allocator.cpp` | `Engine/BufferAllocator.cs` | 复用策略（`assignBuffers`） |
| `src/onnx/onnx_importer2.cpp` | `Onnx/OnnxImporter.cs` + `Handlers/` | 属性映射、opset 兼容、常量 scale 折进 params |
| `src/layers/nary_eltwise_layers.cpp` | `Ops/NaryEltwise*.cs`、`Tensors/TensorOps.cs` | 广播规则、比较算子语义 |
| `src/layers/conv2_layer.cpp` + `cpu_kernels/convolution.cpp` | `Ops/Conv*.cs`、`Kernels/Conv*` | shape 推断、kernel 选择（**kernel 实现优先 SimdYolo**） |
| `src/layers/resize2_layer.cpp` | `Ops/ResizeLayer.cs` | nearest/floor/scales（R7 高危区） |
| `src/layers/slice2/reshape2/split2/expand2/topk2` | 同名 `Ops/` | 常量输入反查 |
| `src/layers/layer_norm_layer.cpp` / `gridsample_layer.cpp` / `topk_layer.cpp` | `Ops/`（自写，SimdYolo 无） | M3 重点 |
| `src/layers/cpu_kernels/fast_gemm.*` | `Kernels/Gemm*` | 仅当 SimdYolo 版本更差时才采纳 |
| `src/graph_fusion_*.cpp` | M6 候选 | 首版不跟 |

### 11.5 改动分类处理规则

| 上游改动类型 | 处理 |
|---|---|
| 数值 / shape **bugfix**（Resize scales、LayerNorm eps、GridSample 边界、广播规则） | **必修**，并补回归用例 |
| 新增 conformance 测试 | 抄用例进 `tests/OnnxSharp.Tests`（是免费的边界清单） |
| 性能优化（tiling / Winograd / 融合） | 先看 SimdYolo 是否已有等效或更好的实现；没有再采纳 |
| 新增算子 | 记入 M6 候选，**不改首版范围** |
| API / 枚举 / 命名重命名 | 忽略（不追 OpenCV 的对外 API） |
| int8 / fp16 / QDQ / LLM（SDPA、KVCache） | 忽略（非目标） |
| 纯测试 / CI / 文档 | 忽略 |

### 11.6 跟踪记录（模板 + 首批样例）

| 日期 | 提交 | 主题 | 影响模块 | 结论 |
|---|---|---|---|---|
| 2026-11-07 | `43f8b1f5` | 基线（`5.x` HEAD，2026-11-06） | — | 记为语义基准 |
| 2026-11-07 | `490dc835` | fix: dnn resize scales downsample | `resize2_layer.cpp` | **已评估、需采纳写法**：步长是 `1/scale` 而非 `in/out`，二者仅在 `in*scale` 非整数时不同。yolo11n 是 2× 上采样（整数）→ 当前数值不受影响；但实现按 `1/scale` 写，且 handler 要把常量 `scales` 折进 `LayerParams`（对应 `zoom_factor_*`） |
| 2026-11-07 | `4f4d6125` | dnn conformance: `CastLike`（FP8/FP4/INT4） | `onnx_importer2` | 忽略（非目标类型） |
| 2026-11-07 | `ee27ba20` | `AveragePool` with dilations conformance | `pool` | 忽略（不用 `AveragePool`） |

### 11.7 节奏

- **每个里程碑开始前**跑一次 §11.3 的第 3 条命令，把新条目按 §11.5 归类；
- **每个里程碑结束时**把「已采纳 / 已忽略」写入 §11.6，采纳项必须带回归用例；
- 基线提交号变更时同步更新 §11.1 与本项目版本号；
- 与 SimdYolo 的优先级冲突（例如上游做了融合、SimdYolo 已用别的办法更快）时，按 `plan.md §1.5` 的规则交给用户拍板。
