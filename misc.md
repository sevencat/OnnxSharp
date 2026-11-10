# OnnxSharp 杂项

> 与 `plan.md` 分离的"事实库"：实测数据、源码地图、环境、坑、脚本、备忘。
> 本文件允许频繁追加，不追求结构性。

---

## 1. 开发环境（实测）

| 项 | 值 |
|---|---|
| CPU | **AMD Ryzen 7 8845HS**（8C/16T，Zen4，AVX2 + **AVX-512**，L1D 32 KiB / L2 1 MiB / L3 16 MiB） |
| 内存 | 47 GB |
| .NET SDK | `7.0.410`, `8.0.416`, `10.0.101`, `10.0.402` |
| .NET 运行时 | `Microsoft.NETCore.App 7.0.20 / 8.0.22 / 10.0.1 / 10.0.13` |
| Python | `3.11.17` |
| onnx | `1.23.2` |
| onnxruntime | `1.31.1`（golden oracle + 性能对照组） |
| cv2 | **5.0.0**（仅 `bench/` 里作 OpenCV dnn 性能对照组，**不作 oracle**） |
| OpenCV 参考源码 | `F:\code\opencv`（**5.2.0-dev**） |
| SimdYolo | `0.11.3`（`CPU SIMD (256 bits)`，性能对照组） |
| 本项目路径 | `F:\code\OnnxSharp` |

> ⚠️ 测量机是**共享环境**（后台常驻约 1.5 核负载），同代码连跑三次能差 ±30%。做 A/B 必须交替跑（见坑 #37），绝对数字只作趋势看。
>
> 机器上限用 `fma` 实测（纯 FMA 微循环，不是理论推算）：**单线程 91 GFLOPS、16 线程 753 GFLOPS**。
> GEMM 单算子能到 300~500 GFLOPS，即 16 线程峰值的 40%~66%；**别再用「单核 136 GFLOPS = 峰值」这类推算下判断**，实测单线程只有 91。

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

### 3.10 RF-DETR 的 MatMul 构成（2026-11-07 实测，已修正）

> **上一版这里写的「140 个 MatMul 全是 batched attention、MLP 已导出成 Conv」是错的。**
> 按 `(a[-2:], b[-2:])` 重新统计后，MLP 与投影都是货真价实的 `MatMul` 节点。

`rfdetr-small` 的 140 个 `MatMul`，合计 **54.1 GFLOP**：

| 形状（前置 batch） | 组数 | 合计 FLOPs | 出现位置 |
|---|---|---:|---|
| `(257,384)×(384,1536)`，batch **4** / `(257,1536)×(1536,384)`，batch **4** | 各 12 | 各 14.55 G | ViT-S 主干 MLP |
| `(257,384)×(384,384)`，batch **4** | 12 | 10.00 G | 主干 qkv |
| `(1028,384)×(384,384)` | 若干 | 3.64 G | 全局 block 的 qkv |
| `(6, 1028, 64, 1028)` / `(6, 1028, 1028, 64)` | 各 6 | 各 2.43 G | 主干 attention 的 `QKᵀ` 与 `AV` |
| `(24, 257, 64, 257)` / `(24, 257, 257, 64)` | 各 24 | 各 1.83 G | 主干 windowed attention |
| `[300,1,256]×[256,256]` | 9 | 可忽略 | decoder q/k/v 投影（**m=1 的假 batch，见下**） |
| `(8, 300, 32, 300)` / `(8, 300, 300, 32)` | 3 组 | 可忽略 | decoder cross-attention |

> ⚠️ **算 FLOPs 必须乘上前置 batch**。主干 MLP/投影都是 batch 4（windowed attention 的 4 个窗口并行），
> 漏掉就得到 16.6 GFLOP，进而算出「in-situ 只有 125 GFLOPS、比裸内核慢 3 倍」的假象。
> 按 54.1 GFLOP 算，in-situ 是 **407 GFLOPS**，已经不低于同形状的独立基准（386 GFLOPS）。

**含义**：
- 卷积优化（Winograd / 面板打包）对 RF-DETR 只影响 **3.4%**，别再往那儿投；
- attention 那几组的 `k` 只有 32/64，**K 方向喂不满流水线**，且 batch 维度是串行循环 → 24 个连续的并行屏障；
- `[300,1,256]×[256,256]` 是最典型的坑：ONNX 里 batch 维是动态的（静态推导为 0，运行时为 1），
  正确切法是 `(batch=300, m=1)`，按 batch 逐个调内核会让 256 KB 权重被打包 300 遍，单算子 4.06 ms。
  **`(batch, m)` 合并成一次 GEMM 后 0.13 ms**，已修复，端到端 -24%；
- 逐元素 `Add 169 + Mul 114 + Div 35 + Sqrt 45` 合计约 20%，每次都是一遍全量读写（实测 56 GB/s，已接近单线程 memcpy 上限 73 GB/s）。**这是当前第一优先级的可优化项**。
- 实测吞吐：独立基准 `257×384×1536` 386 GFLOPS、`1028×64×1028` 276~345 GFLOPS；机器 16 线程 FMA 峰值 753 GFLOPS。
  真实模型里 MatMul 合计 133 ms / 54.1 GFLOP = **407 GFLOPS**，已经没有「in-situ 落后于裸内核」这回事。详见 `bench.md` 结论 3。
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

1. **参数走 input 而非 attribute**：`Slice`、`Split`、`Resize`、`ReduceSum`、`Expand`、`Tile`、`TopK`(k)、`Pad`、`Clip` 的参数是张量输入（多来自 initializer）→ `LayerParams` 必须支持"从 const 输入取属性"。
2. **`Split`**：opset 13+ 的 `split` 是可选 input；缺省时按输出个数均分，除不尽时**最后一组更小**。
3. **`Slice`**：`starts/ends/axes/steps` 均可为负；`ends` 可能是 `INT64_MAX`（表示到末尾）。
4. **`Resize`**：`coordinate_transformation_mode=asymmetric` + `nearest_mode=floor` → `src = floor(dst * scale)`；与常见"四舍五入"实现不同，必须照 floor。
5. **`Reshape`**：`allowzero=0` 时 `0` 表示"复制输入对应维"，`-1` 表示推断；`allowzero=1` 时 `0` 是真正的 0。两个目标模型都带 `allowzero` 属性。
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

31. **`TensorProto.DataType` 从 `FLOAT = 1` 开始，不是 2**（`UNDEFINED = 0` 占位）。完整表：`1 FLOAT / 2 UINT8 / 3 INT8 / 4 UINT16 / 5 INT16 / 6 INT32 / 7 INT64 / 8 STRING / 9 BOOL / 10 FLOAT16 / 11 DOUBLE / 12 UINT32 / 13 UINT64 / 16 BFLOAT16`。按「从 0 数」写会整体偏移一位，两个目标模型都会在**第一个 conv 权重**（`model.0.conv.weight`）上报 `unsupported data_type 1`——这是最快的自查信号。首版只接受 1/2/3/6/7/9/10，其余显式报错而不是静默退化。

32. **RF-DETR 是双输出**：`dets:F32[1,300,4]` + `labels:F32[1,300,91]`（早前文档只记了 `dets`）。因此 `Net.Forward()` 的无参重载对 RF-DETR 不成立，验收脚本一律用 `Forward(name)` / `ForwardAll()`。两模型输入都是**静态形状**（RF-DETR 是 `[1,3,512,512]`，不是动态 512 可变）。

33. **`Constant` 节点才是 I64/Bool 的主来源**：RF-DETR 的 459 个 `Constant` 里有 379 个 I64 + 80 个 F32 张量，而 354 个 initializer 全是 F32。importer 必须两条路都支持（initializer→Const Arg、`Constant` 节点→Const Arg），否则 `Shape/Slice/Gather/Where` 全部拿不到索引值。

34. **重复字符串字段里的空串是「占位符」，绝不能过滤掉**。ONNX 用 `""` 表示「这个可选输入不存在」，但**位置必须保留**：yolo11n 的 Resize 是 `Resize(X, "", scales)`，把空串过滤掉后 `scales` 会滑到 index 1（`roi`），handler 按 `inputs[2]` 取 scales 就取不到，报「neither scales nor scales provided」。同理 `Split(X, "", ...)`、`Resize` 的 `sizes`。解析层一律原样保留（见 `OnnxFile.Strings`），由各 handler 用 `Inputs[i].Idx != 0` 判断占位。

35. **`ArgData.Shape` 与 `Tensor.Shape` 不是一回事**。共享缓冲槽会被后续算子反复 `Fit`，所以 `GetArgTensor(arg).Shape` 读到的是**该槽最后一次写入的形状**，而 `GetArgData(arg).Shape` 才是「这个 Arg 被生产时」的形状。凡是「事后」检查/打印中间张量形状的地方（形状对齐测试、`dump-graph`）都必须读 `ArgData`；第一版 `dump-graph` 就是踩了这个坑，打印出满屏相同的形状。

### 6.5 性能测量与内核调优的坑（2026-11-07）

36. **.NET 分层编译会让「按次数预热」的基准低 4 倍**（本轮最贵的一课）。热点方法要被调用几十次才会提升到 tier-1，tier-0 代码没有内联、没有 AVX 向量化。`gemmbench` 原来固定预热 3 次，同一个 `1028×64×1028` 测出 **50 GFLOPS，真实值 490 GFLOPS**。之前据此得出的「MatMul 只有 32~58 GFLOPS」「k 方向撞 L1」等结论**全部作废**。
    ```csharp
    // 正确做法：按时间预热，不是按次数
    var warm = System.Diagnostics.Stopwatch.StartNew();
    do { Run(); } while (warm.Elapsed.TotalMilliseconds < 400);
    ```
    推论：任何只看 5~10 次迭代的数字都不可信；同时 `--iters` 太小时中位数仍会被 tier-0 样本污染。`env COMPlus_TieredCompilation=0` 可以让结果更"干净"，但**那测量的不是实际运行时的性能**，不要用它出结论。

37. **A/B 必须在同一会话里交替跑，否则会被后台负载骗**。测量机后台有 ~1.5 核常驻负载，同一份代码连跑 3 次能得到 197 / 243 / 331 GFLOPS。做法：把两个版本各编一份二进制到 `bench/bin/{plain,unroll}/`，然后**交替**调用，取中位数。只看单侧「改前 vs 改后」的两次独立测量，5~10% 的差异完全是噪声。

38. **已验证无效的内核优化方向（不要再做）**：
   - **B 全量预打包成 `[列块][k][16]` 供所有行块复用**：`1028×64×1028` 从 304 掉到 234 GFLOPS。原因是行块高 40 行，打包成本本来就只占 5%，却多付了一整份 B 的写回 + 每次 `Multiply` 一次 `ArrayPool` 大块租借。
   - **微内核 k 方向 2 步展开**：单线程 136 → 135 GFLOPS，零收益。JIT 的循环展开 + 8 个独立累加器已经把 ILP 吃满，瓶颈在 FMA 端口不在循环开销。
   - **行块高度 `mb` 扫参**（16/32/64/128，含按物理核数而非逻辑核数分配的变体）：微基准 250~340 GFLOPS 之间波动，端到端交替 A/B 无差异。
   - **按流量把切分方向从行切换成列切**：微基准 `257,1536,384` 343 → 499 GFLOPS（+45%，看着很香），端到端交替 A/B 三轮完全一致（273~277 ms 双方相同）。

   先把「单线程跑出多少」测出来再决定方向：机器纯 FMA 上限用 `fma --threads 1` 量（实测 91 GFLOPS 单线程、753 GFLOPS 16 线程），**GEMM 单算子能到 300~350 GFLOPS 就说明微内核已经不是主要矛盾**，剩下的只能在分块与线程调度上找。

40. **独立内核吞吐无法预测端到端（本轮第二次踩，最值钱的一条）**。上面 38 里的后两条、以及 MatMul 的分块实验，都出现过「微基准明显变快、端到端纹丝不动」。
    原因不是测量方法，而是**真实模型里运行的是另一套内存压力**：微基准反复跑同一组数据，权重留在 L3；真模型里十几个算子的中间结果会把它挤出去。
    所以：
    - 微基准只能用来**排除明显不合理的实现**（比如 50 GFLOPS 那种）；
    - 任何方案是否上线，必须用「同会话、两个二进制交替跑端到端」来判定（见坑 37）；
    - 端到端噪声 ±15%~20%，单次测量毫无意义。同一个 `yolo11n` 在不同时段测出 36.5 / 43.4 / 50.2 ms。

41. **`profile` 的两个陷阱**（`Net.ProfileAll` 已按这两条改造）：
    - **必须预热**：前几轮迭代跑 tier-0 代码，而 tier-0 的 MatMul 比 tier-1 慢约 4 倍（逐元素只慢 2 倍），于是计算密集算子的份额被系统性高估 —— 不加预热得到「MatMul 占 66%」，预热后是 60%。
    - **对账要用均值口径**：各算子均值之和（rfdetr-small 251 ms）等于端到端中位数（250~275 ms）。
      每算子最小耗时也有用（看单个算子的稳态性能），但**最小值之和只是下界（215 ms），不能拿去和中位数比** ——
      每个算子的最小值基本不会同时出现，差的 15% 不是「隐藏开销」。
    - **后台负载**会把均值抬高 20% 以上，所以同一份代码不同时段能测出不同绝对值，跨会话不要比绝对值。
    - 计时开销本身可以忽略（每算子约 100 ns），别去怀疑 `Stopwatch`。

39. **`Parallel.For` 的固定开销约 7 µs**，小算子开线程反而更慢。当前阈值：`GemmKernel.ParallelThreshold = 1<<22` 乘加、`TensorOps.ParallelElementThreshold = 1<<16` 元素。副作用是 `300×32×300`（2.9M 乘加）这类形状被判定串行，只能跑 60 GFLOPS —— 更好的做法是「按 batch 并行」而不是整算子串行。

### 6.6 逐元素融合（2026-11-07，全部踩过）

42. **`Vector256<float>[]` 用运行期下标访问 = 全部溢到栈上**。通用「寄存器机」内核（`regs[dst] = op(regs[a], regs[b])`）看着很优雅，
    但 JIT 不可能把「下标是变量」的数组元素提升到 YMM 寄存器，每一步都是「写回栈、下一步再读回来」，
    再叠加每步一次 `switch` 分发 —— 实测比不融合还慢：每条 GELU 融合层 1.65 ms，而拆成五个算子是 1.50 ms，
    **省下的 4 遍读写被内核自身的开销全部吃回去**。
    修法不是优化通用内核，而是**给固定形状的链手写内核**（中间值全是本地变量）：1.65 → 0.30 ms，一步到位。
    推论：算子融合的正确形态是「模式识别 + 代码生成/手写内核」，通用解释器（interpreter loop over steps）在 SIMD 上基本没用。
43. **融合链的点数要设门槛**。2~3 节点的链只省 1~2 遍读写，抵不过 `GCHandle` pin×4、闭包分配、数组索引这些固定开销：
    yolo 系列（链短、张量小）长跑 A/B 里 median 慢 4%。现在规则是
    `chain.Count < 4 && shape.Total() < 1<<18` 就不融合；加上之后 yolo11n 变成 median -5%（不再是负收益）、rfdetr 不受影响。
44. **GELU 在 ONNX 里是 `Div → Erf → Add → Mul → Mul`，顺序容易记错**。
    实际导出是 `x/√2 → Erf → (erf+1) → x·(erf+1) → ·0.5`，
    不是 `Mul(x,0.5)` 在前。做模式匹配时**不要硬编码操作数位置**（ONNX 不保证 `Mul` 的左右序），
    也不要假设「被乘数就是 Div 的被除数」—— 要显式校验两个操作数指向同一个寄存器，否则会静默算错。
    另外注意 `FusedEltwiseLayer` 构造函数里 `Inputs` 还是空的，`nIn` 只能用 `regCount - steps.Length` 反推。
45. **融合的正确性不能只靠 golden 测试**。golden 只覆盖 yolo11n，而 rfdetr 上的 GELU 链是最激进的改动；
    本轮曾出现过 8 节点错误链（同一节点重复 8 次）却没触发任何测试失败。
    现在 `run --dump <dir>` 会把每个输出按 float 落盘，`bench/results/ab_dump.ps1` + `cmp_all.py`
    对 base/fused 做逐值比对（rfdetr-small/medium + yolo11n/yolov8s 全部 max|diff| = 0）。
46. **沿「消费者」前进，不是沿「生产者」**。第一版用 `producer[outIdx]` 找下一个节点，拿到的是节点自己，
    于是构出「同一节点重复 N 次」的链（还真的跑出了结果）。正确做法是建 `consumerOf[argIdx]`
    （一个 arg 只有一个消费者时才能延伸），并且要显式 `chain.Contains(next)` 防环。
47. **「算得对不对」的 bug 好抓，「写得顺不顺」的 bug 很难发现——它不报错、结果也全对，只是慢 2 倍**。
    `GridSampleLayer.Forward` 原来按 `n → oy → ox → c` 遍历，看着很自然（先算一个采样点的坐标，再给所有通道用），
    但输出是 NCHW：最内层的 `c` 每次跳 `outH*outW` 个元素，**每写一个 float 就换一个缓存行**。
    改成 `n → c → (oy,ox)` 后目的地址连续、源平面一轮内驻留，单个算子 1.6 → 0.92 ms。
    - 规律：**最内层循环必须对应张量最内层的连续维度**。NCHW 里就是 `W`，所以 `c` 永远不该在最内层。
    - 同一类 bug 还藏在「先算标量、再广播到所有通道」的写法里 —— 坐标/索引这类与通道无关的量要**预算到数组**，
      否则每个通道重算一遍（本例是 2 次 `Floor` + 2 次 `int` 转换 × 256 通道）。
    - 排查手段只有 `profile` 的 by-type 表：**「算子个数少但耗时占比高」就是它**。
      GridSample 只有 3 个却占 2.3%，而同耗时的 MatMul 有 139 个 —— 单算子效率差两个数量级，一眼可疑。

### 6.7 全表测量的两条铁律（2026-11-07 第七轮踩到）

48. **跑全表前先确认没有满载进程**。第七轮开跑时 `SC2_x64` 在前台跑游戏，5 秒吃掉 5.03 秒 CPU（一个核满载），
    结果前两轮全表整轮作废：同一份代码，ORT 的 yolov8s median 跑出 243 / 194，OpenCV 的 rfdetr-small 跑出 222 / 275，
    而我们自己 264.1 / 243.0 —— 全部落在 ±20% 的噪声里，什么也读不出来。
    确认成本只有 5 秒：
    ```powershell
    $a=(Get-Process SC2_x64).CPU; Start-Sleep 4; (Get-Process SC2_x64).CPU - $a   # 接近 4 说明它占满一个核
    ```
    一般结论：**`bench.md` 里的绝对值只在「同一会话、安静机器」下有效**，跨会话只能比同表内部的**倍率**。
49. **「追平对手」的结论至少要看对手两轮的读数**。第五轮我们记下「rfdetr 对 OpenCV 1.12 / 1.06，已追平」，
    第七轮同一份 OpenCV 代码跑出 191.7 / 243.1（medium 快了 **20%**）—— 说明第五轮是**对方发挥失常**，
    真实差距一直是 1.2× 左右。我们自己的数字两轮几乎没变（231.5 vs 230.4、299.4 vs 321.5），
    所以问题完全出在「拿对方的一次异常读数当基准」。
    - 判定规则：**双方绝对值都落在各自历史区间内**，比值才写进结论；否则只写「本轮 N 倍，待复核」。
    - 反向也成立：**不要因为对手某一轮变快就认为我们回归了**（第五轮 yolov8s 的 1.54→1.78 就是这么来的）。

### 6.8 Vision 门面（letterbox + NMS，2026-11-07）

50. **检测头布局会歧义**。`[1,6,N]` 既可能是 `EndToEnd`（N 恰好 6），也可能是 `4+2` 类的 `ChannelsFirst`。
    自动推断只在 `dim1 == 4+C && dim2 == N` 这种宽泛前提下有效，必须允许显式传 `DetectorOptions.ClassCount` / `OutputLayout`；
    推断不出来时抛 `ArgumentException` 让调用方指定，不要默默猜一个（猜错的结果是「框全对、分数全错」，最难查）。
51. **`Detect` 的三段耗时必须分开量**。`preprocess / forward / decode` 混在一锅里没有任何信息量：
    首次运行 `preprocess` 会测出 14.76 ms（JIT + 首次分配），稳态只有零点几毫秒 ——
    计时必须在 `--iters` 循环内取**最小值**，和 bench 的口径保持一致。
52. **`SetInput` 不要放进计时循环**。放进去后 `forward` 从 36 ms 变成 235 ms（拷贝 + 分层编译叠加），
    结论会从「和 bench 一致」变成「慢了 6 倍」这种完全错误的判断。
53. 与 SimdYolo 逐框对照的结论：同图 1280×720 + yolo11n，双方各 14 框，类别完全一致，
    框坐标最大偏差 **0.005 px**（仅 JSON 两位小数舍入）—— 预处理/解码的语义可以认为已经对齐。
    PPM 与 BMP 两条解码路径输出完全相同，可作为「图像读取没引入色序错误」的回归手段。

> **决策（2026-11-10）：`OnnxSharp.Vision` 不改成基于 ImageFacade 的 Pipeline。**
> 技术上完全可行（`Letterbox.Apply` / `Tensorizer.ToTensor` / `Nms.Suppress` / `DetectionDrawing.DrawDetections`
> 与 Vision 现在的 letterbox / NMS / 绘制一一对应，`LetterboxInfo` 也能直接还原坐标），但代价是核心库要多引
> `Sevencat.ImageFacade` + `Sdcb.SimdCodecs.*` 三个包，直接推翻 README 的头号卖点（零第三方依赖、
> NuGet 无依赖、AOT 友好）。定位划分因此保持不变：
> **`OnnxSharp.Vision` = 零依赖的模型侧门面**（letterbox 几何、YOLO/RF-DETR 解码、NMS）；
> **CLI = 图像 IO 层**（ImageFacade 软后端 + 额外引入的 `Sdcb.SimdCodecs.Jpeg/Png/Bmp`，负责解码 / 画框 / 保存）。
> 需要 ImageFacade 的调用方照 CLI 的写法把两者接起来即可 —— `Preprocess`/`Decode` 本来就是分开暴露的。
> 复审条件：如果哪天「零第三方依赖」不再是卖点（例如要同时接 Skia/OpenCV 图像后端），再评估整体迁移。

### 6.9 RF-DETR 的前后处理（2026-11-07）

54. **RF-DETR 不是 letterbox，是直接拉伸**（`torchvision.resize` 到 512×512，`antialias=False`，半像素中心双线性）。
    所以还原框只需乘原图宽高，没有 pad 要减；**照抄 YOLO 的 letterbox 会让框系统性偏移**（竖图最明显）。
    归一化也不同：ImageNet `mean=[0.485,0.456,0.406] std=[0.229,0.224,0.225]`，不是 `/255`。
55. **逐类 sigmoid，不是 softmax**。一个查询可以同时在多个类别上超过阈值，所以必须把 `(Q, C)` 摊平成
    `Q×C` 个「查询×类别」对再取 top-k；**每查询 argmax 会静默丢掉额外类别**（COCO 上常出现的
    「person + 自行车」同框就会消失）。也正因如此 top-k 必须在阈值**之前**做（官方 `_select_topk` 的顺序）——
    数值上「先阈值再取前 N」等价（阈值单调），实现里用的是后者，省一次全量排序。
56. **背景槽位默认是最后一列（-1）**。COCO 91 类导出里最后一列是背景，不排除会出一堆 class90 的高分框；
    自定义类别数导出、或稀疏 COCO checkpoint（末列是 class 90）时必须改 `BackgroundClassId`（CLI `--background`）。
57. 与官方 `rfdetr.export._runtime.decode.decode_detections` 逐框对照：persons.jpg 1568×1068 + rfdetr-small，
    双方各 **41 框**，类别完全一致，分数差 < 1e-4，坐标最大偏差 **0.00 px**。
    另一条旁证：同图 yolo11n 给出的 person 框（1347.3, 671.9）与 rfdetr（1349.7, 672.1）几乎重合，
    说明两条完全不同的前/后处理链路没有引入共同偏移。
58. **像素输入的 `bgr` / `rowStride` 必须一路透传到门面**。`ImagePreprocessor` 早就支持这两个参数，但
    `YoloDetector`/`RfDetrDetector` 的 `Preprocess`/`Detect` 一开始只暴露了 `bgr`，`Stretch` 的 stride
    还被写死成 0 —— 结果「OpenCV `Mat` 直接喂」在门面层是断的。现在两个门面都按
    `(pixels, width, height, bgr = false, rowStride = 0)` 透传，`rowStride` 以**字节**计、0 表示 `width * 3`。
    验证：persons.jpg 1568×1068，紧凑 RGB 与「BGR + 每行 64 字节填充」两条路径的 yolo11n 结果
    24/24 框、坐标差 **0.000000 px**；rfdetr 41/41 框、top1 分数一致。

### 6.10 OCR（PaddleOCR / PP-OCRv6，2026-11-10）

59. **全动态图不是「多补几个算子」的问题，是「输入维未知 → 形状链折不出来」的问题**。
    det / rec 的输入都是 `x[N,3,H,W]`（`dim_param`），导入期 `Shape → Slice → Concat → Reshape` 只能看到 -1，
    `Reshape`/`Resize` 的目标形状永远「运行时才知道」，常量折叠与 `PropagateShapes` 全停。
    解法不是做符号维度传播，而是**特化成静态图**：`OnnxFile.WithInputShapes` 改写图输入维（节点与权重共享、
    不重新解析），再走原来的折叠。代价是一个 `(H,W)` 一份 `Net`，所以调用方按形状缓存
    （det 32 对齐 / rec 16 分桶；实测导入+折叠约 50 ms，缓存后第二次调用零成本）。
60. **`PackedConv.Multiply` 在无 bias 时是空指针**。原实现只有 `c00` 判了 `bias is null`，
    `c10/c20/c30` 直接读 `bias[row+i]`；YOLO / RF-DETR 的 Conv 全带 bias（BN 已折进权重），一直没暴露。
    PaddleOCR 的 `Conv → BatchNormalization` 结构里 Conv 没有 bias，第一次跑 rec 就 NRE（"Object reference not set"，
    堆栈停在 `PackedConv.Multiply`）。四个累加器都要判空。
61. **复用缓冲 + `Tensor.FromArray` 会炸**：`Tensor.FromArray(data, shape)` 要求 `data.Length == shape.Total()`，
    而 OCR 的 `_detBuffer` / `_recBuffer` 是跨调用复用的（这张图更大就重分配，之后不会缩回去），
    第二次不同尺寸的调用就得到 `ArgumentException: Destination is too short`。
    传 `data.AsSpan(0, count)` 的精确切片，别指望 FromArray 帮你截断。
62. **BN 的两种处置**：YOLO / RF-DETR 里 BN 可以折叠进 Conv；OCR 的 rec 图里 BN 挂在 HardSwish 之后、
    输入是运行时张量，直接实现成算子（每通道预算成一次 `y = a*x + b` 仿射，SIMD 内层）。
    别为了「省一个算子」去做通用 BN 折叠 pass —— 收益是 4 个节点的 0.1 ms。
63. **检测后处理不要照抄 pyclipper**：官方 `DBPostProcess` 用多边形外扩（能贴合倾斜文本），
    本实现是 8 邻域连通域 → 轴对齐框 → 按 `面积 × unclip_ratio / 周长` 外扩。竖排 / 倾斜文本仍能框住，
    但框是正的；要贴边就得引入多边形近似，暂不做。
64. **rec 的宽度必须分桶，不能「一行一份图」**：PP-OCRv6 的 rec 是 `[1,3,48,W]`，W 直接决定输出步数 T
    （`T ≈ W/8`）。按 16 分桶 + 桶外补 0（PaddleOCR 的 padding 就是归一化后补 0，不是 -1），
    一张图通常只用 1~3 个桶，比每行一份图省一个数量级的导入时间。
65. **对齐验证口径**：与 ORT 比的是**原始输出张量**（det `[1,1,H,W]`、rec `[1,T,6906]`），不是最终文本。
    det 1×3×960×960 `max|diff| = 1.4e-7`，rec 1×3×48×320 `7.6e-6`；文本层再对就说明后处理也对。
    （文本层的实测：900×300 合成图两行全对；3200×1800 车牌 limit 1600 出 `浙A ES8381` / `浙A：:3M15X`，
    中点分隔符会被识别成冒号，是 tiny 模型 + 65 px 行高的正常表现。）
66. **图片格式是 CLI 的可选依赖，不是引擎的**：`ImageFacade` 软后端只带渲染（自绘 / 中文标签 / Resize），
    编解码要额外引 `Sdcb.SimdCodecs.Jpeg / Png / Bmp` 并在 `ImageHost.RegisterCodecs()` 里显式注册；
    没注册的格式调用时抛 `FeatureNotSupportedException`。产品库 `OnnxSharp` 一行都不引 —— 它只吃像素。

### 6.22 对标 SimdYolo：YOLO 还能优化哪里（2026-11-11 深夜）

先拿到干净基线（`profile --iters 25` 的 sum-of-min 与 `bench` 的 min 对得上：25.7 / 28.2 ms）：

| 类别 | # | ms | GFLOP | GFLOPS | 占比 |
|---|---|---|---|---|---|
| 3×3 s1 | 28 | 7.88 | 2.39 | 303 | 38% |
| 1×1 s1 | 44 | 7.30 | 2.31 | 316 | 36% |
| 3×3 s2 | 7 | 3.98 | 1.74 | 437 | 19% |
| depthwise 3×3 | 7 | 1.39 | 2.50 | 1799 | 7% |
| **Conv 合计** | 86 | **20.6** | 8.94 | **435** | 80% |
| Concat | 23 | 2.2 | — | — | 8.6% |
| Softmax（DFL） | 2 | 1.1 | — | — | 4.3% |
| MaxPool / Split / Add / Reshape / Transpose / Resize | — | ~3.2 | — | — | 12% |

**先确认「不用再折腾」的三件事**（都做了 A/B，别重复踩）：

1. **面板式 vs im2col**：`ONNXSHARP_PACKED_CONV=0` 关掉面板式，yolo11n min **31.9 → 27.8**（开）——
   面板式仍然值得（快 ~12%）。
2. **大空间 s1 强制走面板式**：新增 `ONNXSHARP_PACKED_S1=1` 试过，min **28.5 vs 27.8**（更差）——
   现有「s1 且空间 ≥ 4096 交回 im2col」的策略是对的（面板式每个 16 列面板都要重读一遍权重，
   160²/64→64 那层要重读 1600 次 × 147 KB）。
3. **Winograd 的阈值**：`convbench` 扫描显示 40²/20² 上 Winograd 明显赢（256→256 40²：
   394 vs im2col 130 GMAC/s），80²/160² 输（206/109 vs 296/160）——现有 `ONNXSHARP_WINO_MAX=4096`
   （≈64²）正好卡在对的位置。

**真正的差距**：我们是 **435 GFLOPS** 聚合，SimdYolo 在 18.1 ms 内跑同样的 8.94 GFLOP ≈ **650 GFLOPS**。
差 1.5×，且个体内核并不差（1×1 单测 316 GMAC/s、depthwise 等效 1799 GFLOPS 已经打满带宽），
所以缺口在「每个算子之间」和「逐 op 开销」上。

#### 下一步的优先级（按 收益/工作量 排序）

| # | 优化 | 结论 | 依据 |
|---|---|---|---|
| ✅ | **常驻 intra-op 线程池**（已实现，见 `bench.md` 同日一节） | **yolo11n 37.1 → 23.0 ms（−38%）** | 一次 `ParallelRange.For` 派发 14.4 µs × 200+ 区域 |
| ✅ | **DFL 折成一个内核**（已实现，见下） | 实测 **~0.4 ms（1~1.5%）** | 5 个算子 → 1 个 |
| ~~—~~ | ~~`Concat` 融进 1×1 `Conv`~~ | **已实现并回滚：慢 1.6~1.9×** | 见下面的复盘 |
| ~~—~~ | ~~微内核 4 行 → 6 行~~（SimdYolo 用 6×16） | **已实测否决：慢 1.3~3.3×** | 12 个累加器被 JIT 挤出寄存器 |
| ~~—~~ | ~~Conv+SiLU+Conv 按 tile 流水~~ | **已实测否决：上限只有 ~0.4 ms** | 中间张量本来就留在 L3 |

#### 复盘：Conv 链的 tile 流水为什么也不值得（实测否决）

静态统计看着很香：yolo11n 里有 **38 条** `Conv → SiLU → Conv` 单消费者链，中间张量合计 **36.8 MB/帧**，
按「省掉一写一读」算 73.6 MB。按 30 GB/s 的 DRAM 带宽估，那是 ~2.4 ms —— 所以本来准备做
「按行条带切 tile、conv1 算 strip+halo、SiLU 就地、conv2 从 L2 读」。

**先做了个 20 行的决定性实验**：一个只有 `Add(x, x)` 的 micro 模型，
输入 `[1,64,160,160]`（6.5 MB，正好是那些大中间张量的量级）：

| 项目 | 实测 |
|---|---|
| `Add(x,x)`（2 读 + 1 写 = 19.7 MB） | **0.10 ms → ≈ 197 GB/s** |

⇒ 中间张量（1~6.5 MB）**本来就留在 L3**，一次写+读只有 ~0.07 ms；73.6 MB 全算上也就 **~0.37 ms**。
tile 流水还要引入行窗口内核、halo 边界、新的融合层（200+ 行），收益上限 1.5%，**不值得**。
这也解释了为什么 **ORT（MLAS）和 SimdYolo 都不做 conv 链融合** —— 它们把力气花在内核里。

> 这条同时改掉一个方法论错误：**「省了 N MB 内存搬运」必须先量本机的有效带宽，不能按 DRAM 峰值估**。
> 本机 L3 常驻时 200 GB/s，是 DRAM 的 4~6 倍，很多「融合省带宽」的设想在这个量级上都是噪声。

#### 复盘：6 行微内核为什么不行（实测否决）

按 SimdYolo 的 6×16 形状写了 `Multiply6Rows`（12 个 YMM 累加器 = 6 行 × 16 列），
`gemmbench --threads 16` 隔离对比（GFLOPS）：

| 形状 (m,k,n) | 4 行 | 6 行 |
|---|---:|---:|
| 64, 256, 6400 | **682** | 328 |
| 64, 576, 6400 | **649** | 195 |
| 64, 1280, 1600 | **448** | 337 |
| 32, 144, 25600 | **356** | 268 |

⇒ 全面更慢 1.3~3.3×。原因：**.NET JIT 保不住 12 个活跃 `Vector256`**（再加 2 个 B 向量和广播
= 15/16 个 YMM），主循环开始 spill —— 和仓库里早先「把激活塞进 GEMM epilogue 反而慢 26 ms」
是同一个坑。实现已删除。

> SimdYolo 能在 6×16 上跑得动，是因为它把权重按 **6 行交错预打包**（`CpuPackedGemm`）：
> 每条 load 取到的是连续的 6 个 float，而不是像我们这样 `Vector256.Create(a[t])` 逐行广播。
> 要复现它的收益得连权重打包一起改；收益上限只有 ~2 ms，风险/工作量不匹配，暂时搁置。

#### 已落地：DFL 单内核（`FuseDflSoftmax` + `Ops/DflOps.cs`）

yolo 的 DFL 被导出成 5 个算子，其中「乘 arange」其实是一个**权重为 `[0,1,…,15]` 的 1×1 卷积**：

```
x[64,N] → Reshape[1,4,16,N] → Transpose[0,2,1,3] → Softmax(axis=1) → Conv(1×1, w=arange) → Reshape[1,4,N]
```

融合层一次遍历就出结果：`out[s,n] = Σ_b softmax_b(x[s·16+b, n]) · b`（先 `1/Σ`、再 `p_b·b`，
与 Softmax 算子的舍入顺序一致）。判据里最硬的一条是**卷积权重必须正好等于 `arange(bins)`** ——
形状对但权重不是 arange 的卷积一律不碰，避免误吃普通 1×1 卷积。

- 效果：算子 166 → 162，`DflSoftmax` 一层 **0.1 ms**（原 `Softmax+Conv+2×Reshape+Transpose` ≈ 0.5 ms）。
- **踩的坑**：第一版每个 8 列就调一次 `VectorMath.Exp`（每块 16 次、共 16800 次小调用），
  光调用固定开销就要 **1.3 ms**；改成每任务 256 列、每个 bin 只调一次 Exp（仍是 8 lane 向量化）后降到 0.1 ms。
- **口径教训**：一开始我用「总时间 A/B」量，负载一抖就得出「更慢」的错误结论。
  正确做法是**比算子表**：融合前后差额只有被删掉那几个算子的 min 之和，这个数字稳定得多。

#### 复盘：`Concat → Conv` 为什么没成（这一条是实测否掉的）

yolo11n 的 23 个 Concat 里 17 个是「唯一消费者 = 1×1 Conv」，看着和 SimdYolo 的
`OnnxConcatConv` 完全同构。两种实现都写了、都验证过数值（YOLO 与 ORT golden：
`max|diff| = 1.34e-3 / mean 3.1e-6`），但**都比不融合慢**：

| 实现 | 做法 | 结果（min，交替 A/B） |
|---|---|---|
| A | 每段一次 GEMM + **整幅 C 累加**（GEMM 新增 `accumulate`：把 C 旧值当累加器初值） | 28.8 vs 29.0 —— **打平** |
| B | 按列块把各段拷进 L2 常驻 `[K,width]` 缓冲，再**一次** GEMM | 45.3~61.0 vs 29.0~44.1 —— **慢 1.6~1.9×** |

原因：
- A 的问题是多一段就要把整个 `[C,H,W]` 输出重读重写一遍（C2f 那块 6.5 MB × 2），省的拼接写覆水难收。
- B 的问题是我们拷进 `[K,width]` 之后，**GEMM 内部还要按 16 列块再打包一遍**（等于拷两次），
  而且每块的 `n` 只有 ~512，GEMM 的列并行与 A 面板复用都丢了；实测拷贝带宽成了瓶颈。
- 真正的解法是把拼接并进 GEMM 的 K 维（给内核加 `ldb` 行跨度 + 块内累加），那是内核级改造，
  预期也就 ~1.5 ms（5%），风险和工时都不划算。**所以这条留在「已知不必做」清单里。**

副作用（保留）：`tools/scripts/gen_golden.py` 生成了 yolo11n 的 golden，
`YoloGoldenTests` 从 SKIP 变成真的在跑（之前一直跳过）。

> 注：SimdYolo 的 `KernelFusion.Merge`（把小的 rep-conv 核居中加进大核）只对「并行分支相加」的
> 导出（YOLOX/v5 rep-conv）有用，yolo11n/v8 的 head 是串行 `Conv→Conv→Conv`，**不适用**。

### 6.21 YuNet 人脸检测（2026-11-11 夜）

模型 `yunet.onnx`（0.22 MB）：106 节点 / 8 种算子（Conv 53、Relu 15、Transpose 12、Reshape 12、
Sigmoid 6、MaxPool 4、Resize 2、Add 2），静态输入 `[1,3,640,640]`，12 个输出头
（cls/obj/bbox/kps × stride 8/16/32）—— **算子全在已支持范围内，引擎一行没改**。

新增 `Vision/YuNetDetector`（门面）+ `Vision/YuNetDecoder`（解码/NMS）+ `Vision/FaceDetection.cs`
（`Face` / `FaceLandmarks` 结果类型）+ CLI `face` 子命令 + `YuNetTests`（3 条用例）。

- **对拍口径**：`cv2.FaceDetectorYN` 做 oracle，同一个 640×640 输入下
  **框差 0.00 px、5 点关键点差 3e-5 px、分数差 5e-7**。
- 稳态性能（640×640，16 线程，30 次迭代）：**median 19.0 ms / min 15.8 ms**。
- lena 512×512：1 张脸 score 0.9135，坐标换回原图后 = `[209.3, 181.5, 142.2 × 206.0]`。

#### 三个坑

1. **`cv2.FaceDetectorYN` 不 resize，它按 32 的倍数 pad 后按原分辨率跑**（`face_detect.cpp`：
   `padWithDivisor` → `blobFromImage(pad_image)`）。而我们的模型是静态 640×640。一开始我拿
   「resize 到 640 的结果」去比「cv2 在 512 上跑的结果」，凭空多出 6 px 框差 / 8 px 关键点差，
   差点去改一个本来就正确的解码公式。**正确做法：两边喂同一个 640×640 输入再比。**
2. **`ImagePreprocessor` 的张量恒为 RGB**，它的 `bgr` 参数描述的是**源像素**顺序；YuNet 要 BGR
   （OpenCV `blobFromImage` 不做 swapRB）。直接传 `bgr: true` 得到的张量是 RGB，实测
   `mean|d| = 49.7 / max|d| = 156`（红蓝通道互换的典型量级），框就整体偏 6 px。
   现在门面里显式 `bgr: !bgr` 并写了注释，还补了一条 `preprocessing matches the reference blob`
   用例把这一步钉住（现在 max|d| = 0）。
3. **解码公式只能抄 `face_detect.cpp`，不能按 SSD prior 猜**：YuNet 是
   `cx = (col + bbox[0]) * stride`、`w = exp(bbox[2]) * stride`（**格子下标、没有 +0.5**，
   也没有 variances），score 是 `sqrt(cls·obj)`。按 SSD 的 `(j+0.5+dx·var)·stride` 去猜，
   关键点会系统性偏半个格子。

#### 不支持的部分

`yunet_int8.onnx` 是 **QDQ** 图（`QuantizeLinear` / `DequantizeLinear`），这两个算子还没实现
（`plan.md` M10-04/05），所以 int8 版现在直接报 `op type 'QuantizeLinear' is not implemented yet`。

### 6.20 性能复核：交替 A/B（含两条撤回，2026-11-11 夜）

本机（8845HS APU）单次测量的抖动极大 —— 同一份代码、同一条命令的 median 从 79 到 110 ms 都出现过
（§6.19 那组「r1→r3 一路升高」就是典型的负载/睿频爬坡）。而 §6.18/§6.19 里有几处结论是
「拿今天的数字比昨天的数字」得出的，**那是无效对比**。所以这次给代码加了两把运行期开关（不用重编）：

- `ONNXSHARP_NO_FUSE=attention,dqmm,layernorm`（可以是任意子集）
- `ONNXSHARP_NO_SHAPECACHE=1`

然后用「交替 A/B × 5~6 轮、取各自最好值」重新量（`profile` 的 sum-of-min 口径，76 帧 encoder）：

| 项目 | 关掉 | 打开 | 效果 |
|---|---|---|---|
| **全部四项** | 66.5 / 67.0 | **43.8 / 44.5** | **−33%（−22.5 ms）** |
| └ `DynamicQuantizeMatMul` 融合 | 59.5 | 45.4 | **−14.1 ms（−23.7%）** |
| └ `FuseAttention` | 52.0 | 44.6 | **−7.4 ms（−13.4%）** |
| └ `FuseLayerNorm` | 44.4 | 43.7 | −0.7 ms（−1.6%） |
| └ 形状/类型推断缓存 | 66.9 | 67.0 | **0（噪声内）** |

端到端（`asr --repeat 10` 的 min）：**101 ms → 84 ms（−17%）**。

**两条撤回**：

1. §6.18 写「形状缓存省 ~17 ms（encoder 103 → 86）」—— 那是拿 `bench` 的 median 比历史数字，
   **不是 A/B**。实测：在 3858 算子的原图、以及融合后 2244 算子的图上，缓存都**省不到 0.1 ms**，
   `bench` 的 `alloc` 也几乎不变（42.7 vs 42.4 MB/iter）。它值的是「行为稳定、没有额外分配」，
   不该再算进性能表。（这也说明这张图每个算子的形状推断本来就很便宜 —— 每算子 ~25 ns 级别。）
2. §6.18 猜 `FuseLayerNorm`「收益约 5~8 ms」—— A/B 实测 **0.7 ms**。81 个 LN 每帧一共才 0.7 ms，
   `LayerNormalization` 内核本身已经很快；手写 9 个算子那版主要输在「算子个数」，时间本来就不多。

**方法学**：往后的性能结论一律以「交替 A/B + 各自最好值」为准；两把开关保留在代码里
（`GraphFusion.Disabled` / `Interpreter.DisableShapeCache`），下次再遇到争议可以直接复跑。

### 6.19 `FuseAttention`：整条 multi-head attention 折成一层（2026-11-11 傍晚）

**先说结论**：融合干净、数值也对（静态夹具 max|diff| = 1.8e-7，动态 encoder 与 ORT 的 diff 与融合前
完全一致），但**收益只有 −8%（51.8 → 47.5 ms 的 encoder sum-of-min），不是预估的 20~30%**。
原因在 profile 里一眼可见：attention 那一整段（`MatMul` 6.8 + `Softmax` 0.9 + 掩码 `Add` ~1.5 +
`Transpose/Reshape` ~1.4）本来就只有 ~11 ms，占 84 ms 的 13%；而 **66% 的时间是
`DynamicQuantizeMatMul` 的 GEMM 本身**，那是融合碰不到的部分（要 int8 内核才降）。

#### 匹配到的结构（Paraformer SANM 自注意力）

```
Split → q,k,v [B,T,320]
q → Reshape[B,T,H,D] → Transpose[0,2,1,3] → Mul(·, 1/√D) ─┐
k → Reshape[B,T,H,D] → Transpose[0,2,3,1] ────────────────┼→ MatMul → Add(mask) → Softmax(-1)
v → Mul(mask) → Transpose[0,2,1] → … → Transpose[0,2,1,3] ─┘
                                     → MatMul(·, Vh) → Transpose[0,2,1,3] → Reshape[B,T,320]
```

- 判据：`Perm` 精确等于 `[0,2,1,3]` / `[0,2,3,1]`（`Transpose` 加了 `ITransposePerm` 内部接口读属性，
  沿用 `ILastAxisArgMax` 的做法）、`Softmax` 的 `axis` 是 -1 或 rank-1（`ISoftmaxAxis`）、
  每个中间张量单消费者、scale 是标量常量、mask 在 `Add` 的另一路。
- **两种形态**（这是本次最费时间的坑）：
  - 静态形状：`Reshape` 目标是常量 ⇒ head 数从 value_info 读得出来 ⇒ 融合层吃 3 维 `[B,T,C]`，
    连 3 个 `Reshape+Transpose` 和输出的 `Transpose+Reshape` 一起折掉。
  - 动态形状（Paraformer 用的这条）：`Reshape` 目标由 `Shape→Gather→Concat` 建出来，head 数读不出来；
    而且 **输出的 `Transpose` 有 3 个消费者**（`Shape_6`、`Shape_7` 在给它算 Reshape 目标）⇒ 不能删。
    于是融合层吃 head 空间的 `Qh[B,H,T,D]`/`Kt[B,H,D,T]`/`Vh[B,H,T,D]`（H/D 由形状给出），
    输出 `[B,H,T,D]`，`Transpose+Reshape` 原样保留。**如果只按「单消费者」硬判，Paraformer 一块都融合不了。**

#### 内核（`Ops/AttentionOps.cs`）

K 先按 head 打包成 `[B,H,T,D]`（原布局是 `[B,H,D,T]`，对点积不友好），然后**按行并行**
（`B·H·T` 个任务，`ParallelRange.For` 分块、每块租 3 个缓冲）：一行 = `scale·q`（预乘，和图上
`Mul(Qh, 1/√D)` 的舍入一致）→ `T` 次 AVX/FMA 点积 → `+mask` → `VectorMath.SoftmaxRow`（与 `Softmax`
算子同一份内核，结果与原来逐位一致）→ `T` 次 `axpy` 累加 `P[j]·V[j]`。全程不物化
`[B,H,T,T]` 的 scores（每块 46k 元素）。

#### 踩的坑

1. **`drop` 和 `replacements` 的顺序**：我把 `softmax` 同时放进了 `drop` 和 `replacements`，而
   `result` 循环里先判 `drop` ⇒ 融合层根本没进图。报错却出现在下游：
   `Transpose: perm has 4 entries, rank is 0`（它的输入张量没人产出）。已改成**先查替换表**。
2. **输出 Arg 写错**：`foldOutput = false` 时融合层要产出**输出 Transpose 的输入**（`pv` 的输出），
   我一开始写成了 Transpose 的输出 ⇒ 同一个 rank 0 报错。
3. **`Span`/固定指针都不能进 lambda**（CS1764）：先把 `fixed` 指针对拷成普通局部 `float*` 再捕获。
4. 动态 `Reshape` 的 `Shape` 链（见上面「两种形态」）——这是「单消费者」判据在真实图上失效的那一条。

#### 结果

| 指标 | 融合前 | 融合后 |
|---|---|---|
| encoder `profile` sum-of-min | 51.8 ms | **44.8~47.5 ms（−8~14%）** |
| 算子数 | 2410 | 2244 |
| `Softmax` / `MatMul` | 40 / 81 | **0 / 1** |
| 掩码 `Add` | 202 | 162（40 个掩码加法折进融合层） |
| `FusedAttention` | — | 40 个，5.2 ms |
| 剩余大头 | `DynamicQuantizeMatMul` 66% | **74.3%（33.3 ms）**——融合碰不到的部分 |
| 端到端（`asr --repeat` 的 min） | 81 ms（enc 62） | **73~84 ms（enc 50~60）** |
| 与 ORT 的 encoder diff | mean 8.22e-4 / max 9.25e-2 | **完全一致** |

新增夹具 `tools/scripts/make_attention_model.py`（静态形状的标准 MHA + ORT 参考输出）与
`AttentionTests`（`max|diff| = 1.8e-7`，并断言 `Softmax`/`Transpose`/`Reshape` 全被折掉），
两条形态各有覆盖：静态形状走 `AttentionTests`，动态 head-split 走 `AsrEncoderTests`/`AsrEndToEndTests`。

### 6.18 对齐 ORT 的融合 + 去掉逐算子固定开销（2026-11-11 下午）

这一轮把 6.17 里「下一步该做的两件事」都做完了，**并推翻了 6.17 的一个结论**（见文末「更正」）。

- **① 按输入签名缓存形状/类型推断**（`Interpreter` + `LayerInfo`）：
  每帧每个算子都会调 `GetMemoryShapes/GetTypes`，而 ASR/YOLO 这类图每帧输入形状完全一样。
  现在把结果按「输入 shape + type 签名」存在 `LayerInfo`（`ShapeCacheValid` / `Cached*`），
  命中就直接 `BindAll` 复用，连 `TensorShape`/`int[]` 的分配都省掉。
  encoder **~103 → 86 ms**（原本 layer 时间之和 67 ms、总 91 ms，那 ~24 ms 的差就是它）。
- **② `DynamicQuantizeMatMul` 融合**（`GraphFusion.FuseQuantizedMatMul` + `Ops/QuantizedMatMulOps.cs`）：
  把 `DQL → MatMulInteger → Cast → Mul(a_scale·w_scale) → [Add(bias)]` 折成一层。
  A 只过两趟内存（min/max、边量化边展开成 F32，**不物化 u8**），GEMM 后在一个 epilogue 里
  乘 `a_scale * w_scale[col]`、加 bias。encoder 上的效果：`DQL` 161→1、`Cast` 170→10、
  `Mul` 526→125、`Add` 524→202，`MatMulInteger` 38.0 ms + `DQL` 8.0 ms → 融合层 33.9 ms。
  数值与原链**逐位一致**（整数累加在 F32 里精确，乘加顺序照抄图里的顺序），
  ASR 文本与 `quant: quantized graph matches onnxruntime` 用例都不变。
  - 坑 A：权重是 **per-channel 量化**的 —— `onnx::MatMul_8982_scale` 是 `(960,)` 向量、
    `zero_point` 也是 `(960,)`，不是标量。第一版按标量写死在参数里，epilogue 直接算错；
    改成「scale 走图输入、长度 == N 就按列、长度 == 1 就当标量」。
  - 坑 B：`a_scale` 只喂那个「组合 scale」的 `Mul`、组合 scale 只喂一次 —— 这两条必须查，
    否则会拆掉别的消费者。
- **③ 修 `producerOf`**：`FuseLayerNorm` / `FuseQuantizedMatMul` 里「生产者表」当时是用
  `op.Inputs` 建的（那是消费者表），所以永远查不到真正的生产者；改成用 `op.Outputs` 后
  `FuseLayerNorm` 从 **1/81** 变成 **81/81**（`ReduceMean`/`Sqrt`/`Div` 全部消失）。
  另外它原来要求中间 `Add` 有 value_info 才能校验通道数，裁剪过的子图没有 —— 退化成
  「`gamma` 与 `beta` 等长」的自洽检查就够了（`LayerNormalization` 的输出形状本来就等于输入形状）。
- **顺手修的构建问题**：`Kernels/Int8Gemm.cs` 用了 `AvxVnni`（CA2252 预览 API），
  **net8.0 目标一直编不过**（之前用 `Select-String 'error CS'` 过滤，漏掉了 `error CA2252`）；
  加文件级 `#pragma warning disable CA2252`（本来就有运行期 `IsSupported` 守卫）。
  `MatMulArgMaxKernel` 的 `stackalloc` 原本在循环里（CA2014），移到循环外。
  现在 `dotnet build` 双目标 **0 错 0 警**。

#### 更正（6.17 的两处错误结论）

| 6.17 的说法 | 实测 |
|---|---|
| `FuseLayerNorm` 让 encoder 177 → 115 ms（−35%） | **当时一次都没融合成功**（`producerOf` 建错表 + value_info 条件）。那 62 ms 的落差是机器负载波动。修好后 81/81 全融合，收益约 5~8 ms |
| 「下一步：降 per-op 固定开销，预计省 10~15 ms」 | 实测省 **~17 ms**（103 → 86），估计略保守 |

#### 最终性能（同一台机、同一时段背靠背，min 值）

| 阶段 | 6.17 | 现在 | ORT | 现在 vs ORT |
|---|---|---|---|---|
| fbank+LFR+CMVN | 8 ms | 4~7 ms | 8.3 ms | **≈1.0×** |
| encoder 段（76 帧） | 103~115 ms | **62~86 ms** | 39 ms | 1.6~2.2× |
| decoder 段（深切） | 21 ms | **14~21 ms** | 6.4 ms | 2.2~3.3× |
| **整句（热态）** | **136 ms** | **81~87 ms** | **55.6 ms** | **~1.5×** |

演进：559 → 182 → 136 → **81 ms**（本机同时跑着 4 个 VS Code 进程，数字取多次 `--repeat` 的 min）。

### 6.17 ASR 端到端性能收尾与 LayerNorm 融合（2026-11-11 上午）

- **decoder 深切**：decoder 的 cross-attention K/V 其实是 **encoder 的量化输出**
  （`/encoder/after_norm/Add_1_output_0_{quantized,scale,zero_point}`），把这三个张量 + `token_num`
  提升成图输入后，decoder 图 **7828 → 2076 节点**，不再重跑 encoder。ORT 验证与全模型 `logits`
  **逐位一致**（`max|d| = 0`）。
  - 坑：一开始只提升 loop 的输出，图仍是 7827 节点 —— 因为 **`token_num` 也是 decoder 的输入**
    （它给 `make_pad_mask` 用），保留 `token_num` 输出会把 alphas/encoder 支路整棵拉回来。
- **图缓存**：`ParaformerRecognizer` 按 `(frames, dim)` / `(tokens, encFrames, dim)` 缓存 `Net`
  （形状特化发生在 `ReadModel`，所以不同长度必须各存一份），并用一把锁串行化 `SetInput+ForwardAll`。
  冷启动 1.7 s → 995 ms，热态不再重载。
- **`FuseLayerNorm`（学 SimdOnnx）**：手写 LayerNorm（9 个算子：
  `ReduceMean/Sub/Pow/ReduceMean/Add/Sqrt/Div/Mul/Add`）折成单层 `LayerNormalization`。
  ASR 文本不变。
  - ⚠️ **更正（见 §6.18）**：这一轮它其实**一次都没融合成功**（`producerOf` 用 `Inputs` 建表），
    当时记的「encoder 177 → 115 ms（−35%）」是机器负载波动；修好后 81/81 全融合，收益约 5~8 ms。
  - 匹配要点：每个中间张量单消费者、`gamma/beta` 长度 == 最后一维、`eps` 是标量常量、
    `Div(Sub, Sqrt)` 的方向、以及 `Pow(x,2)` 与 `Mul(x,x)` 两种平方写法都要认。
- **int8 两条路径一致性**：新增用例在 `k = 1280`（最长 FFN 层）上验证
  「F32 展开替代」与精确整数乘 **max|diff| = 0** —— 这样 VNNI 机器切到整数内核时结果不变。

#### SimdOnnx 的 14 个融合 pass vs 我们（2026-11-11 调研）

| SimdOnnx | 我们的状态 | 备注 |
|---|---|---|
| `FuseLayerNorm` | ✅ 已实现（§6.18 修好后 81/81 全融合） | 收益约 5~8 ms（6.17 记的 −35% 已更正） |
| `FuseBiasAdd`（Conv/MatMul + Add(bias) 折进权重） | ❌ 未做 | 我们的卷积已带 bias；MatMul+Add 可考虑并入 `MatMulArgMax` 或 `Gemm` |
| `FuseAttention`（QKV→softmax→输出整链） | ❌ 未做 | 收益最大但工作量也最大（我们的 encoder 有 52 个 Softmax/106 个 MatMul） |
| `FuseGelu` / `FuseGatedAct`（SwiGLU） | ❌ 未做 | ASR 模型里没有；LLM 才用得上 |
| `FuseConvResidual` / `FuseResizeAdd` / `FuseChannelGate` / `FuseScaledTanh` / `FuseSpaceToDepth` / `FuseConvTBias` | ❌ 未做 | 都是 CV 专用模式，我们的 YOLO/RF-DETR 里暂未出现 |
| `FuseMemory`（Conv→MaxPool 共享缓冲） | ❌ 未做 | 我们有缓冲池，收益不如他们明显 |
| 死节点消除 / 零拷贝视图 | ❌ 未做显式 pass | 我们靠导入期可达性 + 常量折叠覆盖一部分 |

#### 最终性能（同一台机、同一时段背靠背）

| 阶段 | OnnxSharp | ORT | 倍数 |
|---|---|---|---|
| fbank+LFR+CMVN | 8 ms | 8.3 ms | **1.0×** |
| encoder 段（76 帧） | 103~115 ms | 39 ms | 2.6~3.0× |
| decoder 段（深切） | 21 ms | 6.4 ms | 3.3× |
| **整句（热态）** | **136 ms** | **55.6 ms** | **2.4×** |
| 冷启动（含 JIT + 两次图加载） | 995 ms | ~56 ms + session 加载 | — |

剩下的差距**不是 VNNI**（这一条之前写错了：ORT 在同一台机上也没有 VNNI）。
用 ORT 自己的 profiler（`tools/scripts/bench_ort_profile.py`）看到的真实原因是：

| ORT 的做法 | 我们的现状 |
|---|---|
| `DynamicQuantizeLinear + MatMulInteger` 融成单 kernel **`DynamicQuantizeMatMul`**（210 次/run，18.5 ms）：A 边量化边参与 GEMM，**不物化 u8 激活、不产出 int32 张量、scale/bias 在 epilogue 里做** | `DQL`（写 u8）→ `MatMulInteger`（读 u8、展开成 F32、GEMM、写 int32）→ `Cast`（int32→f32）→ `Mul(scale)` → `Add(bias)`，5 个算子 5 趟内存 |
| 手写 LayerNorm 融合（`LayerNormalization` 131 次/run） | 我们这一轮也加了（`FuseLayerNorm`），但当时没真正生效，§6.18 修好后 81/81 |
| 图编译后的执行计划，per-node 开销接近 0 | 每个算子每次调用都要重算 `GetMemoryShapes/GetTypes`、重新绑定缓冲 |

我们的 encoder 分解（`profile`，10 次迭代的 min）：总计 **91 ms**，
其中 `MatMulInteger` 38.4 ms（57%）、`DynamicQuantizeLinear` 8.0 ms（11.8%）、`MatMul` 7.0 ms、
`Add` 5.5 ms，而 **layer 时间之和只有 67 ms** —— 剩下 ~24 ms 是逐算子固定开销。

所以下一步该做的是「像 ORT 一样融合 + 降 per-op 开销」，而不是等 VNNI 机器。

### 6.16 端到端语音识别打通（M9-07/08/09，2026-11-11 早晨）

- **decoder 段**：`tools/scripts/split_decoder.py` 去掉两个 `Loop`/`Sequence*`、剪掉不可达节点、把 loop 的输出
  提升成图输入 → `decoder.onnx`（7828 节点）。ORT 验证切图与全模型 `logits` **逐位一致**；
  我们的引擎跑通（5302 op），**21/21 token 的 argmax 与 ORT 一致**。
- **补的引擎缺口**：`LogSoftmax`、**运行期 `Slice`**（`Slice(..., max_label_len)`）、
  **标量输入的 dtype**（`Dims.Length == 0` 时原来直接丢掉了类型）。
- **token 解码**：`Speech/TokenTable.cs`，规则 = argmax → 跳过 `<blank>` → 折叠连续重复 → 遇 `</s>` 停。
- **识别门面 + CLI**：`Speech/Paraformer/ParaformerRecognizer` + `asr <wav> --models <dir>`。
- **真实语音验证**：`tools/scripts/tts_huihui.ps1` 用 Windows 自带的 **Microsoft Huihui（微软慧慧）**
  合成 16 kHz wav（4.54 s）→ 76 帧特征 → 16 个 token → 文本
  **「今天气真不错我们出去走一走吧」**，与 ORT 全模型的参考文本**逐字一致**。
- 首次端到端 1.7 s（JIT + 两次图加载各 ~0.5/0.2 s）；测试全量 **93/96 passed, 3 skipped**。
- **目录约定**（对齐 `Ocr/PaddleOCR`）：`Speech/` 只留**模型无关**的东西
  （`WavReader` / `FbankExtractor` / `LfrCmvn` / `SpeechFeatureExtractor` / `TokenTable`），
  模型专属实现放子目录 —— 现在有 `Speech/Paraformer/`（`ContinuousIntegrateAndFire` + `ParaformerRecognizer`），
  以后的 Whisper / Zipformer / streaming paraformer 各开一个兄弟目录，互不干扰。

### 6.15 纯 C# 语音前端（M9-02，2026-11-11 凌晨）

- **新增 `Speech/`**：`WavReader`（PCM 8/16/24/32 + float32，多声道混单声道，不做重采样）、
  `FbankOptions` / `FbankExtractor`（kaldi 风格 fbank）、`LfrCmvn`（funasr 的 LFR + CMVN）、
  `SpeechFeatureExtractor`（wav → `[T, 560]`，参数可从模型 metadata 自动读）。
- **ONNX reader 补了 metadata**：`OnnxFile.Metadata`（`ModelProto.metadata_props`，字段 14）。
  Paraformer 把 `lfr_window_size=7`、`lfr_window_shift=6`、`neg_mean`/`inv_stddev`（560 维，
  正好是 80 维平铺 7 次）都放在那里，前端不用再要 CMVN 文件。
- **金标准**：`tools/scripts/fbank_probe.py` 用 **kaldi-native-fbank**（sherpa 内部同一个库，
  `pip install kaldi_native_fbank`）合成 6 s 扫频 + 噪声的 wav，落 598×80 的 fbank 与 100×560 的 LFR。
  为搞清细节还把 sdist 拉下来读了 `feature-fbank.cc` / `mel-computations.cc`。
- **结果**：fbank **median|diff| 4.8e-7、有声 bin ≤ 9.7e-5**；LFR+CMVN **median 0**、max 3.4e-4；
  CLI `features` 端到端 22.6 ms（6 s 音频），与 Python 参考 54.9% 元素逐位相同。
- 新增回归 `fbank: matches kaldi-native-fbank frames`、`fbank: LFR + CMVN matches funasr semantics`，
  全量 **91/94 passed, 3 skipped**。

### 6.14 CIF 用 C# 重写（M9-06，2026-11-11 凌晨）

- **新增 `src/OnnxSharp/Speech/Paraformer/ContinuousIntegrateAndFire.cs`**：把模型里 80 节点的 `Loop` body
  逐节点移植成十几行循环，产出「每步候选帧 / 每步累加 alpha / 触发下标 / token 级 embeddings /
  残留 frame 与 alpha」。API 只支持单条语音（`[T, dim]` + `[T]`），批量在外面循环。
- **金标准**：`tools/scripts/sherpa_cif_probe.py` 把 `/predictor/Concat_5_output_0`（enc）、
  `/predictor/Add_output_0`（alphas）、`fires`、`frames`、`frame.7`、`integrate.7`、`token_num`
  一起加成 graph output，用 ORT 跑一遍落盘。
- **结果**：`frames` / `fires` / `frame.7` / `integrate.7` **全部 `max|diff| = 0`**（逐位一致），
  token 数 **21 = ORT 21**。新增回归 `cif: candidates/fires/leftovers match the ONNX loop body`
  与 `cif: token count matches the model output`。
- **踩过的三个语义点**（照抄 body 才不会错）：
  1. `completion = 1 - integrate` 用的是**上一步的** integrate（不是加完 alpha 之后的）；
  2. `fire` 时 `frame` 是**替换**成 `remainds * enc[t]`，不是「先发出、再清零」；
  3. token 选择是 `frames[fires >= 1]`（`fires` 存的是**累加后的 alpha**），
     不是 `fires >= 0.5`、也不是按 `alphas` 排序。
- **顺带**：`MatMulArgMaxKernel` 的 `m ≥ 2` 路径改成 **4 行分块**（B 只读 `ceil(m/4)` 遍），
  之前逐行做会把 B 重扫 m 遍，CTC 形状下不如「先 GEMM 再 argmax」。
- 另外记一笔：**bench 结论会被后台负载污染**。同一份代码在机器空闲时 encoder 104 ms，
  VS Code/微信 占满核心时是 152~190 ms（`gemmbench` 从 ~500 掉到 ~100-200 GFLOPS）；
  以后对比数字要连同「当时的微基准」一起报，别拿不同负载下的两组数比。

### 6.13 MatMul+ArgMax 与 encoder 性能收尾（2026-11-10 深夜）

- **新增 `ArgMax` 算子**（axis / keepdims / select_last_index，I64 输出，支持 F32/I32/I64 输入）——
  之前完全没有，是 decoder / CTC 贪心解码的前置。
- **`Kernels/MatMulArgMaxKernel`**：`ArgMax(MatMul(a,b)+bias)` 不物化 logits；m ≥ 2 按行并行，
  m == 1（逐 token 解码）按列并行后合并（「值大者胜、相等取小下标」，与 `select_last_index=0` 一致）。
- **图融合**：`GraphFusion.FuseMatMulArgMax` 把 `MatMul →(Add 常量 bias)→ ArgMax(axis=-1, keepdims=0)`
  替换成单层 `MatMulArgMax`（新建层 + `graph.Reset`，因为输出形状/类型都变了）。
  校验：中间张量单消费者、非图输出、ArgMax 属性匹配（层内接口 `ILastAxisArgMax` 暴露给融合器）。
- **实测**：与 ORT 逐值一致（`1×512×8000` 单 token、`24×256×8000` CTC 两档）。
  **性能如实说**：与「先 GEMM 再 argmax」基本持平（同一进程内 0.9~1.3×，纯噪声）；
  原因是手写 GEMV 循环（8 路累加器）打不过寄存器分块 + 打包的 `GemmKernel`。
  所以内核分两层：logits ≤ 4M 元素走「GEMM 到池化缓冲 + 扫描」（省一次调度与分配），
  超过才走不物化的流式路径（长语音 CTC 的几 MB 写读才真的有收益）。
  CLI：decode 0.2 ms / ctc 0.9 ms。
- **M10-02d 完成**：1-D `Conv` 改「按输出向量化 + 按输出通道并行」（41 → 9 ms）、
  `DynamicQuantizeLinear` 的 min/max 向量化（16.8 → 11.5 ms）。
  encoder 段 **170 → 104 ms**（累计 **104×**），与 ORT 的 30.9 ms 差 3.4×。
- **测试**：新增 `MatMulArgMaxTests`（3 例：ArgMax 语义、融合 vs ORT、融合 vs 两趟吞吐），
  全量 **87/90 passed, 3 skipped**。

### 6.12 int8 量化算子与 ASR encoder 段（2026-11-10 晚）

这一轮的目标是「把 Paraformer 的 encoder 段跑起来」，过程中把「int8 加速」的假设全部证伪了，
顺手补了三个运行期形状的 escape hatch。

- **新增算子**：`DynamicQuantizeLinear`（per-tensor min/max → scale/zp）、`MatMulInteger`
  （u8/i8 输入，零点标量或 per-column）、`Pad`（constant / edge / reflect，支持负 pads = 裁剪）+
  **1-D `Conv`**（FSMN 记忆块，输入 `[N,C,T]`）。算子数 72 → 75。
- **与 ORT 的一致性**：`tools/scripts/make_quant_model.py` 生成两组微模型（u8 权重 + 标量 zp、
  s8 权重 + per-column zp），`run --compare` 的 `max|diff| = 0.000E+000`；
  这个用例固化成回归 `quant: quantized graph matches onnxruntime`。
- **ASR encoder 段**：`tools/scripts/split_before_control_flow.py`（7881 → 5715 节点）+
  `sherpa_encoder_probe.py`（ORT 金标准）+ `probe_tensors.py`（逐层二分）。
  `speech[1,96,560] → /predictor/Concat_5_output_0 [1,97,320]`，10.8 s/帧；
  与 ORT 主输出 mean|diff| 8e-4（中间层 1e-6~3.6e-5，CIF 放大），padded 帧完全一致。
- **被证伪的三个假设**：① int8 权重 + F32 计算（0.10×，见坑 #76）；② 权重带宽是瓶颈（FP32 也只有
  73 GFLOPS，见 #77）；③ VNNI 可用（本机 Zen4 的 `AvxVnni.IsSupported = False`）。
  结论：**量化图的正确性已具备，速度交给 M10-02c（小 GEMM 效率）**。
- **测试**：新增 `QuantOpsTests`（3 例，含 ORT 端到端对拍）与 `PadConv1DTests`（4 例），
  全量 **80/83 passed, 3 skipped**。

### 6.11 SAM3 / 共享图像预处理（2026-11-10）

67. **`Clip` 的 min/max 是可选输入，空串占位会变成 `Arg(0)`**。按 `inputs.Count > 2` 判断「有没有 max」，
    会把「缺失」当成「值为 0 的标量」：`Clip(2, min=1)` 变成 `Clip(2, 1, 0) = 0`，
    SAM3 的 `Div` 随后除零 → ∞ → `pred_logits` 全 NaN。凡是可选输入都要 `Inputs[i].Idx != 0` 兜底
    （ShapeOps 里的 `Slice`/`Reshape` 已按这个约定修过一遍）。
68. **`FusedEltwise` 的融合判据不能只比元素总数**。`[1,1,256]` 与 `[256]` 元素数相同但广播语义不同，
    融合层只做线性遍历（不理解广播），放行后运行期抛 `input #1 shape [1,1,256] does not match [256]`。
    现在要求**严格同形**，或元素数为 1（splat）。
69. **`Flatten` 的 `axis` 允许等于 rank**（ONNX 规范是 `[-r, r]`，此时拍平成 `[total, 1]`）。
    `OpUtil.NormalizeAxis` 只接受 `[-r, r-1]`，直接在 Flatten 里自己归一化即可。
70. **FP16 导出转 FP32 有四个坑**：① 权重是 F16 initializer；② `ConstantOfShape` 的 `value` 是 F16 **属性**；
    ③ 图里的 `Cast → F16` / `Cast(F16) → F32` 边界要换成 `Identity`；④ `value_info` 里还写着 F16 会让 ORT 加载失败。
    另外 **外部权重必须一模型一文件**：`onnx.save` 是 append 写外部数据、offset 又从 0 重新算，
    同名文件会让后转换的模型读到前一个模型的权重（我们踩过：四个 SAM3 模型共用 `weights.bin`，
    geometry/decoder 全部输出垃圾）。
71. **NaN 定位要「逐算子」，不要「事后扫缓冲」**。引擎的缓冲槽是共享的（生命周期着色），
    事后遍历 Arg 看到的是「某个时刻写过 NaN 的槽」，名字对不上。正确做法：
    `Net.ForwardAll(detectNaN: true)` 在每次 `Layer.Forward` 之后检查该算子的输出，直接报
    「op 'xxx' (Gemm) output #0 [1,256] contains NaN at index 0」。SAM3 就是靠它一步定位到 `text_proj/Gemm`
    的上游 `Clip`。
72. **SAM3 的几个约定**（都是实测确认的）：框提示是**归一化 cxcywh + label=1**（label 只能是 0/1，2 越界）；
    `pred_masks` 是 **logits**（要先 sigmoid 再 0.5 阈值）；掩码 288² 覆盖 1008 的嵌入画布，
    源图坐标 = `mask × (1008/288) × (源尺寸/嵌入尺寸)`；框提示下 200 个 query 会在同一目标上重复命中，
    **必须做框级 NMS**（实测同一人 79 条 → 5 条）。
73. **图像预处理收口成 `Imaging.ImageTensor.ToChw`** 时最容易错的是「通道平面步长」：
    `plane` 必须用**目标缓冲的高度** `destinationHeight × destinationStride`，
    而不是「缩放区域高度」——后者在 letterbox（上下有填边）时会让三个通道互相覆盖，
    表现为检测分数整体偏移（实测 0.7994 → 0.5725），而且不报错。A/B 用真实模型输出对比最可靠。
74. **`.NET` 里没有 `Avx512Vnni`，只有 Intel 专属的 `AvxVnni`**（2026-11-10 用编译验证，不是查文档）：
    在 .NET 10.0.101 里写 `Avx512Vnni.IsSupported` 直接 `CS0103: 名称不存在`，
    `System.Runtime.Intrinsics.dll` 的字节里也搜不到该字符串；而 `AvxVnni` 存在、本机（Zen4）`IsSupported=False`。
    原因：`AvxVnni` 暴露的是 Intel 专属的 **AVX-VNNI（256 位 VEX 编码，12 代酷睿起）**；
    Zen4/Zen5 与 Ice Lake/SPR 只有 **AVX512-VNNI**，.NET 没有包装 → **VNNI 快路径只在 Intel 12 代+ 生效**。
    可用的替代是 `Avx512BW.MultiplyAddAdjacent`（512 位 `vpmaddubsw`，实测 `3×5×4=60` 语义正确）。
75. **非 VNNI 机器上「安全的 int8 计算」打不过 FP32**。按每 32/64 个 MAC 的指令数算：
    `vpdpbusd` 1 条（4×）、`vpmaddubsw`+`vpmaddwd` 单趟 2 条（2×，但 i16 会饱和：255×128×2 > 32767）、
    7 位拆分版 ~8 条（**0.5×，比 FP32 慢**）。所以 Zen4 / 老 Xeon 上的正确路线是
    **「int8 权重存储 + F32 计算」**（按列块在 scratch 里反量化，DRAM 流量降 ~4×），而不是 int8 MAC。
    另外：`[kb][col][4]` 这种为 `vpdpbusd` 设计的打包布局**同样适配 `vpmaddubsw`+`vpmaddwd`**
    （16 个 i16 lane 正好是 8 列 × 4 个 k），两套内核可以共用一份打包。
76. **上一条（#75）的「int8 权重 + F32 计算」实测是错的，别照做**。真按列块反量化再调 `GemmKernel`：
    `96×320×1280` = **10.88 ms vs FP32 1.07 ms（0.10×）**，两次都是 7 GFLOPS。
    慢的不是 `vpmovzxbw` 解包（per call 也就 40 万次），而是**把「小 GEMM 的固定开销（B 面板打包 +
    `Parallel.For` 调度）」乘上了列块数**（1280/64 = 20 次调用）。教训：测「带宽优化」之前先确认
    **不是固定开销主导**。代码留在 `Kernels/Int8DequantGemm.cs` 作反例与回归基准。
77. **ASR 形状的 GEMM 并不慢 —— 我上一版记的「73 GFLOPS」是单线程数字**（已更正）。
    `gemmbench --shape 96,320,1280 --threads 16` 实测 **496 GFLOPS**（`96×320×320` 是 112 GFLOPS）。
    当初那个 73 是在测试进程里量的（同一进程里 int8 路径 7 GFLOPS，比值 0.10× 仍然成立）。
    教训：**A/B 比值可以在测试进程里取，绝对吞吐必须用 CLI + 明确线程数**；
    另外 `gemmbench --shape` 之前根本没生效（`ParseOptions` 把 `--shape` 收进 `Shapes` 列表，
    而 `GemmBench` 读的是通用字典），已修，现在支持重复传多个 `--shape`。
78. **运行期形状只有一条路：`DynamicOutputShapes()` + 在 `Forward` 里 `outputs[i].Fit(...)`**。
    Paraformer 的 `Range(0, ReduceMax(speech_lengths))`、`Reshape(Shape→Gather→Concat)`、
    `ConstantOfShape(pads)` 都属于「形状要跑了才知道」，`GetMemoryShapes` 里 `RequireConstInput`
    必然抛「needs constant folding」。让这三个算子按「输入是否常量」二选一：
    常量走静态推断 + 常量折叠，运行时走动态形状，互不影响（`ConstFolder` 也按同一判据跳过）。
79. **sherpa 的 ONNX 导出不是拓扑序**（`check_model` 报 `Nodes in a graph must be topologically sorted`）。
    ORT 容忍，我们的 importer 不容忍（生产者必须先定义）—— 切图工具里必须自己 Kahn 排序一遍。
    连带后果：`onnx.utils.extract_model` 不可用（它第一步就 `check_model`），只能手写
    「反向 BFS + 拓扑排序 + 补 `value_info`」。
80. **onnx 1.23 默认写 IR version 14，本机 ORT 只认到 13**（`Unsupported model IR version: 14`）。
    脚本里显式 `model.ir_version = 10`。
81. **切图时不能把 `Sequence*` 产出的值当边界**。CIF 的 `Loop` 有 6 个输入，其中 2 个是
    `SequenceEmpty`（循环携带的空序列）：它们既不能作为子图输出（引擎不支持序列），
    也不该进 BFS（否则白带一堆序列算子）。判据：`producer.op_type in {SequenceEmpty, SequenceInsert, ...}` 就跳过。
82. **`Pad` 的负 pads 是「裁剪」，不是「填负数」**。常量模式先整块填常数再按行拷贝时，
    必须按 `[0, outDim)` 夹紧首尾轴：`[3,3]` + `pads=[-1,0,0,0]` 在旧实现里会把 `coords+pads = -1`
    写成越界地址（现在有 `pad: negative pads crop` 单测挡着）。末轴的起始/结束也要夹紧，
    否则会整段拷到输出外面。
83. **`OpUtil.ReadInts` 原来不认 U8 / I8**，而量化算子的 `zero_point` 恰好就是 uint8
    （`cannot read integers from U8 tensor`）。凡是「读量化参数」的路径都要能读 u8/i8 标量。
84. **量化图的对拍要看「逐层」而不是最终输出**。Paraformer encoder 的中间张量差 1e-6~3.6e-5，
    但 CIF 的 frame-count（`cif_output/MatMul` → `floor`）是混沌放大器，最终主输出
    mean|diff| 8e-4、max 8.7e-2，padded 帧却完全一致。
    `tools/scripts/probe_tensors.py` 把任意中间张量追加成 graph output，ORT / 引擎各跑一次，
    直接给「哪个张量开始变坏」；盯着最终输出瞎猜会浪费一整天。
85. **`Cast` 的通用实现是纯标量的**（逐元素 `switch` + `double` 中转，~200 M 元素/秒）。
    Paraformer encoder 里 170 个 `Cast(i32 → f32)` 就占了 **33%**。`Tensors/CastKernel.cs` 给
    `i32↔f32`、`u8/i8→f32`、`u8/i8→i32`、`f32→u8`、`i32→u8` 开了 AVX2 通道后降到 **0.8%**。
    注意 ONNX 的 float→int 是**向零截断**（`Avx.ConvertToVector256Int32WithTruncation`），不是四舍五入。
86. **`Pow` 的向量化通道实际上退化成逐 lane `MathF.Pow`**（~14 ns/元素，35 ms 占 12%）。
    LayerNorm 的 `Pow(x, 2)` 在 `ElementWiseBinary` 里直接改写成 `Mul(x, x)` 即可（0.2 ms）；
    顺手把 `Pow(x,1)` → 拷贝、`Pow(x,0.5)` → `Sqrt`。这类「指数是标量常量」的改写要放在
    通用广播循环之前，否则永远走不到。
87. **广播逐元素还缺一档：「每段一个标量」**。我们原有 ① 同形 ② 标量 ③ 尾部广播（`[384]` 对
    `[1,1024,384]`）三条快路径，但 `[1,97,1] op [1,97,320]`（LayerNorm 减均值、掩码相乘）落到通用
    循环（逐元素算索引，~13 ns/元素，ASR 图里 33 ms）。补 `IsInnerConstantBroadcast` +
    `ApplyInnerScalar` 后降到 ~10 ms —— 这是 SimdOnnx 的 `EwInner` 给我的提示。
    **踩坑**：这条路径隐含假设「其中一个操作数的形状 == 输出形状」，我第一版没判这个前提，
    遇到 `[1,97,1] op [1,1,320]`（两边都不等于输出）时用错了 `full` 的形状，直接越界读崩在
    ASR 用例里。现在两条入口都加了 `x.Shape.Equals(c.Shape)` 前置条件，并有
    `broadcast: …` 三个形状的回归用例钉住。
88. **「加索引算偏移」的小算子要先算清摊薄对象**。`ApplyInnerScalar` 每段只调一次 `ApplyScalar`
    （段长 320 时开销可忽略），但如果段长只有 4~8，逐段调用的固定开销会反噬 —— 这种形状还是走
    通用循环或 `ApplyBroadcast` 更稳。
89. **`Span` 和 `fixed` 局部变量都不能进 lambda**（`CS9108` / `CS1764`）。写并行内核时：
    `fixed (float* p = span) { float* p0 = p; Parallel.For(..., i => 用 p0); }` ——
    必须再赋给一个**普通**指针局部，直接把 `p`（fixed 局部）捕获会被编译器拒绝。
90. **`MatMul + ArgMax` 融合的收益完全取决于 logits 体积**。逐 token 解码 `m=1, n=8000` 只有 32 KB，
    融合只省一次算子调度（0.2 ms 量级）；CTC 那种 `m = 帧数(几百) × n = 词表(几千)` 才是几 MB 的
    写读，实测 **1.8~2.0×**。别指望它在所有 shape 上都加速，做 A/B 再吹。
91. **融合「替换整层」比「就地加输出」简单**：激活融合靠 `op.Outputs[0] = act.Outputs[0]`
    （形状/类型相同才能这么做）；`MatMul → ArgMax` 的输出形状/类型都变了（F32 `[m,n]` → I64 `[m]`），
    只能新建一个 `MatMulArgMax` 层、把 ArgMax 的输出 Arg 挂上去，再 `graph.Reset(...)`。
    另外融合前必须查三件事：中间张量**只有一个消费者**、**不是图输出**、ArgMax 的属性
    是「最后一维 + `keepdims=0` + `select_last_index=0`」（后者通过层内接口暴露给融合器，
    因为 `LayerInfo` 不保存 `LayerParams`）。
92. **`OnnxFile.Parse` 里有两个 `new OnnxFile`**（先造一个临时对象解 initializer，最后再造一个返回）。
    给 `Metadata` 加字段时只在第一个里赋值，结果 `Metadata.Count == 0`，表现为「LFR 参数拿了默认 1/1、
    特征输出 `[598, 80]` 而不是 `[100, 560]`」。凡是新增的「文件级」属性都要两处都带。
93. **梅尔滤波器必须全程 float（连 `MelScale` 也是）**。用 double 算会让「恰好落在三角边上的 FFT bin」
    的**包含关系翻转**（`mel > left` 差 1e-7 就换结果），实测在响亮 bin 上造成 **~3e-4 的相对能量差**；
    照抄 float 之后与 kaldi-native-fbank **54.9% 元素逐位相同、median|diff| 4.8e-7**。
94. **别用「静音 bin 的对数差」当精度指标**。能量贴着 log 下限（`float epsilon` 1.19e-7）时，
    1e-8 的绝对能量差就能放大成 1e-2 的对数差 —— 但那是 ASR 里完全无意义的区间。
    判据要分三层：中位数（浮点级一致）、有声 bin 的最大差、以及静音区间单独放宽，
    否则会为了「修」这个假失败去改本来正确的实现。
95. **标量（rank 0）输入的 dtype 会被丢掉**：`OnnxImporter.ApplyValueInfo` 原来 `Dims.Length == 0` 直接 return，
    于是 `max_label_len`（int64 标量）被当成 F32，跑到 `Slice` 才炸「expected 4 B for [] F32」。
    现在 dtype 无条件记下，只有**形状**交给运行时。
96. **`Slice` 的 starts/ends 可能来自运行时**（Paraformer decoder 的 `Slice(..., max_label_len)`）。
    和 `Reshape`/`Range` 一样处理：`DynamicOutputShapes()` 里判「有没有非常量输入」，`Forward` 里用
    真实张量算出输出形状再 `Fit`；常量路径保持原样（不影响常量折叠）。
97. **decoder 那个 `Loop` 不是 transformer**：body 106 个节点全是「批内 Gather/NonZero/GatherND 取本句
    embedding → ScatterND 写进 padded buffer → `If` 更新最大长度」。单条音频下它等价于
    「memory = CIF 的 `[1, tokenNum, 320]`、max_label_len = tokenNum」，**不需要实现 `Loop` 本身**；
    切图时把这两个输出提升成图输入即可（切完与全模型 `logits` 逐位一致，已用 ORT 验证）。






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

# 编译（双目标 net8.0 + net10.0）
dotnet build OnnxSharp.slnx

# 测试（零依赖运行器；退出码 0 = 全通过）
dotnet run --project tests/OnnxSharp.Tests

# CLI：冒烟 / ONNX 结构统计（M1 已可用）
dotnet run --project tools/OnnxSharp.Cli -- info F:\ai\models\yolo\yolo11n.onnx
dotnet run --project tools/OnnxSharp.Cli -- info F:\ai\rfdetr\onnx_model\rfdetr-small.onnx

# CLI：导入成图并打印每个节点的推断形状（纯 shape 推断，不跑 kernel）
dotnet run --project tools/OnnxSharp.Cli -- dump-graph F:\ai\models\yolo\yolo11n.onnx
dotnet run --project tools/OnnxSharp.Cli -- dump-graph F:\ai\models\yolo\yolo11n.onnx --type Conv --limit 5

# 生成 / 刷新对照数据（形状 oracle + ORT golden）
python tools/scripts/dump_shapes.py F:\ai\models\yolo\yolo11n.onnx --out tests/data/yolo11n.shapes.tsv
python tools/scripts/gen_golden.py F:\ai\models\yolo\yolo11n.onnx F:\ai\models\yolo\golden\yolo11n

# 后续子命令
dotnet run --project tools/OnnxSharp.Cli -- run F:\ai\models\yolo\yolo11n.onnx --input test.bin
dotnet run --project tools/OnnxSharp.Cli -- bench F:\ai\rfdetr\onnx_model\rfdetr-small.onnx --iters 20

# 性能基准（四引擎 × 6 模型，写 bench/results/latest.json，结果抄进 bench.md）
python bench/run_all.py --iters 30 --warmup 10 --rounds 2 --threads 16

# 单算子内核定点测量（--batch 模拟 batched attention；内部按时间预热，勿改成按次数）
$cli = 'tools\OnnxSharp.Cli\bin\Release\net10.0\OnnxSharp.Cli.exe'
& $cli gemmbench --shape 1028,64,1028 --batch 6 --iters 20
& $cli convbench --shape 1,128,80,80,128,3,3,1,1,1

# 微模型（attn-6x1028x64 / matmul-1025x384x1536），产物在 bench/models/，可随时重建
python bench/make_micro_models.py
& $cli bench bench/models/attn-6x1028x64.onnx --input bench/inputs/i1x6x1028x64.f32 --iters 60 --warmup 10
```

> **性能数字一律只抄进 `bench.md`**，本文件与 `plan.md` 只写方法、坑与结论。

M1 + M2（Conv）完成时的实测输出：

```
$ dotnet build OnnxSharp.slnx
已成功生成。0 个警告 0 个错误

$ dotnet run --project tests/OnnxSharp.Tests
running 25 test(s)
... 25/25 passed, 0 skipped, 0 failed

$ dotnet run --project tools/OnnxSharp.Cli -- info F:\ai\models\yolo\yolo11n.onnx
file      : F:\ai\models\yolo\yolo11n.onnx (10.2 MB)
producer  : pytorch    opset: 19
loaded in : 22 ms
nodes     : 320
inits     : 196 (10.1 MB)
inputs    : images:F32[1,3,640,640]
outputs   : output0:F32[1,84,8400]

$ dotnet run --project tools/OnnxSharp.Cli -- dump-graph F:\ai\models\yolo\yolo11n.onnx --limit 1
imported  : 51 ms    ops: 320    slots: 10
  [   1] Conv         /model.0/conv/Conv     -> F32 [1,16,320,320]

$ dotnet run --project tests/OnnxSharp.Tests
        compared 333 tensors against oracle (333 entries)
  PASS  shapes: yolo11n matches onnx.shape_inference
        forward 5382 ms   max|diff|=1.862E-003   mean|diff|=3.293E-006
  PASS  yolo11n: end-to-end matches onnxruntime golden
```

> 注意：**26 个 `TensorProto.DataType` 变体里 `FLOAT = 1`**（`UNDEFINED = 0` 占位）。
> 实现时按「从 0 开始」写偏移一位，两个模型都会在第一个 conv 权重上报 `unsupported data_type 1`。
> `misc.md` 坑 #31 已登记。

> `max|diff|` 1.9e-3 出现在量级 ~267 的输出上（相对误差 ~7e-6），是累加顺序差异，不是算错。
> 5.4 s 是标量直接卷积的耗时，性能优化在 M4。

---

## 8. 许可与合规

- OpenCV 5 的 dnn 文件头为 OpenCV 经典 **3-clause BSD**（`// This file is part of OpenCV project.` + 顶层 `LICENSE`）。照抄/翻译算法需在源码中保留原始版权声明与许可文本。
- ONNX proto 定义（`opencv-onnx.proto` / `onnx.proto`）为 **Apache-2.0**（文件内 SPDX 标注）。
- 本项目 `OnnxSharp` 建议同样采用 Apache-2.0，并在 `NOTICE` 中列出 "Portions derived from OpenCV (BSD-3-Clause)" 与 "ONNX protobuf schema (Apache-2.0)"。

---

## 9. 当前代码结构（实测，M0 ~ M3 完成 + M4 进行中）

```
F:\code\OnnxSharp\
├─ plan.md / misc.md / bench.md          三件套：计划 / 事实库 / 性能（性能唯一事实源）
├─ upstream.lock                         上游 OpenCV dnn 版本锁（JSON）
├─ OnnxSharp.slnx                        src / tests / tools 三个 Folder
├─ bench\                                四引擎对照基准（结果抄进 bench.md）
│  ├─ run_all.py                         编排 OnnxSharp / ORT / OpenCV dnn / SimdYolo × 6 模型
│  ├─ run_python_bench.py                ORT 与 cv2.dnn 单独跑
│  ├─ models.json                        模型清单（路径 / 输入形状 / 是否支持）
│  └─ make_micro_models.py               生成定点测量用微模型（attn / 单 MatMul）
├─ src\OnnxSharp\OnnxSharp.csproj        net8.0;net10.0, AllowUnsafeBlocks, Nullable
│  ├─ Tensors\DType.cs                   元素类型 + SizeOf/IsInteger/IsFloat
│  ├─ Tensors\Layout.cs                  Layout 枚举（仅 NCHW，接口预留）        [M0.5]
│  ├─ Tensors\TensorShape.cs             形状（值类型，params int[]）
│  ├─ Tensors\Tensor.cs                  张量（Fit 复用 / AsSpan<T> / Clone / Layout）
│  ├─ Tensors\TensorOps.cs               广播 + 逐元素（F32）+ 三条快路径        [M4]
│  ├─ Tensors\GemmKernel.cs              GEMM：4×16 AVX2/FMA 微内核 + B 面板打包  [M4]
│  ├─ Tensors\TransposeKernel.cs         块向量转置                             [M4]
│  ├─ Tensors\VectorMath.cs              共用 AVX2 工具（水平归约等）
│  ├─ Kernels\ConvKernels.cs             直接卷积（group/stride/dilation/pad）+ MaxPool [M2+M4]
│  ├─ Kernels\PackedConv.cs              面板打包卷积（16 空间位置，SliceDepth=128）[M4]
│  ├─ Kernels\Winograd.cs                Winograd F(2×2,3×3) 8-tile 向量化        [M4]
│  ├─ Core\Arg.cs                        Arg / ArgKind / ArgData（+ Layout / BufferIndex）
│  ├─ Core\Graph.cs                      prog 线性表 + Version + Reset/RemoveAll   [M0.5]
│  ├─ Core\LayerInfo.cs                  LayerInfo + Layer + Net 反查 + In() + GetLayouts [M0.5]
│  ├─ Core\LayerParams.cs                参数集合
│  ├─ Core\DictValue.cs                  属性值（int/float/string/数组/tensor）
│  ├─ Core\LayerRegistry.cs              显式字典注册（无反射，AOT 友好）
│  ├─ Core\Interpreter.cs                解释循环 + 动态形状分支 + 类型签名         [M0.5]
│  ├─ Core\NetImpl.cs                    Arg 表 + 槽/池 + Declare/Require + PrepareForGraph [M0.5]
│  ├─ Core\Net.cs                        门面（ReadModel / Forward / ForwardAll）
│  ├─ Core\ConstFolder.cs                常量折叠（消除 shape 链）               [M3]
│  ├─ Core\GraphFusion.cs                图级融合 pass（Mul 常量折叠 / Silu / 逐元素链）[M4]
│  ├─ Core\OnnxSharpException.cs
│  ├─ Engine\BufferAllocator.cs          生命周期区间着色，共享缓冲槽             [M0.5]
│  ├─ Onnx\Proto.cs                      protobuf 消息视图（零拷贝）             [M1]
│  ├─ Onnx\OnnxFile.cs                   模型结构 + TensorProto 解码             [M1]
│  ├─ Onnx\OnnxImporter.cs               图导入：initializer/Constant→Const Arg、value_info、属性 [M1]
│  ├─ Ops\OpUtil.cs                      广播/轴/常量读取等公共工具              [M1]
│  ├─ Ops\ShapeOps.cs                    Shape/Reshape/Transpose/Concat/Split/Slice/Gather [M1]
│  ├─ Ops\DnnOps.cs                      Conv/MaxPool/Resize/Softmax/MatMul/LayerNorm/GridSample [M1+M2+M3]
│  ├─ Ops\AttnOps.cs                     Softmax/LayerNorm/Attention 相关          [M3]
│  ├─ Ops\BroadcastOps.cs                Expand/Tile/Where/广播构造                [M3]
│  ├─ Ops\IndexOps.cs                    Gather/GatherElements/TopK/Slice          [M3]
│  ├─ Ops\MiscOps.cs                     Cast/Erf/Sin/Cos/比较/逻辑 等杂项         [M3]
│  ├─ Threading\ParallelRange.cs         parallel_for_ 等价物（小任务直接串行）
│  ├─ Ops\NaryEltwiseLayer.cs            Add/Sub/Mul/Div/Max/Min
│  ├─ Ops\ActivationLayer.cs             Relu/Sigmoid/Sqrt/Exp/Neg/Abs
│  └─ Ops\FusedEltwiseLayer.cs           逐元素链融合执行器（寄存器机 + GELU 手写内核）[M4]
├─ tests\OnnxSharp.Tests\                零依赖测试运行器；29 个用例（net10.0）
│  ├─ Harness.cs                         TestCase / Assert / RunAll / Skip
│  ├─ ArgTableTests.cs                   Declare/Require/TryGetArg/常量元信息
│  ├─ BufferAllocatorTests.cs            槽复用、并发分支、图输出存活、保守策略
│  ├─ EngineTests.cs                     常量驱动形状、temp 池、动态形状、图变更
│  ├─ OnnxReaderTests.cs                 protobuf/类型解码 + 两模型结构断言   [M1]
│  ├─ OnnxShapeTests.cs                  333 张量形状 vs onnx.shape_inference [M1]
│  ├─ TransposeKernelTests.cs            Transpose 全 rank/全 perm 对拍       [M4]
│  ├─ YoloGoldenTests.cs                 yolo11n / yolov8 端到端 vs ORT golden [M2]
│  └─ TestLayers.cs                      ReadConstShapeLayer / DynamicFillLayer
├─ tools\scripts\dump_shapes.py         生成形状 oracle（onnx.shape_inference）
├─ tools\scripts\gen_golden.py          生成 ORT golden 张量               [M2]
└─ tools\OnnxSharp.Cli\                  info / dump-graph / run / bench / gemmbench / convbench
```

命名空间约定：`OnnxSharp.Tensors` / `OnnxSharp.Core` / `OnnxSharp.Ops` / `OnnxSharp.Threading`。
（注意：**不要**用 `OnnxSharp.Graph` 作命名空间，会与 `Graph` 类型名冲突。）

**M0.5 改动清单（✅ 已完成 2026-11-07）**

| 文件 | 原问题 | 落地改法 |
|---|---|---|
| `Core\LayerInfo.cs` | `GetMemoryShapes` 只有 shapes，读不到常量值 | ✅ `public NetImpl? Net { get; internal set; }` + `In(int)` + `IsConstInput(int)` |
| `Core\Interpreter.cs` | ① 无动态形状分支；② 每次 forward 新建临时张量；③ 重 finalize 只比 shapes | ✅ 补动态分支（不预 fit）、改用 `FitTemps` 全局池、判据加 `inpTypes` |
| `Core\NetImpl.cs` | `GetArg` 对未知名字静默新建 Temp Arg | ✅ 拆 `Declare` / `Require` / `TryGetArg`；接 `BufferAllocator` + 槽池 |
| `Core\Graph.cs` | `Prog` 无法删除节点 / 重建 | ✅ `Reset()` / `RemoveAll()` / `Version`（驱动缓冲重算） |
| `Tensors\Tensor.cs` | `Strides` 恒 null；`long→int` 有溢出路径 | ✅ 明确首版只做「紧凑 + 别名」；`Fit` 溢出检查带字节数 |
| `Engine\BufferAllocator.cs` | 不存在 | ✅ 生命周期区间着色（`Analyze` + `Assign`），槽只增不减 |
| `Tensors\Layout.cs` | 不存在 | ✅ 仅 `Layout.NCHW`，纯接口预留 |

**已知取舍**

- 分配器采用**严格 `<`** 判定：某 Arg 的「最后读」与其后 Arg 的「首次写」落在同一 op 时不复用，避免任何「先写输出再读输入」的算子破坏输入；代价是峰值多一代张量。
- 测试运行器是**零第三方依赖**的控制台程序（xunit 未在本地 nuget 缓存、且离线），后续需要时再迁移到 `dotnet test`。

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
| 2026-11-07 | 新增仓库根 **`upstream.lock`**（JSON：remote/branch/commit/scope/verify 命令/history），与 §11.1 互为镜像 |
| 2026-11-07 | **M1 完成**：`Onnx/OnnxImporter.cs`（initializer/Constant→Const Arg、value_info→shape/dtype、属性解析）、`Ops/OpUtil.cs` + `Ops/ShapeOps.cs` + `Ops/DnnOps.cs`（yolo 所需 15 类算子）、CLI `dump-graph`、`OnnxShapeTests`（333 张量全对齐）、`Net.ReadModel` 打通。新增坑 #34（空串占位输入）、#35（`ArgData.Shape` ≠ `Tensor.Shape`） |
| 2026-11-07 | **M2 第一批完成**：`Kernels/ConvKernels.cs` 直接卷积；`tools/scripts/gen_golden.py` + `YoloGoldenTests` 打通 ORT 对照；**yolo11n 端到端跑通，max\\|diff\\|=1.9e-3 / mean\\|diff\\|=3.3e-6，耗时 5.4 s**（标量版，性能留给 M4）。25/25 PASS |
| 2026-11-07 | **M1 第一批完成**：`Onnx/Proto.cs`（零拷贝 protobuf 视图）、`Onnx/OnnxFile.cs`（结构 + TensorProto 解码，含外部数据路径校验）、CLI `info` 子命令、`OnnxReaderTests`（7 个用例）。实测 yolo11n 22ms/320 节点/196 init，rfdetr-small 91ms/1539 节点/354 init。新增坑 #31（DataType 从 1 起）、#32（RF-DETR 双输出）、#33（Constant 才是 I64 来源） |
| 2026-11-07 | **M0.5 落地完成**：`Engine/BufferAllocator.cs`（生命周期着色）、`LayerInfo.Net` + `In()/IsConstInput()`、`Graph.Version/Reset/RemoveAll`、`NetImpl` 槽池 + `FitTemps` + `Declare/Require/TryGetArg`、`Interpreter` 动态形状分支 + 类型签名、`Tensors/Layout.cs` 预留、`tests/OnnxSharp.Tests`（零依赖 16 用例）。`dotnet build` 0 警告 0 错误，16/16 PASS |
| 2026-11-07 | **建立 `bench.md`（性能唯一事实源）+ `bench/` 脚本体系**：`run_all.py` 一条命令跑 4 引擎 × 6 模型（OnnxSharp / ORT CPU / OpenCV 5 dnn / SimdYolo，YOLO×4 + RF-DETR×2），输出 median/min 双指标；实测 yolo11n 43.4 ms、rfdetr-small 317.5 ms。新增坑 #36（分层编译按次数预热会低估 4 倍） |
| 2026-11-07 | **逐元素图级融合落地**（`GraphFusion.FuseElementWiseChains` + `Ops/FusedEltwiseLayer.cs`）：GELU `Div→Erf→Add→Mul→Mul` 编译成单层，rfdetr-small 12 条链、节点 857→801，端到端 **-7.4% / medium -9.1%**，输出 max\|diff\|=0。新增 §6.6 坑 #42~#46（运行期下标的 `Vector256[]` 会被 JIT 打回内存、短链必须设门槛、GELU 实际算子顺序、golden 覆盖不足要用 `run --dump` 逐值 A/B、沿消费者而非生产者遍历）；§9 目录树补 `FusedEltwiseLayer.cs`；CLI `run` 新增 `--dump <dir>` |
| 2026-11-07 | **修 `GridSampleLayer` 的循环顺序 bug**：通道 `c` 原本在最内层而输出是 NCHW（写步长 `outH*outW`，每元素换缓存行），改为 `c` 外提 + 采样坐标预算到 `ArrayPool`。3 个算子 1.6 → 0.92 ms/个（-48%），端到端 rfdetr-small -1.4% / medium -1.1%（min），逐值 diff=0。新增坑 #47（「写得顺不顺」类 bug 不报错只是慢，"算子个数少但占比高"是唯一线索）；记录本机 median 抖动 ±8%，小改动只能用 min 或 profile 判定 |
| 2026-11-07 | **第七轮全表复测 + 修正「已追平 OpenCV」的错误结论**：rfdetr 231.5 / 299.4 ms，对 OpenCV（191.7 / 243.1）为 **1.21× / 1.23×**，对 ORT 0.60× / 0.66×；yolo 侧 vs ORT 0.58~0.89、vs OpenCV 1.16~1.28、vs SimdYolo 1.06~1.63。上一轮记的 1.12 / 1.06 是 **OpenCV 自身发挥失常**（medium 从 304.4 变到 243.1，-20%）造成的假象。新增坑 #48（比值只在双方都落在各自历史区间时可信；跑全表前先用 `(Get-Process).CPU` 增量确认没有满载进程——本轮 `SC2_x64` 占核导致前两轮作废） |
| 2026-11-07 | **上一轮：RF-DETR 瓶颈定位到 batched attention GEMM**（结论后被修正）。新增 §6.5 坑 #36~#39（分层编译预热、A/B 必须交替、两个已否决的内核方向、`Parallel.For` 7 µs 阈值）；`gemmbench` 加 `--batch` 并改为按时间预热；新增 `bench/make_micro_models.py` |
| 2026-11-07 | **本轮：修正上一轮两个错误结论 + 修掉最大单点浪费**。① §3.10 重写：140 个 MatMul 里 MLP（`[257,384]×[384,1536]`）与投影都是真 MatMul，合计 **54.1 GFLOP**（前置 batch 4），in-situ 407 GFLOPS，GEMM 已到顶；② 机器上限用 `fma` 重新量：单线程 91 GFLOPS、16 线程 753 GFLOPS；③ `MatMul` 在 B 不广播时合并 `(batch, m)`（decoder q/k/v 投影是 `[300,1,256]×[256,256]`，权重曾被打包 300 遍，4.06 → 0.13 ms），rfdetr-small **356 → 270 ms（-24%）**、medium -12%；④ `ProfileAll` 加预热与每算子最小耗时（坑 #41）；⑤ 新增 `fma`、`gemmbench --threads`、逐元素带宽与链式 MLP 微模型、`bench` 的 `alloc`/`gc`/`gc pause` 诊断；⑥ 排除 GC（0.7 ms/iter）与降频（长跑不衰减）两个嫌疑，剩下的计时差指向 host 侧每 forward 重算 2064 个算子的形状并分配 113 MB；新增坑 #40（独立内核吞吐无法预测端到端） |
| 2026-11-10 | **M8 SAM3（框提示）跑通 + 图像预处理抽象**：① 引擎补 `IsNaN` / `GreaterOrEqual` / `InstanceNormalization` / `Einsum`(bqc,bchw->bqhw) / `RoiAlign`，算子 67 → **72**；② 修三个真 bug（`Clip` 可选 min/max 占位被当 0 → SAM3 `Div` 除零 NaN、`FusedEltwise` 放行 `[1,1,256]` vs `[256]`、`Flatten` 的 `axis == rank`）；③ 新 `tools/scripts/onnx_fp16_to_f32.py`（F16 权重/属性 + Cast 边界 + value_info + 每模型独立外部权重）；④ 新门面 `src/OnnxSharp/Sam3/`（`Sam3Segmenter`：vision → geometry → decoder、按形状缓存特化图、`Sam3Nms` 框级 NMS、掩码 logits→概率、`MaskScaleX/Y`）；⑤ 新 `src/OnnxSharp/Imaging/`（`ImagePixels` / `ImageGeometry` / `AxisMap` / `ImageTensor.ToChw`），Vision(YOLO/RF-DETR)、OCR、SAM3 三处预处理全部收口，YOLO 检测结果与重构前逐位一致；⑥ 调试能力：`run --debug` 逐算子检 NaN、`--input` 支持 I64/Bool、`--shape name=dims` 多输入；⑦ CLI 新增 `sam3`（`--box/--conf/--iou/--max-items/--format/--out/--json/--debug`）；⑧ 实测：decoder 与 ORT 逐值一致，行人框提示 top-1 **0.932**、整链 30 s；⑨ 坑 #67~#73；⑩ 测试 66/69 PASS（ImagingTests 6 + Sam3Tests 4） |
| 2026-11-10 | **M7 OCR 落地（PP-OCRv6 det/cls/rec 跑在自家引擎上）**：① 引擎新增 `OnnxFile.WithInputShapes` + `Net.ReadModel(path, 输入形状/字典)`，把全动态图（`x[N,3,H,W]`）**特化**成静态图，形状链才能在导入期折叠（`plan.md §5.6`）；② 补 `HardSigmoid` / `BatchNormalization`（推理模式）/ `Identity`，复用 `GlobalAveragePool` / `ReduceMean` / `ConvTranspose` / `AveragePool` / `Squeeze`，注册算子 42 → **67**；③ 修 `PackedConv.Multiply` 无 bias 的空指针（`Conv+BN`）；④ 新门面 `src/OnnxSharp/Ocr/PaddleOCR/`（命名空间 `OnnxSharp.Ocr.PaddleOCR`，全部 OCR 代码都在这一层）：`OcrLine` / `OcrPixelFormat` / 像素→NCHW / CTC 解码 + `PaddleOcrModel` / `PaddleOcrPipeline` / `DbPostprocess` / `PaddleOcrOptions`，检测按 32 对齐、识别按 16 分桶并按形状缓存特化图；⑤ 与 ORT 对齐：det `max|diff|` 1.4e-7、rec 7.6e-6；900×300 合成图两行全对，3200×1800 车牌 `浙A ES8381` / `浙A：:3M15X`；⑥ CLI 新增 `ocr`，图像 IO 改走 **ImageFacade + 额外引入的 `Sdcb.SimdCodecs.Jpeg/Png/Bmp`**（`ImageHost.RegisterCodecs()` 显式注册，`--out` 画框存图），`detect` 也支持 `--out`；⑦ 新增 `tools/scripts/extract_ocr_models.ps1` 与 `OcrTests`（8 用例）；⑧ 坑 #59~#66（动态图特化、无 bias 卷积、缓冲复用与 FromArray、BN 处置、DB 后处理、rec 分桶、对齐口径、格式包边界）；⑨ 语音说明进 README「库用法 6」：sherpa-onnx 模型是 int8 + `Loop/NonZero/GatherND`，不在本引擎范围内 |
| 2026-11-11 | **`FuseAttention`：整条 MHA 折成一层**（详见 §6.19）：新增 `Ops/AttentionOps.cs`（`FusedAttention`：K 打包 `[B,H,T,D]` → 按行并行做 `scale·q·kᵀ(+mask) → softmax → ·V`，不物化 `[B,H,T,T]`）+ `GraphFusion.FuseAttention`；`Transpose`/`Softmax` 各加一个内部接口（`ITransposePerm`/`ISoftmaxAxis`）供匹配读属性。两种形态：静态形状连 `Reshape+Transpose` 一起折，动态 head-split（Paraformer，输出的 Transpose 还被 `Shape` 链用着）留在 head 空间。encoder 40 块全融合（`Softmax` 40→0、`MatMul` 81→1、掩码 `Add` 202→162、算子 2410→2244），`profile` sum-of-min **51.8 → 44.8~47.5 ms（−8~14%）**、端到端 min **81 → 73~84 ms**；新增静态夹具 `make_attention_model.py` + `AttentionTests`（max\|diff\|=1.8e-7），95/98 测试全绿。**如实记一笔**：收益远低于预估的 20~30%，因为 attention 本来只占 13%，66% 是 `DynamicQuantizeMatMul` 的 GEMM |
| 2026-11-11 | **ASR 性能对齐：两个融合 + 去 per-op 开销**（详见 §6.18）：① `Interpreter`/`LayerInfo` 按「输入 shape+type 签名」缓存 `GetMemoryShapes/GetTypes`（encoder ~103 → 86 ms）；② 新 `FuseQuantizedMatMul` + `Ops/QuantizedMatMulOps.cs`：`DQL→MatMulInteger→Cast→Mul→[Add]` 折成 `DynamicQuantizeMatMul`，A 边走量化边展开、scale/bias 走 epilogue（融合层 33.9 ms 取代 `MatMulInteger` 38.0 + `DQL` 8.0 ms）；权重是 **per-channel** 量化（scale/zp 都是 `(960,)` 向量），不能当标量；③ **修 `producerOf` 建表错误**（`Inputs` → `Outputs`），`FuseLayerNorm` 从 1/81 变 **81/81**，并更正 §6.17 的「−35%」实为负载波动；④ 修 net8.0 编不过的 CA2252（VNNI 预览 API）与 CA2014（循环内 `stackalloc`），双目标 0 错 0 警。结果：encoder 69~86 ms、整句热态 **136 → 87 ms**（vs ORT 55.6 ms，**2.4× → 1.56×**），ASR 文本与 94/97 测试不变 |

---

## 11. 上游 OpenCV 5 dnn 变更跟踪

> **长期机制**：后续的算子更新、性能优化、bug 修复，一律先看 OpenCV 5.x `modules/dnn` 的提交，再决定是否移植。
> 参考库是**只读**的：只 `fetch` 与 `diff`，不在里面改代码、不切分支。

### 11.1 跟踪版本台账（**单一事实源**）

本项目「跟到了哪个版本」**只记在这一节**，其它文档一律引用本节，不重复写 commit 号。
机器可读的同一份记录在仓库根的 **`upstream.lock`**（JSON，供脚本/CI 读取）；两处必须同步，冲突时以本节为准。

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
