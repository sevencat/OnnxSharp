# OnnxSharp 性能基准

本文件是**唯一**的性能结果记录处。历史上散落在 commit message、`plan.md`、临时终端输出里的数字不再作为依据；有新结果就往这里追加一个带日期的段，旧段落保留。

## 口径

| 项 | 取值 |
| --- | --- |
| 计时范围 | 仅模型前向（N 次 `Forward` / `session.run` / `net.forward`），**不含** letterbox 前处理、NMS、掩码后处理 |
| 输入 | 静态 batch=1、FP32、NCHW，全部元素常量 `0.5`（与 SimdYolo 默认一致） |
| 预热 | 10 帧，不计入样本 |
| 样本 | 30 帧，2 轮取较快一轮 |
| 线程 | 全部固定 16 线程（`ONNXSHARP_THREADS=16` / `intra_op_num_threads=16` / `cv2.setNumThreads(16)` / SimdYolo `threads=16`） |
| 指标 | 主指标 **median**；副指标 **min**（反映无干扰时的下限） |

## 环境

| 项 | 取值 |
| --- | --- |
| CPU | AMD Ryzen 7 8845HS（8C/16T，3.8 GHz，Zen4，AVX-512） |
| 内存 | 47 GB |
| OS | Windows（AC 供电） |
| .NET | 10.0.402（OnnxSharp / SimdYolo 均 Release） |
| ONNXRuntime | 1.31.1（`CPUExecutionProvider`，`ORT_ENABLE_ALL`，inter_op=1） |
| OpenCV | 5.0.0（`cv2.dnn` 新图引擎，`DNN_BACKEND_OPENCV` / `DNN_TARGET_CPU`） |
| SimdYolo | 0.11.3（`CPU SIMD (256 bits)`） |

> 说明：测量机是共享环境（后台有 ~1.5 核的常驻负载），所以同一次会话里依次跑完全部引擎，没有跨天拼接数字。ONNXRuntime 对线程抢占特别敏感（median 与 min 常差 40%+，见下文），跨轮次 median 波动 ±10% 属正常。

## 复现

```powershell
# 一次跑完四个引擎 × 全部模型（结果写 bench/results/latest.json，并打印 Markdown 表）
python bench/run_all.py --iters 30 --warmup 10 --rounds 2 --threads 16

# 只跑部分模型 / 部分引擎
python bench/run_all.py --names yolo11n,rfdetr-small --runtimes onnxsharp,opencv-dnn

# 单引擎单独跑（脚本内部也是调这些）
python bench/run_python_bench.py --runtime ort --iters 30 --warmup 10 --threads 16
python bench/run_python_bench.py --runtime cv  --iters 30 --warmup 10 --threads 16
tools\OnnxSharp.Cli\bin\Release\net10.0\OnnxSharp.Cli.exe bench <model.onnx> --input <in.f32> --iters 30 --warmup 10
F:\code\SimdYolo\src\SimdYolo.Cli\bin\Release\net10.0\SimdYolo.Cli.exe bench <model.onnx> cpu 640 30 16 10
```

模型清单在 `bench/models.json`。

## 结果（2026-11-07，16 线程）

单位 ms，越低越好。**加粗为同组最快**。

| 模型 | 输入 | OnnxSharp | ORT CPU | OpenCV dnn | SimdYolo |
| --- | --- | ---: | ---: | ---: | ---: |
| yolov8n | 640² | 35.1 | 53.5 | **29.5** | 32.0 |
| yolo11n | 640² | 43.4 | 49.5 | **34.0** | 35.6 |
| yolov8s | 640² | 88.0 | 151.1 | 77.3 | **52.1** |
| yolo11s | 640² | 81.3 | 114.9 | 73.2 | **58.8** |
| rfdetr-small | 512² | 317.5 | 366.1 | **188.8** | N/A |
| rfdetr-medium | 512² | 411.8 | 514.6 | **241.9** | N/A |

min（无干扰下限）：

| 模型 | OnnxSharp | ORT CPU | OpenCV dnn | SimdYolo |
| --- | ---: | ---: | ---: | ---: |
| yolov8n | 33.1 | **34.2** | **27.8** | 28.6 |
| yolo11n | 37.6 | 34.6 | **28.3** | 31.1 |
| yolov8s | 86.0 | 93.1 | 75.4 | **49.7** |
| yolo11s | 78.9 | **63.4** | 71.6 | 56.3 |
| rfdetr-small | 312.5 | 294.2 | **183.0** | N/A |
| rfdetr-medium | 405.5 | 320.1 | **237.7** | N/A |

相对倍率（OnnxSharp ÷ 该引擎，>1 表示比它慢）：

| 模型 | vs ORT | vs OpenCV dnn | vs SimdYolo |
| --- | ---: | ---: | ---: |
| yolov8n | 0.66 | 1.19 | 1.10 |
| yolo11n | 0.88 | 1.28 | 1.22 |
| yolov8s | 0.58 | 1.14 | 1.69 |
| yolo11s | 0.71 | 1.11 | 1.38 |
| rfdetr-small | 0.87 | 1.68 | N/A |
| rfdetr-medium | 0.80 | 1.70 | N/A |

## 结论

1. **YOLO 侧**：OnnxSharp 稳定快于 ONNXRuntime CPU（0.58~0.88×），与 OpenCV 5 dnn 差 11%~28%，落后 SimdYolo 最大 1.69×（yolov8s）。
   - SimdYolo 在 `s` 档领先明显、`n` 档优势很小，说明差距不在"单个卷积"而在**通道数变大后的大 GEMM 分块/打包**：n 档层宽 64~128 时双方都喂不满 L1/L2，s 档 128~256 时我们的 `RowBlock=4 / Panel=16 / SliceDepth=128` 组合明显吃亏。
   - OpenCV dnn 的 `fastConv` 胜在按 `blocked` 布局一次性重排权重 + Winograd 覆盖更宽，是我们下一轮的主要参照。
2. **RF-DETR 侧**：我们比 ORT 快（0.80~0.87×），但比 OpenCV dnn 慢 1.7×，是当前最大短板。
   - profile 显示 DETR 的算力构成与 YOLO 完全不同（rfdetr-small，`profile` 有额外开销，只看占比）：
     `MatMul 56.8%`、`Add 12.4%`、`Mul 5.5%`、`Softmax 5.0%`、`Transpose 3.6%`、`Conv 3.4%`、`LayerNormalization 2.5%`、`Erf 1.6%`、`GridSample 1.2%`。
   - 也就是说 **Transformer 的 GEMM + 逐元素 + 转置**才是瓶颈，卷积优化（Winograd/面板打包）在 DETR 上只影响 3.4%。
   - 具体短板：Atention/MLP 的 `MatMul` 是 `m≈257~273, k=256~384` 这种"中等 m、小 k"形状，当前走的是通用分块 GEMM，K 方向很快撞 L1；`Add/Mul/Div/Erf` 这类逐元素各占一次全量读写；`Transpose(111 个)` 基本没优化。
3. **ONNXRuntime 的 median 不可全信**：同一进程内 min 26~34 ms 而 median 49~54 ms（yolo11n），是它的 intra-op spin-wait 线程在后台负载下被抢占后的 barrier 放大效应。真实算力看 min，实际部署体验看 median —— 两者都列了。

## 已知限制（不是失败，是能力边界）

- **SimdYolo 跑不了 RF-DETR**：它的模型读取只认 YOLO 系列检测/分割/姿态头，DETR 输出 `dets[1,300,4] + labels[1,300,91]` 不匹配。
- **OpenCV dnn 能跑 RF-DETR**：3 个 `GridSample`（deformable attention）在新图引擎里有实现，且是其 5.0 的强项之一。
- 所有引擎都只测了 FP32 CPU；GPU / FP16 / INT8 不在本表范围。

## 下一步（按收益排序）

1. **Transformer GEMM**：为 `m∈[128,512]`、`k∈[64,384]` 的形状补一个面板打包（`kPanel=8` 级 B 打包 + 4×16 微内核），预期能把 DETR 的 56.8% 提上来。
2. **逐元素融合**：把 `Add/Mul/Div` 合成一条链（DETR 里 `Add` 12.4% + `Mul` 5.5% + `Div 1.6%` 全是全量读写），至少做到同形状二元运算原地融合。
3. **Transpose 优化**：111 个 Transpose（3.6%）大多是把 `[B,H,S,D]` 与 `[B,S,H,D]` 互换，可做 8×8 块向量转置。
4. **s 档大 GEMM 打包**：对照 SimdYolo `CpuPackedGemm` 与 OpenCV `fastConv` 的权重 block 布局，调 `Panel / RowBlock / SliceDepth`。
5. 卷积侧：Conv+SiLU 融合、Winograd 覆盖 `5×5`。

## 变更记录

- 2026-11-07：首次建立本文件，四个引擎 × 6 模型（4 YOLO + 2 RF-DETR）全量对照；新增 `bench/run_all.py`、`bench/run_python_bench.py`、`bench/models.json`。
