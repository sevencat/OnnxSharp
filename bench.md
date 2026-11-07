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

### 测量方法上的坑（务必先读）

.NET 默认开启分层编译，热点方法要**调用若干十次**之后才会被提升到 tier-1 优化代码。按「迭代次数」预热（例如预热 3 帧）根本不够：同一个 GEMM 形状，预热不足时实测只有 50 GFLOPS，预热充分后是 490 GFLOPS，**差 4 倍以上**。所以：

- `gemmbench` / `convbench` 已改成**按时间预热**（至少跑满 400 ms 才开始计时）。
- `bench` 的 `--warmup` 是帧数，模型每帧会调用同一个 kernel 几十次，10 帧足够；但**单算子模型**（1~6 个节点）每帧只调用几次，必须把 `--iters` 拉到 60 以上，否则中位数会被 tier-0 的慢样本污染。
- 结论：任何只看 5~10 次迭代的数字都不可信。

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

```powershell
# 微模型（单算子 / 少算子）用于定点测 GEMM 内核，产物在 bench/models/，可随时重建
python bench/make_micro_models.py

# GEMM 定点测量（--batch 模拟 batched attention；内部按时间预热）
tools\OnnxSharp.Cli\bin\Release\net10.0\OnnxSharp.Cli.exe gemmbench --shape 1028,64,1028 --batch 6 --iters 20
```

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
   - profile 显示 DETR 的算力构成与 YOLO 完全不同（rfdetr-small，预热后，`profile` 有额外开销，只看占比）：
     `MatMul 61.7%`、`Add 8.7%`、`Mul 4.9%`、`Transpose 4.9%`、`Softmax 3.9%`、`Conv 3.4%`、`LayerNormalization 2.5%`、`Erf 1.6%`、`GridSample 1.2%`。
   - 也就是说 **Transformer 的 GEMM + 逐元素 + 转置**才是瓶颈，卷积优化（Winograd/面板打包）在 DETR 上只影响 3.4%。
   - 节点构成（rfdetr-small，共 2064 节点）：`Constant 459`、`Add 169`、`MatMul 140`、`Reshape 122`、`Mul 114`、`Transpose 113`、`Slice 62`、`LayerNormalization 49`、`Sqrt 45`、`Cast 36`、`Div 35`。
   - **140 个 MatMul 全是 batched attention**（没有一个是 MLP 的 `[N,384]×[384,1536]`，那些已被导出成 `Conv`）：
     `[6,1028,64]×[6,64,1028]`、`[6,1028,1028]×[6,1028,64]`、`[24,257,64]×[24,64,257]`、`[24,257,257]×[24,257,64]`、`[8,300,32]×[8,32,300]`、`[8,300,300]×[8,300,32]`。
3. **GEMM 微内核本身没问题，问题在并行扩展性**（`gemmbench`，单算子模型与真实模型逐个对齐验证过，误差 <1%）：

   | 形状（batch） | 每 batch ms | 等效 GFLOPS |
   | --- | ---: | ---: |
   | `1028×64×1028`（×6） | 0.49 | 276 |
   | `1028×1028×64`（×6） | 0.39 | 347 |
   | `257×64×257`（×24） | 0.054 | 156 |
   | `257×257×64`（×24） | 0.041 | 205 |
   | `300×32×300`（×8） | 0.096 | 60（未过并行阈值，单线程） |
   | `300×300×32`（×8） | 0.066 | 87（同上） |

   - **单线程**下 `1028×64×1028` 是 136 GFLOPS，正好是 Zen4 单核 AVX2+FMA 的峰值（16 FLOP/cycle × 4.2 GHz ≈ 134）—— 微内核已经吃满单核。
   - 16 线程只有 490 GFLOPS，约为 8 核理论值的 45%。原因有二：① 26~33 个行块分给 16 线程要跑两波，尾块浪费；② 每个行块都要把整个 B 重读一遍（`26 × 262 KB`），实际是 DRAM 带宽受限。
   - 已试过但**没有收益**的两条路（别再重复）：
     - 把整个 B 预打包成 `[列块][k][16]` 供所有行块复用：304 → 234 GFLOPS（打包本来就被 40 行摊薄到 5%，反而多付一份 B 的写回）。
     - 微内核 k 方向 2 步展开：单线程 136 → 135 GFLOPS（JIT 已经展开，没有收益）。
4. **ONNXRuntime 的 median 不可全信**：同一进程内 min 26~34 ms 而 median 49~54 ms（yolo11n），是它的 intra-op spin-wait 线程在后台负载下被抢占后的 barrier 放大效应。真实算力看 min，实际部署体验看 median —— 两者都列了。

## 已知限制（不是失败，是能力边界）

- **SimdYolo 跑不了 RF-DETR**：它的模型读取只认 YOLO 系列检测/分割/姿态头，DETR 输出 `dets[1,300,4] + labels[1,300,91]` 不匹配。
- **OpenCV dnn 能跑 RF-DETR**：3 个 `GridSample`（deformable attention）在新图引擎里有实现，且是其 5.0 的强项之一。
- 所有引擎都只测了 FP32 CPU；GPU / FP16 / INT8 不在本表范围。

## 下一步（按收益排序）

1. **GEMM 的多线程扩展性（当前最大收益点）**：微内核已经吃满单核，缺的是「16 线程只跑出 45%」。方向：
   - 二维分块（m 块 × n 块都做 cache blocking），让每个线程的 B 面板只读一遍，而不是每个行块重读整个 B；
   - batched attention 改成**一个** Parallel 区域里按 `(batch, 行块)` 展平调度，消掉 24 个连续的并行屏障（`257×…×24` 只有 156 GFLOPS，是最差的一组）；
   - 把 `300×32×300` 这类 `m·n·k < 4M` 的形状从「串行」改成「按 batch 并行」。
2. **逐元素融合**：`Add 169 + Mul 114 + Div 35 + Sqrt 45 + Erf` 全是全量读写，占 DETR 约 20%。先做同形状二元链的原地融合。
3. **Transpose 优化**：113 个 Transpose（4.9%），主要是 `[B,H,S,D] ↔ [B,S,H,D]`，可做 8×8 块向量转置。
4. **s 档大 GEMM 打包**：对照 SimdYolo `CpuPackedGemm` 与 OpenCV `fastConv` 的权重 block 布局，调 `Panel / RowBlock / SliceDepth`。
5. 卷积侧：Conv+SiLU 融合、Winograd 覆盖 `5×5`。

## 变更记录

- 2026-11-07：首次建立本文件，四个引擎 × 6 模型（4 YOLO + 2 RF-DETR）全量对照；新增 `bench/run_all.py`、`bench/run_python_bench.py`、`bench/models.json`。
- 2026-11-07（第二轮）：`gemmbench` 支持 `--batch` 并按时间预热（此前预热不足导致数字偏低 4 倍）；定位 RF-DETR 瓶颈为 batched attention GEMM，实测微内核已达单核峰值但多线程只有 45%；验证并否决「B 全量预打包」「k 方向 2 步展开」两个方案。
