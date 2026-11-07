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
# attn / matmul / add / mulscalar / mlp 五类
python bench/make_micro_models.py

# GEMM 定点测量（--batch 模拟 batched attention；内部按时间预热）
tools\OnnxSharp.Cli\bin\Release\net10.0\OnnxSharp.Cli.exe gemmbench --shape 1028,64,1028 --batch 6 --iters 20
# 指定线程数，用来区分「内核差」和「并行调度差」
tools\OnnxSharp.Cli\bin\Release\net10.0\OnnxSharp.Cli.exe gemmbench --shape 1028,64,1028 --threads 1,2,4,8,16

# 机器算力上限（纯 FMA 微循环，用来看 GEMM 离上限还有多远）
tools\OnnxSharp.Cli\bin\Release\net10.0\OnnxSharp.Cli.exe fma --threads 1,8,16

# 逐算子热点（务必给 --warmup：tier-0 代码会让计算密集算子虚高数倍）
tools\OnnxSharp.Cli\bin\Release\net10.0\OnnxSharp.Cli.exe profile <model.onnx> --input <in.f32> --iters 20 --warmup 10
```

> `profile` 的绝对值也别信：后台负载会让均值比真实值高 20% 以上。
> 输出里同时给了每算子的最小耗时，**把各算子 min 相加（rfdetr-small 约 215 ms）才和 `bench` 的中位数对得上**。

## 结果（2026-11-07，16 线程）

单位 ms，越低越好。**加粗为同组最快**。

| 模型 | 输入 | OnnxSharp | ORT CPU | OpenCV dnn | SimdYolo |
| --- | --- | ---: | ---: | ---: | ---: |
| yolov8n | 640² | 42.4 | 48.6 | **31.1** | 35.0 |
| yolo11n | 640² | 36.5 | 44.6 | **31.7** | 39.3 |
| yolov8s | 640² | 90.2 | 159.0 | 81.1 | **58.5** |
| yolo11s | 640² | 88.3 | 135.0 | 75.1 | **59.0** |
| rfdetr-small | 512² | 275.1 | 495.7 | **195.3** | N/A |
| rfdetr-medium | 512² | 348.8 | 611.2 | **253.7** | N/A |

min（无干扰下限）：

| 模型 | OnnxSharp | ORT CPU | OpenCV dnn | SimdYolo |
| --- | ---: | ---: | ---: | ---: |
| yolov8n | 35.0 | 33.2 | 29.8 | **28.9** |
| yolo11n | 32.9 | 29.1 | **28.7** | 34.3 |
| yolov8s | 88.3 | 113.6 | 78.4 | **54.8** |
| yolo11s | 83.3 | 93.9 | 72.7 | **53.9** |
| rfdetr-small | 268.1 | 349.2 | **190.9** | N/A |
| rfdetr-medium | 329.2 | 492.1 | **248.8** | N/A |

相对倍率（OnnxSharp ÷ 该引擎，>1 表示比它慢）：

| 模型 | vs ORT | vs OpenCV dnn | vs SimdYolo |
| --- | ---: | ---: | ---: |
| yolov8n | 0.87 | 1.36 | 1.21 |
| yolo11n | 0.82 | 1.15 | 0.93 |
| yolov8s | 0.57 | 1.11 | 1.54 |
| yolo11s | 0.65 | 1.18 | 1.50 |
| rfdetr-small | 0.55 | **1.41** | N/A |
| rfdetr-medium | 0.57 | **1.37** | N/A |

> **跨会话不要比绝对值**：同一份代码，机器状态不同时 yolo11n 能测出 36.5 / 43.4 / 50.2 三个数（±20%）。
> 上表全部来自同一次 `run_all.py` 会话，所以引擎之间的**倍率**可信，绝对值只作趋势。
> 与上一版相比 rfdetr 的倍率从 1.68/1.70 改善到 1.41/1.37，原因见结论 2。

## 结论

1. **YOLO 侧**：OnnxSharp 稳定快于 ONNXRuntime CPU（0.57~0.87×），与 OpenCV 5 dnn 差 11%~36%，落后 SimdYolo 最大 1.54×（yolov8s）。
   - SimdYolo 在 `s` 档领先明显、`n` 档优势很小，说明差距不在"单个卷积"而在**通道数变大后的大 GEMM 分块/打包**：n 档层宽 64~128 时双方都喂不满 L1/L2，s 档 128~256 时我们的 `RowBlock=4 / Panel=16 / SliceDepth=128` 组合明显吃亏。
   - OpenCV dnn 的 `fastConv` 胜在按 `blocked` 布局一次性重排权重 + Winograd 覆盖更宽，是我们下一轮的主要参照。
2. **RF-DETR 侧**：我们比 ORT 快（0.55~0.57×），比 OpenCV dnn 慢 1.37~1.41×，是当前最大短板。
   - `profile`（rfdetr-small，预热 10 轮、20 轮统计、按每算子最小耗时汇总到 215 ms）的构成：
     `MatMul 61.9%`、`Add 8.7%`、`Transpose 5.6%`、`Mul 5.4%`、`Conv 3.4%`、`Softmax 3.3%`、`LayerNormalization 2.3%`。
   - 也就是说 **Transformer 的 GEMM + 逐元素 + 转置**才是瓶颈，卷积优化（Winograd/面板打包）在 DETR 上只影响 3.4%。
   - 节点构成（rfdetr-small，共 2064 节点）：`Constant 459`、`Add 169`、`MatMul 140`、`Reshape 122`、`Mul 114`、`Transpose 113`、`Slice 62`、`LayerNormalization 49`、`Sqrt 45`、`Cast 36`、`Div 35`。
   - **140 个 MatMul 不全是 batched attention**（此前记错了）。按 `(a[-2:], b[-2:])` 统计真实形状，MLP 与投影都是货真价实的 MatMul 节点：
     `[257,384]×[384,1536]` ×12（主干 fc1，单算子 1.26 GFLOP）、`[257,1536]×[1536,384]` ×12（fc2）、
     `[1028,384]×[384,384]` ×6 等投影、以及 attention 的 `[6,1028,64]×[6,64,1028]`、`[24,257,64]×[24,64,257]`、`[8,300,32]×[8,32,300]`。
     全部 140 个合计约 **16.6 GFLOP**，占模型约 92%。
   - **最大的一处浪费（已修复）**：DETR decoder 的 q/k/v 投影运行时是 `[300,1,256]×[256,256]`（300 个 batch、每 batch m=1）。
     引擎按 batch 逐个调内核，于是 256 KB 的权重被重新打包 300 遍，单个算子 4.06 ms；
     而 `(batch, m)` 合并后只是一次 `300×256×256` 的 GEMM，0.13 ms。9 个这样的算子，端到端 **356 → 270 ms（-24%）**。
3. **GEMM 微内核本身没问题，问题在并行扩展性**（`gemmbench` 与 `fma` 均为按时间预热后的稳态值）：

   | 形状（batch） | 每 batch ms | 等效 GFLOPS |
   | --- | ---: | ---: |
   | `257×384×1536`（×1） | 0.88 | 343 |
   | `257×1536×384`（×1） | 0.97 | 313 |
   | `1028×64×1028`（×6） | 0.49 | 276 |
   | `1028×1028×64`（×6） | 0.39 | 347 |
   | `257×64×257`（×24） | 0.054 | 156 |
   | `300×32×300`（×8） | 0.096 | 60（未过并行阈值，单线程） |

   - 机器上限：纯 FMA 微循环 `fma` 实测单线程 91 GFLOPS、16 线程 **753 GFLOPS**（16 线程靠 SMT 超过 8 个物理核）。
     GEMM 最好也只到约 500 GFLOPS（`257,1536,384` 的列切分版），常规形状 300~350 GFLOPS，即机器峰值的 40%~46%。
   - 26~33 个行块分给 16 线程要跑两波、尾块浪费，且每个行块都要把整个 B 重读一遍（`26 × 262 KB`），是 DRAM 带宽受限。
   - **已试过但端到端没有收益的方案（别再重复）**：
     - 把整个 B 预打包成 `[列块][k][16]` 供所有行块复用：微基准 304 → 234 GFLOPS（反而多付一份 B 的写回）。
     - 微内核 k 方向 2 步展开：单线程 136 → 135 GFLOPS（JIT 已经展开）。
     - 行块高度 `mb` 扫参（16/32/64/128）：微基准 250~340 GFLOPS 波动，端到端无差异。
     - 按流量把切分方向从行切换成列切：微基准 `257,1536,384` 343 → **499 GFLOPS（+45%）**，但端到端交替 A/B 三轮完全一致（273~277 ms 双方相同）。
   - 这三条合起来是这个项目最重要的一条经验：**独立内核吞吐无法预测端到端**（三次独立验证）。
     微基准只能用来排除明显不合理的实现，任何优化都必须用「同会话、交替运行两个二进制」的端到端 A/B 来判定。
4. **ONNXRuntime 的 median 不可全信**：同一进程内 min 29~33 ms 而 median 44~49 ms（yolo11n），是它的 intra-op spin-wait 线程在后台负载下被抢占后的 barrier 放大效应。真实算力看 min，实际部署体验看 median —— 两者都列了。

## 已知限制（不是失败，是能力边界）

- **SimdYolo 跑不了 RF-DETR**：它的模型读取只认 YOLO 系列检测/分割/姿态头，DETR 输出 `dets[1,300,4] + labels[1,300,91]` 不匹配。
- **OpenCV dnn 能跑 RF-DETR**：3 个 `GridSample`（deformable attention）在新图引擎里有实现，且是其 5.0 的强项之一。
- 所有引擎都只测了 FP32 CPU；GPU / FP16 / INT8 不在本表范围。

## 下一步（按收益排序）

1. **逐元素融合**：`Add 169 + Mul 114 + Div 35 + Sqrt 45 + Erf` 全是全量读写，占 DETR 约 20%。
   逐元素单个算子实测 56 GB/s，已到单线程 memcpy 上限（73 GB/s）的 77%，**没有带宽余量，只能靠减少趟数**。
   先做同形状二元链的原地融合。
2. **搞清「独立内核快 ≠ 端到端快」**（见结论 3）：MatMul 仍占 61.9%（133 ms / 16.6 GFLOP ≈ 125 GFLOPS），
   而同样的形状独立跑是 313~343 GFLOPS。把链式 MLP 微模型（24 个 MatMul，实测 280 GFLOPS 与独立一致）
   逐步加回真实模型里的其它算子，找出到底是谁把缓存挤掉了 —— 在此之前继续调 GEMM 内核都是白费。
3. **Transpose**：113 个（5.6%），已是 8×8 向量分块 + 同布局 memcpy 快路径，剩余空间有限。
4. **s 档大 GEMM 打包**：对照 SimdYolo `CpuPackedGemm` 与 OpenCV `fastConv` 的权重 block 布局，调 `Panel / RowBlock / SliceDepth`。
5. 卷积侧：Conv+SiLU 融合、Winograd 覆盖 `5×5` —— 但 DETR 上 Conv 只占 3.4%，收益上限很低。

## 变更记录

- 2026-11-07：首次建立本文件，四个引擎 × 6 模型（4 YOLO + 2 RF-DETR）全量对照；新增 `bench/run_all.py`、`bench/run_python_bench.py`、`bench/models.json`。
- 2026-11-07（第二轮）：`gemmbench` 支持 `--batch` 并按时间预热（此前预热不足导致数字偏低 4 倍）；定位 RF-DETR 瓶颈为 batched attention GEMM，实测微内核已达单核峰值但多线程只有 45%；验证并否决「B 全量预打包」「k 方向 2 步展开」两个方案。
- 2026-11-07（第三轮）：纠正上一轮两个错误结论 —— ①「MatMul 全是 batched attention、MLP 已导出成 Conv」不成立，MLP 的 `[257,384]×[384,1536]` 就是 MatMul 节点，合计 16.6 GFLOP；② 单线程 GEMM 不是 136 GFLOPS，机器 16 线程 FMA 峰值 753 GFLOPS、GEMM 最好约 500。
  修掉两个测量坑：`profile` 增加 `--warmup`（不预热时 tier-0 代码让 MatMul 份额虚高到 66%）、改用每算子最小耗时汇总（均值被后台负载重尾放大 20%+）。
  修复最大单点浪费：MatMul 在 B 不参与广播时把 `(batch, m)` 合并成一次 GEMM，decoder q/k/v 投影 4.06 → 0.13 ms，rfdetr-small -24%、medium -12%。
  验证并否决「按流量改切分方向」（内核 +45%，端到端 0）；新增 `fma` / `gemmbench --threads` 与逐元素带宽、链式 MLP 两个微模型。
  全表重新测量（同会话），rfdetr 对 OpenCV 的倍率从 1.68/1.70 降到 1.41/1.37。
