# OnnxSharp 性能基准

本文件是**唯一**的性能结果记录处。历史上散落在 commit message、`plan.md`、临时终端输出里的数字不再作为依据；有新结果就往这里追加一个带日期的段，旧段落保留。

## 口径

| 项 | 取值 |
| --- | --- |
| 计时范围 | 仅模型前向（N 次 `Forward` / `session.run` / `net.forward`），**不含** letterbox 前处理、NMS、掩码后处理 |
| 输入 | 静态 batch=1、FP32、NCHW，全部元素常量 `0.5`（与 SimdYolo 默认一致） |
| 预热 | `run_all.py` 默认 5 帧（`warmup=5`），不计入样本 |
| 样本 | 默认 40 帧（`iters=40`），跑 2 轮取 median 较低的一轮 |
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
tools\OnnxSharp.Cli\bin\Release\net10.0\OnnxSharp.Cli.exe run  <model.onnx> --input <in.f32> --dump out/   # 输出按 F32 落盘（逐值 A/B 用）
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

> `profile` 的输出怎么和 `bench` 对账：**各算子均值之和**才对得上端到端中位数
> （rfdetr-small：251 ms vs 250~275 ms）。**各算子最小值之和只是下界**（215 ms），
> 不能拿去和中位数比 —— 每个算子的最小值基本不会同时出现，差 15% 是正常的，不要当成「有隐藏开销」。

## 结果（2026-11-07 第四轮，16 线程）

单位 ms，越低越好。**加粗为同组最快**。全部来自同一次 `python bench/run_all.py`（iters=40、warmup=5、rounds=2 取最优轮）。

| 模型 | 输入 | OnnxSharp | ORT CPU | OpenCV dnn | SimdYolo |
| --- | --- | ---: | ---: | ---: | ---: |
| yolov8n | 640² | 36.9 | 59.3 | **30.4** | **30.4** |
| yolo11n | 640² | 33.7 | 46.5 | **28.9** | 30.0 |
| yolov8s | 640² | 92.2 | 149.4 | 77.2 | **51.8** |
| yolo11s | 640² | 83.1 | 140.6 | 74.0 | **53.9** |
| rfdetr-small | 512² | 230.4 | 432.6 | **204.9** | N/A |
| rfdetr-medium | 576² | 321.5 | 638.4 | **304.4** | N/A |

min（无干扰下限）：

| 模型 | OnnxSharp | ORT CPU | OpenCV dnn | SimdYolo |
| --- | ---: | ---: | ---: | ---: |
| yolov8n | 33.6 | 30.0 | 28.8 | **27.5** |
| yolo11n | 32.0 | **25.7** | 27.9 | 26.2 |
| yolov8s | 86.7 | 89.6 | 72.7 | **49.2** |
| yolo11s | 79.4 | 102.3 | 70.2 | **49.7** |
| rfdetr-small | 220.0 | 317.6 | **194.0** | N/A |
| rfdetr-medium | 304.0 | 458.6 | **263.5** | N/A |

相对倍率（OnnxSharp ÷ 该引擎 median，>1 表示比它慢）：

| 模型 | vs ORT | vs OpenCV dnn | vs SimdYolo |
| --- | ---: | ---: | ---: |
| yolov8n | 0.62 | 1.21 | 1.21 |
| yolo11n | 0.72 | 1.17 | 1.12 |
| yolov8s | 0.62 | 1.19 | **1.78** |
| yolo11s | 0.59 | 1.12 | 1.54 |
| rfdetr-small | 0.53 | **1.12** | N/A |
| rfdetr-medium | 0.50 | **1.06** | N/A |

> **跨会话不要比绝对值**：同一份代码，机器状态不同时 yolo11n 能测出 32~50 ms（±20%）。
> 上表全部来自同一次会话，所以引擎之间的**倍率**可信，绝对值只作趋势。
> 与上一版（275.1 / 348.8）相比 rfdetr 的 vs OpenCV 倍率从 1.41 / 1.37 改善到 **1.12 / 1.06**，几乎追平，原因见结论 2（融合 + MatMul 合并 batch）。

### 4.1 与上一轮的 OnnxSharp 自比（同一口径，median / min）

| 模型 | 上一轮 median | 本轮 median | 变化 | 上一轮 min | 本轮 min |
| --- | ---: | ---: | ---: | ---: | ---: |
| yolov8n | 42.4 | 36.9 | -13% | 35.0 | 33.6 |
| yolo11n | 36.5 | 33.7 | -8% | 32.9 | 32.0 |
| yolov8s | 90.2 | 92.2 | +2% | 88.3 | 86.7 |
| yolo11s | 88.3 | 83.1 | -6% | 83.3 | 79.4 |
| rfdetr-small | 275.1 | 230.4 | **-16%** | 268.1 | 220.0 |
| rfdetr-medium | 348.8 | 321.5 | **-8%** | 329.2 | 304.0 |

> yolo 侧的 ±13% 属于跨会话噪声（本轮机器更空），**不应算作优化收益**；
> rfdetr 的 -16% / -8% 与结论 2 的交替 A/B（-7.4% / -9.1%）量级一致，其中 medium 偏小是因为它的 576² 形状让 GEMM 占比更高。

## 结论

1. **YOLO 侧**：OnnxSharp 稳定快于 ONNXRuntime CPU（0.59~0.72×），与 OpenCV 5 dnn 差 **6%~21%**，落后 SimdYolo 最大 1.78×（yolov8s）。
   - SimdYolo 在 `s` 档领先明显、`n` 档优势很小，说明差距不在"单个卷积"而在**通道数变大后的大 GEMM 分块/打包**：n 档层宽 64~128 时双方都喂不满 L1/L2，s 档 128~256 时我们的 `RowBlock=4 / Panel=16 / SliceDepth=128` 组合明显吃亏。
   - OpenCV dnn 的 `fastConv` 胜在按 `blocked` 布局一次性重排权重 + Winograd 覆盖更宽，是我们下一轮的主要参照。
   - **yolov8s 的 1.78× 是唯一还在恶化的项**（上一轮 1.54×）：本轮绝对值 92.2 vs 上一轮 90.2 基本没变，是 SimdYolo 那边从 58.5 掉到 51.8 拉大了倍率，属于对方变快而不是我们变慢。但 `s` 档确实是唯一没被本轮优化覆盖的区域。
2. **RF-DETR 侧**：我们比 ORT 快（0.50~0.53×），比 OpenCV dnn 只慢 **1.06~1.12×**（上一轮 1.37~1.41×），已经从"最大短板"变成"基本追平"。
   - `profile`（rfdetr-small，预热 10 轮、统计 20 轮；下面是按**每算子最小耗时**汇总的**占比**，总时长合计 215 ms 只是下界）：
     `MatMul 61.9%`、`Add 8.7%`、`Transpose 5.6%`、`Mul 5.4%`、`Conv 3.4%`、`Softmax 3.3%`、`LayerNormalization 2.3%`。
   - **融合后的分布（2026-11-07 第四轮实测，`profile ... --top 1` 的 by-type 段，sum-of-min 211 ms）**：
     `MatMul 64.3%`（139 个 / 135.7 ms）、`Add 6.6%`（155 个 / 13.9 ms）、`Transpose 6.1%`（111 个 / 12.9 ms）、
     `Conv 3.6%`、`Softmax 3.4%`、`Mul 2.3%`、`LayerNormalization 2.3%`、`GridSample 2.3%`、
     `FusedEltwise 1.8%`（12 个 / 3.8 ms）、`Reshape 1.6%`、`Where 1.1%`。
     - `Div` 从 35 个降到 7 个（其余被融进链里），`Add + Mul` 从 14.1% 降到 8.9%，代价是新增 `FusedEltwise 1.8%` —— 净省约 3.4 个百分点，与端到端 -7% 吻合。
     - **MatMul 的占比反而升到 64.3%**，这正是把别的部分做快之后的必然结果；它本身已经 407 GFLOPS（见下），
       所以 rfdetr 剩下的空间只有 `Add`、`Transpose`、`Conv`、`Softmax` 这几个 3~7% 的碎块，**再也没有单点能给出 10%**。
     - 注意 `Add 155 个 / 13.9 ms` 是**融合后剩下的**节点：它们要么每个中间结果有多个消费者（残差、attention 的加回），
       要么链长不足门槛，已经不是「漏融合」而是「结构上不能融合」。
   - **`GridSample` 修复后（2026-11-07 第六轮）**：3 个 `cross_attn/GridSample` 从 `1.6 ms/个`（合计 4.8 ms、2.3%）
     降到 `0.92 ms/个`（合计 **2.5 ms、1.2%**，**-48%**），同口径 by-type 表里
     `MatMul 64.8%`、`Add 6.7%`、`Transpose 6.2%`、`Conv 3.8%`、`Softmax 3.4%`、`Mul 2.4%`、`LayerNorm 2.4%`、
     `FusedEltwise 1.8%`、`Reshape 1.6%`、`GridSample 1.2%`。
     端到端交替 A/B **6 轮**（同会话，只取 min —— 本机 median 不可用，yolo 同一份代码都能抖 ±8%）：
     rfdetr-small **225.4 → 222.2 ms（-1.4%）**、rfdetr-medium **290.9 → 287.8 ms（-1.1%）**，
     与「省 2.3 ms / 225 ms ≈ 1%」的预期一致。4 个模型 base/fused 输出 `max|diff| = 0`。
     - 剩下的 0.92 ms 仍偏慢（C=256、spatial=300 → 76800 元素，约 83 M 元素/s），
       瓶颈已从"写不连续"变成"每元素 4 次随机读 + 4 次边界判断"。再往上要预计算四个邻居的平面内偏移
       （4×300 个 int，L1 装得下），把内层压成纯 lerp —— 预计还能再省一半，但端到端只有 0.6%，**优先级低于 Transpose**。
   - 也就是说 **Transformer 的 GEMM + 逐元素 + 转置**才是瓶颈，卷积优化（Winograd/面板打包）在 DETR 上只影响 3.4%。
   - 节点构成（rfdetr-small，共 2064 节点）：`Constant 459`、`Add 169`、`MatMul 140`、`Reshape 122`、`Mul 114`、`Transpose 113`、`Slice 62`、`LayerNormalization 49`、`Sqrt 45`、`Cast 36`、`Div 35`。
   - **140 个 MatMul 不全是 batched attention**（此前记错了）。按 `(a[-2:], b[-2:], 前置 batch)` 统计，MLP 与投影都是货真价实的 MatMul 节点：

     | 形状（前置 batch） | 组数 | 合计 FLOPs |
     | --- | ---: | ---: |
     | `(257,384)×(384,1536)`，batch 4（主干 fc1） | 12 | 14.55 G |
     | `(257,1536)×(1536,384)`，batch 4（fc2） | 12 | 14.55 G |
     | `(257,384)×(384,384)`，batch 4（qkv） | 12 | 10.00 G |
     | `(1028,384)×(384,384)`（全局 block 的 qkv） | 若干 | 3.64 G |
     | `(6,1028,64)×(6,64,1028)` / `(6,1028,1028)×(6,1028,64)` | 各 6 | 各 2.43 G |
     | `(24,257,64)×(24,64,257)` / `(24,257,257)×(24,257,64)` | 各 24 | 各 1.83 G |

     全部 140 个合计 **54.1 GFLOP**。
   - ⚠️ **`batch 4` 这一列不能漏**：windowed attention 是 4 个窗口并行，MLP/投影的每个 MatMul 都带前置 batch=4。
     上一版把总量算成 16.6 GFLOP（漏了系数 4），于是得出「in-situ 只有 125 GFLOPS、比裸内核慢 3 倍」的错误结论。
     按 54.1 GFLOP 算，**in-situ 是 407 GFLOPS，比同形状的独立基准（386 GFLOPS）还略高** —— MatMul 已经到顶，没有可挖的空间。
   - **最大的一处浪费（已修复）**：DETR decoder 的 q/k/v 投影运行时是 `[300,1,256]×[256,256]`（300 个 batch、每 batch m=1）。
     引擎按 batch 逐个调内核，于是 256 KB 的权重被重新打包 300 遍，单个算子 4.06 ms；
     而 `(batch, m)` 合并后只是一次 `300×256×256` 的 GEMM，0.13 ms。9 个这样的算子，端到端 **356 → 270 ms（-24%）**。
   - **逐元素图级融合（2026-11-07 新增，已落地）**：DETR 每个 transformer block 的 GELU 在图上就是
     `Div → Erf → Add → Mul → Mul` 五个节点，每个节点都对 `[4,257,1536]`（1.58 M 元素）整读整写一遍。
     `GraphFusion` 现在把满足「中间结果只有一个消费者、不是图输出、链上同形」的逐元素链编译成一个
     `FusedEltwiseLayer`（寄存器机式：`reg[dst] = op(reg[a], reg[b])`），中间值全程留在 YMM 寄存器，整条链只读外部输入一遍、写最终输出一遍。
     rfdetr-small 上 12 条 GELU 链，节点数 857 → 801。
     - 端到端交替 A/B（同会话，base/fused 两个二进制各 3 轮 × 20 帧取 median）：

       | 模型 | base min | fused min | min 收益 | base median | fused median | median 收益 |
       | --- | ---: | ---: | ---: | ---: | ---: | ---: |
       | rfdetr-small | 243.0 | 224.9 | **-7.4%** | 254.1 | 236.1 | **-7.1%** |
       | rfdetr-medium | 320.2 | 291.2 | **-9.1%** | 336.6 | 305.5 | **-9.2%** |
       | yolo11n (5 轮×50 帧) | 33.3 | 32.5 | -2.4% | 36.0 | 34.2 | -5.0% |
       | yolov8s (5 轮×50 帧) | 85.7 | 86.1 | ±0 | 88.7 | 88.4 | ±0 |

     - 输出逐值比对（`run --dump` 落盘后按 float 比较）：4 个模型 **max|diff| = 0**，和未融合逐位一致。
     - **真正的坑在实现，不在算法**：第一版通用寄存器机内核比不融合还慢（每条 GELU 1.65 ms vs 五个算子合计 1.50 ms）。
       原因是 `Vector256<float>[] regs` 是**运行期下标**访问的数组，JIT 无法把它提升到 YMM 寄存器，
       每步都要写回栈再读出来，再叠加每步一次 `switch` 分发，把省下的带宽全部吃掉。
       解决办法是给 GELU 这一种固定形状写手写内核（中间值是**本地变量**），每条 1.65 → **0.30 ms**，
       `FusedEltwise` 总耗时 21.5 → 3.5 ms。
     - **短链要设门槛**：2~3 个节点的链省下的流量本来就少，抵不过 pin/闭包/数组索引的固定开销。
       yolo 系列的长跑 A/B 里这类小链让 median 变慢 4%（min 持平）；加上「链长 < 4 时张量必须 ≥ 256 K 元素」的门槛后
       yolo11n 变为 min -2.4% / median -5.0%，不再有回归。
     - **融合后已经贴到带宽墙，这条链再无空间**：五算子拆分耗时是 `Div 0.30 + Erf 0.25 + Add 0.37 + Mul 0.20 + Mul 0.38 = 1.50 ms`，
       而单层张量 1.58 M × 4 B = 6.3 MB，10 遍读写 = 63 MB / 1.50 ms = **42 GB/s**；
       融合后只剩 1 读 1 写 = 12.6 MB / 0.30 ms，仍是 **42 GB/s** —— 完全重合。
       也就是说这 12 条链已经跑到该机器的逐元素带宽上限，`Erf` 的算力开销（0.25 ms）与访存完全重叠、不可见。
       继续在 GELU 上做文章没有意义，剩下的只能是**再减少别的算子的趟数**或回到 GEMM。
3. **GEMM 微内核本身没问题，问题在并行扩展性**（`gemmbench` 与 `fma` 均为按时间预热后的稳态值；测量机状态会漂 ±10%，下表是同一时段的一组）：

   | 形状（batch） | 每 batch ms | 等效 GFLOPS |
   | --- | ---: | ---: |
   | `257×384×1536`（×1） | 0.79 | 386 |
   | `257×1536×384`（×1） | 0.97 | 313 |
   | `1028×64×1028`（×6） | 0.49 | 276 |
   | `1028×1028×64`（×6） | 0.39 | 347 |
   | `257×64×257`（×24） | 0.054 | 156 |
   | `300×32×300`（×8） | 0.096 | 60（未过并行阈值，单线程） |

   - 机器上限：纯 FMA 微循环 `fma` 实测单线程 91 GFLOPS、16 线程 **753 GFLOPS**（16 线程靠 SMT 超过 8 个物理核）。
     GEMM 常规形状 300~400 GFLOPS，即机器峰值的 40%~53%。
   - **真实模型里已经跑到 407 GFLOPS**（54.1 GFLOP / 133 ms），没有落后于独立基准，说明「在真实图里跑得慢」这个方向可以排除。
   - 26~33 个行块分给 16 线程要跑两波、尾块浪费，且每个行块都要把整个 B 重读一遍（`26 × 262 KB`），是 DRAM 带宽受限。
   - **已试过但端到端没有收益的方案（别再重复）**：
     - 把整个 B 预打包成 `[列块][k][16]` 供所有行块复用：微基准 304 → 234 GFLOPS（反而多付一份 B 的写回）。
     - 微内核 k 方向 2 步展开：单线程 136 → 135 GFLOPS（JIT 已经展开）。
     - 行块高度 `mb` 扫参（16/32/64/128）：微基准 250~340 GFLOPS 波动，端到端无差异。
     - 按流量把切分方向从行切换成列切：微基准 `257,1536,384` 343 → **499 GFLOPS（+45%）**，但端到端交替 A/B 三轮完全一致（273~277 ms 双方相同）。
   - 上面四条合起来是这个项目最重要的一条经验：**独立内核吞吐无法预测端到端**。
     微基准只能用来排除明显不合理的实现，任何优化都必须用「同会话、交替运行两个二进制」的端到端 A/B 来判定。
     原因不是测量方法，而是真模型里的缓存/内存压力与微基准不同（微基准反复跑同一组数据，权重留在 L3）；
     另外算 FLOPs 时**一定要把前置 batch 乘进去**（本轮就因为漏了 batch 4 把 54 GFLOP 算成 16.6，凭空造出「慢 3 倍」的假象）。
4. **ONNXRuntime 的 median 不可全信**：同一进程内 min 29~33 ms 而 median 44~49 ms（yolo11n），是它的 intra-op spin-wait 线程在后台负载下被抢占后的 barrier 放大效应。真实算力看 min，实际部署体验看 median —— 两者都列了。

## 已知限制（不是失败，是能力边界）

- **SimdYolo 跑不了 RF-DETR**：它的模型读取只认 YOLO 系列检测/分割/姿态头，DETR 输出 `dets[1,300,4] + labels[1,300,91]` 不匹配。
- **OpenCV dnn 能跑 RF-DETR**：3 个 `GridSample`（deformable attention）在新图引擎里有实现，且是其 5.0 的强项之一。
- 所有引擎都只测了 FP32 CPU；GPU / FP16 / INT8 不在本表范围。

## 下一步（按收益排序）

1. ~~**逐元素融合**~~ —— **已完成（2026-11-07，见结论 2）**：GELU 链 `Div→Erf→Add→Mul→Mul` 已融合，
   rfdetr-small/medium 端到端 **-7.4% / -9.1%**（min），输出逐位不变。
   注意结论 2 里的带宽核算：融合后已经贴在 42 GB/s 的逐元素带宽上限上，**这条路到头了**，
   不要再往「更聪明的逐元素融合」投入。`Add 169 + Mul 114` 里剩下的多数节点不是链（每个中间结果有多个消费者），
   要动只能动 `Sqrt + Div` 这类成对出现的规范化模式（下一轮可查，量级远小于 GELU）。
2. ~~**GridSample 循环顺序 bug**~~ —— **已完成（2026-11-07，见结论 2）**：`c` 外提 + 采样坐标预算，
   3 个算子 4.8 → 2.5 ms（**-48%**），端到端 rfdetr-small **-1.4%**、medium **-1.1%**（min，6 轮交替 A/B），
   逐值 `max|diff| = 0`。剩下的 0.92 ms/算子要再快得预计算邻居偏移，端到端只值 0.6%，**不再投入**。
3. **Transpose**：113 个（**6.2%**，12.7 ms），已是 8×8 向量分块 + 同布局 memcpy 快路径，
   但单个仍要 0.11 ms —— DETR 的 `[4,257,1536]` 这类大转置应能压到 0.05 ms 级。**现在是 RF-DETR 剩余的最大单块**。
4. **静态形状图缓存逐算子的形状/类型推断**：`Interpreter.Run` 每 forward 对全部 809 个算子重算
   `GetMemoryShapes` / `GetTypes` / `BindArgTensor` / `FitTemps`，实测**每 forward 分配 113 MB**。
   代价被证明很小（GC 暂停 0.7 ms/iter，同形状连续跑 3000 次 383 vs 386 GFLOPS 不衰减），
   但静态图缓存这些结果是**纯收益**，顺手可做。
5. **s 档大 GEMM 打包**：对照 SimdYolo `CpuPackedGemm` 与 OpenCV `fastConv` 的权重 block 布局，调 `Panel / RowBlock / SliceDepth`。
   这是 yolov8s/yolo11s 落后 1.5~1.8× 的唯一解释，也是唯一还没被本轮优化覆盖的区域。
6. 卷积侧：Conv+SiLU 融合、Winograd 覆盖 `5×5` —— 但 DETR 上 Conv 只占 3.8%，收益上限很低。
7. **不要再做的**：任何以「减少逐元素趟数」为目标的改动（已贴 42 GB/s 带宽墙）；
   任何以「提高 GridSample 单算子速度」为目标的改动（端到端只值 0.6%）；
   以及任何需要用微基准（而非交替 A/B）证明收益的 GEMM 分块调整（见结论 3）。

## 变更记录

- 2026-11-07：首次建立本文件，四个引擎 × 6 模型（4 YOLO + 2 RF-DETR）全量对照；新增 `bench/run_all.py`、`bench/run_python_bench.py`、`bench/models.json`。
- 2026-11-07（第二轮）：`gemmbench` 支持 `--batch` 并按时间预热（此前预热不足导致数字偏低 4 倍）；定位 RF-DETR 瓶颈为 batched attention GEMM，实测微内核已达单核峰值但多线程只有 45%；验证并否决「B 全量预打包」「k 方向 2 步展开」两个方案。
- 2026-11-07（第三轮）：纠正上一轮两个错误结论 —— ①「MatMul 全是 batched attention、MLP 已导出成 Conv」不成立，MLP 的 `[257,384]×[384,1536]` 就是 MatMul 节点，合计 16.6 GFLOP；② 单线程 GEMM 不是 136 GFLOPS，机器 16 线程 FMA 峰值 753 GFLOPS、GEMM 最好约 500。
  修掉两个测量坑：`profile` 增加 `--warmup`（不预热时 tier-0 代码让 MatMul 份额虚高到 66%）、改用每算子最小耗时汇总（均值被后台负载重尾放大 20%+）。
  修复最大单点浪费：MatMul 在 B 不参与广播时把 `(batch, m)` 合并成一次 GEMM，decoder q/k/v 投影 4.06 → 0.13 ms，rfdetr-small -24%、medium -12%。
  验证并否决「按流量改切分方向」（内核 +45%，端到端 0）；新增 `fma` / `gemmbench --threads` 与逐元素带宽、链式 MLP 两个微模型。
- 2026-11-07（第四轮）：**逐元素图级融合落地**。新增 `Core/GraphFusion.FuseElementWiseChains` 与 `Ops/FusedEltwiseLayer`
  （寄存器机式 `EltwiseStep[]`，中间值留在 YMM），rfdetr-small 12 条 GELU 链、节点 857 → 801，
  端到端 rfdetr-small **-7.4%**、rfdetr-medium **-9.1%**（min，交替 A/B 三轮），4 个模型输出 max|diff| = 0。
  三个必须记住的教训：① 通用内核里**运行期下标的 `Vector256[]` 会被 JIT 打回内存**，固定形状的链（GELU）必须手写内核，
  1.65 → 0.30 ms；② 2~3 节点短链融合在小张量上是**负收益**（yolo median +4%），要按链长/元素量设门槛；
  ③ 融合后 42 GB/s 已贴带宽墙，GELU 这条线到头。
  `run` 新增 `--dump <dir>`（把每个输出按 F32 落盘），配合 `bench/results/ab_dump.ps1` + `cmp_all.py` 做逐值 A/B；
  交替 A/B 脚本见 `bench/results/ab_fuse.ps1`（4 模型）与 `ab_fuse_yolo.ps1`（小模型长跑）。
  全表重新测量（同会话），rfdetr 对 OpenCV 的倍率从 1.68/1.70 降到 1.41/1.37。
- 2026-11-07（第三轮补）：再纠正一处算错的 FLOPs —— 主干 MLP/投影的 MatMul 带**前置 batch 4**（windowed attention 的 4 个窗口），
  MatMul 总量是 **54.1 GFLOP** 而不是 16.6，于是 in-situ 是 **407 GFLOPS**（高于同形状独立基准 386），
  「真实图里 MatMul 慢 3 倍」不成立。同时排除了 GC（0.7 ms/iter）与降频（长跑不衰减）两个嫌疑；
  `bench` 新增 `alloc` / `gc` / `gc pause` 三行诊断输出。
- 2026-11-07（第五轮，融合后复测）：全表重跑（同会话 `run_all.py`，iters=40/warmup=5/rounds=2）。
  **rfdetr 对 OpenCV 的倍率进一步降到 1.12 / 1.06**，从「最大短板」变为基本追平；对 ORT 是 0.53 / 0.50。
  yolo 侧倍率 vs ORT 0.59~0.72、vs OpenCV 1.12~1.21，同时修正了 `rfdetr-medium` 的输入尺寸
  （此前误记为 512²，实际 **576²**）与口径表里的 iters/warmup（此前写 30/10，与脚本默认 40/5 不符）。
  融合后的 by-type profile 见结论 2：`Div` 35→7 个、`Add+Mul` 14.1%→8.9%、新增 `FusedEltwise 1.8%`，
  而 **`GridSample` 3 个算子占 2.3%** 暴露成一个新的单点异常，已列为下一步第一优先。
  yolov8s 对 SimdYolo 的倍率涨到 1.78× 是因为对方从 58.5 变快到 51.8（我们是 90.2→92.2，属噪声），不是回归。
- 2026-11-07（第六轮）：**修 `GridSampleLayer.Forward` 的循环顺序 bug**。原实现把通道 `c` 放在最内层，
  而输出是 NCHW，写地址步长 `outH*outW` → 每写一个元素换一个缓存行。
  改为 `c` 在外、(oy,ox) 在内（输出连续写、源平面驻留），并把 `Unnormalize` 的采样坐标按 batch 预算到
  `ArrayPool` 租的数组（原来每个 `c` 都重算）。
  3 个 `cross_attn/GridSample`：`1.6 → 0.92 ms/个`（合计 4.8 → 2.5 ms，**-48%**，占比 2.3% → 1.2%）。
  端到端 6 轮交替 A/B（只取 min）：rfdetr-small 225.4 → 222.2（**-1.4%**）、rfdetr-medium 290.9 → 287.8（**-1.1%**）。
  4 模型 `max|diff| = 0`，测试 29/29。
  本轮同时确认**本机 median 已不可用于 A/B**：同一份代码重跑 6 次，yolo11n median 49.2~53.5、yolov8s 96.9~106.9，
  幅度 ±8%，与 GridSample 的真实收益同量级 —— 小改动只能看 min 或逐算子 profile。
