# OnnxSharp 性能基准

本文件是**唯一**的性能结果记录处。历史上散落在 commit message、`plan.md`、临时终端输出里的数字不再作为依据；有新结果就往这里追加一个带日期的段，旧段落保留。

## 口径

| 项 | 取值 |
| --- | --- |
| 计时范围 | 仅模型前向（N 次 `Forward` / `session.run` / `net.forward`），**不含** letterbox 前处理、NMS、掩码后处理 |
| 输入 | 静态 batch=1、FP32、NCHW，全部元素常量 `0.5`（与 SimdYolo 默认一致） |
| 预热 | `run_all.py` 默认 5 帧（`warmup=5`），不计入样本 |
| 样本 | 默认 40 帧（`iters=40`），跑 2 轮取 median 较低的一轮 |
| 线程 | **4 线程**（`ONNXSHARP_THREADS=4` / `intra_op_num_threads=4` / `cv2.setNumThreads(4)` / SimdYolo `threads=4`）。<br>**2026-11-07 ~ 11-11 的历史段落是 16 线程口径**，且其中 onnxsharp 的数字实为 **32 线程**（`run_all.py` 的 env 没传给子进程，`ONNXSHARP_THREADS` 从未生效，见第八轮）。跨段落比较前先确认线程数 |
| 指标 | 主指标 **median**；副指标 **min**（反映无干扰时的下限） |
| ASR（sherpa / Paraformer） | **不在四引擎表内**：输入是 wav（先过 fbank），按 `feat / enc / dec` 分段计时。口径与结果见 **§31** |

## 环境

| 项 | 取值 |
| --- | --- |
| CPU | **AMD Ryzen 9 8945HX**（16C/32T，基频 2.5 GHz，Zen4，AVX-512）· HP OMEN 16-ap0xxx 笔记本 |
| 内存 | 31.1 GB |
| OS | Windows（AC 供电；电源方案 `QuickCpuScheme`，`PROCTHROTTLEMIN = 100%` —— 满载撞温度墙的根因，见 §8.1） |
| .NET | 11.0.100 SDK，目标 `net10.0`（OnnxSharp / SimdYolo 均 Release） |
| ONNXRuntime | 1.32.0（`CPUExecutionProvider`，`ORT_ENABLE_ALL`，inter_op=1） |
| OpenCV | 5.0.0（`cv2.dnn` 新图引擎，`DNN_BACKEND_OPENCV` / `DNN_TARGET_CPU`） |
| SimdYolo | **0.11.4**（本地源码构建 `F:\code\SimdYolo\src\SimdYolo.Cli\bin\Release\net10.0\`，commit `824e574`，`CPU SIMD (256 bits)`）。<br>升级时注意：`git pull` 只换源码，**必须重新 `dotnet build -c Release`** —— 本次就出现过源码已是 0.11.4 而 `bin` 里还是 11-10 的旧 dll 的情况 |

> **机器已于 2026-11-15 更换**：文中 11-07 ~ 11-11 的所有段落是旧机器
> **Ryzen 7 8845HS（8C/16T、47 GB）** 上测的，与当前 8945HX（16C/32T）的绝对值**不可比**；
> 那一批数字只保留「同一次会话内的相对倍率」价值。
>
> 说明：测量机是共享环境（后台有常驻负载），所以同一次会话里依次跑完全部引擎，没有跨天拼接数字。ONNXRuntime 对线程抢占特别敏感（median 与 min 常差 40%+，见下文）。

### 测量方法上的坑（务必先读）

.NET 默认开启分层编译，热点方法要**调用若干十次**之后才会被提升到 tier-1 优化代码。按「迭代次数」预热（例如预热 3 帧）根本不够：同一个 GEMM 形状，预热不足时实测只有 50 GFLOPS，预热充分后是 490 GFLOPS，**差 4 倍以上**。所以：

- `gemmbench` / `convbench` 已改成**按时间预热**（至少跑满 400 ms 才开始计时）。
- `bench` 的 `--warmup` 是帧数，模型每帧会调用同一个 kernel 几十次，10 帧足够；但**单算子模型**（1~6 个节点）每帧只调用几次，必须把 `--iters` 拉到 60 以上，否则中位数会被 tier-0 的慢样本污染。
- 结论：任何只看 5~10 次迭代的数字都不可信。

## 复现

```powershell
# 一次跑完四个引擎 × 全部模型（结果写 bench/results/latest.json，并打印 Markdown 表）
# 脚本默认已是 4 线程；要 16 线程口径必须显式 --threads 16（会撞温度墙，见第八轮）
python bench/run_all.py --iters 30 --warmup 10 --rounds 2

# 只跑部分模型 / 部分引擎
python bench/run_all.py --names yolo11n,rfdetr-small --runtimes onnxsharp,opencv-dnn

# 单引擎单独跑（脚本内部也是调这些）
python bench/run_python_bench.py --runtime ort --iters 30 --warmup 10
python bench/run_python_bench.py --runtime cv  --iters 30 --warmup 10
tools\OnnxSharp.Cli\bin\Release\net10.0\OnnxSharp.Cli.exe bench <model.onnx> --input <in.f32> --iters 30 --warmup 10 --threads 4
tools\OnnxSharp.Cli\bin\Release\net10.0\OnnxSharp.Cli.exe run  <model.onnx> --input <in.f32> --dump out/   # 输出按 F32 落盘（逐值 A/B 用）
F:\code\SimdYolo\src\SimdYolo.Cli\bin\Release\net10.0\SimdYolo.Cli.exe bench <model.onnx> cpu 640 30 4 10

# 测之前先确认没在降频（>80% 才可用，<60% 直接作废）
(Get-Counter '\Processor Information(_Total)\% Processor Performance').CounterSamples.CookedValue
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

## 结果（2026-11-07 第七轮，16 线程）

单位 ms，越低越好。**加粗为同组最快**。全部来自同一次 `python bench/run_all.py`（iters=40、warmup=5、rounds=2 取最优轮）。

| 模型 | 输入 | OnnxSharp | ORT CPU | OpenCV dnn | SimdYolo |
| --- | --- | ---: | ---: | ---: | ---: |
| yolov8n | 640² | 35.8 | 57.5 | 30.9 | **27.6** |
| yolo11n | 640² | 36.7 | 41.2 | **28.6** | 34.7 |
| yolov8s | 640² | 92.4 | 158.5 | 77.4 | **56.7** |
| yolo11s | 640² | 85.1 | 111.2 | 73.1 | **58.7** |
| rfdetr-small | 512² | 231.5 | 382.7 | **191.7** | N/A |
| rfdetr-medium | 576² | 299.4 | 455.0 | **243.1** | N/A |

min（无干扰下限）：

| 模型 | OnnxSharp | ORT CPU | OpenCV dnn | SimdYolo |
| --- | ---: | ---: | ---: | ---: |
| yolov8n | 33.4 | 35.8 | 29.0 | **24.4** |
| yolo11n | 33.3 | 27.8 | **27.3** | 31.2 |
| yolov8s | 88.0 | 92.9 | 74.9 | **53.2** |
| yolo11s | 81.6 | 64.4 | 71.2 | **55.3** |
| rfdetr-small | 219.8 | 300.4 | **187.3** | N/A |
| rfdetr-medium | 287.0 | 297.5 | **237.0** | N/A |

相对倍率（OnnxSharp ÷ 该引擎 median，>1 表示比它慢）：

| 模型 | vs ORT | vs OpenCV dnn | vs SimdYolo |
| --- | ---: | ---: | ---: |
| yolov8n | 0.62 | 1.16 | 1.30 |
| yolo11n | 0.89 | 1.28 | 1.06 |
| yolov8s | 0.58 | 1.19 | 1.63 |
| yolo11s | 0.77 | 1.16 | 1.45 |
| rfdetr-small | 0.60 | **1.21** | N/A |
| rfdetr-medium | 0.66 | **1.23** | N/A |

> **跨会话不要比绝对值**：同一份代码，机器状态不同时 yolo11n 能测出 32~50 ms（±20%）。
> 上表全部来自同一次会话，所以引擎之间的**倍率**可信，绝对值只作趋势。
> **本轮修正上一轮的一个错误结论**：上一轮写的「rfdetr 对 OpenCV 已追平（1.12 / 1.06）」是**对方状态不佳导致的假象**。
> 本轮 OpenCV 在同样代码下跑出 191.7 / 243.1（上一轮 204.9 / 304.4，medium 快了 20%），
> 而我们自己的 231.5 / 299.4 与上一轮持平 —— 真实差距是 **1.21 / 1.23**，不是 1.06。
> 教训：**比值只有在「双方同一轮都发挥正常」时才可信**，单看一两个比值不足以宣称追平；要看两个引擎的绝对值是否都落在各自的历史区间内。

### 4.1 与上一轮的 OnnxSharp 自比（同一口径，median / min）

| 模型 | 上一轮 median | 本轮 median | 变化 | 上一轮 min | 本轮 min |
| --- | ---: | ---: | ---: | ---: | ---: |
| yolov8n | 36.9 | 35.8 | -3% | 33.6 | 33.4 |
| yolo11n | 33.7 | 36.7 | +9% | 32.0 | 33.3 |
| yolov8s | 92.2 | 92.4 | ±0 | 86.7 | 88.0 |
| yolo11s | 83.1 | 85.1 | +2% | 79.4 | 81.6 |
| rfdetr-small | 230.4 | 231.5 | ±0 | 220.0 | 219.8 |
| rfdetr-medium | 321.5 | 299.4 | **-7%** | 304.0 | 287.0 |

> yolo 侧的 ±9% 属于跨会话噪声，**不应算作优化收益，也不应算作回归**。
> rfdetr 的 ±0 / -7% 里，GridSample 修复的贡献只有 **-1.1% / -1.4%**（第 6 轮交替 A/B 实测），
> 剩下的差异来自机器状态（medium 上一轮 321.5 本身就偏高）。**不要从这张表读出 7% 的收益。**

## 结论

1. **YOLO 侧**：OnnxSharp 稳定快于 ONNXRuntime CPU（0.58~0.89×），与 OpenCV 5 dnn 差 **16%~28%**，落后 SimdYolo 最大 1.63×（yolov8s）。
   - SimdYolo 在 `s` 档领先明显、`n` 档优势很小，说明差距不在"单个卷积"而在**通道数变大后的大 GEMM 分块/打包**：n 档层宽 64~128 时双方都喂不满 L1/L2，s 档 128~256 时我们的 `RowBlock=4 / Panel=16 / SliceDepth=128` 组合明显吃亏。
   - OpenCV dnn 的 `fastConv` 胜在按 `blocked` 布局一次性重排权重 + Winograd 覆盖更宽，是我们下一轮的主要参照。
   - `yolov8s` 对 SimdYolo 的 **1.63×** 是本轮最大短板，但要注意双方都在抖（我们 92.2 / 92.4 两轮几乎相同，SimdYolo 51.8~56.7 摆幅 10%），
     倍率在 **1.63~1.78** 之间浮动，**不要按单轮数字做判断**。按两轮各自最好的绝对值算是 88.0 vs 53.2 = **1.65×**。
   - `yolo11n` 本轮 median 36.7（上一轮 33.7）是噪声：min 33.3 vs 32.0 只差 4%，而 median 抖 9% —— n 档本身只有 30 ms 量级，**噪声占比最大的就是它**。
2. **RF-DETR 侧**：我们比 ORT 快（0.60~0.66×），比 OpenCV dnn 慢 **1.21~1.23×**。
   - **这是修正后的真实差距**。上一轮记的 1.12 / 1.06 是 OpenCV 那边状态不佳造成的假象（见 4.1 上面的说明）。
     按两轮各自最好的绝对值算：small 219.8 vs 187.3 = **1.17×**、medium 287.0 vs 237.0 = **1.21×**。
   - 即「RF-DETR 仍落后 OpenCV 约 20%」，这个结论在 GridSample 修复后没有变化（它只值 1.1%~1.4%）。
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

> **2026-11-15 注**：本节基于 16 线程口径 + 旧机器（8845HS），其中「s 档大 GEMM 打包是唯一解释」
> 与「剩下差距全在小 GEMM」两条已被 §8.4 的 4 线程实测推翻（GEMM 已打平）。
> **后续排期以 `planv2.md` 为准**，本节仅作历史记录保留。

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
  > **（第七轮修正：1.06 不成立，是 OpenCV 那一轮发挥失常，真实为 1.21~1.23。）**
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
  - 2026-11-07（第七轮，GridSample 修复后全表复测）：同会话 `run_all.py`（iters=40/warmup=5/rounds=2）。
   **rfdetr：231.5 / 299.4 ms，对 OpenCV（191.7 / 243.1）为 1.21× / 1.23×，对 ORT（382.7 / 455.0）为 0.60× / 0.66×。**
   yolo 侧 vs ORT 0.58~0.89、vs OpenCV 1.16~1.28、vs SimdYolo 1.06~1.63。
   - **修正上一轮的错误结论**：上一轮写的「rfdetr 对 OpenCV 已追平（1.12 / 1.06）」不成立——
     OpenCV 的 rfdetr-medium 从 304.4 变到 243.1（-20%），是**对方上一轮发挥失常**，而我们自己 321.5 → 299.4 的变化也在其自身噪声内。
     真实差距是 **1.17~1.23×**。这是本文档第二次出现「拿对方一次异常读数当基准」的错误，
     规矩定死：**只有两个引擎的绝对值都落在各自历史区间内，比值才写进结论。**
   - 本轮开跑前发现 `SC2_x64` 满载占核，前两轮全表**作废**（已删除，不进仓库）：
     同一份代码 ORT 的 yolov8s median 跑出 243 / 194，OpenCV 的 rfdetr-small 跑出 222 / 275。
     **跑全表前先看 `(Get-Process).CPU` 增量**，否则整轮数据只能当噪声。

## 2026-11-11 夜：bench 顺序偏置修正 + yolo11n 复测 + YuNet

**先修方法学**：`run_all.py` 原来是「同一个引擎连跑 N 轮再换下一个」。本机是功耗受限的 APU，
排在前面的引擎总能吃到更高的 boost 档 —— 实测同一份代码、同一个会话：我们 38.2 / 36.7 基本不变，
而 ORT 从历史基线 41 变成 **107**、OpenCV 从 28.6 变成 **138**、SimdYolo 从 34.6 变成 **68**（median）。
现在改成**每轮把引擎顺序轮转一格**，再取各引擎自己的最好值。

修正后 yolo11n（iters=40 / warmup=5 / threads=16 / rounds=4，每轮轮转）：

| 引擎 | median | min |
|---|---:|---:|
| onnxsharp | 30.0 | 28.0 |
| onnxruntime-cpu | 21.1 | 19.9 |
| opencv-dnn | 24.3 | 23.1 |
| simdyolo | 18.1 | 15.9 |

⇒ **修正一条旧结论**：以前写的「OnnxSharp 稳定快于 ONNXRuntime CPU」是 **median 口径 + 顺序偏置**
共同造成的假象（ORT 的 median 对后台负载特别敏感，min 才是它的真实算力）。按 **min** 看，
ORT 比我们快 ~1.4×、SimdYolo 快 ~1.8×、OpenCV 快 ~1.2×。
yolov8n / yolov8s 本轮被后台负载污染（我们 236 ms、ORT 126 ms），**不采信**，等空闲时重跑。

**YuNet（新增模型，单引擎数字）**：`bench --iters 40 --warmup 10 --threads 16`，
`yunet.onnx`（0.22 MB，91 个执行算子，640×640）：**median 19.0 ms / min 15.8 ms**。本轮没有跨引擎对照。
## 2026-11-11 深夜：常驻 intra-op 线程池（降逐 op 开销）

**问题**：`parbench` 显示一次 `ParallelRange.For` 空调用 = **14.4 µs**（TPL 的 `Parallel.For`）。
yolo11n 一次前向有 200+ 个并行区（88 个卷积 × 2 个派发点），累计 2~3 ms —— 正好是 `profile` 的
sum-of-min 与 `bench` 的 min 之间那 2.5 ms 的缺口。

**做法**：`ParallelRange` 换成常驻线程池（对齐 ONNXRuntime 的 intra-op pool）：调用线程自己也参与、
工作线程用 `Interlocked` 动态领任务；错误经 `ExceptionDispatchInfo` 回传；已经在并行区里的线程再调用时
降级为串行（防死锁）。`ONNXSHARP_THREADPOOL=0` 可回退到旧 TPL 路径（A/B 与排障用）。
同时把 `GemmKernel` 里两处直接调 TPL 的并行区也接到这个池上。

**同会话 A/B**（旧版用 `git worktree` 从 HEAD 单独编译，两个二进制交替跑）：

| 版本 | min（3 次） | best |
|---|---|---|
| HEAD（TPL） | 39.8 / 37.1 / 52.3 | 37.1 ms |
| 常驻线程池 | 23.0 / 23.4 / 23.7 | **23.0 ms** |

⇒ **−38%**，而且方差从 37.1~52.3 收到 23.0~23.7（负载下 TPL 的调度抖动一并消失）。
数值不受影响：yolo11n golden、ASR encoder、attention 的对拍 diff 与改动前一模一样。

**线程数复扫**（池化后 16 线程不再是瓶颈）：4 → 36.5 ms、**8 → 21.6 ms**、12 → 22.1、16 → 23.4。
这台 8 核 16 线程 APU 上 **8 线程最优**（超线程争用同一 FP 单元），和代码里原有注释一致。
`bench.md` 的对外口径仍固定 16 线程，保证和历史数据可比。

**坑**：第一版工作线程用 `SpinWait.SpinOnce()`，它 ~10 次之后就 `Thread.Sleep(1)`，
而 Windows 默认计时器粒度 15.6 ms —— 一次派发能等到 15~30 ms（`parbench` 直接飙到 30 ms/次）。
改成「紧凑自旋 `Thread.SpinWait` → `Thread.Yield` → 最后才 `Sleep(1)`」后恢复正常。
**同轮顺手否决的一项**：把微内核从 4 行改成 6 行（SimdYolo 的 6×16 形状）—— `gemmbench` 隔离对比下
全面更慢 1.3~3.3×（64×576×6400：649 → 195 GFLOPS、64×256×6400：682 → 328）。
原因：.NET JIT 保不住 12 个活跃 `Vector256` 累加器，主循环 spill。详见 `misc.md §6.22`。

## 2026-11-11 更深夜：RF-DETR（`MatMul+Add(bias)` 融合 + 列切分判据）与四引擎复测

**模型位置更正**：rfdetr 现在在 `F:\code\TinyLabel\train\testrftedr\runs\rfdetr-small.onnx`
（旧路径 `F:\ai\rfdetr\onnx_model\` 已不存在，`bench/models.json` 已同步；medium 也没有了）。

**逐项 A/B**（`bench` 的 min，交替跑，5 轮取各自最好值；HEAD 用 `git worktree` 单独编译）：

| 版本 | rfdetr-small 512² |
|---|---|
| HEAD（2e88cf9） | 170.7 ms |
| 当前 + 关掉 `MatMul+Add` 融合（`ONNXSHARP_NO_FUSE=matmulbias`） | 183.2 ms |
| 当前（融合开 + `m≤512` 且列块 ≥4×线程走列切分） | **155.1 ms** |

⇒ 相对 HEAD **−9%**。分解：`Add 155 个 → 46 个`（15.3 → 2.0 ms）、`ops 809 → 700`，
MLP 的 `257×384×1536` 从行切分 476 GFLOPS 换到列切分 718（gemmbench，n=1536 有 96 个列块）。
数值：dets `2.3e-4` / labels `5.0e-5`（同输入同会话对 ORT）。

**否决记录**（详情 `misc.md §6.23`）：
- `SelectRowBlock` 改成「行块数 = 线程数」→ 172.2 vs 156.1 ms（慢 9%），已回退；
- 大 T 注意力融合（含两段 GEMM 版内核）→ 融合层 66 ms vs 原图 MatMul+Softmax ~33 ms，
  加 `AttentionFusionMaxSeq=128` 直接不融合（ASR 的 T=76 不受影响）。

**四引擎复测**（`python bench/run_all.py --iters 20 --rounds 1..2`，同会话轮转引擎顺序）：

| 模型 | onnxsharp | onnxruntime-cpu | opencv-dnn | simdyolo |
|---|---:|---:|---:|---:|
| yolo11n 640² | 27.4（24.6） | 23.4（21.1） | 26.5（24.4） | 26.3（21.4） |
| yolov8n 640² | 27.6（25.8） | 23.4（21.3） | 24.9（24.0） | 34.7（18.1） |
| yolo11s 640² | 61.6（59.7） | 45.0（42.3） | 57.2（52.9） | 46.8（40.1） |
| yolov8s 640² | 66.8（64.4） | 50.7（49.6） | 54.3（53.1） | 44.4（36.2） |
| rfdetr-small 512² | 159.8（151.9） | 135.4（125.1） | 440.9（135.0）※ | 跑不了 |

格式是 `median（min）`。※ OpenCV 在 rfdetr-small 上 median 440 ms 而 min 135 ms，
同一进程内抖了 3 倍（`MatMul` 在这个形状上疑似线程切分不均），只作量级参考；
SimdYolo 在 yolov8n 上也抖（34.7 / 18.1），遇到 DETR 模型直接抛异常。

**结论**：ORT 仍是 CPU 上最快的（我们落后 15~30%）；OpenCV 与我们互有胜负；
RF-DETR 与 ORT 的比值从 1.21× 收到 1.18×（按 min 是 1.21×），剩下差距几乎全在
transformer 的小 GEMM（MLP 的 m=257、24 个 head-batch 的 QK/PV）。

## 2026-11-15 第八轮：温度墙 → 口径改成 4 线程；GEMM 归零，差距全在 GEMM 之外

### 8.1 为什么之前的数字不可信：不是后台负载，是热降频

16 线程同代码同模型，min 在 **108 ↔ 780 ms** 之间跳（7 倍）。满载期间每 400 ms 采样
`\Processor Information(_Total)\% Processor Performance`：

| 状态 | perf% | rfdetr-small min |
|---|---:|---:|
| 空闲 | 138% | — |
| **16 线程满载 25 iters** | **20.96 / 21.05 / 21.00（15 点，极稳）** | 777.6 ms |
| 冷却 60 s 后 | ~130% | 108.0 ms |

根因：电源方案 `QuickCpuScheme` 的**最小处理器状态 = 100%**（AC/DC 都是 100），CPU 常驻最高频，
16 线程一满载立刻撞温度/功耗墙，频率被压到标称的 21%。`138/21 ≈ 6.6`，正好对上 7 倍。

**规矩定死**：跑基准前先看 perf%，< 60% 的整轮数据直接作废；长期方案是把
`PROCTHROTTLEMIN` 调回 5%（`powercfg /setacvalueindex SCHEME_CURRENT SUB_PROCESSOR PROCTHROTTLEMIN 5`）。
**现在默认口径改为 4 线程**：4 线程功耗远低于墙，同代码两轮重跑差异 **< 0.5%**（p95 与 min 差 < 4%），
历史数据一律 4 线程重测。

> ⚠️ **4 线程也不是绝对安全**：2026-11-15 晚测 SimdYolo 0.11.4 时，`yolov8s` + `yolo11s`
> 各 3 轮连续跑 **约 40 s**，perf% 从 138 掉到 **20.9**，`yolo11s` median 冲到 526 ms（正常 77 ms）。
> 冷却后立即恢复。所以 **s 档模型（尤其中间那个引擎是 simdyolo）不能连着跑多轮**，
> 每测完一个 s 档模型就查一次 perf%，低于 100 就等。

### 8.2 顺带查出的 bug：`ONNXSHARP_THREADS` 从未生效

`run_all.py` 里 `env = dict(os.environ, ONNXSHARP_THREADS=str(threads))` 算出来了，
但 `run()` 的签名是 `run(cmd, cwd=ROOT)`，**env 没传给 `subprocess.run`**。于是：
**`bench.md` 11-07 ~ 11-11 所有 onnxsharp 列标注的「16 线程」实为 32 线程**
（`Environment.ProcessorCount = 32`）。已修（`run(cmd, cwd, env)`），
那一批 onnxsharp 数字只保留「同会话内的相对倍率」价值，绝对值等 4 线程重测。

### 8.3 4 线程全表（median，括号内 min）

`python bench/run_all.py --iters 20 --rounds 2`（引擎顺序每轮轮转），同一次会话：

| 模型 | onnxsharp | onnxruntime-cpu | opencv-dnn | simdyolo（0.11.4） |
|---|---:|---:|---:|---:|
| yolov8n 640² | 46.0（45.6） | **28.3（27.9）** | 54.5（48.5） | 31.4（29.9） |
| yolo11n 640² | 39.1（37.5） | **23.6（23.5）** | 48.6（46.8） | 31.8（29.2） |
| yolov8s 640² | 121.0（120.0） | **76.7（76.1）** | 138.8（135.5） | 78.7（76.6） |
| yolo11s 640² | 111.7（103.0） | **65.1（63.8）** | 133.4（127.6） | 77.2（76.7） |
| rfdetr-small 512² | 262.5（252.4） | **189.4（187.9）** | 326.3（317.8） | 跑不了 |

> **simdyolo 列已换成 0.11.4**（2026-11-15 21:23 重建 `F:\code\SimdYolo` 后单独重测，
> 其余三列未变）。0.11.4 对这四个 YOLO 模型的收益基本在噪声内：
> yolo11s 快 5%（81.3 → 77.2），yolov8s 慢 3%（76.2 → 78.7），n 档持平（min 差 < 2%）。
> **注意**：该次重测踩了温度墙 —— `simdyolo` 在 s 档 4 线程连跑 40 s 就把 perf% 从 138 打到
> **20.9**（median 瞬间涨到 526 ms），冷却到 138% 后复测才拿到 77 ms。**跑 simdyolo 的 s 档必须单模型、跑完即查 perf%。**

倍率（onnxsharp ÷ 该引擎 median，>1 表示比它慢）：

| 模型 | vs ORT | vs OpenCV | vs SimdYolo |
|---|---:|---:|---:|
| yolov8n | 1.63 | **0.84** | 1.47 |
| yolo11n | 1.66 | **0.80** | 1.23 |
| yolov8s | 1.58 | **0.87** | 1.54 |
| yolo11s | 1.71 | **0.84** | 1.45 |
| rfdetr-small | 1.39 | **0.80** | N/A |

**两个口径依赖的现象，不要混着引用**：

- 4 线程下 OpenCV dnn 退化得比谁都厉害（yolo11n 28.6 → **48.6**），我们反而领先它 16~20%；
  16 线程时是它领先我们。**「谁快」在这台机器上是线程数决定的，不是内核决定的。**
- 我们对 ORT 的劣势从 16 线程的 1.15~1.30× 扩大到 **1.39~1.71×**。ORT 几乎不吃线程
  （yolo11n 23.4 → 23.6，16 → 4 线程），我们从 27.4 → 39.1。它早已撞上带宽墙，我们还没有。

### 8.4 关键结论：GEMM 已经不落后了，差距全在 GEMM 之外

同会话 profile（rfdetr-small，4 线程，占比；profiling 会放大绝对值，只看结构）：

| op_type | onnxsharp | ORT | 折算绝对值（我们 252 / ORT 189） |
|---|---:|---:|---|
| MatMul + Gemm | **61.2%** | 79.2%（Gemm+FusedMatMul+MatMul） | **154 ms vs 150 ms** |
| Conv | 10.9% | 4.1% | **27 vs 8 ms** |
| Transpose | 4.3% | 2.7% | 11 vs 5 ms |
| Mul + Add | 6.5% | 2.4% | 16 vs 4.5 ms |
| LayerNormalization | 2.3% | 1.9% | 6 vs 3.6 ms |
| Softmax | 2.2% | 2.3% | 5.5 vs 4.3 ms |
| Where | 2.0% | 0（被折叠） | 5 vs 0 ms |
| GridSample | 1.4% | 0.5% | 3.5 vs 0.9 ms |
| FusedEltwise + Relu + Silu | 2.9% | 0（Conv+SiLU 融合） | 7 vs 0 ms |
| GatherElements | 1.0% | ≤0.3% | 2.5 vs 0.6 ms |

**GEMM 类绝对耗时几乎打平（154 vs 150 ms），39 ms 的差距全部在非 GEMM 的 98 ms 里。**
这修正了 11-07 的结论「剩下差距几乎全在 transformer 的小 GEMM」——按 4 线程口径不成立：
差距是 **Conv（-19 ms）+ 逐元素/激活（-20 ms，其中 Conv+SiLU 融合一项就 -7 ms）+ Transpose（-6 ms）
+ Where/Gather/Reshape 这类本该被常量折叠或融合掉却没掉的算子（-8 ms）**。

### 8.5 ORT 的图优化对 RF-DETR 是负收益（可复现）

`--opt-level disable`（1537 节点）vs `all`（776 节点），4 线程，各跑两轮：

| 优化等级 | 轮 1 min | 轮 2 min | median |
|---|---:|---:|---:|
| disable | **182.07** | **182.77** | 183.1 |
| all | 186.87 | 187.14 | 187.7 |

**关掉全部图优化反而快 2.6%，两次复现一致。** 融合省下的 700 个节点被 Gemm 化引入的
253 个 `Reshape` 和 NCHWc 的 `ReorderInput/Output` 拷贝吃掉了。而 `MatMul::PrePack`
（`orttraining/orttraining/training_ops/cpu/math/matmul.cc:301`）是 **kernel 级、session 初始化时
对常量权重做的，与 `graph_optimization_level` 无关** —— 所以 disable 档也照样快。

⇒ 对 `planv2.md` 的直接含义：**ORT 的优势不在图融合，在运行时（预打包 / 内存模式 / 线程池）。
我们已经有 `GemmKernel.PackedB`（MatMul 常量权重）和 `PackedConv.Cache`（Conv 权重）两条等价物，
所以「补预打包」不是待办 —— 待办是 GEMM 之外的 98 ms。**

## 2026-11-15 第九轮：单层微基准 —— §8.4 的卷积 27 ms 被推翻，实际只有 11 ms

§8.4 的算子分解是拿 `profile` 占比 × 端到端 min 折算出来的，而 `profile` 的每层回调开销
**按层数摊**：MatMul 有几百层，Conv 只有 9 层，两者被放大的倍数完全不同，占比必然失真。
这一轮用 `convbench` / `gemmbench` 直接测单层（4 线程，iters 20，取 min）。

### 9.1 先修工具：`convbench --shape` 从来没生效过

`ParseOptions` 把 `--shape` 收进了 `CliOptions.Shapes` 列表，而 `ConvBench` 用
`options.GetString("--shape")` 去取（走的是 `_values` 字典），永远拿到 null，
于是**不管传什么形状都在跑 yolo11n 的默认层**。已改为读 `options.Shapes`，并支持重复传多个
`--shape`。`gemmbench` 早就是这个写法（它做了两种兼容），只有 `convbench` 漏了。

> 教训：任何「跑出来都一样」的基准工具，先验证参数真的进去了。

### 9.2 RF-DETR-small 的 9 个 Conv（真实形状，来自 `dump-graph --type Conv` + 权重 dims）

| 层 | 权重 | 类型 | M×N×K |
|---|---|---|---|
| patch_embed | `[384,3,16,16]` s16 | 输入投影 | 1024×384×768 |
| projector cv1 | `[256,1536,1,1]` | 1×1 | 1024×256×1536 |
| stages.0.0 ×6 | `[128,128,3,3]` p1 | 3×3 全通道（**非 DW**） | 1024×128×1152 |
| projector cv2 | `[256,640,1,1]` | 1×1 | 1024×256×640 |

Conv 合计 **1.78 GMAC**。4 线程实测（各取最快内核）：

| 层 | im2col | packed | Winograd | 采用 |
|---|---:|---:|---:|---:|
| patch_embed | **2.05** | 2.39 | — | 2.05 |
| cv1 1×1 | **2.66** | — | — | 2.66 |
| 3×3 128→128 ×6 | 1.56 | 1.13 | **0.90** | 5.40 |
| cv2 1×1 | **1.12** | — | — | 1.12 |
| **合计** | | | | **11.23 ms** |

- 6 个 3×3 层**全部命中 Winograd**（`spatial=1024 ≤ MaxSpatialPixels=4096`），
  167 GMAC/s，对 4 核 AVX2 理论上限（~211 GMAC/s）是 **79%**。
- 三个候选假设全部证伪：① `PackedConv` 面板数 64 ≥ 4×线程数 16，不空转；
  ② Winograd 的 16 个 component / 4 线程已接近线性；③ 没有层因 `ParallelThreshold` 退化。
- **`§8.4` 的 27 ms 是折算错误，实际 11.2 ms（占 262 ms 的 4.3%）**。Conv 的可优化上限
  因此从「-6~7%」降到 **-3~4%**，`planv2` 的 P1 已据此降级。

### 9.3 GEMM 才是 69%，且微内核效率已经 70~74%

`shape_inference` 统计：全模型 GEMM 共 **27.1 GMAC**（Conv 只有 1.78），其中 86% 集中在 6 类形状。

| 形状（batch, M, N, K） | 个数 | GFLOPS | 单次 ms | 合计 ms |
|---|---:|---:|---:|---:|
| 4× 257×384×384（QKV） | 33 | 313 | 0.97 | 31.9 |
| 4× 257×384×1536（fc1） | 12 | 296 | 4.10 | 49.2 |
| 4× 257×1536×384（fc2） | 12 | 302 | 4.02 | 48.2 |
| 1× 1028×384×384（dense） | 12 | 290 | 1.05 | 12.6 |
| 24× 257×64×257（PV） | 9 | 236 | 0.84 | 7.6 |
| 24× 257×257×64（QK） | 9 | 243 | 0.86 | 7.8 |
| | | | | **157.3** |

加上剩余 14% 的小 GEMM，GEMM 总计约 **180 ms（69%）**。4 线程 AVX2 理论峰值 ~422 GFLOPS，
实测 290~313，即 **70~74%** —— 剩下的 26% 要靠 AVX-512 微内核或更大分块，不是「小 GEMM 没优化」。

### 9.4 线程扩展性接近线性，P6 的前提也不成立

`convbench`（128→128 @32² k3）跨线程档：

| 线程 | 1 | 2 | 4 | 8 |
|---|---:|---:|---:|---:|
| Winograd ms | 3.46 | 1.77 | 0.90 | 0.56 |
| 加速比 | — | 1.95× | 3.84× | 6.18× |

1→2 是 1.95×、2→4 是 1.97×、4→8 是 1.61×。**没有出现「块数不足导致退化」**，
`planv2` P6「4 线程下块数 < 线程数会整层退化」的根因假设可以划掉。

### 9.5 修正后的端到端分解（rfdetr-small 262 ms，4 线程）

| 类别 | 实测 | 占比 | 4 线程效率 |
|---|---:|---:|---:|
| GEMM | ~180 ms | 69% | 70~74% of AVX2 峰值 |
| 逐元素（Mul/Add/Div/Where/激活） | ~32 ms | 12% | 未测，疑似带宽/多 pass |
| Conv（9 层） | 11.2 ms | 4.3% | 79% |
| Transpose（111 个） | ~11 ms | 4.2% | 未测 |
| Softmax + LayerNorm | ~12 ms | 4.6% | 未测 |
| GridSample / Gather / Reshape 等 | ~15 ms | 6% | 未测 |

**新结论：`planv2` 原先的 P1（Conv）与 P6（并行粒度）都建立在这轮被推翻的前提上。**

> **本表已被 §11 取代**：「逐元素 ~32 ms」复现不出来，同口径实测只有 ~10 ms。下表其余各项
> 与本轮复查一致（MatMul 数字后来微调为 186~188 ms）。

---

## 2026-11-15 第十轮：分配归因 —— 共享槽位的 `Tensor.Fit` 把大缓冲缩容了

第 1 轮讨论时断言过「ORT 的三大机制（PrePack / memory pattern / 常驻线程池）我们都已等价实现」。
本轮把这句拿去做验证，结果 **memory pattern 这一项我们此前是有真缺陷的**：不是没做复用规划，
而是复用规划的落点 `Tensor.Fit` 会把已经变大的缓冲缩回去。

### 10.1 先加归因，再改代码

`Interpreter.Run` 增加 `allocs` 参数，用 `GC.GetTotalAllocatedBytes(precise:true)` 差值逐层归因，
探针**包住循环开头到 `Forward` 结束**（第一版只包 `Forward`，只归到 1.4 MB，漏掉了形状推断 /
`BindAll` / `PrepackWeights` / `Finalize`）。CLI：`profile --alloc`（逐算子 KB 列）、
`bench --alloc`（另打 `fit realloc` 计数与现场日志）。

首测 rfdetr-small（20 iter，4 线程）：

| 指标 | 改前 |
|---|---:|
| 每帧分配 | **54 823 KB** |
| `Fit` 重分配 | **19 次/20 帧** |
| gen2 GC | **5 次/20 帧** |

### 10.2 根因：`Fit` 把 dtype 变化当成了重分配理由

原实现：

```csharp
if (!reuse || Buffer.Length < need || DType != dtype) { ... Buffer = new byte[need]; }
```

`Buffer` 是 `byte[]`，`DType` 只是解释视图，长度够就该直接复用。加上 `DType != dtype` 后：

- 网络里有共享槽位（`NetImpl._slots`）被不同 dtype 的张量轮流使用。
- `ReallocLog` 抓到同一个 slot `#03c85018` 在 `[1,128,32,32] I64 (2400 B)` 与
  `[1,6,64,1028] F32 (25 MB)` 之间来回切换。
- 每次切到 I64 就 `new byte[2400]`，**把 25 MB 的缓冲缩成 2.4 KB**；下一帧 F32 再重新分配 25 MB。

即：不仅白分配，还把「复用的容量」销毁了，等于每帧重来。

修复后（`Tensor.Fit`）：

```csharp
if (!reuse || Buffer.Length < need) { ... Buffer = new byte[need]; }
DType = dtype;
```

### 10.3 效果

| 指标 | 改前 | 改后 |
|---|---:|---:|
| 每帧分配 | 54 823 KB | **1 397 KB** |
| `Fit` 重分配 | 19 次/20 帧 | **0** |
| gen2 GC | 5 次/20 帧 | **0** |

### 10.4 等价性与性能

- **逐字节等价**：`git stash` 前后分别 `run --dump`，`rfdetr out0/out1`、`yolo11n out0`
  三个输出全部 IDENTICAL（4800 / 3600 / 2 822 400 字节）。
- **端到端**：rfdetr min 252.6 → 254.2 ms，在噪声内（perf% 前后均 137~138）。
- **回归**：yolo11n 38.0 / 38.9、yolov8n 46.3 / 46.9、rfdetr 254.0 / 261.9，均在前几轮波动范围内。
- `gc pause` 实测仅 **0.1~0.2 ms** → **分配不是耗时主因**。这项修复是消除隐患
  （GC 抖动与内存峰值），不是提速手段，不能计入「阶段 A 的 -13%」。

### 10.5 结论

1. `planv2 §2.3`「ORT 三机制我们都有」的说法要降级为：**前两项（PrePack、常驻线程池）等价，
   第三项（memory pattern）此前有 bug，现已修复并验证等价。**
2. 「分配合规」之后，逐元素 32 ms 的成因只能从**带宽 / 多 pass** 上找，不能再用「每帧在分配」解释。
   这正是 P1'（`eltwisebench`）要回答的问题。

---

## 2026-11-15 第十一轮：逐元素「32 ms」是错的，实测只有 ~10 ms

`planv2` 的 P1'/P2/P3 三项（合计承诺 -8%）都建立在「逐元素 ~32 ms」这个数字上。
本轮先复测这个前提，结论是**前提不成立**，于是这一批课题直接作废。

### 11.1 先确认 profile 的分解可信

以前担心 `profile`「会放大绝对值」，所以 §8.4 只用**份额**去折算。现在加了预热，
`sum-of-min` 与 `bench` 的 min 能对上了：

| | bench min | profile sum-of-min |
|---|---:|---:|
| rfdetr-small 4 线程 | 253.5 ms | 252.5 ms |

差 0.4%，可以当绝对时间用。于是拿到一份可信分解（3 iter，4 线程）：

| type | ops | min ms | 占比 |
|---|---:|---:|---:|
| MatMul | 139 | 186.4 | **73.8%** |
| Conv | 9 | 11.8 | 4.7% |
| Transpose | 111 | 10.5 | 4.2% |
| Softmax | 18 | 7.9 | 3.1% |
| **Mul** | 76 | **5.4** | 2.1% |
| LayerNormalization | 49 | 5.2 | 2.0% |
| Reshape | 119 | 4.4 | 1.7% |
| **FusedEltwise** | 12 | **4.2** | 1.7% |
| GridSample | 3 | 3.5 | 1.4% |
| **Where** | 1 | **2.6** | 1.0% |
| **Add** | 46 | **1.2** | 0.5% |
| **Div** | 7 | 0.7 | 0.3% |
| **Relu / Silu** | 8+8 | **0.2 + 0.2** | 0.2% |

**逐元素合计 ≈ 12 ms（4.7%）**，不是 32 ms。§9.5 / §8.4 里 Mul 8.6 ms、Add 7.8 ms、
Relu 6.4 ms、Silu 6.0 ms 那一组数**在本机复现不出来**（可能是更旧的代码状态或未预热的
`avg`；预热与否对 `min` 无影响，已单独验证）。

### 11.2 加 `dump-graph --inputs`：算真实流量

只看 ms 判断不了「贵不贵」，得知道搬了多少字节。`dump-graph` 原来只打印输出形状，
本轮加 `--inputs` 同时打印输入形状，于是能区分标量广播（2 趟）、尾广播（≈2 趟）、全长（3 趟）：

| type | ops | 输出元素 | 真实流量 | 实测 | 等效带宽 |
|---|---:|---:|---:|---:|---:|
| Mul | 76 | 6.8 M | 153.1 MB | 4.9 ms | **31 GB/s** |
| Add | 46 | 10.9 M | 126.6 MB | 1.2 ms | **106 GB/s** |
| Relu | 8 | 3.0 M | 17.0 MB | 0.2 ms | 85 GB/s |
| Silu | 8 | 2.1 M | 10.0 MB | 0.2 ms | 50 GB/s |
| Where | 1 | 0.26 M | 2.0 MB | 2.6 ms | **0.8 GB/s** |

Mul 54% 的输入是 `F32 [1]` 或 `F32 [384]`，所以它只有 ~2 趟流量，153 MB 里已经算过。
Mul 的 31 GB/s 偏低（Add 是 106 GB/s），差在 24 个 `layer_scale/Mul`（`[4,257,384] × [384]`）
走了尾广播路径；见 11.4。

### 11.3 `eltwisebench`：内核本身贴着缓存带宽

新加 `eltwisebench`（`--shape` 可指定，默认取 rfdetr 的真实形状），每项给 ms / GB/s / ns·元素，
并**自带带宽参考** —— `copy`（1 读 1 写 memmove）是这台机器在该线程数下的上限，别的不跟它比就没意义。
`[4,257,384]`（a+b+c = 4.52 MB，L3 常驻）4 线程：

| 内核 | ms | GB/s | ns/元素 |
|---|---:|---:|---:|
| copy（memmove） | 0.057 | 55.2 | 0.145 |
| add | 0.043 | 111.2 | 0.108 |
| mul | 0.040 | 119.0 | 0.101 |
| mul-scalar | 0.051 | 61.7 | 0.130 |
| **mul-tail（`[384]`）** | **0.534** | **5.9** | **1.354** |
| relu | 0.024 | 131.6 | 0.061 |
| silu | 0.090 | 35.0 | 0.228 |
| **copy1（逐元素 4 字节 CopyTo）** | **0.315** | **10.0** | **0.799** |

两条读数值得记：

- **连续内核 100+ GB/s，已经超过 `copy` 本身**（三趟流比两趟更容易掩盖延迟）→ 逐元素**不是**亏在算法上。
  图里 Mul 只有 31 GB/s 是**缓存不命中**（生产者隔了几百个算子），不是内核缺陷。
- **`copy1` = 0.8 ns/元素**，是 `copy` 的 5~6 倍。`Where` 正是这个写法
  （`OpUtil.CopyBytes(..., 1)` 逐元素），262 K 元素实测 2.6 ms = **10 ns/元素**
  （多出来的部分是掩码 `ReadBools` + 三套 stride 推进）。**单点 1% 的纯浪费，改法明确。**

### 11.4 尾广播：分段循环搬进内核（-28%）

`ApplyBroadcast` 原来是「逐段调用 `ApplyContiguous`」，1028 段就 switch 1028 次、
压 1028 次 `fixed`。改成一次 switch 选内核、分段循环放在内核内部：

| | ms | ns/元素 |
|---|---:|---:|
| 改前（逐段 dispatch） | 0.712 | 1.803 |
| 改后（内核内分段） | 0.534 | 1.354 |

图中 `Mul` 5.4 → **4.9 ms**，端到端在噪声内（这 0.5 ms 不到 0.2%）。
**遗留问题**：即便搬进内核，仍比连续内核慢 13×（1.35 vs 0.10 ns/元素），
且**与段大小无关**（inner 从 384 扫到 4112，ns/元素 恒定 1.3~1.7）——说明不是分段调度开销，
而是每元素级别的代价（怀疑 4K 别名或内核未被向量化，未定位）。图里这些节点实际只要
~0.13 ms/个（比微基准快 4 倍），所以**没有继续追**。

### 11.5 结论

1. **P1'（逐元素为什么贵）关闭：它不贵。** 12 ms，其中 Mul 已贴带宽，剩下的
   `Where`（2.6 ms）与 `copy1` 模式是唯一明确的浪费，合计收益上限 ≈ 2.5 ms（1%）。
2. `planv2` 的 P2（Conv+SiLU 融合）、P3（逐元素碎片）随之作废 —— 它们的杠杆是按 32 ms 估的。
3. 剩余预算里 **MatMul 独占 73.8%（186 ms）**，而 `gemmbench` 说明它已到 AVX2 微内核的 70~74%，
   图里 290 GFLOPS ≈ 微基准水平，**没有逐层损耗**。要再往下走只能动微内核本身（P6'）。
4. 非 GEMM 合计 65 ms（ORT 约 39 ms），分散在 Transpose 10.5 / Softmax 7.9 / LN 5.2 /
   Reshape 4.4 / FusedEltwise 4.2 / GridSample 3.5 / Where 2.6，单项都不超过 4%
   —— 属于「有空顺手做」的档，不再单列课题。


## 2026-11-15 第十二轮：GEMM 微内核 MR 4→6（端到端 -5%）

`planv2` 的 P6' 是唯一大项。本轮先纠正两个测量前提，再动内核。

### 12.1 先把「峰值」钉死

以前说「微内核 70~74%」是拿 290 GFLOPS 比一个含糊的峰值。现在有 `flopsbench`
（纯 FMA 环，`--threads` 可调）当硬上限：

| | GFLOPS | 备注 |
|---|---:|---|
| 1 线程 | 111.0 | = 2×FMA × 8 lane × 2 × 3.45 GHz，**贴着理论值** |
| 4 线程 | 427.1 | 4 核线性扩展 |

同期 `% Processor Performance` = 137.8（≈3.45 GHz），且**GEMM 满载 6 秒内该值不动**
（`gemmbench --iters 400` 期间连续采样恒定 137.8）→ GEMM 期间不存在降频，427 是硬上限。

于是效率可以直接算：`1028×384×1536` 打包路径 3.400 ms = **356.6 GFLOPS = 83.5%**；
未打包 3.759 ms = 322.6 GFLOPS = 75.6%。（上一轮说的 70~74% 用的是旧的 4 行内核。）

### 12.2 两个被推翻的猜测

- **「大 k（n 小）会掉一半」是降频假象。** 某次 `1028,1536,384` 只跑到 47 GFLOPS，
  复测同样命令是 297.8。凡是一次进程里连跑 3~4 个形状，**后面的形状会撞墙**
  （同一进程内 4 个 `Best()` 累计 ~4 s 重载即可触发）。所以：
  **每个形状单独起进程 + 之间冷却 10~20 s + 取 min**，是本机唯一可信姿势。
- **对齐不是问题。** `gemmbench --misalign 1` 对 `257,64,257×24` 无影响
  （逐 batch 0.033 ms 不变）。图里 `byte[]` 切片只保证 4 字节对齐，但内核是 32 字节
  非对齐 load，Zen4 上代价可忽略。

### 12.3 内核：MR=4 → MR=6

原来 `Multiply4Rows`（4 行 × 16 列 = 8 个累加器）每个 k 需要 2 条 B load + 4 条 A 广播
= 6 条 load（3 拍）对 8 条 FMA（4 拍）—— 只有 25% 余量。改成 `Multiply6Rows`
（6 行 × 16 列 = 12 个累加器）：8 条 load（4 拍）对 12 条 FMA（6 拍），余量 50%，
每行的 B 流量降到 2/3。这也正是 OpenBLAS Haswell sgemm 内核的 6×16 分块。

累加仍是 k 递增的单条 FMA 链 → **逐位一致**（见 12.5）。

微基准（每形状单独进程、取 min of 20，GFLOPS 为未打包路径）：

| 形状 | MR=4 | MR=6 | 提升 |
|---|---:|---:|---:|
| 1028,384,1536 | 295.8 | 321.1 | +8.6% |
| 1028,1536,384 | 297.8 | 323.4 | +8.6% |
| 1028,384,384 | 299.8 | 324.9 | +8.4% |
| 300,256,2048 | 310.0 | 339.0 | +9.4% |
| 1028,1024,1024 | 272.9 | 289.9 | +6.2% |

端到端（交替 A/B，`--warmup 2 --iters 4`，每轮之间冷却 15 s，取 min）：

| | 3 次 min ms | 最好 |
|---|---|---:|
| MR=6 | 234.3 / 235.1 / 237.0 | **234.3** |
| MR=4 | 246.0 / 246.3 / 247.1 | 246.0 |

**-5.0%（246.0 → 234.3 ms）**。`yolo11n` 37.9 ms 不变（38.0）—— 它是卷积模型，
MatMul 只有几个，符合预期。

### 12.4 批量 GEMM 合并：确实有用（1.048×）

注意力 `QK^T`/`PV` 是 `[4,6,257,64]×[4,6,64,257]`：24 个 batch、每个只有 8.4 MFLOP。
逐 batch 调 `Multiply` 会开 24 次并行域（每次 17 个列块摊 4 线程 = 5:4:4:4），
新增 `MultiplyBatched` 把 `batch × 列块` 铺成 408 个任务**一次派发**。

端到端交替 A/B（其余代码相同，只切 `densePerBatch` 分支）：
**238.1 vs 249.6 ms，1.048×**。因此保留该路径（原先只是「收益有限」的猜测，
现在有端到端读数）。

顺带确认 `gemmbench` 对同一形状：逐 batch 0.033×24 = 0.79 ms，合并 0.68 ms（1.16×）。

### 12.5 等价性

`bench/ab/*` 与本轮改动前后的 dump 逐位比对（4 线程、同一输入）：

| 模型 | 输出 | 结果 |
|---|---|---|
| rfdetr-small | out0 / out1 | IDENTICAL |
| yolo11n | out0 | IDENTICAL |

另外单独验证过 MR=6 与 MR=4 的输出 MD5 完全相同（`ONNXSHARP_GEMM_MR=4` 退回旧内核）。

### 12.6 profile 的 per-op 小算子仍然不可信（未解决的口径问题）

同一个 `attention/attention/MatMul`（203 MFLOP），三种口径给出三个数：

| 口径 | ms | 折算 |
|---|---:|---|
| `gemmbench 257,64,257 --batch 24`（合并） | 0.68 | 298 GFLOPS |
| `ONNXSHARP_MM_TRACE=1` 图内稳态 | 0.80 | 254 GFLOPS |
| `profile --warmup 4` 的 min | 1.47 | 138 GFLOPS |
| `profile --warmup 10` 的 min | 3.60 | 56 GFLOPS |

预热越多反而越大 → 不是 JIT 提升不足，而是**运行期间它在变大**（同一份代码、同一形状）。
`sum-of-min` 与 `bench min` 总量能对上（232.9 vs 234.3），说明这些偏差在别处被抵消，
per-op 单项不能单独引用。**判断某条路径值不值得做，一律用「关掉该路径」的端到端 A/B**，
12.4 就是这么做的。

### 12.7 结论与剩余空间

1. MR=6 落地，端到端 **-5.0%**；内核从 75.6% → 83.5%（打包路径）。
2. 打包路径已经 83.5%，未打包 75.6%，差 8 个点就是「每帧重新打包 B」的成本
   （`1028,384,1536` 打包/不打包 = 3.400 / 3.759 ms）。常量权重已走 `Pack` 缓存，
   未打包路径只剩注意力这类动态 B。
3. 剩下的 16.5% 不是调参能拿的：JIT 生成的内层循环已经是
   `2 load + 6 broadcast + 12 FMA + 5 标量` / k（`DOTNET_JitDisasm=Multiply6Rows`
   已核对，12 个累加器全在寄存器里，用到了 AVX-512 的 ymm16~31）。
   再往上要么 **NR=32 / MR=12 的 24 累加器内核**（寄存器够，但需要 AVX-512 分派），
   要么 **BLIS 式 A 打包 + k 分块**。按 12.6 的口径教训，这两项都要先有可靠的
   per-op 读数才值得投入，当前**暂不启动**。
   > 同日已做：MR=12 的 24 累加器内核，打包路径 83.5% → **92.2%**，见 §13。

### 12.8 复现命令

```powershell
$env:ONNXSHARP_THREADS='4'
$cli='tools/OnnxSharp.Cli/bin/Release/net10.0/OnnxSharp.Cli.dll'

# 峰值参考
dotnet $cli flopsbench --iters 100 --inner 500000 --threads 4

# 单形状（每个形状一个进程，之间冷却）
dotnet $cli gemmbench --iters 20 --shape 1028,384,1536

# 内核 A/B（退回 4 行内核）
$env:ONNXSHARP_GEMM_MR='4'

# 逐算子形状/耗时
$env:ONNXSHARP_MM_TRACE='1'
```


## 2026-11-15 第十三轮：MR 6→12（24 累加器，打包路径到 92%）

§12 留下 16.5% 的缺口，判断是「每 k 的 load 与标量开销摊得太薄」。这一轮按
**BLIS/OpenBLAS 的老办法**把 MR 再翻一倍：6×16 → 12×16。

### 13.1 为什么 12 行（而不是 8 行）

| MR×NR | 累加器 | 每 k：load / FMA | load 占比 | 需要的 ymm |
|---|---:|---|---:|---:|
| 4×16 | 8 | 6 / 8 | 75% | 6 |
| 6×16 | 12 | 8 / 12 | 67% | 15 |
| **12×16** | **24** | **14 / 24** | **58%** | **27** |

12 行还有个附带收益：同一个 B 面板被重流的次数减半（行组数 m/6 → m/12），
而 k=384 时面板正好 24 KB —— 这种「刚好压住 L1」的尺寸最怕被反复重流。

24 个累加器 + 2 条 B + 1 条 A 广播 = 27 个 ymm，**只有 AVX-512 的 32 个向量寄存器装得下**
（AVX2 只有 16 个，直接溢出到栈），所以 `UseMr12` 在运行时按 `Avx512F.VL.IsSupported` 分派，
AVX2 机器仍走 6 行内核；`ONNXSHARP_GEMM_MR=6` / `=4` 可强制降级。

### 13.2 踩到的坑：12 个 bias 向量把寄存器吃满了

第一版照抄 6 行内核的写法（`var b0v = Vector256.Create(bias[0]) … b11v`），
24 + 12 + 3 > 32 → 溢出，实测只比 6 行快 2%且方差 6.8%。
改成**只在初始化表达式里就地广播**（`initBias ? Vector256.Create(i0) : ...`，`i0` 是 float 标量），
持久向量降到 27 个，性能立刻跳到 11%。

### 13.3 结果

微基准（交替 A/B、每档之间冷却 20 s、每档 15 iter，两次独立复现）：

| 形状 1028,384,1536 | 未打包 ms | 未打包 GFLOPS | 打包 ms | 打包效率 |
|---|---:|---:|---:|---:|
| MR=6 | 3.814 / 3.817 | 318.0 / 317.7 | 3.424 / 3.438 | 83.0% |
| **MR=12** | **3.473 / 3.475** | **349.0 / 349.2** | **3.079 / 3.096** | **92.2%** |

打包路径 1.2126 GFLOP / 3.079 ms = **393.8 GFLOPS = 92.2%**（峰值 427.1）；未打包 81.7%。

端到端（rfdetr，交替 A/B，`--warmup 2 --iters 4`，两次独立实验共 6 对）：

| 对 | MR=6 | MR=12 | MR12 快 |
|---|---:|---:|---:|
| 1 | 257.7 | 248.4 | 3.7% |
| 2 | 246.6 | 235.4 | 4.7% |
| 3 | 241.3 | 227.9 | 5.9% |
| 4（反向） | 233.7 | 221.8 | 5.4% |
| 5（反向） | 234.5 | 244.2 | **-4.0%** |
| 6（反向） | 235.0 | 223.0 | 5.4% |

5/6 对 MR=12 更快，中位 **5.4%**；反向顺序（先跑 MR=12）也保持 5.4%，不是顺序偏置。
最好读数 **221.8 ms**（MR=4 时代是 246.0，累计 **-9.8%**）。
第 5 对的反例伴随整机发热（同一时段 MR=6 也从 233 升到 234.5、MR=12 从 221 跳到 244），
按 §12.2 的口径纪律判为噪声。

`yolo11n` 同协议 **37.9 → 36.4 ms**（-4%），方向一致（它也有几处常量权重 MatMul）。

### 13.4 等价性

MR=12 与 MR=6 的 `--dump` 逐位比对：`rfdetr out0/out1` **IDENTICAL**
（累加仍是 k 递增的单条 FMA 链，只是行块变宽，求和顺序不变）。

### 13.5 复现

```powershell
$env:ONNXSHARP_THREADS='4'
$cli='tools/OnnxSharp.Cli/bin/Release/net10.0/OnnxSharp.Cli.dll'
dotnet $cli gemmbench --iters 15 --shape 1028,384,1536   # 默认 MR=12
$env:ONNXSHARP_GEMM_MR='6'                               # 退回 6 行内核
```

### 13.6 还剩什么

打包路径 92.2%、未打包 81.7%（差 10 个点仍是「每帧重新打包动态 B」）。GEMM 总量按
83.5% → 92% 折算还剩约 12 ms（端到端 ~5%），已经没有「两位数」的空间：
再往上要靠 BLIS 式 A 打包 + k 分块把未打包路径也拉平，收益上限 10 ms 量级。

---

## 14. 2026-11-15 P4：把 attention 的 Transpose 折进 GEMM 的读面板（rfdetr −4.5%）

### 14.1 结构（探针实测，不是猜的）

`ONNXSHARP_DEBUG_FUSE=1` 打出 rf-detr 每层 encoder 的原始链：

```
q = Reshape(qproj)  [B,T,H,D] ─T[0,2,1,3]→ [B,H,T,D] ─Mul(·,s)→ A ┐
k = Reshape(kproj)  [B,T,H,D] ─T[0,2,3,1]→ [B,H,D,T] ─Mul(·,s)→ B ┴→ MatMul     (QK^T)

P = Softmax(QK^T)   [B,H,T,T] ───────────────────────────────────→ A ┐
v = Reshape(vproj)  [B,T,H,D] ─T[0,2,1,3]→ [B,H,T,D] ─────────────→ B ┴→ MatMul  (PV)
```

`[B,T,H,D]` = `[4,257,6,64]`，每个转置都是 1.58 MB 的纯搬运（layer.3 的 2× 分辨率那层是
`[1,1028,6,64]`；`B=4` 的层实际是 `[4,257,6,64]`）。全模型 111 个 Transpose，这条链占 48 个
（每层 4 个：QK^T 的 A、B，PV 的 B，以及 PV 输出回 `[B,T,H,D]` 的那个）。

### 14.2 做法

`GraphFusion.FuseTransposeIntoMatMul`（`ONNXSHARP_NO_FUSE=attnview` 可单项关闭）：
把 MatMul 的 A/B 换成「转置的源张量 + 一套 stride」，GEMM 直接按 stride 读，转置层删掉。
放在 `FuseAttention` **之后** —— 否则合成 MHA 图会先被本 pass 拆散，`attention: fused MHA
matches onnxruntime` 那个单测立刻失败（整块融合比单点折叠更值）。

| 位置 | 折掉的转置 | `gemm_view` 里的读法 |
|---|---|---|
| QK^T 的 A | `[0,2,1,3]` | 行距 `H*D=384`、head 步长 `D=64`、outer 步长 `T*H*D`（每行仍是 64 个连续 float） |
| QK^T 的 B | `[0,2,3,1]` | 行距 `1`、列距 `H*D=384`、head 步长 `D`（列方向 gather） |
| PV 的 B | `[0,2,1,3]` | 行距 `H*D=384`、列距 `1`、head 步长 `D`（每行仍是 64 个连续 float） |
| PV 的 A | 无需折 | Softmax 输出本来就连续，`aRow=K`、`aBatch=T*K` |

内核侧的改动（上一轮已落地，本轮只是接上视图）：`PackColumns` 增加 `srcColStride`，
`MultiplyColumnRange` / `MultiplyBatched` 增加 `aStride` / `bColStride` / `bStride`，
`MatMulLayer` 读 `gemm_view`（12 个整数）走视图分支。默认参数下这些分支一条指令都不会多执行。

**`Mul(·, s)` 保留，只换输入源。** `s = 1/√8 = 0.3536` 是常量，但
`(q·s)·(k·s) ≠ s²·(q·k)` 在浮点上（`s` 不是 2 的幂，两次舍入 vs 一次），
折进 B 打包就破坏了「逐位一致」。所以策略是：转置折掉、乘法留下 —— 把 `Mul` 的输入从
「转置输出」改接到「转置的源」（逐元素乘标量与转置可交换，每个元素的乘法与结果逐个不变）。
`Mul` 输出的物理布局随之从 `[B,H,T,D]` 变成 `[B,T,H,D]`，而视图层的 stride 正是按源布局写的。

### 14.3 结果

节点数：Transpose **111 → 75**（−36，即 12 层 × 3），融合命中 24 处（12 层 × 2 个 MatMul）。

交替 A/B（同一个 exe，`ONNXSHARP_NO_FUSE=attnview` 关这条 pass；`--warmup 4 --iters 10`，12 轮）：

| 轮 | 基线 min | 融合 min | 差 |
|---:|---:|---:|---:|
| 1 | 247.7 | 224.3 | −9.4% |
| 2 | 254.3 | 223.2 | −12.2% |
| 3 | 675.5 | 224.7 | 基线整轮降频，作废 |
| 4 | 233.7 | 223.7 | −4.3% |
| 5 | 254.6 | 368.0 | 融合整轮降频，作废 |
| 6 | 233.8 | 223.2 | −4.5% |
| 7 | 228.6 | 223.7 | −2.1% |
| 8 | 446.3 | 221.4 | 基线整轮降频，作废 |
| 9 | 1217.1 | 222.3 | 基线整轮降频，作废 |
| 10 | 248.3 | 227.8 | −8.3% |
| 11 | 234.1 | 247.4 | +5.7% |
| 12 | 1056.3 | 247.9 | 基线整轮降频，作废 |

有效 7 轮：配对中位 **−4.5%**（6/7 轮更快），中位耗时 234.1 → 223.7 ms，
最好读数 228.6 → 221.4 ms（−3.1%）。

yolo11n 持平（37.5/36.9、36.8/36.5、36.8/37.3，中位 36.8 → 36.9）—— 它 0 命中，见 14.7。

> 环境：本机 `PROCTHROTTLEMIN=100`，`% Processor Performance` 在 137%（冷）与 35%（热）之间跳，
> 12 轮里有 5 轮整轮不可用。判定用中位 + 最好读数，单轮数值不要单独引用。

### 14.4 等价性

5 个模型 `--dump` + MD5 与 `bench/ab/mr12_*` 逐位一致（rfdetr 的两个输出
`359D8BA3…` / `687A1986…`；yolo11n / yolov8n / yolov8s / yolo11s 关/开融合同一 exe 比对）。

### 14.5 踩到的坑：B 没有外层 Mul 时必须换输入源

第一版把视图层的输入直接写成 `matmul.Inputs[1]`，对 QK^T 是对的（那里 `Inputs[1]` 是新的 `Mul` 输出，
`Mul` 还活着），但对 PV 是错的：PV 的 B 是**裸的 Transpose**，融合时被 `drop` 掉，
`matmul.Inputs[1]` 指向一个已经没有生产者的 arg —— 读到的是一块没人写的槽。

症状很有欺骗性：**QK^T 单独折没问题，只有 PV 折了之后输出全错**（max |diff| ≈ 0.99，
1200 个 float 全变），而 yolo 系 0 命中、看起来「也一致」。二分方式是加临时环境变量只启用
`bTransposed==true` / `==false` 各跑一次。修正后：

```csharp
viewLayer.Inputs.Add(aScale?.Outputs[0] ?? aSource);   // 有外层 Mul 就用 Mul 的输出
viewLayer.Inputs.Add(bScale?.Outputs[0] ?? bSource);   // 裸 Transpose 才直接用源张量
```

### 14.6 复现

```powershell
$env:ONNXSHARP_THREADS='4'
$cli='tools/OnnxSharp.Cli/bin/Release/net10.0/OnnxSharp.Cli.dll'
$m='F:/code/TinyLabel/train/testrftedr/runs/rfdetr-small.onnx'
$i='bench/inputs/i1x3x512x512.f32'

# 命中数 / 命中详情
$env:ONNXSHARP_DEBUG_FUSE='1'; dotnet $cli run $m --input $i      # 期望 24 行 [attnview]

# 等价性
$env:ONNXSHARP_NO_FUSE='attnview'; dotnet $cli run $m --input $i --dump bench/ab/dump-base
Remove-Item Env:ONNXSHARP_NO_FUSE; dotnet $cli run $m --input $i --dump bench/ab/dump-new
Get-FileHash bench/ab/dump-base/*.f32, bench/ab/dump-new/*.f32 -Algorithm MD5

# A/B
$env:ONNXSHARP_NO_FUSE='attnview'; dotnet $cli bench $m --input $i --warmup 4 --iters 10
Remove-Item Env:ONNXSHARP_NO_FUSE; dotnet $cli bench $m --input $i --warmup 4 --iters 10
```

### 14.7 还剩什么

- **attention 链里还剩 1 个/层**：PV 输出 `[B,H,T,D] → Transpose[0,2,1,3] → [B,T,H,D] → Reshape`
  回 `[B,T,H*D]`。要折它得让 GEMM 的 **C 写面板**也带 stride（写 `[B,T,H,D]` 布局），
  12 个节点约 1.4 ms（0.6%）。属于「输出侧折叠」，可以接着做。
- **yolo 系 0 命中的原因**：ultralytics 的 attention 是 `[B,H,T,D]` 布局下的**矩阵转置**
  （`perm=[0,1,3,2]`），它的 A 是列主序（行距 1、列距 T），当前内核的 A 只支持单一 `aRow`
  （列步长固定 1）。要吃得下得给 12/6/4/1 行内核都加 A 的双 stride。但 yolo11n 里这 2 个
  MatMul 一共占 1.4%（约 0.5 ms），**不值得**。
- backbone / decoder 里剩下的 63 个 Transpose 没有 MatMul 消费者（是 `Reshape` 前的布局调整、
  decoder 的 permute 等），不在 P4 范围。

---

## 15. 2026-11-15 P3/P5：Where 按行选择 + Reshape 零拷贝（合计 -2.0%）

### 15.1 Where：2.6 ms → 0.1 ms

rfdetr 里只有一个 Where，形状是 `cond Bool[1,1024,1]` + `x F32[]`（**标量**）+ `y F32[1,1024,256]`，
即「每行 256 个元素共享一个条件值」。原实现是纯逐元素：每个元素一次 `OpUtil.CopyBytes`
（内部两次 `Buffer.AsSpan` + `CopyTo`），1024×256 = 262144 次 → 2.6 ms，折合 4 GB/s。

按行快路径（`WhereLayer.ForwardRows`）：尾维上「条件广播（stride 0）+ 两个源各自连续（1）或广播（0）」
时，外层走 `dims[0..^1]`、内层一次写 256 个元素 —— 每行一次判定，然后整块 memcpy 或整行填充。
填充用「等宽整数」`Span<short/int/long>.Fill`，位模式与 float/int64 完全一致，省掉按 dtype 分派。
不满足条件的形状退回原逐元素路径（行为不变）。

### 15.2 Reshape：4.3 ms → 0.2 ms（槽位复用，零拷贝）

Reshape 只改形状标签，原实现却做整块 memcpy（119 个 reshape 共 4.3 ms，平均每次 3.6 MB）。

新增 `LayerInfo.InPlaceInput`（默认 `-1`），`ReshapeLayer` 返回 `0`，`BufferAllocator` 让输出 arg
与输入 arg **共用同一个槽**（槽 → Tensor 是一对一，于是两者是**同一个 Tensor 对象**），
`ReshapeLayer.Forward` 里检测 `ReferenceEquals(inputs[0], outputs[0])` 直接 return。

两条前提在 allocator 内检查，缺一就退回独立槽位：

1. 输入必须是**中间结果**（`firstWrite != NeverWritten`）。图输入/常量有自己的存储，
   改它们的形状会污染调用方的张量（同一份 buffer 被下一次 `Run` 复用）。
2. 本算子必须是该输入的**最后读者**（`lastRead == i`）。否则下游读到的是被 Reshape 改过的形状
   （数据没变、形状标签变了），形状推断会错位。

动态形状的 Reshape 不复用（要 `Fit` 成不同大小）。`ONNXSHARP_NO_INPLACE=1` 可整体关闭。

### 15.3 结果

| 项 | 前 | 后 |
|---|---:|---:|
| Where ×1 | 2.6 ms | 0.1 ms |
| Reshape ×119 | 4.3 ms | 0.2 ms |
| profile sum-of-min | 211.0 ms | 208.5 ms |
| yolo11n sum-of-min | 36.9 ms | 36.5 ms（Reshape 6 个 0.3 → 0.0） |

**端到端 bench 已经测不动了**：本机同一配置下「后跑的那个进程」平均慢 11 ms（4.7%），
比本轮收益还大。用 6 对交替顺序反演（new 先跑的 3 对合计 −15.6 ms、base 先跑的 3 对 +6.4 ms）
得到 Δ ≈ **−4.6 ms（−2.0%）**，与单项 profile 的 −2.5…−6.6 ms 一致。
**以后端到端结论以 profile 的 sum-of-min 为准**，bench 只用来做「有没有大回归」的粗筛。

一致性：rfdetr 两个输出 + 4 个 yolo 模型（`ONNXSHARP_NO_INPLACE` 关/开）逐位一致；
单测 116/118 passed, 0 failed。

---

## 16. 2026-11-16 全引擎对比（P6' + P4 + P3 + P5 汇总成绩）

四引擎同会话、同机、4 线程、静态 batch=1、FP32、输入全 0.5、预热后取 N 次，只统计纯前向。
**median（ms）**：

| 模型 | onnxsharp | onnxruntime-cpu | opencv-dnn | simdyolo |
|---|---:|---:|---:|---:|
| yolov8n | **43.0** | 27.9 | 50.3 | 33.0 |
| yolo11n | **37.2** | 23.4 | 46.3 | 30.4 |
| yolov8s | **116.2** | 77.0 | 138.5 | 82.1 |
| yolo11s | **108.6** | 67.3 | 133.2 | 77.9 |
| rfdetr-small | **217.4** | 188.8 | 326.0 | N/A |

### 16.1 与冻结基线比（`progressv2 §2`，2026-11-15）

| 模型 | 基线 | 现在 | 变化 | vs ORT（前 → 后） |
|---|---:|---:|---:|---:|
| yolov8n | 46.0 | 43.0 | **−6.5%** | 1.63× → **1.54×** |
| yolo11n | 39.1 | 37.2 | **−4.9%** | 1.66× → **1.59×** |
| yolov8s | 121.0 | 116.2 | **−4.0%** | 1.58× → **1.51×** |
| yolo11s | 111.7 | 108.6 | **−2.8%** | 1.72× → **1.61×** |
| rfdetr-small | 262.5 | 217.4 | **−17.2%** | 1.39× → **1.15×** |

- rfdetr 的 −17.2% = P6'（MR=6/12，−9.8%）+ P4（Transpose 折进 GEMM，−4.5%）+ P3/P5（Where + Reshape，−2.0%）。
- yolo 系只吃到 −3…−7%：它们的 GEMM 占比低，且 attention 是 `[B,H,T,D]` 内的矩阵转置（P4 0 命中）；
  Reshape/Where 在 yolo 里本来就小（yolo11n 的 Reshape 只有 0.3 ms）。
- **对 OpenCV dnn 全面领先**（0.67× ~ 0.85×）；**对 SimdYolo 仍慢 22%~42%**（1.22× ~ 1.42×），
  差距主要在 GEMM 与 Conv 的打包/微内核，不在算子覆盖率。

### 16.2 测量注意（这次又踩到了）

`run_all.py` 的轮转只能部分抵消顺序偏置：`--runtimes` 多引擎一起跑时，**后跑的引擎会被前面
引擎的余热拖慢**，本轮就出现过 ORT 的 rfdetr median 254.8（而 min 189.9）、我们自己的 rfdetr
median 337.9（min 243.1）这种自相矛盾的数字 —— 同一个引擎 min 与 median 差 100 ms 时，
**该轮数据必须作废，改用单引擎独立重测**。

最终做法：**每个引擎单独一次会话、跑前冷却 25~40 s、`% Processor Performance` 确认回到 137%+**，
取该引擎自己的最好 median。上表即按此口径重测（rfdetr：ours 217.4/214.1、ORT 188.8/186.7；
SimdYolo s 档：82.1/75.4 与 77.9/75.9）。

> `bench/results/latest.json` 每次运行会被覆盖，只保留最后一次的成绩；本表是 6 次独立运行的汇总。

### 16.3 新发现：yolo 系的瓶颈是 Conv，不是 GEMM（P1 的否决只在 rfdetr 成立）

`profile yolo11n`（4 线程，sum-of-min 36.4 ms）：

| 类型 | ops | min ms | share |
|---|---:|---:|---:|
| **Conv** | 87 | **30.7** | **84.4%** |
| Concat | 23 | 1.9 | 5.3% |
| MaxPool | 3 | 1.0 | 2.8% |
| Split | 11 | 0.6 | 1.7% |
| 其余（Mul/Softmax/Add/Resize/MatMul/Transpose…） | — | 1.9 | 5.2% |

ORT 整个 yolo11n 只花 23.4 ms —— 也就是说**我们光 Conv 就比 ORT 的全模型还慢 7 ms**。
`planv2` 里 P1「Conv 低线程效率」被否决的依据是「Conv 实测 11.2 ms（4.3%）」，
那个数字是 **rfdetr-small** 的（DETR 的 backbone 轻、GEMM 重）；**yolo 系完全相反**：
Conv 占 84%，是唯一的主项。

结论：
- **rfdetr 的路线（GEMM 微内核 + 融合）不该往 yolo 上套**，yolo 要提速只能改 Conv。
- 87 个 Conv 里最大单个才 1.94 ms（`model.3/conv`），是**大量中等尺寸 conv 的效率问题**，
  不是某个巨大 conv；对应的是 Winograd/Im2Col 的选择阈值与打包开销，属于 P1 的原始议题，
  但需要按 yolo 的形状分布重做一遍（rfdetr 形状太少，当时的结论不适用）。
- 这一项未排期：当前阶段目标是 rfdetr ≤210（差 3.5%）。

## 17. 2026-11-16 P7-D 验证：并行阈值不是瓶颈，但 attention 的 QK/PV 有 1.8× 的冷数据损失

来自 `planv2 §P7.2` 的参照清单：ORT 的 MLAS 用 `M·N·K / 65536` 决定是否开线程，
我们原来固定 `1<<22`（等效要求 64 个 MLAS 任务）。怀疑 rfdetr 的 attention 小 k 因此全程单线程。

### 17.1 先把它变成可测量的（新增工具，默认行为不变）

- `GemmKernel.ParallelThreshold` 由 `const` 改为可写属性，默认仍是 `1<<22`，
  可用 `ONNXSHARP_GEMM_THRESHOLD=65536` / `1<<16` / `64K` 覆盖。
- `gemmbench --threshold 1<<16` 直接扫。
- `gemmbench --attn 1`：复刻 rfdetr attention 的真实形状与视图（4 样本 × 6 head、T=257、D=64，
  `aRow=384 aBs=64 bRow=1 bCol=384`，`outer=4 × inner=6` 的派发序列与 `MMv` 一致），
  用来把端到端 ±10% 的噪声隔离掉。
- `MMv` 的 trace 行补上 `aRow/aBs/bRow/bCol/bBs` —— 否则无从知道图里到底走哪个 stride。

### 17.2 结论：阈值不是瓶颈，P7-D 关闭

rfdetr 全部 139 个 MatMul 的 `m·n·k` 最小也有 19.6 M（decoder 的 `300×256×256`），
QKV 151 M、QK 101 M、fc1 606 M —— **全都远高于 `1<<22`（4.19 M）**，
把阈值降到 `65536` 不会改变任何一层的并行决策。

端到端交替 A/B（同二进制、只差 `ONNXSHARP_GEMM_THRESHOLD`）：min 217.8 → 237.4，
差异在噪声内且方向随机。**没有可测收益，不改默认值。**

### 17.3 真正的发现：QK/PV 在图里比同参数探针慢 1.8×

`gemmbench --attn 1`（数据全热）与 `profile`（图里真实数据流）同参数对比：

| 算子 | 图里实测 | probe 同参数 | 差距 |
|---|---:|---:|---:|
| QK^T（`257×64×257` × 24） | 1.73 ms / 117 GFLOPS | 0.95 ms / 213 GFLOPS | **1.8×** |
| PV（`257×257×64` × 24） | 1.03 ms / 197 GFLOPS | 0.54 ms / 377 GFLOPS | **1.9×** |

12 层合计 **33.1 ms**。差距不在打包、也不在派发（probe 里 `batch=6/12/24` 分别是
248/249/245 GFLOPS，与派发次数无关），而在**数据是冷的**：每层的 attention 矩阵是
`24×257×257×4 B = 6.3 MB`，QK 写、Softmax 读+写、PV 再读，一轮 25 MB 全在 L2（1 MB/核）之外。

**上限**：QK+PV 若能做到 probe 的 70%，12 层 33.1 → 23 ms，端到端约 **−4%**。
这需要把 QK→Softmax→PV 融合成「不落地 6.3 MB attention 矩阵」的分块实现（大改动，未排期）。

### 17.4 否决：`PackColumns` 改成「按行优先」不划算

看到 `bCol=384`（K^T 视图，每取一个 float 跳 1536 B）时，直觉是 cache line 利用率只有 6%，
于是把遍历顺序反过来（外层 c 顺序读、内层 t 写面板）。微基准（`--btrans`，`bColStride=k=64`）
确实从 5.314 → 0.728 ms，**但在真实参数 `bColStride=384` 下毫无差别**：

| 模式 | QK | PV |
|---|---:|---:|
| 原顺序（t 外 c 内） | 0.958 / 1.035 ms | 0.538 / 0.613 ms |
| 行优先（c 外 t 内） | 0.952 / 1.018 ms | 0.586 / 0.615 ms |

原因是原顺序在内层已经复用了那 16 条 cache line（`t+1` 只前进 4 B），
所谓「6% 利用率」只在 `bColStride == k`（列块正好铺满一行）时成立。
**已回退**，`gemmbench --btrans` 保留为今后的探针。

## 18. 2026-11-16 P6' 余项验证：BLIS 式 A 打包是**负收益**，P6' 关闭

`planv2 §P6'` 把最后 8 个点（打包路径 92.2% → 100%）归给「BLIS 式 A 打包 + k 分块」，
上限 10 ms。再往下拆，这 8 个点里能指认的只有一条：`m = 1028` 时 A 是 1.58 MB，
**超过 L2（1 MB/核）**，而 `Multiply6Rows` 在 k 循环里同时读 6 条相距 `k×4 = 1536 B` 的行，
是 6 条独立的 load stream。打包成 `[k][6]` 后每步只碰 1 条 line —— 这就是要验证的假设。

### 18.1 做法

- `PackARows`：A 的 6 行一组 → `[k][6]` 面板（24 B/步）。
- `Multiply6RowsPacked`：与 `Multiply6Rows` 逐指令同构，只把 6 条跨行广播换成 1 条连续读，
  累加顺序不变（逐位一致）。
- MR 取 **6** 而不是 12：`Multiply12Rows` 的 24 个累加器要 AVX-512VL 的 ymm16~31，
  A 面板版若沿用 MR=12，在测量时会把「寄存器溢出」和「A 打包」混在一起（实测慢 5.5×，
  是 spill 而不是 panel 的问题）。
- `gemmbench --apack 1`：**交替**跑生产路径与 A 面板路径、各自取 min。机器上有 IDE 在抢核，
  分开测的绝对值会在 3 ms 和 20 ms 之间跳，只有交替比值可用。
- B 两侧都走 `PackColumns`（否则测的是「B 是否打包」）。

### 18.2 结果（x = 生产路径 min ÷ A 面板 min，>1 才是 A 面板更快）

| m×k×n | 轮 1 | 轮 2 | 轮 3 |
|---|---:|---:|---:|
| 1028×384×1536（fc1） | 0.88 | 0.85 | 0.73 |
| 1028×384×384（QKV） | 0.68 | 0.67 | 0.71 |
| 1028×256×1024 | 1.33 | 0.84 | 0.96 |

9 次交替里有 8 次 A 面板更慢，384 列时稳定在 **−29% ~ −33%**。

### 18.3 为什么是负收益

1. 打包本身要重写一遍 A（1.58 MB），而原路径只是**读** 6 条 stream —— Zen4 的 L2 预取器
   跟得住 8~16 条 stream，6 条不构成问题。
2. 逐字节流量完全相同（每步 24 B）。A 面板只把「每步碰几条 line」从 6 降到 1，
   在 L3/DDR 带宽不是瓶颈时这一项不值钱。
3. 面板多占 1.58 MB，把 B 面板（24 KB，L1）和 C 的驻留挤出去。

### 18.4 A 驻留的真实上限只有 4%

同一 `gemmbench --shape` 口径下，只改 m（A 的大小）：

| m×k×n | A 大小 | 打包路径 | GFLOPS |
|---|---:|---:|---:|
| 1028×384×1536 | 1.58 MB（超 L2） | 2.998 ms | 404 |
| 120×384×1536 | 184 KB（L2 内） | 0.337 ms | 420 |

**4%** 就是「让 A 完全驻留」能拿回的全部。而 MC 分块要为此付 B 的重复读
（2.36 MB × m/MC 遍），净收益 < 2%，还要重写五环 + 面板 + 逐位一致验证。
投入产出不成立。

### 18.5 结论

**P6' 关闭。** 打包路径 **92.2%**（393.8 / 427.1 GFLOPS），剩余 8% 里
4% 是 A 驻留（上面证明拿不到）、其余是微内核的指令发射上限。
rfdetr 的 GEMM 侧不再有 >1% 的单点。

测量纪律再记一条：**交替测**。这轮如果沿用「先测 A、再测 B」，
会把 IDE 抢核的时间差误读成 20% 的优化（第一轮 3.4 vs 5.2、第三轮 8.4 vs 5.0，方向相反）。

## 19. 2026-11-16 yolo A/B（Conv 去物化 + 行块 4→8）：**命中 0 层，已回退**

`planv2 §P7.2` 给 yolo 的 Conv 排了两条：**A 去物化**（不把 `[K][16]` 面板写进缓冲，
内核按 offsets 直读输入）与 **B 行块 4→8**（AVX-512VL 的 ymm16~31 装得下 8 行 × 2 = 16 个累加器）。

### 19.1 实现（已回退，代码见 `6138066^`）

- `RowBlock` 由 `const 4` 改为按层解析（`ResolveRowBlock` + `RowsPerBlockFor(oc, K)`），
  `PackedWeights.RowBlock` 纳入缓存校验。
- `Multiply8`：19 个 YMM（16 累加器 + 2 条 B + 1 广播），每 K 步的 2 条 B 载入摊到 8 行。
- `Multiply8Direct` + `TryTapOffsets`：每个核位置只存一个起始下标，16 个像素就地连续读，
  **面板完全不物化**；跨行 / 贴边 / `strideW != 1` 的面板回退到打包路径。
- 顺带放开准入：直读可用时 `K ≤ 4096`（原规则是 `K×16 ≤ 32768`，即 `K ≤ 2048`）。
- 开关：`ONNXSHARP_PACKED_ROWS=4|8`、`ONNXSHARP_PACKED_DIRECT=0`。

### 19.2 单层 `convbench`（4 线程，交替 5 轮取各自 min）

改动只影响「`K > 2048` 且 `oc ≥ 128`」的层，其余层的两条配置逐位相同，实测数字也完全一致：

| 形状 | 基线（4 行打包） | A+B（8 行直读） | 说明 |
|---|---:|---:|---|
| ic=3 oc=16 640² k3s2 | 2.40 | 2.43 | 门槛未命中 |
| ic=16 oc=32 320² k3s2 | 1.19 | 1.19 | 同上 |
| ic=32 oc=64 160² k3s2 | 0.92 | 0.92 | 同上 |
| ic=64 oc=128 80² k3s2 | 0.84 | 0.84 | 同上 |
| ic=128 oc=256 40² k3s2 | 0.92 | 0.92 | 同上 |
| ic=256 oc=256 40² k3s1 | im2col 8.26 | **6.54** | 新准入，−21% |
| ic=256 oc=256 20² k3s1 | im2col 2.08 | **1.97** | 新准入，−5% |

只看这张表，A+B 在它够得着的两层上是赢的（40² 那层 144 vs 114 GMAC/s）。
**但这两层在 yolo11n 里根本走不到 `PackedConv`。**

### 19.3 为什么一层也命不中：Winograd 先把 3×3/s1 全接走了

选路顺序是 `Depthwise → Winograd(3×3, s1, p1, 空间 ≤ 4096) → PackedConv → im2col`。
yolo11n 里 `K > 2048` 的层（`ic ≥ 256` 的 3×3）**全部在 20×20**（`spatial = 400 ≤ 4096`），
于是全被 Winograd 接走；能走到 `PackedConv` 的只剩 s2 层（`ic = 3/16/32/64/128`，`K ≤ 1152`），
行块仍是 4、直读门槛碰不到。同参数三种算法的实测：

| 形状 | im2col | packed（基线） | winograd |
|---|---:|---:|---:|
| ic=256 oc=256 40² k3s1 | 8.25 ms（114 GMAC/s） | —（K=2304 被准入挡下） | **4.74 ms（199 GMAC/s）** |
| ic=256 oc=256 20² k3s1 | 2.09 ms（113） | —（同上） | **1.78 ms（132）** |
| ic=128 oc=128 80² k3s1 | 5.48 ms（172） | — | 5.37 ms（176） |
| ic=64 oc=64 160² k3s1 | **5.94 ms（159）** | — | 8.68 ms（109，且空间 >4096 已被拒） |

即使把准入放开到 `K = 2304`，40² 那层 6.54 ms 仍比 Winograd 的 4.74 ms 慢 **38%**。
**A/B 的适用面被 Winograd 挤成了空集。**

### 19.4 模型内验证：归一化后在噪声内（但根因已由 §19.3 定性）

交替 4 轮取 `/model.20/conv`（目标层）与 `/model.7/conv`（两配置代码完全相同，作对照）的 min，
再用对照层归一化掉机器抖动：

| 轮次 | 目标层 old → new | 对照层 old → new | 归一化 new/old |
|---|---|---|---|
| 1 | 2.53 → 2.37 | 3.29 → 2.86 | 0.89 |
| 2 | 2.28 → 3.39 | 3.04 → 4.09 | 1.06 |
| 3 | 2.61 → 3.57 | 3.60 → 3.73 | 1.20 |
| 4 | 2.79 → 3.34 | 3.66 → 3.61 | 1.07 |

中位 1.06，方向是「略慢」，但幅度在噪声内。**根因不靠这张表判**：层的命中面已经由 §19.3 排除，
这里只能说明「模型内没有可测收益」。

### 19.5 单独记下来两条副产物（以后别再走一遍）

1. **8 行在打包路径上没有收益**：oc=128 的 80²→40² 是 0.85 vs 0.85、oc=256 的 40²→20² 是 0.92 vs 0.92；
   oc 小时还明显更慢（整层只有 2~4 个 8 行块，8 次 bias 广播 + 16 次存回的固定开销和短 K 同量级）：
   640²→320² 1.23 → 3.28 ms、320²→160² 1.19 → 5.88 ms。
2. **直读只在「冷 cache」口径下领先**：`convbench` 每层从冷内存起跑，省掉的那趟打包（K×16×4 B）
   值 21%；模型内上游刚写过的输入还在 L2/L3，这趟省下来的量测不出来。
   → 以后凡是「省一趟搬运」的优化，先用 `profile` 的逐算子 min 判，别用 `convbench` 的绝对秒数。
3. **本机噪声新基线**：同一形状同一配置，`convbench` 的 min 在 0.85 ~ 1.30 ms、1.19 ~ 5.88 ms 之间跳；
   yolo11n 的 `sum of min` 在 33.9 ~ 158 ms 之间跳。**低于 5% 的差异只能靠同进程交替 + 归一化对照。**

### 19.6 结论

**A/B 回退**（`PackedConv.cs` 回到 `6138066^`），C（ic 块外提）同样以「`K > 2048` 且 `oc ≥ 128`」
为前提，命中面一样是空集，**不启动**。

yolo 的 Conv 侧现状：3×3/s1 已交给 Winograd（199 GMAC/s，4 核 AVX2 峰值的 ~83%），
s2 层由 `PackedConv` 扛（96 ~ 140 GMAC/s），1×1 走 im2col GEMM（177 ~ 197 GMAC/s）。
**剩下的空间在 s2 层**（128 GMAC/s vs 1×1 的 197），要做就得是 stride-2 的 Winograd
（偶/奇子格拆分）或 ORT 那种 NCHWc 间接卷积 —— 与 A/B/C 不是同一件事，需要单独立项。

## 20. 2026-11-16 `ab` 工具（同进程交替 A/B）+ P8 判定：**+2.2%，采纳**

### 20.1 为什么必须换口径

直到 §19 为止，所有结论都是「跑两次进程、比绝对值」。本机这个口径的噪声已经大到不可用：
同一形状同一配置的 `convbench` min 在 0.85 ~ 1.30 ms 之间跳，yolo11n 的 `sum of min` 在
33.9 ~ 158 ms 之间跳（§19.5）。**rfdetr 差 3.5% 就是 ~8 ms，落在噪声里。**

`ab` 的做法：**两个引擎放进同一个进程，逐轮交替、每轮交换先后顺序**。

```
OnnxSharp.Cli ab <model.onnx> --input <f.bin> --a K=V[,K=V] --b K=V[,...] \
                [--iters 5] [--rounds 5] [--warmup 4] [--no-check]
K: attn_fuse(ONNXSHARP_ATTN_FUSE_MAXSEQ) | attn_block(ONNXSHARP_ATTN_BLOCK_SEQ)
   | gemm_threshold(ONNXSHARP_GEMM_THRESHOLD)
```

几个必要的细节：

- **两套配置各建一个引擎**。图融合（attention 合并）发生在建图期，只能重建 —— 所以
  `GraphFusion.AttentionFusionMaxSeq` 从 `static readonly` 改成可写属性；`AttentionOps.BlockKernelMinSeq`
  同样可写，并且**层在构造时捕获**，否则第二个引擎会把第一个的值覆盖掉。
- **每次计时前重放配置**：建图期旋钮此时是空操作，运行期旋钮（`gemm_threshold`）必须重放。
- **未列出的键 = 环境默认**（不是「沿用另一侧」）—— 否则漏写一个键会让两侧悄悄变成同一配置。
- **统计量用配对**：每轮取两侧 min，算比值，最后取**比值的中位数**；噪声地板取
  `max(1%, 3×MAD)`。平均值会被单个后台尖峰带偏（见自检里 mean 233 vs median 209）。
- **图算子直方图 diff**：先确认开关真的改了图，避免测了个空操作。
- **逐输出差异**：max / mean / p99 / >1e-3 的个数 + 最差 5 个下标。只给全局 max 没有诊断价值
  （见 §20.3）。

### 20.2 自检：同配置必须判「噪声内」

`--a attn_fuse=4096 --b attn_fuse=4096`（两侧完全一样），6 轮：

| 轮 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| A/B | 1.001 | 1.011 | 1.002 | 1.010 | 1.006 | 1.004 |

跨度 1.0%，判定「差 0.4% ≤ 噪声地板，噪声内」。**这个工具能分辨的最小信号约 1%。**

预热必须够多：`--warmup 1` 时第一轮 247.8 vs 208.5（+19%，tier-0 代码），会把结论带成
「不可判定」。默认 4 次预热后第一轮就稳了。

### 20.3 P8 判定：融合 vs 不融合

| 配置 A | 配置 B | A min | B min | 配对中位 | 跨度 | 判定 |
|---|---|---:|---:|---:|---:|---|
| `attn_fuse=0` | `attn_fuse=4096` | 225.5 ms | 220.8 ms | **1.022** | 2.9% | **B 快 2.2%**（噪声地板 1.6%） |

图差异：`FusedAttention +12`、`MatMul −24`、`Softmax −12`、`Transpose −12`、`Mul −24`、`Reshape −48`
（664 → 556 个算子）。另一次独立运行给出 1.055，两次都超过噪声地板，**取保守值 2.2%**。
预估上限是 −4%，落在下限。

### 20.4 数值：差异来自 top-k 近邻换序，不是融合的 bug

| 对照 | dets >1e-3 | max\|d\| | labels >1e-3 | max\|d\| |
|---|---|---:|---|---:|
| 融合 vs 不融合 | 108/1200 | 5.0e-1 | 238/900 | 5.1e-1 |
| **块内核 vs 逐行内核（都融合）** | 60/1200 | 4.7e-1 | — | — |

第二行是关键对照：**同为融合路径、只换求和顺序**，输出差异是同一量级、同一形态。
形态都是「相邻 query 的 4 个 det 值 + 3 个 label 值成对翻转」（dets 736~742 / labels 552~557），
即两个分数接近的 query 在 top-k 里换了位置 —— 12 层 transformer 把 1e-7 的求和顺序差放大到
1e-3 量级 logits，近邻就翻过来了。**这是模型自身的条件数，不是 P8 的数值问题**；
换成参考实现同样会翻。因此 P8 **默认开启**（`ONNXSHARP_ATTN_FUSE_MAXSEQ` 默认 4096）。

### 20.5 判定之后的新热点

`profile`（`--iters 6`，sum-of-min **219.7 ms**）：

| # | 算子 | min ms | 占比 |
|---|---|---:|---:|
| 1~12 | `FusedAttention` × 12（encoder layer 0~11） | 7.18 ~ 7.22 | **40%** |
| 13~ | `mlp/fc1`、`fc2` 等 MatMul × 24 | 3.4 ~ 3.7 | 各 1.5~1.7% |

每层 804 MMAC / 7.2 ms = **112 GMAC/s**，只有同机 GEMM 峰值（197）的 57%。
也就是说 P8 把「搬 6.3 MB」的问题解决了，剩下的瓶颈变成了**融合内核自身的算术密度**：
S 块仍然是 `[T][T]` 落地（T=1024 时 4 MB），Q 侧没有分块复用。
下一步是 flash-attention 式（Q 分块 + 在线 softmax，S 块不落地），上限约
`7.2 → 4.2 ms`（按 197 GMAC/s 计），对端到端 **−16%** —— 但工作量和风险都远大于 P8。

> **§21.1 更正**：上面这张表的读法是错的。12 个 `FusedAttention` **不同构**——只有 3 个是
> `T=1028`（≈6.0~6.6 ms），另 9 个是 `b=4, T=257`（≈1.6~1.7 ms）。「各 7.18 ms」是把大层
> 的数字当成了 12 层。attention 的真实占比是 **16.4%**（35.5 ms），不是 40%；
> flash 重写的上限也因此从 −16% 修正为 **不到 −5%**。详见 §21。

## 21. 2026-11-16 P9 attention 内核三步：**+2.2%**（顺带纠正 §20.5 的占比误读）

计划见 `planv2.md §P9`。四步里做完三步（P9.1/P9.2/P9.3），P9.4/P9.5 用实测天花板关闭。

### 21.1 先纠正：12 个 `FusedAttention` 不同构，attention 只占 16.4%

`ONNXSHARP_ATTN_TRACE=1` 下临时加了 `[attnthr]` 探针（每次 `ComputeBlocks` 打一行入参 + 实际
工作线程数；一次性，收工后已删），rfdetr-small 一次 forward 的 12 行分两族：

| 层 | 形状 | 行块 | 每层 MMAC | 每层 ms | GMAC/s |
|---|---|---:|---:|---:|---:|
| layer.3 / 6 / 9 | `b=1 h=6 t=1028 d=64` | 127 × 9 块 | 811 | 6.0 ~ 6.6 | **134** |
| 其余 9 层 | `b=4 h=6 t=257 d=64` | 256 × 2 块 | 203 | 1.6 ~ 1.7 | **127** |

两族吞吐一致（134 vs 127），**没有并行退化、没有形状异常**——就是工作量差了 4 倍
（`T²` 差 16 倍、`b` 差 4 倍）。`profile` 的 sum-of-min 里 attention 合计 **35.5 ms / 219 =
16.4%**，与 `[attnthr]` 的 `3×6.0 + 9×1.6 = 32.5 ms` 对得上。

这条更正很重要：它把 flash-attention 重写的**收益上限**从「−16%」压到
「不到 −5%」（16.4% × 最多 30%），而成本一点没变。

### 21.2 P9.1 行块预算扫描：512 KB 已是最优，默认不变

`ScoreBudget`（`ONNXSHARP_ATTN_SCORE_BUDGET`，单位 KB）参数化后三组独立配对
（`ab --rounds 5 --iters 5 --warmup 4`，A 恒为 512）：

| 对照 | 配对中位 | 判定 |
|---|---:|---|
| 512 vs 256 | 0.992 | 256 慢 0.8%，噪声内 |
| 512 vs 128 | 0.993 | 128 慢 0.7%，噪声内 |
| 512 vs 1024 | 0.991 | 1024 慢 0.9%，噪声内 |

三组方向一致（都指向 512 更好），但幅度都在 1% 地板以下。**「1.08 MB 工作集超出 1 MB L2
会有悬崖」的猜想不成立** —— 说明分块内核不是 L2 带宽受限，而是算术受限，与 §21.6 的
短 K 分析一致。旋钮保留（默认 512），但没有换取默认值。

### 21.3 P9.2 softmax 归一化后置：端到端 **+0.6%**

`Σ_j (e_j/Σ)·v_j` 改算 `(1/Σ)·Σ_j e_j·v_j`：S 块里保留 exp 值，`1/Σ` 乘到 PV 之后的
输出行上（每行 `d=64` 个值，而不是整行 `T=1028`）。省掉每块一次 522 KB 读 + 522 KB 写的
`Scale` 趟。`ONNXSHARP_ATTN_LAZY_SOFTMAX=0` 回退。

- 端到端：6 轮 5 轮同向，中位 **1.007**，跨度大（0.985~1.076），**落在噪声地板内**。
- 段内（`[attnblk]` 同 run 配对）：`soft+pv` 两次都是 **−3~4%**（soft 少一趟，pv 多一行
  缩放），方向稳定。
- 数值：`max|d| = 9.3e-5`，`>1e-3` 计数 **0/1200（dets）、0/900（labels）** —— 比 §20.4 的
  kernel-swap 对照（60/1200）干净一个数量级，只有重结合误差。

严格少做一趟 522 KB 的读写，数值无害，故**保留**（默认开）。

### 21.4 P9.3 K^T / V 面板跨行块复用：**+1.0%，且逐位相同**

原来每个行块都要重打一遍 `K^T`（1028 列 × 64 的逐元素 gather）和 `V`（263 KB 连续拷贝）。
并行区间在 `index = bh·rowBlocks + block` 上是连续的，所以每个工作线程只需要
「记住上一次的 `bh`」——不需要跨线程共享面板（那要同步，6 份面板还会一起挤 L2）。
`GemmKernel.PackFromSource` 把打包与乘法拆开，`ONNXSHARP_ATTN_PACK_REUSE=0` 回退。

| 项 | 结果 |
|---|---|
| 配对中位 | **1.010**（6 轮 1.008~1.013，跨度 0.5%，全部同向） |
| 数值 | **`max|d| = 0.00E+000`（两个输出都是 0，逐位相同）** |
| 段内 | `qk` 0.153→0.134 / 0.157→0.143；`pv` 0.107→0.091 / 0.112→0.099 |

这是唯一一步能要求 `max|d| = 0` 的改动（纯重排，无算法变化），也确认了上两步的数值基准是干净的。
顺带发现 `V` 面板复用也值 ~12 µs/块 —— 连续拷贝并不像原先估计的那样免费。

### 21.5 累计判定**

`--a 'attn_lazy_soft=0,attn_pack_reuse=0' --b 'attn_lazy_soft=1,attn_pack_reuse=1'`
（6 轮）：**配对中位 1.022，跨度 0.7%，6 轮全部同向 → +2.2%，超过噪声地板。**

绝对值（`bench --iters 8 --warmup 4`）：min **220.7 ms** / median 225.6 ms（本机跨度大，
只作记录）。计划里的停止线是「三步合计 ≥4% 就收工」，实际 **2.2%**，差一半。

收工后删掉一次性 `[attnthr]` 探针再复测一遍（同 6 轮 5 iter 4 warmup、`ONNXSHARP_THREADS=4`）：
配对中位 **1.016**（1.011~1.021，6 轮同向），噪声地板 1.0%，`out1 max|d| = 3.86E-005`
（`>1e-3` 计数 0/900）。同一方向，幅度随机器状态在 1.6%~2.2% 之间浮动，**结论不变**。

> 踩过的坑：`ab` / `bench` 不设 `ONNXSHARP_THREADS` 时用满核，绝对耗时掉到 140 ms 量级，
> 配对方向也被噪声打乱（跨度 7.8%，判定「噪声内」）。所有 P9 数字都必须是 **4 线程**口径。

### 21.6 P9.4 / P9.5 关闭：天花板已经量出来了

`gemmbench --threads 4`，两个形状都是 16.7 MFLOP（对应 QK 与 PV）：

| 形状 | 角色 | GFLOPS | GMAC/s |
|---|---|---:|---:|
| `127 × 64 × 1028` | QK（短归约 k=64） | 222 | 111 |
| `127 × 1028 × 64` | PV（长归约 k=1028） | **334** | 167 |

**纯 GEMM 就复现了 attention 里 QK/PV 的 1.5 倍差距**，与打包、softmax、S 块落地都无关：
k=64 时每个 (列块, 行块) 微内核只有 8 个 k 步，序言/尾声（24 累加器清零 + 24 存 C）摊不掉。

- P9.4 的理论收益 = `qk 占比 46% × 33%` ≈ attention 的 15% ≈ **端到端 2.5%**，
  但只能靠**新写一个短 K 微内核**（MR=8/NB=32，AVX-512 下 32 个累加器）去换，
  而 P6' 已判定微内核侧「没有 >1% 的单点改善」。**不划算，关闭。**
- P9.5（flash 式在线 softmax）：S 落地 + 三趟 softmax 现在只剩 `soft` 的 16% 里的一部分，
  上限 **<2%**，而代价是 rescale、边界处理、三路内核并存。**关闭。**

### 21.7 复现

```powershell
$exe='tools\OnnxSharp.Cli\bin\Release\net10.0\OnnxSharp.Cli.exe'
$m='F:/code/TinyLabel/train/testrftedr/runs/rfdetr-small.onnx'; $i='bench/inputs/i1x3x512x512.f32'

# 分层形状（3 个大层 b=1,t=1028 + 9 个窗口层 b=4,t=257）与段内耗时
# 注：§21.1 用的 [attnthr] 探针是一次性的（打印 workers/units），已随 P9 收工移除。
$env:ONNXSHARP_ATTN_TRACE='1'; & $exe run $m --input $i --repeat 4 2>&1 | Select-String attnblk

# P9.1 预算
& $exe ab $m --input $i --a attn_budget=512 --b attn_budget=256
# P9.2 / P9.3 / 累计
& $exe ab $m --input $i --a attn_lazy_soft=0 --b attn_lazy_soft=1
& $exe ab $m --input $i --a attn_pack_reuse=0 --b attn_pack_reuse=1
& $exe ab $m --input $i --a 'attn_lazy_soft=0,attn_pack_reuse=0' --b 'attn_lazy_soft=1,attn_pack_reuse=1'

# P9.4 天花板
& $exe gemmbench --shape 127,64,1028   --iters 5 --threads 4
& $exe gemmbench --shape 127,1028,64   --iters 5 --threads 4
```

## 22. 2026-11-16 yolo 局部 NCHWc：三个前置问题**已量，全部通过**

`planv2 §P7.7` 立项前挂了三个问题：reorder 边界成本、块宽 16/32、与 `BufferAllocator` 共存。
本节把它们挨个量掉，并顺手纠正了 §19 的一个盲区（只盯 s2，漏了 3×3/s1 的 13.4 ms）。

口径：yolo11n、`bench/inputs/i1x3x640x640.f32`、4 线程、`profile --iters 6 --warmup 8 --top 300`。

### 22.1 问题 1：reorder 边界 —— **5 处、10.5 MB、0.336 ms（0.9%）**

方法：在 ONNX 图上做布局传播（`bench/layout_probe.py`）。规则是「算子能否在 c 布局上原地跑」：

| 类别 | 算子 | 处理 |
|---|---|---|
| 内核 | `Conv` | 吃/吐 c 布局（权重按块打包） |
| 逐元素 | `Mul` `Add` `Sub` `Div` `Sigmoid` `Relu` `Clip` `BatchNormalization` | 与布局无关，直接传递 |
| 通道维 | `Concat` `Split` `Slice` | 块宽对齐则免 reorder（拼接 = 块序列拼接） |
| 空间 | `MaxPool` `Resize` `Pad` | 每块内独立，需 c 版内核但免 reorder |
| 硬边界 | `Transpose` `Reshape` `MatMul` `Softmax` `Gather` … | 必须 reorder |

图规模：320 节点 / **88 个 Conv** / 196 initializer。传播结果：

- **298 个张量**可保持 c 布局（几乎整条主干 + neck + head 的 conv 段）
- **5 处 reorder，全部是 `Reshape`**：

| 字节 | 形状 | 消费者 |
|---:|---|---|
| 400.0 KB | `[1,256,20,20]` | `/model.10/m/m.0/attn/Reshape`（C2PSA 的 attention 入口） |
| 3600.0 KB | `[1,144,80,80]` | `/model.23/Reshape`（P3 head） |
| 900.0 KB | `[1,144,40,40]` | `/model.23/Reshape_1`（P4 head） |
| 225.0 KB | `[1,144,20,20]` | `/model.23/Reshape_2`（P5 head） |
| 131.2 KB | `[1,1,4,8400]` | `/model.23/dfl/Reshape_1` |

合计 **5256 KB 单向 / 10.5 MB 双向**。按 §P4 标定的 32 GB/s（111 个 transpose = 11 ms，
单张量 ~1.6 MB）折算 **0.336 ms ≈ 端到端 0.9%**。

两个值得记的细节：

1. **C2PSA 的 `Transpose`×3 / `MatMul`×2 / `Softmax`×2 不在边界上**。它们前面本来就有一个
   `Reshape` 把张量打回 NCHW，attention 分支整段是 plain —— 不需要为它写 c 版，也不产生额外
   reorder（前面那 400 KB 的 Reshape 已经算过了）。
2. 如果**不**做 `MaxPool`/`Resize` 的 c 版（阶段 1 的合理裁剪），再加 5 处（SPPF 的 3 个池化输入
   各 409 KB + 2 处上采样输入），总计约 **+2 MB / +0.13 ms**。仍在 0.5 ms 以内。

> 结论：reorder 不是拦路虎，**阶段 1 可以连 `MaxPool`/`Resize` 都不做**。

### 22.2 问题 2：块宽 —— **16，且必须把 3 通道输入层排除在 c 之外**

| 块宽 | `oc % BW == 0` 的层 | MAC 加权填充 | 裁决 |
|---:|---|---:|---|
| 16 | **86 / 88** | 见下 | ✅ |
| 32 | 70 / 88（16 个 `oc=48/80/112`） | ~10× | ❌ 直接否决 |

块宽 32 死因很直接：ultralytics 的通道数是 16 的整数倍而不是 32 的，补到 32 等于白跑一半 MMA。

块宽 16 的填充要拆开看，**这里有两个会误导人的平均值**：

- 「按 conv 算术平均」= 26.34% —— 被两个极端层拉高；
- 「按 MAC 加权」= 117% —— 更荒谬，因为 **95% 的填充来自 `/model.0/conv`（ic=3）**，
  它 640²×16 输出、`ic` 补到 16 等于 5.3× 空转。

而首层**根本不需要 c 输入**：它读 NCHW 图像、写 c 输出，`ic=3` 走一条普通标量/AVX 路径即可，
这段填充是假的。剔除后真实填充 ≈ **707 MMAC**，来源是 C3k2 里的两对 `oc=8 / ic=8` 瓶颈卷积
（`/model.2/m.0/cv1`、`/model.2/m.0/cv2` 一类），8 → 16 是 2× 填充。这 4 层实测合计 ~1.6 ms，
即便 2× 也只值 ~0.4 ms，而且**更划算的做法是让它们保持 NCHW**（多 1 次 reorder，比 2× 算术便宜）。

> 结论：块宽 **16**（正好 1 个 `zmm`）；c 布局的边界清单里额外排除
> 「`ic<16` 的输入层」与「`oc=8` 的瓶颈层」，它们当普通 NCHW 层用，各付一次 reorder。

### 22.3 问题 3：与 `BufferAllocator` 共存 —— **零改动**

三条已具备的事实（不是计划，是现状）：

1. `Layout` 枚举 + `Tensor.Layout` + `Arg.Layout` + `LayerInfo.GetLayouts` **已经预留**；
   加一个 `NCHWc16 = 1` 成员即可，不需要新的张量类型。
2. 槽分配只看「生产/消费区间不重叠 + `Buffer.Length` 够大」，**与布局无关**。同一槽复用给
   不同布局完全合法 —— 字节数都是 `元素数 × 4`。
3. `Tensor.Fit` 每次都把 `Layout` 复位成 `NCHW`、`Strides` 置空，**复用槽时布局自动失效**，
   不存在「上一个张量的 c 布局泄漏给下一个」的风险。

唯一要补的两处：`Fit` 增加一个 layout 参数（默认 `NCHW`，由 Conv/Concat 的 Fill 指定），
以及 `GetLayouts` 输出「哪些张量在什么时刻是 c 布局」的时间表（供调度器插 reorder）。

> 结论：这一项的风险等级从「架构改造」降到「加一个枚举成员 + 两个方法签名」。

### 22.4 收益推算 —— 顺带发现 §19 漏掉的最大一块

先做家族分解（`bench/conv_family.py`，profile 的逐算子时间 × 权重形状）：

| 家族 | 层数 | MMAC | 实测 ms | GMAC/s | 当前内核 |
|---|---:|---:|---:|---:|---|
| 3×3 s1 | 35 | 1194.4 | **13.44** | **88.9** | Winograd（仅空间 ≤4096）+ im2col |
| 3×3 s2 | 7 | 870.0 | 8.15 | 106.7 | `PackedConv` |
| 1×1 s1 | 45 | 1162.6 | 10.19 | 114.1 | im2col GEMM |
| 合计 | 87 | 3227.0 | 31.78 | 101.5 | （端到端 36.4 ms，Conv 占 86.8%） |

**这张表推翻了一个隐含假设**：§19 整节都在看 s2，因为那是 §P7.2 A/B/C 的命中面；但 conv 里
最大的一块是 **3×3/s1 的 13.44 ms**。35 个层里只有空间 ≤4096（40²/20²）的走 Winograd
（§19.3 实测 199 GMAC/s），160²/80² 的十几个层退回 im2col，把平均拉到 89。

what-if（**目标是假设，不是实测**；参照是 ORT MLAS 的 24-zmm NCHWc 内核）：

| 家族 | 现在 | 保守目标 | 省 | 乐观目标 | 省 |
|---|---:|---:|---:|---:|---:|
| 3×3 s2 | 106.7 | 170 | 3.03 ms | 200 | 3.80 ms |
| 1×1 s1 | 114.1 | 150 | 2.44 ms | 180 | 3.73 ms |
| 3×3 s1（未走 Winograd 的层） | ~89 | 150 | 3.66 ms | 180 | 4.55 ms |
| reorder | — | — | −0.34 ms | — | −0.34 ms |
| **合计** | | | **−8.8 ms** | | **−11.7 ms** |
| **端到端** | 36.4 | | **31.3 ms（−14%）** | | **27.6 ms（−24%）** |

保守档只承诺 s2 + 1×1（−5.1 ms ≈ −14%，另加 reorder）；3×3/s1 那一行算「上行」不承诺，
因为 Winograd 与 im2col 的选路阈值要不要一起动，是另一个决策。

**独立旁证（比上面的推算可靠）**：`§4` 的表里 OpenCV dnn 实测 **28.6 ms**，比我们快 28%。
OpenCV 的 conv 走的正是 `NC8HW8`（块宽 **8**，还没有 AVX-512 的 24 累加器内核）。
这条说明「c 布局在 yolo11n 上能拿到两位数百分比」不是纸面推算 —— 有人已经拿到了。

### 22.5 结论

| 前置问题 | 答案 | 判定 |
|---|---|---|
| reorder 边界 | 5 处 / 10.5 MB / **0.336 ms（0.9%）**；不做 MaxPool/Resize 也只 +0.13 ms | ✅ 通过 |
| 块宽 | **16**（86/88 整除）；排除 `ic=3` 输入层与 `oc=8` 瓶颈层 | ✅ 通过 |
| BufferAllocator | **零改动**（`Layout` 接口已预留、槽按字节复用、`Fit` 自动复位） | ✅ 通过 |

**建议立项，但按「先证一个数」执行**：阶段 1 只做 **s2 的 NCHWc 内核**（7 层、8.15 ms、
占 conv 的 26%），验收标准是「这 7 层从 107 → ≥170 GMAC/s」，`convbench` 微基准 + 端到端
`ab` 双读数。**拿不到 1.5× 就停，后面的 1×1 与 3×3 一律不做。**

不做的事：3×3/s1 的 Winograd 选路不动；C2PSA 的 attention 分支保持 plain；`Slice` 不 c 化
（它全在 head 的 Reshape 之后，本来就是 plain）。

### 22.6 复现

```powershell
# 前置问题 1/2：布局传播 + 块宽利用率（需 onnx）
& 'E:\misedir\installs\python\3.11.17\python.exe' bench/layout_probe.py F:/ai/models/yolo/yolo11n.onnx 16
& 'E:\misedir\installs\python\3.11.17\python.exe' bench/layout_probe.py F:/ai/models/yolo/yolo11n.onnx 32

# §22.4 家族分解
$env:ONNXSHARP_THREADS='4'; $exe='tools\OnnxSharp.Cli\bin\Release\net10.0\OnnxSharp.Cli.exe'
& $exe profile F:/ai/models/yolo/yolo11n.onnx --input bench/inputs/i1x3x640x640.f32 --iters 6 --warmup 8 --top 300 > yolo.log
& 'E:\misedir\installs\python\3.11.17\python.exe' bench/conv_family.py F:/ai/models/yolo/yolo11n.onnx yolo.log
```

> 两个脚本的 `python` 必须走实体路径：`E:\misedir\shims\python.exe` 在这个 shell 里静默返回
> exit 49 且无输出，`python` 命令同样吞掉 stdout。

## 23. 2026-11-16 P7.7 阶段 1：s2 的 NCHWc16 内核 —— **内核 1.82×、端到端 −5.5%**

`§22` 立项判据是「s2 的 7 层 107 → ≥170 GMAC/s，拿不到 1.5× 就停」。本节是阶段 1 的实测结果：
**判据通过（1.82×）**，端到端拿到 **−5.5%（yolo11n）/ −3.9%（yolov8n）**，但只有 `§22.4` 推算值
（−8.4%）的六成，差额的根因也定位了。

### 23.1 内核：blocked 布局 + 16 个 zmm 累加器

新文件 `src/OnnxSharp/Kernels/NchwcConv.cs`（`internal static`，AVX-512 专属）：

| 项 | 取值 |
|---|---|
| 输入布局 | `[C/16][H+1][W+1][16]`，左上 halo 置 0（3×3/s2/pad1 的取样范围恰落在 `[0,H]×[0,W]`） |
| 输出布局 | `[OC/16][OH][OW][16]` |
| 权重布局 | `[OC/16][IC/16][9][icLane][ocLane]`，一次打包、`ConditionalWeakTable` 缓存 |
| 累加器 | 2 个 oc 块 × 8 像素 = **16 个 `zmm`**，`Pixels=8`、`OcGroup=2` |
| 并行 | `ParallelRange.For(pairs * oh)`，`fixed` 指针要转 `nint` 才能进 lambda |

**适用面（`CanUse`）**：`groups=1`、`3×3`、`s=2`、`dil=1`、`pad=1`、`ic%16==0`、`oc%32==0`、
**且 `h`、`w` 都是偶数**。最后一条是踩出来的：blocked 输入只在**左上**留 1 像素 halo，
最后一个输出像素的窗口是 `[2(o-1), 2(o-1)+2]`，只有偶数边长才保证 `2*oh==h`、`2*ow==w`
刚好落在 `[0,h]×[0,w]` 内。**奇数边长会多读一列/一行** —— `101×97` 的 `max|d|` 直接是 **24**，
`7×5` 是 6.2（已实测）。补上偶数判据后这两个形状打印 `--`。

### 23.2 微基准（4 线程，`convbench --iters 20`）

| 形状 | im2col | packed | **nchwc** | +reorder | max\|d\| |
|---|---:|---:|---:|---:|---:|
| 16→32 320²→160² | 92.5 | 104.6 | **0.54 ms（218.5）** | 0.19 | 5.2e-6 |
| 32→64 160²→80² | 123.4 | 128.1 | **0.54 ms（219.6）** | 0.08 | 1.1e-5 |
| 64→64 160²→80² | 122.6 | 126.5 | **1.07 ms（219.9）** | 0.12 | 1.6e-5 |
| 64→128 80²→40² | 128.6 | 138.2 | **0.54 ms（219.8）** | 0.02 | 1.4e-5 |
| 128→128 80²→40² | 128.1 | 136.4 | **1.07 ms（220.0）** | 0.03 | 2.5e-5 |
| 128→256 40²→20² | 88.9 | 124.1 | **0.67 ms（175.2）** | 0.01 | 2.5e-5 |
| 256→256 40²→20² | 104.2 | — | **1.35 ms（174.2）** | 0.02 | 4.8e-5 |
| 128→128 20²→10² | 24.6 | 84.5 | **0.08 ms（174.1）** | 0.01 | 3.2e-5 |

MAC 加权（yolo11n 的 7 个 s2 层，1.180 GMAC）：**112.3 → 204.1 GMAC/s（1.82×）**，
10.50 → 5.78 ms；含 reorder 6.25 ms，即 **1.68×、省 4.25 ms**。
`3→16 640²` 首层按设计不命中（`ic=3` 补到 16 是 5.3× 空转），照旧走 `PackedConv`。

40²/20² 的层只有 174（不是 220），原因是任务数掉到 `pairs*oh = 4×20 = 80`：单行 20 个像素
只有 2 个像素组，`Pixels=8` 的宽度吃不满。仍高于 170。

### 23.3 reorder：3.72 ms → 0.19 ms（第一版写错了，值得记）

第一版 `PackInput` 是「lane 外层、像素内层」的朴素写法，320² 单层要 **3.72 ms**（比 conv 本身还贵）。
根因是**写放大**：每个像素的 64 字节块被 16 个 lane 各写 4 字节，一条 cache line 被扫 16 遍
（RFO × 16）。改成 **8×8 单精度转置内核**（`Avx.UnpackLow/High` + `Shuffle` + `Permute2x128`，
16 条指令搬 8 通道 × 8 像素）后，落盘是连续 32 字节半行，两条 store 填满一条 cache line：

| 形状 | 旧 reorder | 新 reorder |
|---|---:|---:|
| 16→32 320² | 3.72 | **0.19** |
| 64→64 160² | 2.23 | **0.12** |
| 32→64 160² | 0.85 | **0.08** |
| 128→128 80² | 0.77 | **0.03** |

打包与解包是同一个动作的两种行进方向，所以共用 `Transpose8(src, srcStride, dst, dstStride)`：
打包时「行 = 通道、列 = 像素」，解包时反过来。并行粒度：打包按「通道块 × 32 行带」，
解包按「(oc 块, lane 组, 输出行)」；halo 只在 `band == 0` 清一次。

### 23.4 端到端（`ab`，同进程交替，4 线程）

```
yolo11n   A(nchwc=0) min 36.4   B(nchwc=1) min 34.5   A/B 中位 1.055（1.047~1.057）  → B 快 5.5%
yolov8n   A         min 42.5   B          min 40.8   A/B 中位 1.039（1.037~1.042）  → B 快 3.9%
```

等价性：yolo11n `out0 [1,84,8400]` `max|d|=8.7e-3`、`mean=7.7e-6`、`p99=2.4e-4`
（最大差出现在坐标值 ~385 的框上，相对误差 **2.2e-5**，是累加顺序差；yolov8n `max|d|=2.4e-3`）。
每层的 `max|d|` 已在 23.2 表里对 im2col 参照逐个验过，≤5e-5。

### 23.5 为什么只有 −5.5%，不是 −8.4%

`§22.4` 的 what-if 按「s2 8.15 ms → 170 GMAC/s」算得 −3.03 ms（−8.4%）。实测只拿到 −1.9 ms，
差额全部在 **reorder 那一栏**：

- `§22.1` 的 0.336 ms（0.9%）的前提是**图级布局传播** —— 整条 conv 链保持 c 布局，只在 5 个
  `Reshape` 边界转换。**阶段 1 没有做传播**，是「每层进、每层出」，reorder 次数从 5 变成 14。
- 微基准里这 14 次合计 0.47 ms（缓冲滚烫、顺序访问）；但模型里它们紧跟在 conv 之后，
  320² 那层的 blocked 缓冲是 6.6 MB + 3.3 MB，把 L3 顶掉，折算约 **1.3~1.5 ms**。

即内核这块是按预期拿到的（1.82×），端到端缺口来自「每层都付 reorder」这个阶段 1 的临时取舍，
**布局传播能补回来**，而传播所需的三件事（`Fit` 带 layout、`Concat`/`Split` 的 c 版、
`GetLayouts` 时间表）正是 `§22.3` 已经确认的「加两个方法签名」。

### 23.6 又一个测量坑：进程里**第一个**配置永远不可信

这次不是分层 JIT（`DOTNET_TieredCompilation=0` / `TieredPGO=0` / `ReadyToRun=0` 全程开着）：

| 同一形状 `16→32 320²` | im2col | nchwc |
|---|---:|---:|
| 排第一 | 8.19 ms（14.4 GMAC/s） | 3.06 ms（40.7） |
| 排第二（前面加任意形状） | 1.28 ms（92.5） | 0.54 ms（218.5） |

- 与形状无关：把 `16→32 320²` 跑两遍，**两遍都慢**；前面塞一个 `128→128 20²` 就两遍都正常。
- 与 `DOTNET_*` 无关：开着关着都是这个规律。
- 已在 `convbench` 里内置「首个配置读数丢弃」（自动插一个 `128→128 20²` 打头），
  现在不带任何仪式直接跑就是稳定值。

另记一条本机残留噪声：`32→64 160²` 有一次读出 109.8 GMAC/s（约 2×），重跑两次都是
212.9 / 219.6 —— 后面引用数字都取复测值；遇到「同一个形状和相邻层耗时一模一样」先怀疑噪声。

### 23.7 结论与下一步

| 判据 | 目标 | 实测 | 判定 |
|---|---|---|---|
| s2 7 层吞吐 | ≥170 GMAC/s（1.5×） | **204.1 GMAC/s（1.82×）** | ✅ |
| 微基准单层 | — | 174~220 GMAC/s（对照 88~138） | ✅ |
| 端到端 | −8.4%（推算） | **−5.5% / −3.9%** | 采纳，缺口可解释 |

阶段 1 到此收口，进入 **阶段 2**（按 `§22.5` 的执行顺序）：

1. **图级布局传播**（`Fit` 带 layout、`Concat`/`Split`/`MaxPool`/`Resize` 的 c 版、
   `GetLayouts` 输出时间表）—— 把 reorder 从 14 次压回 5 次，预期补回 **−3% 左右**；
2. 1×1 的 c 内核（45 层、1162.6 MMAC、10.19 ms）；
3. 3×3/s1 未走 Winograd 的层（13.44 ms 里的大头）—— 要不要一起动 Winograd 选路，单独决策。

### 23.8 复现

```powershell
$env:ONNXSHARP_THREADS='4'
$env:DOTNET_TieredCompilation='0'; $env:DOTNET_TieredPGO='0'; $env:DOTNET_ReadyToRun='0'
$exe='tools\OnnxSharp.Cli\bin\Release\net10.0\OnnxSharp.Cli.exe'

# 微基准（自带预热丢弃，默认形状表已含 8 个 s2）
& $exe convbench --iters 20

# 端到端 A/B（等价性检查一起打印）
& $exe ab F:/ai/models/yolo/yolo11n.onnx --input bench/inputs/i1x3x640x640.f32 `
    --a nchwc=0 --b nchwc=1 --iters 20 --rounds 6 --warmup 5
```

## 24. 2026-11-16（续）NCHWc 泛化到 **stride 1** —— 端到端 **−21% ~ −28%**

`§23` 的 s2 内核只覆盖 7 层，端到端 −5.5%。顺着「先量再说」的原则先给 `TryConv` 加了分阶段计时
（`ONNXSHARP_NCHWC_PROF=1`），量出来的是：

```
profile yolo11n（6 层命中）: 每次 pack 0.124 / kernel 0.713 / unpack 0.042 ms
                            → reorder 占 NCHWc 总时间的 18.9%，摊到整帧约 1.0 ms（2.9%）
```

即 `§23.7` 计划的「阶段 2 = 图级布局传播」在 yolo11n 上**上限只有 3%**，而且 `dump-graph` 显示
每个 s2 层的消费者是 **3×3/s1 的 Conv 和很后面的 Concat**，要吃掉这 3% 得先把 s1 卷积和 Concat
都改成 c 布局 —— 收益/成本明显不划算。于是掉头：**同一个内核泛化到 stride 1**，吃掉 3×3/s1
那批层（baseline 下它们跑 100~160 GMAC/s）。

### 24.1 改动：一个 `step` 参数

| 项 | stride 2 | stride 1 |
|---|---|---|
| blocked 输入 | `[C/16][h+1][w+1][16]`，**左上** 1 像素 halo | `[C/16][h+2][w+2][16]`，**四周各 1** 像素 halo |
| 取样 | 行 `2y+ky`、列 `2x+kx`，最大 `h`/`w` | 行 `y+ky`、列 `x+kx`，最大 `h+1`/`w+1` |
| 奇偶约束 | 必须 `h`、`w` 偶数（`2oh==h`、`2ow==w`） | 无（halo 独立成行/列） |
| 相邻像素间距 | `2*16*4 = 128` 字节 | `64` 字节 |

- `HaloPad(stride)` 一处给出 1 或 2；`PackInput` 里 stride 1 额外清 **最底一行** 与 **最右一列**
  （原始第 `h` 行 / 第 `w` 列，越界 padding），仍是每行 1 条 `zmm` store，成本可忽略。
- 内核里把硬编码的 `2` 全部换成 `step`：行指针 `(ib*ihp + row*step + ky)*iwp + g*Pixels*step`、
  像素间距 `pix = step*Block`。**指令数完全不变**（16 累加器/9 tap/8 像素），所以 stride 1
  的吞吐和 stride 2 一模一样（见下表 220 GMAC/s）。
- 新增 `Dispatch()`：`CanUse` + 「输出尺寸关系」二合一，`InferShapes`（申请 temp）与
  `Forward`（分派内核）**共用同一个判据**。上一版是两处手写条件，改 stride 支持时正好收掉这个隐患。

### 24.2 微基准（4 线程，`convbench --iters 20`）

| 形状（s1） | im2col | winograd | **nchwc** | +reorder | max&#124;d&#124; |
|---|---:|---:|---:|---:|---:|
| 64→64 160² | 146.7 | 102.2（9.24 ms） | **220.6（4.28 ms）** | 0.24 | 1.6e-5 |
| 128→128 80² | 158.9 | 162.5（5.81 ms） | **220.8（4.28 ms）** | 0.05 | 3.5e-5 |
| 256→256 40² | 107.9 | 183.7（5.14 ms） | **220.8（4.27 ms）** | 0.03 | 4.8e-5 |
| 256→256 20² | 106.1 | 131.5（1.79 ms） | **179.3（1.32 ms）** | 0.01 | 5.1e-5 |
| 64→64 101×97 | 135.6 | 145.3 | **110.7**（3.26 ms） | 0.13 | 1.7e-5 |
| 16→32 33×17 | 26.2 | 23.3 | **117.5** | 0.00 | 3.8e-6 |

对照 Winograd 那一列很有意思：160² 上 Winograd 只有 102 GMAC/s（变换开销摊不动），
真模型里 160²/80² 的 3×3/s1 层走的是 `PackedConv`（约 120~150 GMAC/s）—— 这就是 220 能换来
**1.5~1.8×** 的原因。

`101×97` 那行是唯一低于 im2col 的：`ow = 101` 不是 8 的倍数，12 个像素组只盖住 96 个像素，
剩下 5 个走逐像素尾路径（每行多付 `5/101 ≈ 5%` 的像素，但尾路径没有像素间复用，实测整体 2× 慢）。
真实模型里几乎见不到这种宽度，暂不为它让步；`33×17` 这种小数反而更快（im2col 那 0.10 ms
里全是填充开销）。

**s2 表（`§23.2`）未变**：208~220 GMAC/s（`128→128 80²` 那次读到 208.6，复测 220.0，属本机噪声）。

### 24.3 端到端（`ab`，同进程交替，4 线程）

| 模型 | A（nchwc=0） | B（nchwc=1） | 配对中位 A/B | 判定 |
|---|---:|---:|---:|---|
| yolo11n | 38.6 ms | **31.8 ms** | 1.212 | B 快 **21.2%** |
| yolov8n | 45.9 | **36.6** | 1.251 | B 快 **25.1%** |
| yolov11s | — | — | 1.236 | −23.6% |
| yolov8s | — | — | 1.270 | −27.0% |
| yolo11m_640 | — | — | 1.276 | −27.6% |
| yolov5su | — | — | 1.298 | −29.8% |
| yunet | — | — | — | 未命中（`oc=8/16` 全被 `oc%32` 挡住），**输出逐位相同** |

等价性：yolo11n `max|d|=5.31E-3`（坐标值 ~330，相对 1.6e-5）、`mean=6.1E-6`、`p99=1.8E-4`；
yolov8n `max|d|=2.87E-3`。`bench` 侧 `alloc 129 KB/iter`、`fit realloc 0 KB/iter` —— 没有额外分配。

命中面：`profile`（开 PROF）显示 **30 次调用/帧**，每次 pack 0.048 / kernel 0.804 / unpack 0.036 ms，
reorder 只占 NCHWc 时间的 **9.5%**（≈2.5 ms/帧）。

### 24.4 两条结论

1. **「先做布局传播」这个排序是错的**。量出来的 reorder 上限是 3%（s2 阶段）到 7.9%
   （现在 30 层），而 yolo 的 conv 链并不连续（s2 后面永远跟一个 s1 Conv + 一个远处的 Concat），
   要拿到这 3~8% 得把 Concat/Split/MaxPool 和邻居卷积全改成 c 布局。**先扩内核覆盖面才是对的**：
   同一份代码把 stride 1 收进来，端到端从 −5.5% 直接到 −21~−28%。
2. **`Dispatch()` 这类判据必须单点**。`InferShapes` 申请 temp 与 `Forward` 分派内核一旦各写一份，
   任何一侧扩了适用面，另一侧就静默失配（申请了不用 → 白占内存；该用没用 → 掉回慢路径且无人察觉）。

### 24.5 代价与残留

- **内存**：blocked 输入缓冲是 `[C/16][h+pad][w+pad][16]`，160² 的 s1 层单块 **16.8 MB**
  （320² 的 s2 层 6.6 MB）。temp 池按索引复用（`_temps[i]`，只增不减），所以峰值 ≈ 2 块
  （输入 + 输出），不是 30 层累加。但 1080p 级输入上这块会到 100 MB 量级，将来要么分块
  （按行带流式 pack+conv），要么给 `Tensor` 加 blocked 视图 —— 记在这里，别忘。
- **还差 `oc % 32 != 0` 的层**：yolo 里大量 block 是 8/16/32 通道，`OcGroup=2` 的 32 通道要求
  把 `oc=8/16` 的层挡在外面。放宽到 `OcGroup=1`（16 通道）能再吃一批，但累加器降到 8 个，
  吞吐会掉 —— 要用 `convbench` 单独量过再定。
- **1×1 层没动**：它无 halo，天然和 c 布局衔接，是下一块的候选；但它的输入大多来自 3×3
  或 Concat，仍要付 reorder。实际规模见 `§25`：**15.5 ms/帧（占 49%）**，`§24` 里写的
  「10.19 ms / 45 层」是逐层 min 手工归属的偏小值，已作废。

### 24.6 复现

```powershell
$env:ONNXSHARP_THREADS='4'; $env:DOTNET_TieredCompilation='0'; $env:DOTNET_TieredPGO='0'; $env:DOTNET_ReadyToRun='0'
$exe='tools\OnnxSharp.Cli\bin\Release\net10.0\OnnxSharp.Cli.exe'

& $exe convbench --iters 20                      # s1/s2 都在默认形状表里
& $exe ab F:/ai/models/yolo/yolo11n.onnx --input bench/inputs/i1x3x640x640.f32 `
    --a nchwc=0 --b nchwc=1 --iters 20 --rounds 6 --warmup 5

# 分阶段耗时（pack / kernel / unpack）与命中层数
$env:ONNXSHARP_NCHWC_PROF='1'
& $exe profile F:/ai/models/yolo/yolo11n.onnx --input bench/inputs/i1x3x640x640.f32 --iters 10 --top 3
```

## 25. 2026-11-16（更晚）1×1 的账：**15.5 ms/帧，占 49%** —— 先记数，不动手

3×3 收完之后（`§24`，31.8 ms/帧），剩下的钱在谁身上？新加一个仪表 `ONNXSHARP_CONV_PROF=1`
（`ConvProf`：`Im2Col` 里 pointwise 整段 / 其中的 `GemmKernel.Multiply` / 卷积后融合激活，
外加逐形状聚合的「每帧贡献」），`profile` 末尾打印。

### 25.1 帧预算（yolo11n，4 线程，`profile --iters 12`）

| 项 | ms/帧 | 占比 |
|---|---:|---:|
| **1×1（pointwise）整段** | **15.5** | 49% |
| 3×3（NCHWc + 少量 Winograd） | ~10.5 | 33% |
| 其中 GEMM | 15.3（占 pointwise 的 99%） | — |
| 其中非 GEMM（Span 切片 + 派发） | **0.13** | 0.4% |
| 卷积后融合的 SiLU | 3.7 | 12% |
| Concat / MaxPool / Split / 其余 | ~6 | 19% |

**先把 `§24.5` 里「1×1 = 10.19 ms / 45 层」这个数改掉**：那是按 `profile --top` 逐层 min
手工归属出来的，偏小；`ConvProf` 直接累计整条 pointwise 路径是 15.5 ms。以后引用 15.5。

结论一句话：**包装开销不是问题（0.13 ms），GEMM 本身才是全部**。

### 25.2 GEMM 本身不差

按「每帧贡献」排前面的大形状（每帧次数 · 单次 ms · GMAC/s）：

| m×k×n | 次数 | 单次 | GMAC/s |
|---|---:|---:|---:|
| 128×192×1600 | 5 | 0.24 | 160.7 |
| 256×384×400 | 4 | 0.25 | 156.6 |
| 64×256×6400 | 1 | 0.75 | 140.0 |
| 256×256×400 | 4 | 0.17 | 153.0 |
| 80×80×6400 | 2 | 0.30 | 136.6 |
| 64×128×400 | 5 | 0.08 | **39.9** |

前三档 136~161 GMAC/s ≈ 213 GMAC/s 峰值的 65~77% —— 这个量级已经没什么可榨的。
`64×128×400` 掉到 39.9 是因为 `m·n·k = 3.3 M` 低于 `GemmKernel.ParallelThreshold = 4.19 M`，
走串行。**把阈值降到 64K 的 A/B 没测出结论**：当时 CodeBuddy 进程占了 30% CPU，
`ab` 的噪声地板涨到 48.6%，判定「噪声内」。等安静窗口重测（命令在 25.4）。

就地读数可信度交叉验证：`64×128×400` 就地 0.081 ms vs `gemmbench` 独立 0.081 ms —— 完全一致。

### 25.3 两个未决异常（别当成结论）

1. `32×32×25600` 就地 **2.40 ms**，`gemmbench` 独立只要 **0.309 ms**（7.8×）。但同一层在
   `profile --top` 里只有 0.40 ms —— 两个仪表互相矛盾，说明聚合键可能撞车（同形状多次不同层）
   或某次迭代被抢占。**未定论，先不动**。
2. SiLU 3.7 ms（45 层 × ~0.08 ms）已经是 28 GB/s 的读+写，贴着内存带宽。把 SiLU 熔进 GEMM
   的 C 写回能省掉整整一趟 3.2 MB 的读+写（约 **2 ms = 6%**），但要改 `GemmKernel` 的输出路径。
   **已决定放弃** —— 理由见 §25.4（ORT 自己也不这么做）。

### 25.4 对照 ORT 源码后的决定：**放弃 SiLU 熔进 store，方向改为对齐 MLAS 的微内核**

读的是 ORT `main` 的 `mlas/lib/convolve.cpp`、`mlas/lib/activate.cpp`、`mlas/lib/sgemm.cpp`：

| 环节 | ORT 的做法（源码依据） | 我们 | 差异 |
|---|---|---|---|
| 1×1 conv | `MlasConvPrepare` 里 `AllStridesAreOne && AllPaddingIsZero && K == InputChannels` → `MlasConvAlgorithmGemmDirect`，`WorkingBufferSize` 保持 0，直接 `MlasGemm(oc × pixels × ic)` | `ConvKernels.Pointwise` → `GemmKernel.Multiply`，零 temp | **无** |
| 激活 | GEMM 跑完后单独一趟 `MlasActivation(Activation, Output, Bias, oc, pixels, ldc)`，就地、bias 也在这一趟 | 同构（bias 进累加器初值，激活另一趟） | 仅 bias 归属不同 |
| 融合激活清单 | `Identity / Relu / LeakyRelu / Tanh / Logistic / Clip / HardSigmoid / HardSwish` ——**没有 SiLU** | 我们自己实现了 SiLU 融合 | 我们更强 |
| 微内核 | `MLAS_SGEMM_TRANSA_ROWS = 12`（MR=12）；B 面板「16 columns at a time」、不足零填充（NR 基本块 16）；每线程栈上 `PanelA[12×STRIDEK]` + `PanelB[STRIDEN×STRIDEK]`，K 方向按 `STRIDEK` 滚动 | `Multiply12Rows`（MR=12）+ `SelectColumnBlock`=16 + 每线程打包面板 | **形状参数一致**，差在 asm 软件流水与 K 分块滚动 |
| 并行切分 | `ExpandThenGemmSegmented`：`Complexity = oc × OutputSize × K` 对比 `MLAS_SGEMM_THREAD_COMPLEXITY`，再按 `MLAS_SGEMM_STRIDEN_THREAD_ALIGN` 对齐算 `StrideN` 切 N 段 | `colBlock`=16（天然对齐）+ `ParallelThreshold` 一个阈值 | 思路一致，我们缺「按段对齐切 N」的细粒度 |

结论（改掉 §25.3 第 2 条的方向）：

1. **放弃把 SiLU 熔进 GEMM 的 C 写回。** ORT 也不做：它把激活（连 bias 一起）放在 GEMM 之后一整趟
   `MlasActivation`，微内核里没有任何激活；而且 MLAS 的融合激活枚举里压根没有 SiLU。为了约 2 ms
   去动全库最精细的输出路径，没有先例可抄，风险与收益不成比例。
2. **(A) 1×1 走 blocked 也降级。** ORT 对 1×1 的答案就是「直接 GEMM、零工作缓冲」，与我们相同；
   我们的 GEMM 已到 136~161 GMAC/s（峰值的 65~77%）。要做它等于赌一个 ORT 都没试的方向。
3. **真正该抄的是微内核深度，不是架构。** 结构上我们已经同构（1×1 走 GEMM、激活独立一趟、
   MR=12/nb=16、按 N 切段并行）。ORT 快我们 15~30%（`bench.md` 前文 min 口径）的差额，落在
   asm 级的软件流水（k 展开、提前 load）和 **K 分块滚动**（面板留在 L1）这两处，而我们现在是
   一次把整条 k 打包、整条 k 扫完。
4. 未查到的一条：`MLAS_SGEMM_STRIDEK` 的具体数值不在 `sgemm.cpp` 里（只在宏里引用），所以
   KC 的取值不能照抄，得自己在 `gemmbench` 上扫。

### 25.5 复核失败：三个仪表互相对不上（**§25.1 的表先别引用**）

原打算照 ORT 的微内核往下走，先确认 1×1 的绝对产能，结果发现同一件事三个仪表给了三个数：

| 对象 | 仪表 | 读数 | 折算 |
|---|---|---|---|
| 1×1 `192→128 @40²`（= m128 k192 n1600） | 模型内 `ConvProf`（min） | 0.25 ms | 160 GMAC/s |
| 同上 | `gemmbench --shape 128,192,1600` | 1.39 ms | 28 GMAC/s |
| 同上 | `convbench --shape 192,128,40,40,1,1,0` | 1.94 ms | 20 GMAC/s |
| 模型内单层 `/model.23/cv2.0/...Conv`（m256 k256 n400） | `profile --top`（min） | 2.83 ms | 9 GMAC/s |
| 同一层 | `ConvProf`（min） | 0.18 ms | 149 GMAC/s |

同时这台机器当时是**满负荷**的：同一次 `profile` 报 `total 193.3 ms/iter (mean)` / `38.8 ms/iter (sum of min)`，
mean 是 min 的 5 倍。`gemmbench` 单线程 6.23 ms、四线程 1.39 ms（4.5× 线性）也说明它测的是真算力而不是噪声。

所以现在的状态是：**口径未定，§25.1 那张「1×1 = 15.5 ms / 49%」的表不可引用**，等安静窗口用
「单层 in-model 计时 + gemmbench 同形状」两两对齐后再下结论。在此之前不基于它立项（尤其是 GEMM 微内核）。

### 25.6 复现

```powershell
$env:ONNXSHARP_CONV_PROF='1'      # 打开 pointwise / 激活的分阶段计时
& $exe profile F:/ai/models/yolo/yolo11n.onnx --input bench/inputs/i1x3x640x640.f32 --iters 12 --top 2

# 待安静窗口重测：小形状是否该并行（阈值 4.19 M → 64 K）
& $exe ab F:/ai/models/yolo/yolo11n.onnx --input bench/inputs/i1x3x640x640.f32 `
    --a gemm_threshold=4194304 --b gemm_threshold=65536 --iters 25 --rounds 8 --warmup 6

# 交叉验证某一个形状的就地读数（值应相等；不等说明就地计时被别的东西污染）
& $exe gemmbench --iters 2000 --shape 64,128,400
```

## 26. 2026-11-16（动手了）K 分块照 MLAS 落地 —— **实测负收益，默认关闭**

按 `§25.4` 的决定动手实现 MLAS `MlasSgemmOperation` 的循环形状：

```
for N 段:  for K 块:  重打 PanelB(CountK) + kernel(CountK, 累加)
```

落地范围（`GemmKernel`）：`MultiplyColumnRange` / `MultiplyRowBlock` / `MultiplyPacked.Block`
三处驱动循环都加 `k0` 分块；`Multiply12Rows/6Rows/4Rows/Row` 加 `accumulate` 形参（首块写、
后续 `+=`，即 MLAS 的 `ZeroMode`）；`SelectKBlock` 按 `16 KB / (width·4)` 取块高（≈256）。
开关 `ONNXSHARP_GEMM_KC`（0 = 关）。

### 26.1 A/B：真实形状集上全面变慢 12~22%

4 线程，`gemmbench --iters 200`，同一进程条件、同一台机器连续跑：

| m×k×n | KC=0（整条 k 打包） | KC=256（MLAS 式分块） | 差 |
|---|---:|---:|---:|
| 128×192×1600 | 0.212 ms / **371 GFLOPS** | 0.243 / 324 | −12.7% |
| 256×384×400 | 0.215 / **365** | 0.255 / 308 | −15.6% |
| 64×256×6400 | 0.594 / **353** | 0.667 / 314 | −11.0% |
| 1024×768×1536 | 6.686 / **361** | 8.184 / 295 | −18.3% |
| 1024×1280×1280 | 8.966 / **374** | 11.528 / 291 | −22.2% |

原因：我们的面板本来就落在 L2（k=8192、nb=16 → 512 KB/列块，但每块被 11 个行组复用），
L2 带宽足够；而分块每块的**C 读改写**（24 load + 24 add + 24 store per 16 列）+ 内核重启
（bias/broadcast 重建）是实打实的额外开销。MLAS 分块是为了用**定长栈缓冲**避免分配，
不是为带宽 —— 我们本来就从 `ArrayPool` 租缓冲，前提不成立。

**结论：P11 不立项。** 开关默认 0（语义与改前完全一致，模型 `bench` 中位 29.9 ms / min 29.4 ms，
与 `§24` 的 30.1 / 31.7 同档），代码留着只为可复现这次 A/B。

### 26.2 顺带把 §25.5 的仪表矛盾结掉了一半

修完 K 分块后重跑，`gemmbench` 对 `128×192×1600` 给 **0.212 ms / 371 GFLOPS**，模型内
`ConvProf` 给 **0.24 ms**（含派发），两者相差 13% —— **口径自洽**。

也就是说 `§25.5` 里 `gemmbench 1.39 ms / 28 GMAC/s`、`convbench 1.94 ms` 那几行是**坏读数**
（当时机器在 30% 负载下 + `--iters` 与热机状态不同），不是内核问题。据此修正两条：

1. **1×1 的 GEMM 已在 4 线程峰值的 87~90%**（371~374 GFLOPS vs 427 GFLOPS），没有可榨空间。
2. `§25.1` 的「1×1 = 15.5 ms/帧」仍不可引用：按 185 GMAC/s 与 yolo11n 约 0.9 GMAC 的 1×1
   总量反推只有 **~5 ms/帧**。`ConvProf` 的「每帧贡献」列对迭代次数归一化错了（累计值除以的
   帧数比实际迭代数少），要修。**在修好之前，帧预算一律以 `profile --top` 的 sum-of-min 为准。**

### 26.3 复现

```powershell
$env:ONNXSHARP_THREADS='4'
# 负收益复现（先默认关，再打开 256 对比；两条命令必须连续跑）
& $exe gemmbench --iters 200 --shape 128,192,1600 --shape 256,384,400 --shape 1024,1280,1280
$env:ONNXSHARP_GEMM_KC='256'
& $exe gemmbench --iters 200 --shape 128,192,1600 --shape 256,384,400 --shape 1024,1280,1280
```

## 27. 2026-11-16 修仪表：`ConvProf` 的三个 bug + 修正后的帧预算

`§25.1` 的帧预算不可引用（`§25.5`），但原因不是内核，是仪表。查出来三处：

| # | bug | 后果 |
|---|---|---|
| 1 | `ProfileAll` 先跑 `warmup`(5) 帧，`ConvProf` 累计**不清零**，而 CLI 的每帧除数只有 `iters`(12) | 每帧次数虚高 `(5+12)/12 = 1.42×` |
| 2 | 表头「pointwise 整段」打印的是**全进程累计值**，从未除帧数（那次 217.34 ms 是 17 帧之和） | 直接读成「每帧 217 ms」 |
| 3 | `Capacity = 4096`，溢出**静默丢弃** | `--iters` 大时次数偏低，且看不出来 |

修法（`Core/Net.cs` + `Kernels/ConvKernels.cs` + `tools/.../Program.cs`）：

- `ProfileAll(..., Action? onMeasurementStart)`：预热跑完后回调 `ConvProf.Reset`，
  保证**分子帧数 = 分母帧数**；
- `Take(frames)` 内部除帧数，表头改成「每帧整段 … [12 帧实测]」；
- `Capacity` 4096 → 16384，新增 `dropped` 计数，非 0 时打印警告。

### 27.1 yolo11n 帧预算（4 线程，`profile --iters 12`，安静窗口）

| 项 | ms/帧 | share |
|---|---:|---:|
| 1×1 GEMM（pointwise 整段，含派发） | **9.57** | 33% |
| 1×1 融合激活（SiLU，在 pointwise 内部） | 1.35 | 5% |
| 其余卷积（全是 3×3，im2col）= 24.2 − 9.57 − 1.35 | **13.3** | 46% |
| 非卷积（Concat 1.9 + MaxPool 0.9 + Split 0.6 + Resize 0.3 + …） | 4.7 | 16% |
| 合计 sum-of-min | 28.9 | 100% |
| `bench` 中位（同机） | 29.9 | 差 3.5%，两个口径对上 |

对照 `§25.1`：1×1 从「15.5 ms / 49%」降到 **10.9 ms / 38%**（原表既混了 5 帧预热，又没除帧数）。
**这才是修正后的账：1×1 只占三分之一，「剩下 20 多 ms 说不清」的部分有 13.3 ms 是 3×3。**

### 27.2 三个口径互相咬合（`§25.5` 的矛盾结案）

1. 单形状 `128×192×1600`：`ConvProf` 表内 **0.25 ms** ↔ `gemmbench` **0.212 ms**（86%，差值就是派发与 tensor 切片的固定开销）↔ 之前那个 1.39 ms / 28 GMAC/s 确认是**坏读数**。
2. `pointwise 9.57` vs 其中 `GEMM 9.56` → 非 GEMM 开销 0.01 ms，**1×1 的账全在 GEMM 里**，没有派发浪费。
3. 逐形状表「每帧合计」top-10 相加 ≈ 5.9 ms，加长尾 = 9.57 ms，闭合。

### 27.3 下一步该看哪

GEMM 已到 4 线程峰值 87~90%（`§26`），不追。当时的判断是「第一块肉是 3×3 的 im2col（13.3 ms / 46%），
而它没有任何分阶段计时」——**这条已被 `§29` 否证**：3×3 实际 14.0 ms，75% 走 NCHWc 且已到峰值的 81%，
全部打包加起来只有 1.7 ms。真正没跑满的是 **1×1 的窄 m（136 GMAC/s vs 150~160）**和**首层 3→16（42 GMAC/s）**。

### 27.4 复现

```powershell
$env:ONNXSHARP_THREADS='4'
$env:ONNXSHARP_CONV_PROF='1'
& $exe profile F:/ai/models/yolo/yolo11n.onnx --input bench/inputs/i1x3x640x640.f32 --iters 12 --top 14
# 对齐校验：表内的单形状 ms 应与 gemmbench 同形状相差 <20%
& $exe gemmbench --iters 200 --shape 128,192,1600 --shape 256,384,400 --shape 64,256,6400
```


## 28. 2026-11-16 四引擎同会话复测（4 线程口径）

`run_all.py` 在本机需要显式给解释器（PATH 上的 `python` 是失效 shim，exit 49）：

```powershell
& 'E:\misedir\installs\python\3.11.17\python.exe' bench/run_all.py --iters 30 --warmup 10 --threads 4 --rounds 2
```

| 模型 | onnxsharp | onnxruntime-cpu | opencv-dnn | simdyolo |
|---|---:|---:|---:|---:|
| yolo11n 640² | 29.6 | **23.5** | 46.4 | 29.9 |
| yolov8n 640² | 34.4 | **27.8** | 48.5 | 29.4 |
| yolov8s 640² | 87.4 | **76.4** | 136.5 | 76.2 |
| yolo11s 640² | 77.2 | **63.5** | 129.4 | 77.5 |
| rfdetr-small 512² | 200.0 | **187.2** | 319.1 | N/A |

（median；min 与 median 差 <1%，本机这轮很稳。）

**读法三条**：

1. **4 线程口径下 ORT 全面领先我们 7~26%**（yolo11n 最差 1.26×，rfdetr-small 最好 1.07×）。
   这与 `§11-11 深夜`（16 线程）里「我们 27.4 vs ORT 23.4」是同一结论 —— ORT 的绝对值稳定在 23.5 左右，
   我们 4 线程 29.6、16 线程 27.4，**换线程数救不了这 6 ms**。
2. **我们与 SimdYolo 基本打平**（yolo11n 29.6 vs 29.9、yolo11s 77.2 vs 77.5），
   说明「落后 ORT 25%」不是我们独有的实现粗糙，而是 ORT 的 conv 路径确实更强。
3. **OpenCV 在 4 线程下崩到 46.4**（16 线程历史口径是 26.5），两套数据不可混比 —— 它的并行效率随线程数变化极大。

### 28.1 差距定位：只能落在 3×3（**该判断已被 `§29` 否证**）

我们自己两个仪表是闭合的（`§27`）：`bench min 29.8` ↔ `profile sum-of-min 29.4`，其中
**Conv 24.2 ms（83.9%）**、非卷积 4.7 ms。所以 6 ms 的差距必须从 Conv 里找：

| Conv 子项 | ms/帧 | 现状 |
|---|---:|---|
| 1×1 GEMM + 融合 SiLU | 10.9 | 已达 4 线程峰值 87~90%（`§26`），封板 |
| **3×3** | **13.3** | 当时**无分阶段计时**，被当成「唯一能解释 6 ms 的候选」 |

反证：把 1×1 全砍掉（不可能）也只有 10.9 ms，而 ORT 总耗时 23.5 —— 若我们的 3×3 与 ORT 同速，
两端相加必然 < 23.5，所以 **3×3 必然比 ORT 慢**。

> ⚠ **这段推理的结论错了**。`§29` 把四条 conv 路径拆开后：3×3 的 14.0 ms 里 10.5 ms 走的是
> **NCHWc**（内核 9.44 ms，reorder 只占 10%），整体 150 GMAC/s ≈ 峰值的 81%，**不是 im2col 打包浪费**。
> 差距的真实位置见 `§29.3`。

### 28.2 警告：ORT 的 `--profile` 口径不能跨引擎比

`bench/ort_inspect.py --profile` 在 yolo11n 上给出逐算子合计 **37.25 ms**，而 `run_all` 同机 bench 是 **23.5 ms**
（1.58×）—— ORT 自己的 profiling 开销就是这么重。只能拿它的**相对占比**（Conv 23.85 中 64%、QuickGelu 4.09、
Softmax 2.55、Concat 2.38、ReorderInput/Output 2.4），**不能拿去和我们的 sum-of-min 相加对比**。

## 29. 2026-11-16 把 4 条 conv 路径全拆开：**P12 的前提被否证**

`§27` 之后仍有一块说不清：`ConvProf` 只盖 1×1，3×3 是「24.0 − 10.9」减出来的。这轮把
`PackedConv` / `Im2Col` / `Winograd` 补上分段计时，`NchwcConv` 用它早就有的 `ONNXSHARP_NCHWC_PROF`，
四条路径合起来第一次和 op 级 `Conv` 总量对上。

### 29.1 全路径预算（yolo11n，4 线程，`--iters 12`）

| 路径 | 墙钟 ms/帧 | 内部拆分 | 覆盖层数 |
|---|---:|---|---:|
| 1×1 pointwise | **8.52** + 融合 SiLU **1.31** | GEMM 8.51 / 非 GEMM 0.01 | 87 次 |
| **NCHWc** | **10.46** | 打包输入 0.69 · **内核 9.44** · 解包 0.34（reorder 仅 9.8%） | 30 次 |
| Im2Col（兜底） | 2.02 | 打包 35% · GEMM 65% | 32 次 |
| PackedConv（面板式） | 1.53 | 打包 64% · 微内核 36% | 6400 面板 |
| Winograd | **0.00** | — | **0 次（从未命中）** |
| 合计 | **23.84** | op 级 `Conv` 报 24.0，差 0.7% ✓ | 非卷积 4.8 |

两个读数要注意：1×1 是**单任务墙钟**；Im2Col/Packed 是「逐任务累加」，多线程下 ≈ 墙钟 × 占用核数
（所以它们那两行的累计值 7.36 / 3.78 大于墙钟 2.02 / 1.53）。**打包占总量的真实比例按墙钟摊**：
Im2Col ≈ 0.71 ms、Packed ≈ 0.97 ms，**合计 1.7 ms，占整帧 5.7%**。

### 29.2 结论：3×3 不是 im2col 打包浪费

| 判断（`§28.1` / `planv2` P12 依据） | 实测 |
|---|---|
| 3×3 = 13.3 ms「没有分段计时，疑似浪费」 | 3×3 = **14.0 ms**，其中 75% 走 NCHWc |
| im2col 的 9× 数据膨胀是大头 | 全部打包 = **1.7 ms**（5.7% 帧） |
| 3×3 只跑到 ~30 GMAC/s | **150 GMAC/s**，是 4 线程 GEMM 峰值（185）的 **81%** |
| Winograd 还能救 3×3 | 3×3 s1 空间都 > 64×64 上限，**0 次命中**；且文档记录的旧对比里它本就打不过面板式 |

也就是说：**direct conv / NCHWc 化的收益上限约 1.7 ms，不值得立项**。P12 到此结案。

### 29.3 那么 6 ms 在哪：1×1 与「怪形状层」

按 `bench/conv_family.py` 的 MAC 口径（总数 3.23 GMAC）：

| 家族 | MAC | 实测 | 有效吞吐 | 占帧 |
|---|---:|---:|---:|---:|
| 1×1 | 1162.6 MMAC | 9.83 ms（含激活） | **136 GMAC/s（74%）** | 34% |
| 3×3 s1 | 1194.4 | 8.81 | 136 | 30% |
| 3×3 s2 | 870.0 | 5.94 | 147 | 20% |
| 3×3 s2（首层 3→16） | 44.2 | 1.04 | **42.5（异常）** | 3.6% |

- **1×1 是唯一既大又没跑满的**：8.51 ms 的 GEMM 换 136 GMAC/s，而 `gemmbench` 对模型里同一批形状
  能到 150~160（`§26`）。差的 15% 全在**窄 m**（16/32/64 行，12→6→4 级联浪费）与 epilogue 上，
  把 1×1 推到 185 GMAC/s 可省 **2.2 ms**。
- 首层 `3→16`（`channels=3` 不满足 16 对齐）掉到 42.5 GMAC/s，独占 1.04 ms —— 单项 3.6%，
  靠 `channels<16` 的专用小内核可省 ~0.8 ms。
- 这两项加起来 ~3 ms，仍不足以抹平 6 ms；剩下的解释只能是 **ORT 用了 Winograd 少做 MAC**
  （3×3 s1 的 1.19 GMAC 用 F(2,3) 只有 4/9，等效省 3.6 ms）—— 我们这条路径现有实现太慢（标量变换），
  要做就得重写变换，属于另一个量级的活。

### 29.4 复现

```powershell
$env:ONNXSHARP_THREADS='4'; $env:ONNXSHARP_CONV_PROF='1'; $env:ONNXSHARP_NCHWC_PROF='1'
& $exe profile F:/ai/models/yolo/yolo11n.onnx --input bench/inputs/i1x3x640x640.f32 --iters 12 --top 1
# 只想要家族口径（MAC 与 GMAC/s）：
& $exe profile ... --top 300 > bench/results/yolo11n_prof4.log
& 'E:\misedir\installs\python\3.11.17\python.exe' bench/conv_family.py F:/ai/models/yolo/yolo11n.onnx bench/results/yolo11n_prof4.log
```

零仪表开销校验：`ONNXSHARP_CONV_PROF=0` 时 `bench` min 30.3 / median 30.8（`§28` 同机 29.6 / 30.2），
差在噪声内；测试 116/118（2 skipped）。

> ⚠ 本节数字都是**对齐修复之前**的口径，绝对值和 `§30` 不可直接混着比（机器也有漂移）。
> 结构性的结论（3×3 走 NCHWc、打包只占 5.7%）不受影响。

## 30. 2026-11-16 **张量缓冲 64 字节对齐**：同形状从 124 → 138 GMAC/s（−1.2 ms，−4.4%）

P13 的第一步本来只是「用 `gemmbench` 扫模型里的真实窄 m 形状」，结果扫出了一个更底层的问题。

### 30.1 线索：同一批形状，图里比微基准慢 12~15%

| m×k×n（1×1 层） | 图内 | gemmbench | 比 |
|---|---:|---:|---:|
| 128×192×1600 ×4 | 159.2 GMAC/s | 185.4 | 0.86 |
| 64×48×25600 | 90.5 | 119.4 | 0.76 |
| 64×256×6400 | 132.7 | 173.4 | 0.77 |
| 128×96×6400 | 149.1 | 176.6 | 0.84 |
| 64×64×6400 | 135.1 | 155.6 | 0.87 |
| 32×32×25600 | 75.8 | 80.8 | 0.94 |

差别只可能在**指针**：微基准的缓冲是 `new float[]` 的起点，图里是 `byte[]` 槽位的切片。
拿 `gemmbench --misalign 4`（把起点推 4 个 float = 16 字节）直接复现：**掉 10~14%**，与上面的比值同量级。

### 30.2 诊断：真实对齐率 A 18% / B 8% / C 0%

新增 `ONNXSHARP_GEMM_ALIGN=1`（`GemmKernel.SampleAlignment`，只在开关打开时多一次 `fixed`）：
统计每次 `Multiply` 的 A/B/C 指向是否 64 字节对齐。

```
gemm 对齐  : 1309 次调用 · 64B 对齐率 A 18% / B 8% / C 0%
```

原因很直白：`Tensor.Fit` 里 `new byte[need]` 只保证 8 字节对齐（数组数据从对象头后 16 字节开始），
于是 32 字节一加载的微内核**每行都跨缓存行**。

### 30.3 修法：多给一个粒度，把 `ByteOffset` 推到对齐位置

`Tensors/Tensor.cs`：

- `Alignment = 64`（`ONNXSHARP_ALIGN=1` 可关掉做 A/B；不是 `const` 只为同会话对照，代价仅在分配路径）。
- `Fit` 分配 `need + Alignment`，用 `GC.AllocateArray<byte>(..., pinned: true)`：**POH 上 GC 不搬**，
  所以「起点到 64 边界」的偏移一次算好终身有效。
- `ByteOffset = AlignedOffset(Buffer)`，`ByteLength` 不变 ⇒ `AsSpan<T>()` 自动吃到对齐。
- 新增 `Bytes()`（含 `ByteOffset` 的原始字节视图），把 5 处「假设偏移是 0」的裸写改成它：
  `OnnxFile` 的 raw / external / typed 三条解码路径、`SafeTensorFormat`、`ParaformerRecognizer` 的 3 处
  `Buffer.BlockCopy`、`Tensor.Clone` 的目标缓冲。
- **同一批假设还藏在测试里**（5 个用例直接 `CopyTo(tensor.Buffer, 0)`），已一并改成 `Bytes()` ——
  这也是为什么改动第一版跑出 5 个数值失败，`Alignment=1` 复跑即恢复，从而锁定原因。

### 30.4 A/B（同会话交替，min ms）

| 模型 | `ONNXSHARP_ALIGN=1` | `=64` | 变化 |
|---|---:|---:|---:|
| yolo11n（3 轮交替） | 29.9 / 29.2 / 29.9 | **28.4 / 28.5 / 28.6** | −1.2 ms（−4.4%） |
| yolov11s | 77.2 | 75.9 | −1.7% |
| rfdetr-small | 221.5 | 219.7 | −0.8% |

逐路径（同会话交替 `profile`，第二次=稳态）：

| 路径 | `ALIGN=1` | `ALIGN=64` |
|---|---:|---:|
| **1×1 GEMM** | 9.34 | **8.44**（−0.90，−10%） |
| 1×1 有效吞吐 | 124 GMAC/s | **138 GMAC/s** |
| Im2Col 墙钟 | 2.13 | 2.11 |
| PackedConv 墙钟 | 1.58 | 1.62 |
| sum-of-min | 29.7 | **28.5** |

Im2Col / PackedConv 不动是合理的：它们的 A 是权重、B 来自 `ArrayPool` 的 `float[]`（本来就对齐）。

四引擎（`run_all.py --names yolo11n --rounds 2`）：**onnxsharp 28.9 / ORT 23.5 / OpenCV 45.9 / SimdYolo 29.3**
—— 与 ORT 的差距从 1.26× 缩到 **1.23×**。

### 30.5 残留与风险

- **B 仍然只有 61% 对齐**：B 是输入激活，很多来自 `Concat` / `Split` / `Reshape` 的**视图**（`ByteOffset` 随父张量走）。
  要对齐它，要么让视图产出也带 64 对齐（改 `Concat` 的输出布局），要么在 GEMM 前复制 —— 都是下一档的活。
- POH 内存：`alloc` 仍是 116 KB/iter（缓冲被 `Fit` 复用，没变多）；但 pinned 落在 POH 且不可压缩，
  长时间跑多模型要继续盯 `FitReallocCount`。
- 方法学外推：任何「起点只保证 8 字节」的张量都会让 SIMD 掉档，不只是 GEMM。

## 31. 2026-11-16 sherpa / Paraformer（ASR）专项 —— 常量折叠后与 ORT 复测

### 31.1 为什么单独一节，而不是并进四引擎表

`bench/run_all.py` 的模型清单只有视觉/OCR 那 6 个，ASR 不在里面：它的输入是 wav（先过 fbank），
输出是 token 序列，端到端没有和 yolo 共通的「一次 forward」口径。这里的对照方式是**逐段对照**：

- 我们：`asr` 命令的内部三段计时（`feat` / `enc` / `dec`）+ `profile` 的纯 forward；
- ORT：`onnxruntime` 直接建 session，喂**同一批落盘的张量**（`.tmp/sherpa/*.f32`），只测 `sess.run`。

两边都是 4 线程（`ONNXSHARP_THREADS=4` / `intra_op=4, inter_op=1, ORT_ENABLE_ALL`），热态多轮取 min。

### 31.2 折叠前后（decoder，814 → 419 节点）

| 项 | 折叠前 | 折叠后 |
|---|---|---|
| 节点数 | 814 | **419**（`Shape` 96→0、`Gather` 96→0、`Unsqueeze` 102→0） |
| forward `huihui.wav`（16 tokens / 76 帧） | 12.62 min / 16.00 median | **10.69 / 14.08** |

（这一行是同进程交替 A/B、各 15 次取 min/median，可信度高于后面 §31.3 的跨进程快照。）

改动本身（rank-0 输入误判 + 长度输入按数值参与掩码）见 `plan.md §A11`；这里只记性能。
**收益不是线性的**：那 395 个节点逐算子自报合计才 ~0.2 ms，真实代价在每节点固定开销
（≈4.8 μs/节点 ≈ 1.9 ms），所以省的是 13~15%。

### 31.3 与 ORT 的分段对比（2026-11-16，交替轮次，min ms）

| 段 | 我们 | ORT | 比值 |
|---|---:|---:|---:|
| encoder（1×76×560，纯 forward，`profile`） | 30.2 / 30.2 / 32.4 | 23.9 / 24.0 / 23.9 | **1.27~1.35×** |
| decoder 单步（16 tokens / enc 76，纯 forward） | 13.2 / 10.3 / 10.7（median 15.0 / 13.6 / 12.9） | 3.56 / 3.62 / 3.55（median 3.59） | **2.9~3.7×** |
| 端到端 `huihui.wav` | 68~70 ms（feat 6~7 / enc 45~55 / dec 8~13） | enc+dec 27.5（无 ORT 版 CIF/argmax） | — |

- **抖动差异大**：ORT 的 min 与 median 几乎重合（<1%），我们的 min/median 差 20~30%。
  这台机器 32 核但 IDE 长期抢 CPU，C# 侧对干扰更敏感（GC / 线程唤醒）。**结论只看交替 A/B 的 −15%**，
  上表 2.9~3.7× 只是折叠后的一次快照。
- decoder 仍是最弱的一环（encoder 已到 1.3×）：419 节点里 `Transpose` 72 + `Concat` 48 + `Reshape` 48 尚未融合，
  每节点固定开销仍约 2 ms。
- `asr` 的 `dec` 段（含 CIF + 输入绑定 + argmax）比纯 forward 高 1~3 ms；`enc` 段比 `profile` 高 4~5 ms
  （多了 3 个输出拷贝 + 量化）。跨表比较时别把两套口径混用。

### 31.4 encoder 随序列长度（`sweep_T.py` 截断输入）

T=8~64 这四行是同日 A10 会话测的、本轮没重测；T=76 是本次交替复测的两个下限值（编码器代码未改动，
与 A10 的 1.28× 一致，可认为没漂）。

| T | 我们 | ORT | 比值 |
|---|---:|---:|---:|
| 8 | 21.0 | 6.1 | 3.44× |
| 32 | 18.2 | 12.8 | 1.42× |
| 48 | 22.1 | 16.9 | 1.31× |
| 64 | 28.1 | 20.8 | 1.35× |
| 76 | 30.2~32.4 | 23.9 | 1.27~1.35× |

短序列由固定开销主导（T=8 时 6.1 vs 21.0），T≥48 后稳定在 1.3× 左右 —— 与 DQM 的 F32 算力差一致。

### 31.5 两个测量坑

1. **CLI `profile` 加载期不做常量烘焙**。直接 `profile decoder.onnx` 跑的还是 814 节点那条老路径
   （逐算子表里 `Shape 96` 仍在就是证据），要看折叠后的数必须先把长度输入烘成 initializer：
   `.tmp/sherpa/bake_decoder.py` 生成 `decoder_baked.onnx`。
2. **ORT 的 `--profile` 口径不能跨引擎比**（同 §28.2），这里 ORT 侧一律是「纯 `sess.run` 计时、不挂 profiling」。

### 31.6 复现

```powershell
$env:ONNXSHARP_THREADS='4'; $exe='tools\OnnxSharp.Cli\bin\Release\net10.0\OnnxSharp.Cli.exe'
# PATH 上的 python 是失效 shim（exit 49 / 无输出），必须给全路径
$py='E:\misedir\installs\python\3.11.17\python.exe'

# 我们：encoder 纯 forward（T=76）
& $exe profile .tmp/sherpa/encoder.onnx `
  --input speech=.tmp/asr/e2e_feats.f32 --input speech_lengths=.tmp/asr/e2e_lengths.i32 `
  --shape speech=1,76,560 --shape speech_lengths=1 --iters 20 --top 0

# 我们：折叠后的 decoder（长度输入先烘成 initializer，与识别器加载期等价）
& $py .tmp/sherpa/bake_decoder.py
& $exe profile .tmp/sherpa/decoder_baked.onnx `
  --input 'onnx::Slice_6785=.tmp/sherpa/ok_memory.f32' `
  --input '/encoder/after_norm/Add_1_output_0_quantized=.tmp/sherpa/dec2_enc_u8.bin' `
  --input '/encoder/after_norm/Add_1_output_0_scale=.tmp/sherpa/dec2_enc_scale.f32' `
  --input '/encoder/after_norm/Add_1_output_0_zero_point=.tmp/sherpa/dec2_enc_zp.u8' `
  --shape 'onnx::Slice_6785=1,16,320' `
  --shape '/encoder/after_norm/Add_1_output_0_quantized=1,76,320' --iters 20 --top 0

# ORT 对照
& $py .tmp/sherpa/bench_ort.py 4 20            # encoder，打印 min/median/mean
& $py .tmp/sherpa/bench_ort_decoder.py         # decoder 单步

# 端到端（内部分段计时）
& $exe asr .tmp/asr/huihui.wav --models .tmp/sherpa --repeat 30
```

> `.tmp/sherpa/dec2_*.f32|bin` 这些探针是早期一次真实推理落盘的，混用过不同样本
> （`speech_lengths` 与 `enc_len` 不自洽），`bench_ort_decoder.py` 里已按「enc 行数 / `token_num`」自洽修正，
> 自己造输入时别踩回去。

## 32. 2026-11-16 decoder 的 3× 差距溯源：**不在图结构，在整数 GEMM**（图优化路线终止）

§31 之后的第一计划是「419 节点里 `Transpose` 72 + `Concat` 48 + `Reshape` 48 继续融合」。先量再动手，
结论是**这条路线无效**，同时把 3× 差距定死在一个不可消除的地方。

### 32.1 前提否证：Transpose/Concat/Reshape 合计 0.5%

折叠后（`decoder_baked.onnx`）的算子构成与耗时（`profile --iters 20`）：

| 类型 | 个数 | min ms | share |
|---|---:|---:|---:|
| DynamicQuantizeMatMul | 50 | 3.3 ~ 3.9 | 30~37% |
| MatMul | 25 | 1.1 ~ 2.3 | 10~21% |
| MatMulInteger | 12 | 2.1 | 19% |
| LayerNormalization | 51 | 0.7 ~ 1.0 | 7~9% |
| Add | 60 | 0.5 ~ 1.1 | 4~10% |
| Softmax / LogSoftmax | 12 / 1 | 0.1 / 0.1 | ~1% |
| Cast / Pad / Split / Conv / Relu | 12 each | 0.3 / 0.2 / 0.2 / 0.3 / 0.1 | ~8% |
| **Transpose** | **36** | **0.1** | **0.5%** |
| **Reshape** | **48** | **0.0** | **0.0%** |
| **Concat** | **0** | — | — |

- 折叠后 `Concat` 已经一个不剩（raw 图里那 60 个被折叠/融合掉了），`Shape`/`Gather`/`Unsqueeze` 也是 0。
- `Transpose` 36 个只剩 0.1 ms —— 它们全是 stride 视图（§15.2 的零拷贝 Reshape + 视图 Transpose），
  本来就不贵。**「再融合 170 个搬运节点」这个前提是错的**（那是 raw 图的数字，不是折叠后的）。

> 逐算子的绝对值不可信（同一份模型两次 profile，DQM 3.3 / 3.9、MatMul 1.1 / 2.3，同 §12.6），
> 但「GEMM 三类合计 60~70%、搬运类 ≤0.5%」这个结构性结论两次都成立。

### 32.2 我们的 GEMM 已经在 F32 峰值附近

同会话 `flopsbench --threads 4`：**单线程 108.5 GFLOPS，4 线程合计 434.1 GFLOPS**（IDE 在后台，真实峰值更高）。

decoder 每帧的 MAC 数（12 层 + output layer，按图里真实形状累加）：

| 段 | 形状 | MMAC/层 |
|---|---|---:|
| self-attn q | 16×320×320 | 1.6 |
| self-attn k_v | 16×320×1280 | 6.6 |
| self-attn out | 16×1280×320 | 6.6 |
| cross-attn q | 16×320×320 | 1.6 |
| cross-attn k_v | 76×320×1280 | 31.1 |
| cross-attn out | 16×1280×320 | 6.6 |
| ffn w_1 / w_2 | 16×320×1280 / 16×1280×320 | 13.1 |
| **合计** | 12 层 + output(16×320×8404) | **849 MMAC/帧 ≈ 1.70 GFLOP** |

7.7 ms 的 GEMM 段对应 **≈221 GFLOPS（51% 峰值）**；按 §32.4 的单形状口径，大形状已经到
`gemmbench` 的 86%（373/434），差距来自小 m/n 的固定开销与权重轮换（§32.5）。

### 32.3 ORT 的 3.57 ms 用 F32 解释不了

| 口径 | GEMM 时间 | 折算 |
|---|---:|---:|
| 我们 | 7.7 ms | 221 GFLOPS |
| ORT（总 3.57 减去 non-GEMM 估计 0.5） | ~3.1 ms | **~550 GFLOPS** |

~550 > 同会话 F32 峰值 434。也就是说：**ORT 这个 decoder 不是跑在 F32 域上的**。
一致的解释是它的 `DynamicQuantizeMatMul` 走 MLAS 的 **u8s8 整数内核**（`MlasGemmU8S8`），
在 Zen4 上可用 AVX512-VNNI 的 `vpdpbusd`，吞吐是 F32 的 2× —— 4 核理论 ~868 GFLOPS，与 550 吻合。

旁证（同一编码器、两个模型）：

| 模型 | 计算类型 | 我们/ORT |
|---|---|---:|
| encoder（F32） | F32 GEMM/Conv | **1.27~1.35×** |
| decoder（量化） | u8×s8 | **2.9~3.7×** |

同为 GEMM 密集，唯独量化模型差 2.4 倍 —— 差异只能来自整数内核。

### 32.4 为什么我们不能上整数内核

> **本节结论已被 §34 推翻**：在 `net11.0` 目标下 `AvxVnni.IsSupported == true`（本机 Zen4 的 AVX-VNNI
> 可用），256 位 `vpdpbusd` 已实装进 `Int8Gemm`，`76×320×1280` 从 314 → 733 GFLOPS，
> decoder 对 ORT 从 2.97× 收到 1.67×。下面这段只在 `net10.0`（无 `AvxVnni`）时成立，保留作对照。

`plan.md` 的「int8 GEMM 指令可行性」已定：本机 Zen4 **没有 VEX 编码的 AVX-VNNI**（`AvxVnni.IsSupported == false`），
而能用的 EVEX `vpdpbusd`（AVX512-VNNI）**.NET 没有暴露 intrinsic**；退而用 `Avx2` 的
`sadbw`/`maddubs` 也没有对应 API（`MultiplyAddAdjacent` 只覆盖 i16 的 `vpmaddwd`，吞吐与 FMA 相同，无收益）。
ORT 是 C++，直接吃 intrinsics 表。

本轮又把两条「整数也许更快」的路实测了一遍，都不通：

| 开关 | 走的路径 | `dec` 段（huihui，min） |
|---|---|---:|
| 默认（`Int8Kernel.Tier == Scalar` ⇒ F32 展开） | F32 packed GEMM | **8~11 ms** |
| `ONNXSHARP_INT8_F32GEMM=0` | 标量 int8 内核（正确性 oracle） | **352~356 ms（40×）** |
| `ONNXSHARP_INT8_TIER=avx512bw` | i16 展开 + `vpmaddwd` | 见 `plan.md §M10-A`：比 F32 慢 4× |

所以默认值是对的（没有 VNNI 就退 F32），两个开关都只该在对照实验里用。

⇒ **decoder 的 3× 在 .NET + 本机指令集下不可消除**。除非 P/Invoke 自写 native 内核（超出引擎范围），
或换到有 AVX-VNNI 的机器（Zen5 / Intel 12 代+）。

### 32.5 顺带否掉两条：MR 与「小 m 需要更窄行块」

`gemmbench`（packed 列，DQM 走的就是 packed 路径）在 decoder 的 5 个真实形状上扫 `ONNXSHARP_GEMM_MR`：

| 形状 | MR=12 | MR=6 | MR=4 |
|---|---:|---:|---:|
| 16×320×320 | **0.036** | 0.036 | 0.037 |
| 16×320×1280 | **0.038** | 0.037 | 0.037 |
| 76×320×1280 | **0.150** | 0.170 | 0.170 |
| 16×1280×320 | **0.037** | 0.037 | 0.040 |
| 76×320×320 | **0.038** | 0.043 | 0.043 |

- MR=12 已是最优或持平，**m=16 并不需要更窄的行块**（16 < MR 时浪费的那部分被「packed 面板 + 免去 padding」
  抵消了）。这条不用再试。
- `16×320×320` 只有 63 GFLOPS，与 MR 无关 ⇒ 瓶颈是**固定开销**（24 个/帧 × 0.037 ms ≈ 0.9 ms），
  不是内核形状。

### 32.6 残留：图内比微基准慢 25%（权重轮换冷）

| 形状 | 图内 DQM 的 gemm 段 | gemmbench packed | 比 |
|---|---:|---:|---:|
| 16×320×320 | 0.037 | 0.036 | 1.03 |
| 76×320×1280 | 0.187 | 0.150 | **1.25** |
| 76×320×320 | 0.048 | 0.038 | **1.26** |

微基准里同一份权重反复跑（L2 热），图里每层换权重（12 层 × 320×1280×4B ≈ 20 MB，L2 1 MB 装不下，
每层重读到 L3）。这是**内存带宽**问题，同 §30 的对齐问题不是一个来源，改法要动权重布局/分块预取，
收益上限 ~0.3 ms（3%），不划算。

### 32.7 结论

1. **decoder 图优化到此为止**：搬运类已 ≤0.5%，GEMM 已到 F32 峰值的 51%（大形状 86%），
   剩余差距是 int8 指令，.NET 拿不到。
2. 若要继续缩 decoder：只有「native int8 内核」一条路，需要评估是否值得为 ASR 引入 C 依赖。
3. 精力应该回到 **F32 模型**（encoder 1.3×、yolo 1.23×）—— 那些差距才是调度/内核可以追的。

### 32.8 复现

```powershell
$env:ONNXSHARP_THREADS='4'; $exe='tools\OnnxSharp.Cli\bin\Release\net10.0\OnnxSharp.Cli.exe'
$py='E:\misedir\installs\python\3.11.17\python.exe'

# 4 线程 F32 峰值（对照 3.57 ms 的可行性）
dotnet tools/OnnxSharp.Cli/bin/Release/net10.0/OnnxSharp.Cli.dll flopsbench --iters 100 --inner 500000 --threads 4

# 图内逐算子构成（注意绝对值不可信，看结构）
& $exe profile .tmp/sherpa/decoder_baked.onnx `
  --input 'onnx::Slice_6785=.tmp/sherpa/ok_memory.f32' `
  --input '/encoder/after_norm/Add_1_output_0_quantized=.tmp/sherpa/dec2_enc_u8.bin' `
  --input '/encoder/after_norm/Add_1_output_0_scale=.tmp/sherpa/dec2_enc_scale.f32' `
  --input '/encoder/after_norm/Add_1_output_0_zero_point=.tmp/sherpa/dec2_enc_zp.u8' `
  --shape 'onnx::Slice_6785=1,16,320' --shape '/encoder/after_norm/Add_1_output_0_quantized=1,76,320' `
  --iters 20 --top 40

# DQM 内部拆解（quant / gemm / epilogue）
$env:ONNXSHARP_DQM_BREAKDOWN='1'; & $exe asr .tmp/asr/huihui.wav --models .tmp/sherpa --repeat 6

# MR 扫描（每个 MR 单独进程）
foreach ($mr in '12','6','4') { $env:ONNXSHARP_GEMM_MR=$mr
  & $exe gemmbench --iters 30 --shape 16,320,320 --shape 76,320,1280 --shape 76,320,320 }

# 整数路径对照（=0 会慢 40×，只用于确认默认值正确）
$env:ONNXSHARP_INT8_F32GEMM='0'; & $exe asr .tmp/asr/huihui.wav --models .tmp/sherpa --repeat 20
$env:ONNXSHARP_INT8_F32GEMM='1'; & $exe asr .tmp/asr/huihui.wav --models .tmp/sherpa --repeat 20
```

## 33. 2026-11-16 收工快照：视觉三模型 + ASR 两段

一条命令跑完、同会话、**串行**（先视觉后 ASR，不并行争 CPU）：4 线程、`--iters 20 --rounds 2 --warmup 5`，
ASR 是 `--repeat 30` 去掉首轮加载后的 29 个样本。口径与全表一致：median 为主、min 为无干扰下限。

### 33.1 视觉（`bench/run_all.py`，median / min，ms）

| 模型 | OnnxSharp | ONNXRuntime | OpenCV dnn | SimdYolo |
|---|---:|---:|---:|---:|
| yolo11n | 29.2 / 28.7 | 23.3 / 23.2 | 45.8 / 45.0 | 32.2 / 29.3 |
| yolo11s | 79.7 / 77.1 | 64.6 / 63.8 | 132.3 / 126.2 | 78.7 / 77.5 |
| rfdetr-small | 198.0 / 196.3 | 187.5 / 187.2 | 315.2 / 311.2 | N/A（CLI 不支持 DETR） |

We ÷ 对方 median：

| 模型 | ÷ORT | ÷OpenCV | ÷SimdYolo |
|---|---:|---:|---:|
| yolo11n | **1.25** | 0.64 | 0.91 |
| yolo11s | **1.23** | 0.60 | 1.01 |
| rfdetr-small | **1.06** | 0.63 | — |

> **对 OpenCV 的 0.60~0.64 不能当成「已超越」**：同一天它跑出 311~315 ms，而 §4 那次是 187~237 ms
> （rfdetr-small 历史 191.7）。也就是说今天 OpenCV 处于坏状态，与 §4 的 **1.21~1.23×** 矛盾。
> 按「跨会话只比倍率方向、绝对值不作数」的约定，**结论仍以 §4 的 1.21 / 1.23 为准**，
> 本节只作为同一会话内的相对快照。
>
> 反过来，我们自己的 rfdetr-small 从 §4 的 231.5 降到 198.0（-14%），但同期 OpenCV 也偏了 64%，
> 说明这是机器状态整体变化，**不要归因为代码优化**（代码自 §30 起未改）。

### 33.2 ASR（`asr` 分段 + `profile` 纯 forward，ms）

| 段 | 我们 | ORT | 倍率 | §31.3 当时 |
|---|---:|---:|---:|---:|
| encoder 纯 forward（1×76×560） | 31.0 / 32.6 / 32.8（3 次） | 23.8 / 23.9 | **1.30×** | 1.27~1.35× |
| decoder 纯 forward（16 tok / enc 76） | 10.6 / 10.7 / 15.1（3 次） | 3.57 / 3.59 | **2.97×** | 2.9~3.7× |
| 端到端 `huihui.wav` | min 44（feat 4 / enc 32 / dec 7），median 65 | enc+dec 27.5 | — | 68~70 |
| 端到端 `huihui2.wav` | min 55（feat 4 / enc 40 / dec 9），median 59 | — | — | — |

- 识别文本两条都正确：`今天气真不错我们出去走一走吧`、`语音识别测试成功这是一个纯c井的实现`。
- **与 §31.3 一致，无回归**：encoder 1.30×、decoder 2.97× 都落在当时的区间内。
- `asr` 的 `dec`（7~9 ms）比 `profile` 的纯 forward（10.6 ms）还小，是因为前者按 `token_num` 收窄、
  后者用固定 16 token 的输入；两者不可直接换算，跨表比较需注意（同 §31.3 的注）。
- decoder 这 2.97× 的成因已定死在 §32：ORT 走 u8s8 整数内核（VNNI），我们只有 F32。

### 33.3 复现

```powershell
$env:ONNXSHARP_THREADS='4'
$py='E:\misedir\installs\python\3.11.17\python.exe'      # PATH 上的 python 是无输出的 mise shim
$exe='tools\OnnxSharp.Cli\bin\Release\net10.0\OnnxSharp.Cli.exe'

# 视觉三模型 × 四引擎（直接打印 §33.1 的表，并写 bench/results/latest.json）
& $py bench/run_all.py --iters 20 --warmup 5 --rounds 2 --threads 4 `
  --names yolo11n,yolo11s,rfdetr-small --runtimes onnxsharp,onnxruntime-cpu,opencv-dnn,simdyolo

# ASR 端到端（分段计时，取 min/median）
& $exe asr .tmp/asr/huihui.wav  --models .tmp/sherpa --repeat 30
& $exe asr .tmp/asr/huihui2.wav --models .tmp/sherpa --repeat 30

# ASR 纯 forward（与 ORT 对照的口径）
& $exe profile .tmp/sherpa/encoder.onnx --input 'speech=.tmp/asr/e2e_feats.f32' `
  --input 'speech_lengths=.tmp/asr/e2e_lengths.i32' --shape 'speech=1,76,560' `
  --shape 'speech_lengths=1' --iters 20 --warmup 5 --top 0

& $py .tmp/sherpa/bench_ort.py 4 20
& $py .tmp/sherpa/bench_ort_decoder.py
```

> `run_all.py` 的 `--runtimes` 里任一引擎失败只会把该格写成 `error`，不会中断整轮
> （本轮 `rfdetr-small × simdyolo` 就是 `error`：SimdYolo 只支持 yolo 家族）。

## 34. 2026-11-16 整数 GEMM 重写：4 行 × 24 列，733 GFLOPS（§32.4 被推翻）

§32 的落点是「decoder 的 3× 差距不可消除，因为 .NET 拿不到整数 VNNI」。这条**只在 `net10.0` 成立**：
`net11.0` 目标下本机 Zen4 的 `AvxVnni.IsSupported` 为 `true`，256 位 `vpdpbusd` 可用。
本轮按 MLAS 的输出块形状重写了 `Int8Gemm` 的 VNNI 内核，矩阵乘段直接翻倍。

### 34.1 形状：1 行 × 32 列 → 4 行 × 24 列

ORT 的 AVX512-VNNI kernel（`QgemmU8X8KernelAvx512Core.asm`）是 **6 行 × 48 列**：
18 个累加 zmm + 3 条 B + 1 条 A，占满 zmm14~zmm31；一个 quad（4 个 k）
是 3 load + 6 bcast + 18 `vpdpbusd` = 27 条指令 / 1152 MAC。

托管侧只有 16 个 ymm，等比缩到 **4 行 × 24 列**（12 + 3 + 1 = 16，刚好占满）：

| 项 | MLAS AVX512VNNI | 旧 `MultiplyVnni` | 新 `MultiplyVnniFourRows` |
|---|---|---:|---:|
| 输出块 | 6 行 × 48 列 | 1 行 × 32 列 | 4 行 × 24 列 |
| 累加器 | 18 × zmm | 4 × ymm | 12 × ymm |
| 每 quad 指令 | 3 load + 6 bcast + 18 dpbusd = 27 | 4 load + 1 bcast + 4 dpbusd = 9 | 3 load + 4 bcast + 12 dpbusd = 19 |
| **MAC/指令** | **42.7** | **14.2** | **20.2** |

关键是**复用**而不是「指令更高级」：B 的加载对 4 行共享、A 的广播对 3 个列组共享，
且每个 kBlock 内的 96 字节 B 是连续的一块；12 条累加链彼此独立，
已足够盖住 `vpdpbusd` 约 4 cycle 的延迟，**不需要再做 K 展开**。

配套三处小改：

1. **epilogue 向量化**（`StoreVector`）：`acc + (offset−bzp)·Σa + azp·(bzp·K−Σb)` 全部走
   `vpmulld`/`vpaddd`，每 8 列一次；旧版是逐列 `GetElement(j)` + 4 次标量运算。
2. **列尾巴 8 列向量**：`n=320 / 1280` 在 24 列块之后恰好剩 8 列，不再逐列。
3. **调度**：`MultiplyVnniDispatch` 按 4 行一组切 `ParallelRange.ForEach`（19 块 / 4 线程，动态领取），
   `m % 4` 的尾巴回落到单行内核；`4×kPadded` 超过 32 KB 时整段退回单行。

### 34.2 吞吐（`tier_probe`，4 线程）

| 形状 | 改前 | 改后 | 倍率 |
|---|---:|---:|---:|
| 16×320×320 | 96.7 | 173.4 | 1.79× |
| 16×320×1280 | 364.2 | **736.4** | 2.02× |
| 76×320×1280 | 314.1 | **732.5** | 2.33× |

`732.5 / 4 = 183 GFLOPS` 单核 = `20.2 MAC/cycle × 4.5 GHz × 2`，
即 256 位 `vpdpbusd` 1 条/cycle 的**理论上限**，多线程效率≈100%。

`16×320×320` 仍是 173（1.6 MMAC 太小，A 打包 + 派发 + 尾块占主导），与 §32.5 的
「小 m 是固定开销」结论一致，不是内核形状的问题。

### 34.3 decoder 与端到端

| 指标 | 改前（§33.2） | 改后 | |
|---|---:|---:|---|
| `MatMulInteger` 12 个（`profile`，min） | 1.3 ms | **0.7 ms** | −46% |
| decoder 纯 forward（16 tok / enc 76） | 10.6 ~ 15.1 | **6.0 ~ 8.6** | |
| ÷ ORT（3.57 ~ 3.61） | **2.97×** | **1.67×** | |
| 端到端 `dec` 段（`asr`，min） | 7 ~ 9 | **6** | |

### 34.4 DQM 换整数内核：实测否证

`decoder` 里 `DynamicQuantizeMatMul` 仍是最大一块（50 个、3.3 ms、39%），
但它**不**走 `Int8Gemm`：它的权重是 per-channel 量化的 F32，走「量化为 F32 偏移 + packed F32 GEMM」。
按 ORT 的口径把它接到整数路径**不划算**，实测拆解如下（`ONNXSHARP_DQM_BREAKDOWN=1`）：

| 段 | 占比 |
|---|---|
| quant（写 m×k 个 float） | 小 |
| **gemm（F32 packed）** | **绝对主体** |
| epilogue | ≈0（已折进 GEMM 尾部） |

即它的成本就是 GEMM 本身：`1.70 GFLOP / 3.3 ms ≈ 512 GFLOPS`，已经在 F32 峰值的量级上；
换成整数路径还要多一趟 `int32 → float` 的输出 pass，收益为负。**DQM 维持现状**。

### 34.5 sherpa 现状（同会话交替，4 线程，ms）

`profile --iters 30 --warmup 5`，`net11`/`net10`/ORT 交替跑 5 轮取 min（机器本底负载偏重，
median 会差 10~30%，见 §31.5）：

| 段 | 我们 net11 | 我们 net10 | ORT | ÷ORT（net11） |
|---|---:|---:|---:|---:|
| encoder 纯 forward（1×76×560） | 30.3 | 29.9 | 24.1 | **1.26×** |
| decoder 纯 forward（16 tok / enc 76） | **6.0** | 6.9 | 3.60 | **1.67×** |

- **encoder 没变**（30.3 vs 29.9，噪声内）：它的 160 个 DQM 同样走 F32 路径，本轮没动它。
  1.26× 就是 §32 说的「F32 域上我们已到峰值」的残留，属于调度/Conv 的事，不是 GEMM。
- **decoder 6.0 vs 6.9**：差额几乎正好是 `MatMulInteger` 的 1.3 → 0.7。
- 端到端 `huihui.wav`（`--repeat 30`，min）：feat 4 / enc 37 / **dec 6**，文本仍为
  `今天气真不错我们出去走一走吧`。
- 单测：`net11.0` 116/118 passed、2 skipped、0 failed（与改前一致，数值逐位不变）。

### 34.6 复现

```powershell
$env:ONNXSHARP_THREADS='4'
$xNew='tools\OnnxSharp.Cli\bin\Release\net11.0\OnnxSharp.Cli.exe'
$xOld='tools\OnnxSharp.Cli\bin\Release\net10.0\OnnxSharp.Cli.exe'

# 内核吞吐（§34.2）：打印 tier 与 3 个形状，只 net11 有 AvxVnni
cd .tmp/tier_probe; dotnet run -c Release -f net11.0

# decoder 纯 forward（§34.3）
& $xNew profile .tmp/sherpa/decoder_baked.onnx `
  --input 'onnx::Slice_6785=.tmp/sherpa/ok_memory.f32' `
  --input '/encoder/after_norm/Add_1_output_0_quantized=.tmp/sherpa/dec2_enc_u8.bin' `
  --input '/encoder/after_norm/Add_1_output_0_scale=.tmp/sherpa/dec2_enc_scale.f32' `
  --input '/encoder/after_norm/Add_1_output_0_zero_point=.tmp/sherpa/dec2_enc_zp.u8' `
  --shape 'onnx::Slice_6785=1,16,320' --shape '/encoder/after_norm/Add_1_output_0_quantized=1,76,320' `
  --iters 30 --warmup 5 --top 0

# encoder 纯 forward + ORT 对照（§34.5）
& $xNew profile .tmp/sherpa/encoder.onnx --input 'speech=.tmp/asr/e2e_feats.f32' `
  --input 'speech_lengths=.tmp/asr/e2e_lengths.i32' --shape 'speech=1,76,560' `
  --shape 'speech_lengths=1' --iters 30 --warmup 5 --top 0
& 'E:\misedir\installs\python\3.11.17\python.exe' .tmp/sherpa/bench_ort.py 4 20
& 'E:\misedir\installs\python\3.11.17\python.exe' .tmp/sherpa/bench_ort_decoder.py

# 端到端
& $xNew asr .tmp/asr/huihui.wav --models .tmp/sherpa --repeat 30
```

## 35. 2026-11-16 PaddleOCR 与 **Sdcb.SimdPaddleOCR** 对照（4 线程）

第一次把 OCR 管线放到外部实现旁边量。对手是 `Sdcb.SimdPaddleOCR 2.0.0`（纯 C# 的 PaddleOCR 推理引擎），
**用同一套模型**（`ChineseV6Tiny` 解包出的 `det/rec/cls/dict`，我们 CLI 用的就是它），
同一份紧凑 RGB24 像素、同一进程内交替运行、4 线程（`ONNXSHARP_THREADS=4`；对方
`DetIntraOpThreads=4` + `LineWorkerCount=4`），解码（ImageSharp）不计时。

### 35.1 端到端（交替顺序、各取 min / median，ms）

| 图 | 配置 | OnnxSharp | SimdPaddleOCR | 倍率（min / median） |
|---|---|---:|---:|---|
| `ocr_test.bmp` 900×300，2 行 | cls on | 34.8 / 43.7 | 18.2 / 20.7 | **1.92× / 2.11×** |
| `ocr_test.bmp` 900×300，2 行 | cls off | 36.1 / 53.6 | 19.3 / 35.6 | 1.87× / 1.50× |
| `plate.bmp` 大图，3 行 | cls on | 59.7 / 87.7 | 34.3 / 51.2 | **1.74× / 1.71×** |

结论：**我们慢 1.7~2.1×** —— 比视觉 / ASR（1.06~1.26×）差得多，是当前最大的单项差距。

> 方差：这台机本底负载重，ours 单轮能飘到 244 / 257 ms（GC + 线程池），所以表里用 min，
> median 只作参考（与 §31.5 口径一致）。`cls off` 那一行反而没降，说明**噪声已经盖住
> cls 的 ~10 ms**，不能据此判断 cls 是否值得关 —— 需进程内分解（见 35.2）。

### 35.2 我们的时间花在哪（`profile`，纯 forward，4 线程 min）

| 段 | 形状 | 次数 | 每次 | 小计 |
|---|---|---:|---:|---:|
| det | 1×3×288×896 | 1 | 19.7 | 19.7 |
| cls | 1×3×80×160 | 2 | 6.5 | 13.0 |
| rec | 1×3×48×384 / 400 | 1 / 1 | 4.6 / 5.5 | 10.1 |

- **det 一项 19.7 ms 就超过对方整条管线（18.2 ms）** —— 差距的地基在 det。
  det 是 PPLCNet 式小通道深度可分离栈（242 节点 / 83 Conv），每次 Conv 只有 16~128 通道，
  NCHWc 打包 + 内存池 + 派发这些**固定开销**盖过了算力。
- **cls 13 ms（37%）**：对 2 行文字跑 2 次 80×160 的小网络，每次 6.5 ms —— 32 个 Conv 摊到
  每个 0.2 ms，同样是固定开销主导。`profile --top` 显示 Conv 占 6.0 / 6.5 ms，没有热点，全是摊薄。
- **rec 10 ms**：两行串行，对方是 4 路 worker 并行（`RecBatchLines` 还能合批）。

### 35.3 正确性

- `ocr_test.bmp`：两行文本 **逐字一致**（`Hello OnnxSharp 123` / `第二行文字 PaddleOCR`），
  框位置一致（43,42,498×64；41,141,447×54）。
- `plate.bmp`：主行 `浙A ES8381` 一致；低置信度小框不同 —— 我们 `e`(0.498) / `A3M15X`(0.807)，
  对方 `8` / `浙A：3M15X`。两个引擎的 det 框有 1~2 px 差异，裁剪进 rec 后输出了不同字符，
  属于同一量级的模型噪声，不是实现错误（同一个 rec 模型、同一字典）。

### 35.4 冷启动

| | 我们 | SimdPaddleOCR |
|---|---:|---:|
| 加载（读盘 + 建会话） | 4~5 ms（只读字节） | 96~100 ms（完整 session + 字典） |
| 首轮（含图特化 + JIT） | 119 ms | 41 ms |
| **合计首次可用** | **~124 ms** | **~141 ms** |

我们是"懒特化"，对方是"加载即建"，总冷启动我们略优，但差距不大。

### 35.5 下一步（如果要把 OCR 拉到 1.3× 以内）

1. **det 的固定开销**：小通道 Conv 豁免 NCHWc / 单线程阈值（`MaxSpatialPixels` 之下已经这么做了，
   但 896×288 不够小）；这是 §29 提到的「打包上限 5.7%」在小通道图上不成立的反例。
2. **cls 合并**：两行→一个 batch（80×160×2），或直接按 det 框宽高比预判跳过（水平框占绝大多数）。
3. **rec 合批**：一次 `Run` 内把多行压成 `1×3×48×Σbucket` 或 `N×3×48×W` 跑，再拆回，
   这是对方最直接的优势。
4. det 若仍是瓶颈，再考虑 PPLCNet 专用 depthwise 路径（per-channel 1 线程、多通道才并行）。

### 35.6 复现

```powershell
$env:ONNXSHARP_THREADS='4'
cd .tmp/ocrbench
dotnet run -c Release -- ..\ocr_test.bmp 30 ..\ocr-models 3            # cls on
dotnet run -c Release -- ..\ocr_test.bmp 20 ..\ocr-models 2 noclass    # cls off
dotnet run -c Release -- ..\plate.bmp    15 ..\ocr-models 2            # 大图 3 行

# 我们的纯 forward 分段（需先造零输入，见 §35.2 的形状）
$cli='..\..\tools\OnnxSharp.Cli\bin\Release\net11.0\OnnxSharp.Cli.exe'
& $cli profile ..\ocr-models\det.onnx --input 'x=..\ocr_det.f32' --shape 'x=1,3,288,896' --iters 20 --warmup 5 --top 12
& $cli profile ..\ocr-models\cls.onnx --input 'x=..\ocr_cls.f32' --shape 'x=1,3,80,160' --iters 20 --warmup 5 --top 12
& $cli profile ..\ocr-models\rec.onnx --input 'x=..\ocr_rec384.f32' --shape 'x=1,3,48,384' --iters 20 --warmup 5 --top 12
```

## 36. 2026-11-16 YuNet：int8 能跑了，以及三条优化尝试的结果

YuNet（`face_detection_yunet_2023mar.onnx`，0.22 MB，opset 11，91 个执行算子，输入 1×3×640×640，
输出 12 个：cls/obj/bbox/kps × stride 8/16/32）。

### 36.1 fp32 基线（4 线程，640×640）

| 段 | min | median |
|---|---:|---:|
| 纯 forward（`bench`，热机） | 5.4~5.9 | 6.1~7.8 |
| 预处理（512² → 640² Stretch + BGR→CHW） | 1.74 | 2.84 |
| forward（`facebench` 里含解码同进程） | 7.92 | 11.89 |
| 解码（12 输出 + NMS） | ~0.5 | ~0.5 |
| **端到端** | **8.4** | **10.9** |

`profile` 的分段：53 层 Conv 占 94%（sum-of-min 5.1 / 总 5.4），
**Conv_0（3→16, 3×3, s2, 320²）一层就 0.81 ms（14.7%）**，其余大层 0.27~0.48 ms。
按 MAC 估，全图约 0.3~0.5 GMAC → 有效 60~100 GMAC/s，对小通道多层网络属正常区间（寄存器/派发主导，不是算力）。

> `facebench` 的 forward 比 `bench` 高 ~2 ms，因为每轮都新建输入张量（1.2M float）+ 与解码共享 Net，
> GC 压力不同；口径分开看。

### 36.2 int8：**能跑**（README 里"M10-04/05 未实现"的旧结论已过时）

`yunet_int8.onnx` 是标准 QDQ/QL 图：53 `QLinearConv` + 16 `DequantizeLinear` + 5 `QuantizeLinear`
+ 6 `QLinearSigmoid` + 2 `QLinearAdd` + 4 `MaxPool` + 2 `Resize`，跑通无报错。

| | fp32 | int8 |
|---|---|---|
| score | 0.9135 | 0.8944 |
| box | (209.3, 181.5) 142.2×206.0 | (209.4, 180.7) 143.6×207.4 |
| forward（min / med） | 5.4 / 6.1 | **22.95 / 23.44** |

精度是同一量级（框中心差 ~1 px、尺寸差 ~1.4 px，score 低 0.019），**但性能慢 2.9~4×**。

原因（`profile` sum-of-min 28.4 ms）：`QLinearConv` 每层都要
`Dequantize(x) → F32 卷积 → Requantize(y)`，也就是在 f32 卷积之外多两次**全张量物化**
（写 4N + 读 4N 各一遍）。53 层累计 14.4 ms（是 fp32 Conv 5.1 ms 的 2.8×），
另有 5 个 `QuantizeLinear` 3.1 ms。YuNet 太小（0.3~0.5 GMAC），省下的带宽抵不过这两趟搬运。

### 36.3 三条尝试（都做了实测）

| 尝试 | 做法 | 结果 |
|---|---|---|
| 数据通路换向 | `ONNXSHARP_PACKED=0` 全走 im2col/Winograd | 5.4 → 5.6 ms，**中性**（面板式与 im2col 在 K=27/M=16 上同样低效） |
| 强制面板 | `ONNXSHARP_PACKED_S1=1`（s1 大空间也打包，复核 §24 的老经验） | 5.7 → 5.8 ms，**中性** |
| 首层专用内核 | 新增 `Kernels/FirstConv.cs`：沿输出宽 8 像素向量化、输出通道内层展开、步长 2 用偶数位抽取 | **更慢**：Conv_0 0.81 → 1.08 ms。每条 `vfmadd` 要配一条 `vbroadcastss`（指令比 1:1），而面板式是 4 行 × 16 列、广播被 16 列复用。已实现但默认关闭（`ONNXSHARP_FIRSTCONV=1` 才开），注释里留了原因 |

### 36.4 结论

- **fp32 已接近这个结构的实测上限**：层多、通道窄、单层 MAC 小，固定开销（打包 + 派发 + 访存）占大头；
  packed/im2col/Winograd 三条路径在这个形状上无差异，说明瓶颈不在某个内核，而在「层数 × 每层固定成本」。
  再往上走要么是布局传播（§24.4 结论：≤8%，不值得），要么是 Conv+Relu 这类图级融合（Relu 已 in-place，收益近零）。
- **int8 想赢 fp32，必须走整数卷积或链式 QDQ 消除**：
  前者把 53 层的 `vpdpbusd` 直接接起来（§34 的 4×24 内核已有），后者在 QLinearConv → QLinearConv 之间
  省掉「RQ→DQ」这一来一回。两条都是 M10 级的改造（前者 ~300 行内核 + epilogue 对拍，后者要动图重写），
  本次没有展开。

### 36.5 复现

```powershell
$env:ONNXSHARP_THREADS='4'
$cli='tools\OnnxSharp.Cli\bin\Release\net11.0\OnnxSharp.Cli.exe'

& $cli face .tmp/yunet/yunet.onnx .tmp/lena.jpg          # fp32：score 0.9135
& $cli face .tmp/yunet/yunet_int8.onnx .tmp/lena.jpg     # int8：score 0.8944

& $cli bench .tmp/yunet/yunet.onnx --input 'input=.tmp/yunet/yunet_input.f32' --iters 40 --warmup 10
& $cli profile .tmp/yunet/yunet_int8.onnx --input 'input=.tmp/yunet/yunet_input.f32' --shape 'input=1,3,640,640' --iters 20 --warmup 5 --top 20

$env:ONNXSHARP_FIRSTCONV='1'    # 复现"首层专用内核更慢"（Conv_0 0.81 → 1.08）

cd .tmp/facebench               # 预处理 / forward / 解码 三段
dotnet run -c Release -- ..\lena.jpg 20 5
```

## 37. 2026-11-17 收工：池化 / Resize 行切分 + Identity 消除 + 物理核默认，全模型复测

本轮不再碰 GEMM 与 Conv（§26~§34 已把 `1×1` 的账算干），改的是**「Conv 之外那 15%」里最便宜的三块**，
外加一个测量之外的默认值错误（默认线程数用逻辑核）。

### 37.1 改动清单

| 文件 | 改动 | 落点 |
|---|---|---|
| `Ops/DnnOps.cs` | `MaxPool` 新增 2×2/s1 滑窗内核（竖向两行 `Avx.Max` 进 row buffer 再横向求 max）与 3×3/s2 内核 | OCR det/rec、YuNet |
| `Ops/DnnOps.cs` | `Resize` 全路径（nearest / bilinear × u8 / f32）按行切分，最近邻 2× 单独走表驱动内核 | yolo 的 upsample、OCR det |
| `Ops/DnnOps.cs` | 池化 / Resize 的并行粒度由「每平面」改成「平面 × 行块」 —— 原来 16 个平面在 16 线程上铺不满就等于串行 | 同上 |
| `Ops/DnnOps.cs` | `GlobalAveragePool` 走整平面快速路径（两路 AVX 累加 + 行切分，部分和用 float 数组归并而非原子加） | SE 模块 0.24 → **0.10 ms** |
| `Core/GraphFusion.cs` | 新增 `EliminateIdentity`：单入单出的 `Identity` 直接重写消费者 | rec 图删 58 个节点 |
| `Threading/ParallelRange.cs` | 默认线程数 `ProcessorCount`（32 逻辑核）→ **物理核数**（Windows 用 `GetLogicalProcessorInformationEx`，=16） | 见 37.2 最后一条 |
| `Ocr/PaddleOCR/PaddleOcrPipeline.cs` | 喂给引擎的输入张量按形状缓存复用（`SetInput` 只引用不拷贝），顺带加 `OcrTimings` 分段 | 每帧省 5 MB 进 LOH |

> `ParallelRange` 那条是**纠错而不是提速**：超线程对这些 FP 内核没有吞吐收益，多出来的 16 个自旋
> worker 只跟机器上别的进程抢时间片。负载 35% 时 OCR 端到端中位数从 **184 ms 回到 35 ms**，
> 而安静窗口下的 min 几乎不变 —— 说明它治的是尾巴，不是均值。

### 37.2 OCR（PaddleOCR，4 线程，同会话 min / median，ms）

| 段 | 优化前 | 现在 | |
|---|---:|---:|---|
| det 1×3×288×896 | 17.89 / 26.12 | **13.18 / 14.56** | −26% |
| cls 1×3×80×160 | 1.86 / 2.77 | **1.55 / 1.77** | −17% |
| rec 1×3×48×384 | 4.36 / 5.32 | **2.90 / 3.57** | −33% |
| rec 1×3×48×400 | 4.82 / 5.21 | **3.68 / 4.07** | −24% |
| **端到端（min）** | 34.3 | **24.2** | −29% |
| vs `Sdcb.SimdPaddleOCR` | 1.66× | **1.25×** | |

分段（单帧 24.3 ms）：`pre 0.53 + det 13.48 + post 0.54 + cls 2.78 + rec 7.00`。

识别文本与置信度（0.957 / 0.982）与优化前逐位一致。

### 37.3 全模型四引擎复测（4 线程，iters 30 / rfdetr 20，warmup 5，rounds 4）

`python bench/run_all.py --threads 4 --rounds 4`，每轮轮转引擎顺序、各引擎取自己最好的一轮；
外部负载约 35%（qbittorrent / xray / 微信常驻），perf% 138（**未撞温度墙**）。

对照列用 **§28**（11-16 同口径四引擎，`--iters 30 --rounds 2`），不要用 §8.3 —— 中间还夹着 NCHWc（§23/§24）、
64 字节对齐（§30）、整数 GEMM（§34）三轮改动，拿 §8.3 作基线会把它们的收益算到本轮头上。

| 模型（median ms） | onnxsharp | onnxruntime-cpu | opencv-dnn | simdyolo | 我们上轮（§28） | Δ |
|---|---:|---:|---:|---:|---:|---:|
| yolo11n 640² | 30.4 | **23.8** | 48.8 | 31.2 | 29.6 | +2.7% |
| yolov8n 640² | 35.8 | **28.2** | 50.8 | 31.3 | 34.4 | +4.1% |
| yolov8s 640² | 90.2 | **79.5** | 151.8 | 83.2 | 87.4 | +3.2% |
| yolo11s 640² | 76.6 | **63.5** | 129.7 | 77.8 | 77.2 | −0.8% |
| rfdetr-small 512² | 198.0 | **187.7** | 314.8 | 跑不了 | 200.0 | −1.0% |

倍率（我们 ÷ 对方，>1 为更慢）：

| 模型 | vs ORT | vs OpenCV | vs SimdYolo |
|---|---:|---:|---:|
| yolo11n | 1.28 | **0.62** | 0.97 |
| yolov8n | 1.27 | **0.70** | 1.14 |
| yolov8s | 1.13 | **0.59** | 1.08 |
| yolo11s | 1.21 | **0.59** | 0.98 |
| rfdetr-small | **1.05** | **0.63** | N/A |

**本轮对 yolo / rfdetr 的收益 = 0（噪声内）**，而且整个会话比 §28 那天一致偏慢 2~4%
（我们 +2.7/+4.1/+3.2/−0.8/−1.0，ORT 23.5→23.8 / 27.8→28.2 / 76.4→79.5 / 63.5→63.5 / 187.2→187.7，
SimdYolo 29.9→31.2 / 29.4→31.3 / 76.2→83.2 / 77.5→77.8）—— **三个引擎同向偏移，说明是机器状态，不是代码**。
只有 OpenCV 的 s 档偏 11%（136.5→151.8），与 §33.1 记的「今天 OpenCV 处于坏状态」同源。

> **为什么 yolo 吃不到这轮改动**：yolo11n 的池化只有 SPPF 的 3 个 5×5/s1（20×20×256），Resize 只有 1 个 2×
> upsample，Identity 在融合后本来就是 0 个 —— 这几处在 §28 的预算里合计不到 1 ms，而本轮的判据是「换掉退化路径」，
> 不是「造出新算力」。受益的是**池化/Resize 占比高**的图：OCR det（FPN 上采样 + 池化，13 ms 级）与 YuNet int8。

### 37.4 YuNet（单引擎，本仓库没有跨引擎脚本）

| | 4 线程 min / median | 16 线程 min / median | 上次（§36） |
|---|---:|---:|---:|
| `yunet.onnx` fp32 640² | 5.7 / 5.9 | 5.2 / 8.2 | 5.4~5.9 / 6.1~7.8 |
| `yunet_int8.onnx` | **20.3 / 20.6** | 23.8 / 32.7 | 22.95 / 23.44 |

- fp32 无变化（符合 §36.4：瓶颈是「层数 × 每层固定开销」，本轮没碰 Conv）。
- int8 快 11%，仍是 fp32 的 3.5×；QDQ 每层多两遍全张量往返的结论不变。
- int8 在 16 线程反而更慢 —— 图太碎，并行收益为负，这和 37.1 的「少开线程」是同一条规律的两种表现。
- 精度无回归：`face` 的 score **0.9135 / 0.8944**，框与 5 个关键点与 §36.2 逐位一致。

### 37.5 目标线复核

| 目标（4 线程 median） | 基线 | 目标 | 现在 | |
|---|---:|---:|---:|---|
| rfdetr-small | 262.5 | ≤210 | **198.0** | **达成**（−24.6%，§28 那轮就到了；本轮 −1.0% 只是维持） |
| yolo11n | 39.1 | ≤27 | 30.4 | 差 **12.6%**（§33.1 时 29.2，差 8%） |

rfdetr 主线可以关。yolo11n 的余量在「大量中等尺寸 Conv」，按 §26~§34 的账只剩微内核深度与
NCHWc 覆盖面两条路，都不是本轮这种「顺手改」的量级。**本轮没有推进这条线**。

### 37.6 遗留

- **median 尖刺**：OCR 单帧仍偶发 100~400 ms（SimdPaddleOCR 也有，幅度小）。已排除输入分配的
  LOH，怀疑是自旋线程与外部负载的时间片争用 —— 物理核默认值把它从「常态」压成了「偶发」，没有根除。
- yolo11n 距目标线还差 12.6%；Conv 之外的零钱已经捡完（本轮证明这部分对 yolo 只有 <1 ms）。
- 本轮收益未做开关式单项剥离；OCR 的对照是「同一段代码改前/改后同会话」，可信度高于 yolo 列。

### 37.7 复现

```powershell
$env:ONNXSHARP_THREADS='4'
$cli='tools\OnnxSharp.Cli\bin\Release\net10.0\OnnxSharp.Cli.exe'

# 四引擎全表（同会话、每轮轮转、各取最好一轮）
python bench/run_all.py --threads 4 --rounds 4 --iters 30

# YuNet
& $cli bench .tmp/yunet/yunet.onnx      --input 'input=.tmp/yunet/yunet_input.f32' --iters 40 --warmup 10
& $cli bench .tmp/yunet/yunet_int8.onnx --input 'input=.tmp/yunet/yunet_input.f32' --iters 40 --warmup 10
& $cli face .tmp/yunet/yunet.onnx .tmp/lena.jpg      # 0.9135
& $cli face .tmp/yunet/yunet_int8.onnx .tmp/lena.jpg # 0.8944

# OCR 端到端 + 分段
cd .tmp; dotnet ..\tools\OnnxSharp.Cli\bin\Release\net10.0\OnnxSharp.Cli.dll ocr ocr_test.bmp `
  --det ocr-models/det.onnx --rec ocr-models/rec.onnx --cls ocr-models/cls.onnx --dict ocr-models/dict.txt
```

回归：`OnnxSharp.slnx` 全量编译通过、零警告；测试 **116/118（2 skipped）、0 failed**。
改动未提交。

## 38. 2026-11-17 与 MLAS 单算子对照：GEMM / int8 GEMM / Conv 三条线

§37 复测后还剩一个问题没回答：图级「我们 ÷ ORT ≈ 1.1~1.3」里，**有多少是我们内核本身不如人**？
ORT 的 GEMM 与 Conv 全在 MLAS 里，把 `F:\code\mlas`（2026-11 主线的 ORT 内部副本）单独编成 dll，
就能把这条问题拆成三个可分的轴来量，而不是继续在图级打转。

### 38.1 口径与五个坑

| 项 | 结论 |
|---|---|
| 线程语义 | `ThreadPool == nullptr` ⇒ `DegreeOfParallelism == 1`（纯串行）；传 `MlasGetBuiltinThreadPool()` ⇒ 物理核 16 |
| 符号 | dll 用 `WINDOWS_EXPORT_ALL_SYMBOLS` 编译，导出的是 MSVC 修饰名，ctypes 按修饰名取（名字从 `mlas.lib` 里 regex 抽） |
| 对齐 | packed B / A / C 缓冲必须 **64 字节对齐**（numpy 只保证 16）：不对齐时 packed 路径直接崩或结果错 |
| `MlasConv` 参数 | `InputChannels` 与 `FilterCount` 都是**每组**的值（`C/group`、`M/group`），传总通道数会越界写堆（`0xc0000374`） |
| `MlasActivation` | 内部直接解引用，不接受 `nullptr`：恒等激活必须显式传 `ActivationKind = 0` |
| 直调不可信 | 直调 `MlasGemmBatch` 单线程只有 14~30 GFLOPS，而 ORT 跑同一份 MLAS 是 55~105 —— 直调缺 ORT 侧的 B 打包缓存与预热。**F32 GEMM 一律用 ORT 单算子 `MatMul` 作口径**，直调只当内核参考 |

本机噪声：整个会话负载在 16~99% 之间跳，1T 读数跨运行漂移 ±13%，16T 读数跨运行 2~3×。
所以 16T 一律 **best-of-2 运行**，且下表只做同会话相对比较、不跨会话引用。
FMA 峰值（`flopsbench`）：单核 110.7 GFLOPS、16 线程 866.6 GFLOPS（54.2/线程）。

### 38.2 F32 SGEMM：1 线程（GFLOPS，同会话）

| m×k×n | 我们 | MLAS(ORT) | 我们/MLAS |
|---|---:|---:|---:|
| 16×27×2240 | 57.6 | 55.9 | 1.03 |
| 32×144×1600 | 68.4 | 84.3 | 0.81 |
| 64×144×800 | 80.6 | 91.8 | 0.88 |
| 64×576×400 | 80.0 | 95.7 | 0.84 |
| 64×1152×1600 | 79.8 | 98.3 | 0.81 |
| 128×576×200 | 69.5 | 95.5 | 0.73 |
| 128×1152×200 | 68.6 | 97.4 | 0.70 |
| **256×1152×100** | 35.3 | 92.9 | **0.38** |
| 64×64×6400 | 73.9 | 96.1 | 0.77 |
| 256×256×6400 | 81.8 | 105.3 | 0.78 |
| 1600×128×256 | 85.7 | 52.8 | 1.62 |
| 1600×256×128 | 87.0 | 105.9 | 0.82 |
| 1025×384×1536 | 85.5 | 106.3 | 0.80 |
| 257×384×1536 | 85.0 | 104.9 | 0.81 |
| **1×384×1536（GEMV）** | 6.4 | 26.5 | **0.24** |
| **512×1024×1024** | 50.3 | 94.8 | **0.53** |

- MLAS 的 1T 基本**打满单核**：84~106 GFLOPS = `flopsbench` 单核峰值的 76~96%。我们 68~87 = 62~79%，
  即 yolo 形状上系统性落后 **19~30%**，性质是**内核效率**而不是调度（线程数一样是 1）。
- 三处断崖与 §26~§34 的账一致：`n=100`（0.38×）、GEMV（0.24×）、`m=512/k=1024`（0.53×）。
  前两个是「N 尾块 + 小 M 固定开销」，第三个是 A 面板复用不够（B 只读一遍、A 读了 n/NC 遍）。
- `1600×128×256` 我们反超 1.6×，但 ORT 这一格自身两次会话读数差 1.5×（52.8 vs 77.9），**存疑，不当结论**。

### 38.3 F32 SGEMM：16 线程（GFLOPS，两侧各自 best-of-2）

| m×k×n | 我们 | MLAS(ORT) | 我们/MLAS |
|---|---:|---:|---:|
| 16×27×2240 ¹ | 61.6 | 81.3 | 0.76 |
| 32×144×1600 | **614.4** | 288.6 | **2.13** |
| 64×144×800 | **638.3** | 376.2 | **1.70** |
| 64×576×400 | 315.1 | 361.0 | 0.87 |
| 64×1152×1600 | 197.7 | 179.8 | 1.10 |
| 128×576×200 | 100.0 | 198.2 | **0.50** |
| 128×1152×200 | 141.9 | 190.8 | 0.74 |
| 256×1152×100 | 54.0 | 90.9 | 0.59 |
| 64×64×6400 | 154.0 | 101.2 | 1.52 |
| 256×256×6400 | 227.0 | 365.5 | 0.62 |
| 1600×128×256 | 148.0 | 117.8 | 1.26 |
| 1600×256×128 | 169.2 | 140.0 | 1.21 |
| 1025×384×1536 | 191.7 | 140.4 | 1.37 |
| 257×384×1536 | 226.3 | 168.8 | 1.34 |
| 1×384×1536 ¹ | 6.4 | 12.7 | 0.50 |
| 512×1024×1024 | 253.0 | 168.2 | 1.50 |

¹ 这两格我们走单线程（`gemmbench` 的 `m·n·k < 1<<22` 阈值），ORT 侧是真 16 线程，不可直接比。

- **多数形状我们 1.1~2.1× 领先**，因为是 ORT 自己的池在大宽形状上没铺开：`1025×384×1536`
  ORT 16T 只比 1T 快 1.32×（140.4 vs 106.3），16 核峰值 866 GFLOPS 只吃到 21%；同一格我们 191.7。
- 我们输的格子有共同特征：**`n ≤ 200`**（0.50~0.74）—— 我们的并行只切 N，`n=100` 时每线程 6 列，
  不如 ORT 的 M/N 二维切分；以及 `256×256×6400`（0.62，B 复用过短）。
- 我们 16T 峰值 638 GFLOPS = 峰值 74%，ORT 最好 376 = 43%。**这条轴我们不落后。**

### 38.4 Conv：MLAS 公共 NCHW `MlasConv` vs `convbench`（GMAC/s）

单线程（同会话，`convbench --threads 1`）：

| 层 | MLAS | im2col | packed | winograd | nchwc |
|---|---:|---:|---:|---:|---:|
| 3→16 640² k3s2 | 18.0 | 16.8 | 17.2 | -- | -- |
| 16→32 320² k3s2 | 27.4 | 25.8 | 26.9 | -- | **54.4** |
| 32→64 160² k3s2 | 36.3 | 35.2 | 32.8 | -- | **54.6** |
| 64→64 160² k3s2 | 36.2 | 34.9 | 32.1 | -- | **54.9** |
| 64→64 160² k3s1 | 45.5 | 36.5 | -- | 29.0 | **54.7** |
| 64→128 80² k3s2 | 42.6 | 38.0 | 35.2 | -- | **55.0** |
| 128→128 80² k3s2 | 42.6 | 35.0 | 34.9 | -- | **55.0** |
| 128→128 80² k3s1 | 49.2 | 39.6 | -- | 45.2 | **55.2** |
| 128→256 40² k3s2 | 47.2 | 34.7 | 36.4 | -- | 43.7 |
| 256→256 40² k3s2 | 47.3 | 23.0 | -- | -- | 44.8 |
| 256→256 40² k3s1 | 51.3 | 35.4 | -- | 46.4 | **55.1** |
| 128→128 20² k3s2 | 37.1 | 16.2 | 26.1 | -- | 44.6 |
| 256→256 20² k3s1 | 50.7 | 23.3 | -- | 33.5 | 44.7 |
| 64→64 160² k1 | 47.2 | 30.9 | -- | -- | -- |
| 128→128 80² k1 | 51.8 | 41.1 | -- | -- | -- |
| 256→256 40² k1 | 52.9 | 44.4 | -- | -- | -- |
| 256→256 20² k1 | 52.6 | 44.1 | -- | -- | -- |

16 线程（两侧各 best-of-2）：

| 层 | MLAS | im2col | packed | winograd | nchwc |
|---|---:|---:|---:|---:|---:|
| 3→16 640² k3s2 | 123.4 | 20.5 | 26.3 | -- | -- |
| 16→32 320² k3s2 | 257.5 | 179.9 | 273.3 | -- | **436.7** |
| 32→64 160² k3s2 | 358.4 | 377.1 | 270.9 | -- | **437.2** |
| 64→64 160² k3s2 | 305.6 | 333.7 | 409.5 | -- | **436.9** |
| 64→64 160² k3s1 | 538.3 | 293.6 | -- | 190.0 | **529.0** |
| 64→128 80² k3s2 | 87.8 | 174.8 | 280.3 | -- | **436.6** |
| 128→128 80² k3s2 | 119.2 | 193.6 | 265.7 | -- | **436.8** |
| 128→128 80² k3s1 | 221.5 | 372.9 | -- | 344.2 | **422.2** |
| 128→256 40² k3s2 | 176.2 | 80.8 | 236.2 | -- | **353.4** |
| 256→256 40² k3s2 | 103.6 | 83.7 | -- | -- | **373.1** |
| 256→256 40² k3s1 | 214.1 | 224.4 | -- | **479.6** | 509.9 |
| 128→128 20² k3s2 | 55.2 | 6.9 | -- | -- | **253.8** |
| 256→256 20² k3s1 | 190.2 | 12.6 | -- | 104.8 | 97.4 |
| 1×1 四层 | 100.8 / 104.7 / 84.9 / 55.1 | 166.0 / 110.5 / 59.0 / 111.1 | -- | -- | -- |

- **单线程**：MLAS 的 NCHW im2col 比我们 im2col 快 1.05~2.2×（它的 segment 版 im2col 更省一遍写回），
  但与我们的 **NCHWc 基本同档**（0.88~2.0×，多数 1.1~1.5×）：我们的 c 内核单线程 43.7~55.2 GMAC/s，
  MLAS 36.2~52.9。**1T 上没有「MLAS 更强」这回事。**
- **16 线程**：MLAS 只有 `64→64 s1` 一层摸到 538（与我们 529 同档），其余 55~373，
  中位约 210；我们 nchwc 254~529，中位约 437。MLAS 的 NCHW 路径在 `64→128 80²`、`128→128 80²`、
  `128→128 20²` 这几层只有 55~119 —— **是它的分块/并行选择问题，不是内核问题**。
- `3→16 640²`、`16→32 320²` 这种「大 N 小 M」层：MLAS 1T 只有 18.0/27.4，16T 123/257；
  我们 NCHWc 1T 就 54.4（`ic=3` 不够 16 通道块，走 im2col 16.8，是已知缺口）。
- ⚠️ **口径边界**：ORT 图内跑的是 **未导出的 `MlasNchwcConv`**，`MlasConv` 是它的 fallback 路径。
  所以这张表只能回答「同一个 NCHW 卷积，谁的实现更快」，**不能**换算成 ORT 整图性能
  （图级仍以 §37.3 的四引擎表为准）。
- 表里 nchwc 列**只含内核**，输入/输出 reorder 另计（`convbench` 输出的 `+reorder`，0.03~0.50 ms）；
  加上 reorder 仍不改变上面的结论。

### 38.5 int8 QGEMM：u8×s8→s32（GMAC/s，与 `tier_probe` 同形状、同零点）

| m×k×n | 我们 1T | MLAS 1T | 我们 16T | MLAS 16T |
|---|---:|---:|---:|---:|
| 16×320×320 | 86.7 | 98.1 | 86.7 ¹ | 21.7 ¹ |
| 16×320×1280 | 98.6 | 122.3 | **345.0** | 75.7 |
| 76×320×1280 | 97.3 | 159.6 | **862.3** | 311.9 |
| 512×1024×1024 | 90.8 | 169.8 | **1066.3** | **1067.1** |

¹ 这两格 16T 比 1T 还慢，是「工作量太小、池调度开销占主导」：MLAS 的单次调用开销约 60 µs，
我们的小形状直接落回单线程。

- **大形状 16T 完全打平**：`512×1024×1024` 双方 0.5035 vs 0.5031 ms（差 0.1%），
  连 `bench.md §34` 记的 ORT 侧 1005 GMAC/s 也对得上。
- 单线程差 1.87×，原因**不是实现质量**，是指令宽度：托管侧只有 256 位 `vpdpbusd`（4 行 × 24 列，
  §34），MLAS 是 AVX512 的 6 行 × 48 列。这条差距在 .NET 侧无解（§32 的结论仍成立）。
- 中小形状我们 16T 反而领先 2.8~4.6×，因为 MLAS 的 `ParallelFor` 在这些规模上调度开销盖过收益。

### 38.6 三条结论

1. **唯一系统性落后的轴是 F32 GEMM 单线程**（−20~30%），而且是纯内核效率：MLAS 已到单核 FMA 峰值的
   76~96%，我们在 62~79%。§26~§34 说的「F32 域已到顶」是**对 16 线程说的**，单线程还有这块。
2. **16 线程不是我们的短板**：SGEMM 多数形状领先 ORT 1.1~2.1×，int8 打平，NCHWc 卷积领先；
   真正要补的是 `n ≤ 200` 的并行切分（只切 N，§38.3 里 0.50~0.74 那几格）。
3. **yolo 的 1.28× 差距不在「内核不如 MLAS」**：把同一层的裸 Conv 拆出来比，我们的 NCHWc 单线程
   与 MLAS 同档、16 线程更好。与 §37.5 的图级结论同向 —— 差距在层数 × 固定开销与覆盖面。

### 38.7 顺带修掉 `convbench --threads`

`--threads` 以前只有 `GemmBench` 吃（设 `ParallelRange.MaxDegreeOfParallelism`），`ConvBench` 只打印
`threads : N` 却从不生效，单线程对照一直靠 `ONNXSHARP_THREADS` 环境变量。现在 `ConvBench` 同样读
`--threads`，与 `gemmbench` 口径统一（本节的 1T 列即用 `convbench --threads 1` 跑出）。

### 38.8 复现

```powershell
$py='E:\misedir\installs\python\3.11.17\python.exe'
$cli='tools\OnnxSharp.Cli\bin\Release\net10.0\OnnxSharp.Cli.exe'
cd .tmp/mlasbench

# F32 GEMM：ORT 单算子 MatMul（= MLAS 的真实口径）与直调 dll（仅内核参考）
& $py ort_matmul_bench.py      # -> ort_matmul.txt
& $py mlas_gemm_bench.py       # 直调 MlasGemmBatch / 含 QGEMM 与 packed

# Conv：MLAS 公共 NCHW MlasConv（脚本内置 3 组与朴素实现对拍，通过才计时）
& $py mlas_conv_bench.py       # -> mlas_conv.txt

# int8：MLAS QGEMM 对 tier_probe 的四个形状
& $py mlas_qgemm_probe.py

# 我们这侧（16 个形状与 ort_matmul_bench.py 一致）
& $cli gemmbench --threads 1  --iters 30 --shape 16,27,2240 --shape 32,144,1600 ... --shape 512,1024,1024
& $cli gemmbench --threads 16 --iters 30 --shape 16,27,2240 --shape 32,144,1600 ... --shape 512,1024,1024
& $cli convbench --threads 1
& $cli convbench --threads 16

# int8 我们这侧
cd ../tier_probe; $env:ONNXSHARP_THREADS='1';  dotnet run -c Release -f net11.0
                  $env:ONNXSHARP_THREADS='16'; dotnet run -c Release -f net11.0
```

峰值对照：`& $cli flopsbench --threads 1`（110.7 GFLOPS）/ `--threads 16`（866.6）。
改动未提交；`.tmp/mlasbench/*.py` 与 `*.txt` 为一次性脚本与原始输出。

## 39. 2026-11-18 首层 `ic=3` 直接卷积：AVX-512 版 1.6×，并修掉 `convbench` 的计时顺序偏置

§38.4 里 `3→16 640² k3s2` 是唯一「我们 1T（16.8）比 MLAS 1T（18.0）还慢、16T 更是 20.5 vs 123.4」
的层。原因不是内核写得差，而是**这层的形状不喂通用路径**：`K = 3×3×3 = 27`、`M = 16`，
im2col 要为 102400 个像素物化 `27×N` ≈ 11 MB 的 col 缓冲（写一遍读一遍 = 22 MB 搬运），
面板式把它摊进 4 行微内核也不省这 22 MB。所以单开一个内核。

### 39.1 内核形状：沿输出宽 16 像素 × 4 通道 × 4 像素块

`Kernels/FirstConv.cs`，与 `DepthwiseConv`/`NchwcConv` 同签名，自身判形状（`ic ≤ 3`、k3×3、
`pad=1`、`stride ∈ {1,2}`、单批次不分组），不满足直接返回 `false` 落回 im2col：

- **向量沿输出宽**：16 个像素正好一个 `zmm`，输出通道做展开。面板（27 条 × 4 块 = 6.7 KB）
  只建一遍常驻 L1，所有输出通道组共用。
- **步长 2 的取样**：一条面板 = 一次 32 元素读 + 一条 `vpermt2ps`（偶数位索引 `0,2,…,30`），
  比逐元素抽快 5 倍。
- **微内核 4×4**：4 个输出通道 × 4 个像素块 = 16 个累加器 + 4 个权重 + 4 条面板 = 24 个 `zmm`。
  关键是循环顺序 —— 一次 `vbroadcastss` 喂 4 条 FMA（1:4），而不是逐块循环里的 1:1
  （那样一半指令在广播）。实测 4T 124→131、16T 237→314 GMAC/s。
- **面板 64 字节对齐**：`stackalloc` 只保证 16 字节，手工 `((nint)p + 63) & ~63`；
  不齐时 512 位存取跨缓存行、store→load 转发不成立。
- **边界**：形状自洽判据是 `out = floor(in/2)` 或 `ceil(in/2)`（奇数边长也放行），
  行尾不满一个 tile 时走逐块 + `[start,end)` 无分支补零路径。
  对拍覆盖 `ic=1/2/3`、`s=1/2`、`w=70/71`（tile 尾巴）、`h=33/65`（奇数行）：`max|d| ≤ 2.4E-007`。

### 39.2 读数（`convbench --iters 200 --shape 3,16,640,640,3,2,1`，与 im2col 交替计时）

| 线程 | im2col | packed | **firstc** | firstc/im2col |
|---|---:|---:|---:|---:|
| 1 | 2.15 ms (20.6) | 2.40 (18.4) | **1.35 (32.8)** | **0.63** |
| 4 | 0.56 (78.3) | 0.62 (71.6) | **0.34 (131.9)** | **0.61** |
| 16 | 0.23 (192.9) | 0.29 (153.7) | **0.14 (314.2)** | **0.61** |

括号内 GMAC/s。对峰值（`flopsbench`：单核 110.7 GFLOPS = 55.4 GMAC/s，16 线程 866.6 = 433）：
1T 32.8 = **59%**（im2col 只有 37%），16T 314 = **72%**；1→16 线程 **9.6×** 扩展（不是早先量到的「3.8×」）。
`max|d| = 0.0E+000`，与 im2col 逐位一致。

### 39.3 顺带修掉的方法学问题：`convbench` 的计时顺序偏置

上面第一版量出来是 1T 1.44 / 4T **0.70** / 16T 0.37，即「1T 领先 1.5×、4T 反而慢 20%、16T 只扩展 3.8×」，
并据此先后否掉了两种布局（把通道放到最外层、把广播提到 tile 层）。**这些结论全是假的。**

`ConvBench` 是「im2col 连测 150 次 → firstc 连测 150 次」。本机是功耗受限的 APU，
firstc 用满 512 位 FMA、功耗最高、降频最狠；等它上机时 CPU 已经被 im2col 烤热，
150 次迭代（~100 ms）全落在低频区，`min` 也取不到凉态样本 —— **恰好把这个内核的读数砍半**。
把两个内核改成**轮流各跑一次**（新增本地函数 `BestRound`）后，同一份二进制：

| 4 线程 | 改前（先后计时） | 改后（交替计时） |
|---|---:|---:|
| im2col | 0.57~0.59 ms | 0.56 ms |
| firstc | 0.70~0.72 ms | **0.34 ms** |

同一个教训在 §19.5 / §33 已经吃过一次（`run_all.py` 也是改成交替）。**凡是同一进程里比两个内核，
必须交替计时，否则读数只反映先后顺序。**

### 39.4 端到端：落在噪声里

```powershell
$cli ab F:/ai/models/yolo/yolo11n.onnx --input bench/inputs/i1x3x640x640.f32 `
     --a firstconv=0 --b firstconv=1 --iters 20 --rounds 6 --warmup 5
```

`max|d|=0.00E+000`（705600 个输出全等）；min 22.1 vs 22.2 ms，配对中位 1.012，而当场噪声地板 31.8%
→ **判不出**。按 39.2 的账，这层 4 线程省 0.22 ms / 22 ms ≈ **1%**，本来就在噪声地板以下；
单元测试（`116/118`，两套 TFM）与 `yolo11n: end-to-end matches onnxruntime golden` 都通过。

### 39.5 复现与开关

```powershell
$cli convbench --threads 1  --iters 200 --shape 3,16,640,640,3,2,1
$cli convbench --threads 4  --iters 200 --shape 3,16,640,640,3,2,1
$cli convbench --threads 16 --iters 200 --shape 3,16,640,640,3,2,1
```

`DnnOps` 里已**无条件**调用（内核自己判形状），开关是 `ONNXSHARP_FIRSTCONV=0`，
`ab`/`--a`/`--b` 认 `firstconv=`（`AbFlags` 表新加一行）。需要 AVX-512（`Avx512F.IsSupported`）。

## 40. 2026-11-18 车牌专用 OCR（受限解码 + 形态校验）与 yolo26s

三件事都不是「换个模型」：OCR 侧是**在既有 PaddleOCR 流水线上做算法特化**（外加一个
「车牌裁剪图直读」入口），检测侧是**新模型形状（480×640 非方形）**的通用化。

### 40.1 为什么不能直接复用通用 CTC 解码

通用 `CtcDecoder` 在 6906 类上做前缀 beam：每帧把全部类别排序取 top-k。车牌字符集只有
`31 省 + 24 字母 + 10 数字 + 6 后缀` 约 70 类，剩下 6800 多类对结果只能是噪声 ——
`S`/`5`、`B`/`8`、`0`/`O` 这类同形竞争会白吃 beam 宽度，还会让 beam 里堆满永远不会成立的字符。
所以特化全在「先砍类别、再按位置约束、最后用形态兜底」：

| 层 | 做法 | 文件（`src/OnnxSharp/Ocr/Plate/`） |
|---|---|---|
| 字符集 | `PlateCharset`：rank ↔ 字符；`AllowedAt(position, length)` 逐位规则（省 / 字母 / 数字 / 后缀，字母与数字排除 `I`、`O`） | `PlateCharset.cs` |
| 概率 | 只把字符集内 ~70 类取 `log`（blank 单算），自动判 softmax / 原始 logits | `PlateDecoder.BuildLogProbabilities` |
| 解码 | CTC 前缀 beam（log 域 Pb/Pnb 两状态）；前缀打包进 `long`（每字符 7 位 rank + 高 8 位长度），非法长度直接剪 | `PlateBeamSearch.cs` |
| 置信度 | 对最终序列做 CTC Viterbi 对齐（2L+1 状态）取**逐字后验**，几何平均 | `PlateAlignment.cs` |
| 形态 | `PlateNumber.TryParse`：7 位燃油 / 8 位新能源 / 尾字 学·警·挂·港·澳 / 首字 使·领，逐位校验后判类型 | `PlateNumber.cs` |
| 流水线 | `PlateReader`：DB 检测 → 宽高比筛（1.6~6.5）→ 逐框受限解码 → 按分排序取 top-N | `PlateReader.cs` |

`PaddleOcrPipeline` 为此拆出两个内部入口 `DetectBoxes`（返回**已缩放到源图**的框）与
`RecognizeLogits`（返回 `[1,T,C]` 原始 logits），通用 `ocr` 与 `plate` 共用，没有第二份预处理代码。

### 40.2 实测：3200×1800 真实图（4 线程，`--limit 960`）

```powershell
$cli plate F:/ai/20260828093055568_121.jpg --det .tmp/ocr-models/det.onnx `
     --rec .tmp/ocr-models/rec.onnx --dict .tmp/ocr-models/dict.txt --limit 960
```

| 输出 | 分数 | 类型 | 框 |
|---|---:|---|---|
| `浙AES8381` | 97.6% | NewEnergy（8 位） | 1755,203 187×64 |
| `新A3M15X` | 62.9% | Fuel（7 位） | 1766,1478 212×85 |

逐字后验直接指出弱字，不用再猜「为什么这张不准」：

```
浙 91% A 100% E 100% S  98% 8 100% 3 100% 8 94% 1 100%
新 17% A  44% 3 100% M  94% 1  68% 5 100% X 81%        ← 低分框，肉眼也糊
```

整图端到端 **825 ms**（`load` 7 ms；DB 在 960×540 上跑，检测主导），输出 2 张。

**三种入口**（决定「要不要先找车牌」）：

| 入口 | 场景 | DB 检测 | 单位 |
|---|---|---|---|
| 默认 | 整图，自己找车牌 | 跑 | 整图 |
| `--crop x,y,w,h` | 上游给了框，框在整图里 | 跳过 | 单块 |
| `--direct` | 上游已把车牌抠成一张图，输入就是车牌 | 跳过 | 整图（即那块） |

`tests/Image/` 的两张纯车牌图（`--direct`）：

| 图 | 输出 | 分数 | 类型 | rec best（beam 8，n=20） |
|---|---|---:|---|---:|
| `仅车牌1.png` 135×46 | `浙ABM770` | **100.0%** | Fuel | 4.45 ms |
| `仅车牌2.png` 175×49 | `浙ACP5872` | **99.6%** | NewEnergy | 7.45 ms |

同一张图走默认通路（DB 先找行再认）是 99.9% / 99.3%：`--direct` 不只省掉一次 DB 前向，
分数还略高 —— DB 的框会把车牌边缘裁掉一点。首次调用 0.2~0.3 s 是 rec 图的导入（形状特化），
稳态才是上表的毫秒数。

反向验证：把 3840×2160 的整图直接喂给 `--direct`（`result : 0 张`）——形态校验 +
逐字几何平均分把它拦住了，不会因为「用户选错通路」而吐出一个假号牌。

单框稳态（`--crop 1755,203,187,64 --iters 20`，交替跑 3 轮）：

| 模式 | rec best | 说明 |
|---|---:|---|
| beam 8 + 结构约束 | 3.63 ~ 5.11 ms | 默认 |
| greedy（`--greedy`） | 4.36 ~ 5.02 ms | **与 beam 分不开** |

同一框在同一进程里只有单模式，跨进程交替读数被降频抹平（`avg` 在 17~30 ms 间跳），
所以**不据此宣称 greedy 更快** —— 要这个结论得给 `plate` 加进程内交替计时（同 §39.3）。
首次调用约 280~330 ms 是 rec 图的导入（形状特化），稳态才是上表。

### 40.3 一张图同时验证三件事（`tests/Image/20260828092321383_121.jpg`，3840×2160）

两辆车、两块蓝牌，正好是三条通路的共用样例：

| 通路 | 命令要点 | 结果 | 耗时 |
|---|---|---|---:|
| yolo26s 检测 | `detect F:\ai\models\yolo26\yolo26s.onnx <img> --conf 0.35 --iters 10` | 2×`car` 0.9049 / 0.8912 | forward 44.93 ms，decode 0.78 ms |
| 车牌（默认，走 DB） | `plate <img> --det ... --rec ... --dict ... --limit 960` | `浙ABM770` 83.5% Fuel、`浙ACP5872` 82.7% NewEnergy | 964 ms |
| 普通 OCR | `ocr <img> --det ... --rec ... --dict ... --limit 960` | `浙A BM770` (0.857)、**`H浙ACP5872`** (0.834) | 617 ms |

三条通路正好互补：

- **yolo26s**：3840×2160 缩 0.1667× 到 640×480、上下各 pad 60，出的 `car` 框换算回原图坐标
  （1951.8,1055.4 / 1831.6,1.8），和车牌框位置对得上。
- **普通 OCR 多认一个字**：第二块牌出来 `H浙ACP5872` —— DB 框把号牌左侧的贴字/边框带了进来。
  车牌路径的受限解码把它剪掉了（`H` 不在逐位允许集里），输出干净的 `浙ACP5872`。
  这就是「字符集 + 逐位规则」最直接的一次收益。
- **车牌分数低于 §40.2 那张图**（83.5% vs 97.6%），逐字后验直接指出原因：省字只有 **33% / 24%**，
  后 7 位都在 90~100%。3840 宽图上 DB 给的框是 258×110（真实号牌约 3.1:1），省字被压窄，
  **是检出框精度问题而不是解码问题** —— 换成上游给框（`--crop`）/ `--direct` 就回到 99%+。

### 40.4 yolo26s：非方形输入

模型 `F:\ai\models\yolo26\yolo26s.onnx`（38 MB）。输出布局仍是 `[1,84,N]` / `[1,N,84]`，
`84 = 4 + 80` 由字段数反推为 COCO，因此 `detect` 不必再传 `--labels coco`：

```powershell
$cli detect F:\ai\models\yolo26\yolo26s.onnx F:/ai/20260828093055568_121.jpg --conf 0.4 --iters 5
```

```
input      : [1, 3, 480, 640]  (3200x1800 -> 640x480 (scale 0.2000x0.2000, pad 0,60))
preprocess : 2.12 ms
forward    : 55.33 ms  (min of 5)
decode     : 0.92 ms
detections : 2   car 0.9103 [1497.0, 0.3, 701.5x298.5] / car 0.8726 [1486.2, 588.5, 780.0x1084.2]
```

- **新形状**：yolo26s 输入是 480×640（yolo11 是 640×640），stride 8/16/32 下锚点
  `60×80 + 30×40 + 15×20 = 6300`。解码按输出形状自适应，letterbox 的 pad 是 `(0, 60)`，
  坐标反算回原图仍准（检测框贴图像上下边）。
- **标签推断**：以前 `detect` 把「第二维 = 84 ⇒ COCO」当路径判断，现在在首次 `Forward` 之后
  由输出形状推断 `yoloFields`，`--labels coco` 变成可选。
- **正确性**：新增 `yolo26s: end-to-end matches onnxruntime golden`
  （golden 由 `tools/scripts/gen_golden.py` 走 ORT 生成，逐元素容差 2e-2，输入 `[1,3,480,640]`）。
  `YoloGoldenTests` 参数化成 `Matches(model, goldenPrefix, inputShape, inputName)`，yolo11 与 yolo26 共用一套对比逻辑。

### 40.5 顺带修掉的东西

- 新增 `PlateReader.ReadWhole(...)` + `plate --direct`：整张图就是车牌时跳过 DB，
  这是「上游检测器先抠车牌、再把裁剪图丢进来识别」的正式入口。
- `plate` 的 `--box` 与 `ocr` 的 `--box`（DB 框分数阈值）撞名：单框入口改名 **`--crop x,y,w,h`**，
  `--box` 恢复为阈值语义（`--crop` / `--direct` 与 `--iters` 一起用才计时）。
- `Ocr/PaddleOCR/DbPostprocess.cs`、`PaddleOcrModel.cs` 的注释是 GBK 字节被当 UTF-8 读出来的乱码
  （原字符已不可逆），按语义重写为 UTF-8 中文；代码一行未动。
- 主帮助补上 `plate` 子命令（之前只在 `plate` 自己的 usage 里）。

### 40.5 回归与复现

- `dotnet build OnnxSharp.slnx -c Release`：0 警告 0 错误。
- 单测两套 TFM：`122/124 passed, 2 skipped, 0 failed`，其中新增 `PlateOcrTests` 5 例
  （字符集/rank 唯一性与逐位规则、号牌解析、受限解码屏蔽字符集外赢家、beam 比贪心多保一字、非法长度被拒）。

本节命令里 `tests\Image\` 下的三张图（`20260828092321383_121.jpg`、`仅车牌1.png`、`仅车牌2.png`）
是本地素材，尚未纳入 git；换成任意等价图片即可复现。

```powershell
$cli = 'tools\OnnxSharp.Cli\bin\Release\net11.0\OnnxSharp.Cli.exe'
& $cli plate <image> --det .tmp/ocr-models/det.onnx --rec .tmp/ocr-models/rec.onnx `
     --dict .tmp/ocr-models/dict.txt --limit 960 --json out.json
& $cli plate <image> ... --crop 1755,203,187,64 --iters 20        # 单框稳态（beam 8）
& $cli plate <image> ... --crop 1755,203,187,64 --greedy           # 贪心对照
& $cli plate <image> ... --no-structure --loose --score 0.3        # 关逐位约束 / 关形态校验 / 放宽阈值
& $cli plate tests\Image\仅车牌1.png ... --direct --iters 20       # 输入本身就是一张车牌
& $cli detect F:\ai\models\yolo26\yolo26s.onnx <image> --conf 0.4 --iters 40
& $cli detect F:\ai\models\yolo26\yolo26s.onnx tests\Image\20260828092321383_121.jpg --conf 0.35
& $cli ocr tests\Image\20260828092321383_121.jpg --det ... --rec ... --dict ... --limit 960
```

## 41. EfficientSAM3（EV-M）：三个 ONNX 全部跑通

`F:\ai\models\esam3\` 下的 `efficientsam3_ev_m_{image_encoder,text_encoder,decoder}.onnx`
是 SAM3 的提示式分割三件套。本轮把它们全部跑通、逐张量与 onnxruntime 对齐，并修掉路上撞到的三个内核 bug。

### 41.1 张量契约

| 模型 | 输入 | 输出 |
| --- | --- | --- |
| image_encoder | `image` F32 `[1,3,1008,1008]`（RGB/255、ImageNet mean-std、NCHW） | `fpn_0` `[1,256,288,288]`、`fpn_1` `[1,256,144,144]`、`fpn_2` `[1,256,72,72]`、`pos_0/1/2` `[1,256,H,W]` |
| text_encoder | `tokens` I64 `[1,16]` | `language_features` `[16,1,256]`、`language_padding` `[1,16]`、`language_embeds` `[16,1,512]` |
| decoder | `fpn_0/1/2`、`pos_2`、`language_features/padding`、`exemplar_boxes` `[8,1,4]`(归一化 `cxcywh`)、`exemplar_labels` I64 `[8,1]`、`exemplar_mask` F32 `[1,8]` | `pred_logits` `[1,200,1]`、`pred_boxes` `[1,200,4]`、`pred_masks` `[1,200,288,288]`、`presence_logit` `[1,1]` |

decoder 的 200 条 query 里靠 `pred_logits` + `presence_logit` 挑赢家，`pred_masks` 是 288×288 的低分辨率 logits。

### 41.2 数值对齐（同输入 vs onnxruntime）

| 模型 | 输出 | max&#124;d&#124; | 相对量级 |
| --- | --- | --- | --- |
| image_encoder | `fpn_0` / `fpn_1` / `fpn_2` | 6.3e-7 / 2.1e-5 / 2.3e-5 | 3e-6 |
| | `pos_0/1/2` | 6.0e-8 | 6e-8 |
| text_encoder | `language_features` | 4.1e-6 | 1e-6 |
| | `language_padding` / `language_embeds` | 0（逐字节一致） | — |
| decoder | `pred_logits` / `pred_boxes` / `presence_logit` | 6.2e-5 / 8.7e-6 / 3.8e-6 | 3e-5 |
| | `pred_masks` | 2.0e-3（值域 ±32） | 6e-5 |

差异只来自累加顺序（GEMM 分块、Winograd、多线程切分），没有算子语义偏差。

### 41.3 修掉的三个 bug（外加一处 20× 提速）

- **`FirstConv` 末块写穿行尾**（`src/OnnxSharp/Kernels/FirstConv.cs`）：满 tile 分支用 `nb == Blocks` 判断，
  但 1008/2 = 504 = 7×64 + 56，末块只剩 56 个像素仍走 4 次 16 宽 Store → 越界写到下一行开头（w=0..7 被污染）。
  改成 `remaining >= Tile` 后 first conv 输出 `max|d| = 0`，image encoder 才跟上 ORT。
- **`ConvTranspose` 相位 GEMM 散写**（`Ops/DnnOps.cs`）：`dstCo = yb + co*outPlane + kh` 把行偏移 `kh`
  当列偏移加了，少乘 `outW`；`fpn_0/fpn_1` 87%/99% 元素错、`max|d|` 到 7.3。改成 `+ (long)kh * outW` 后归零。
- **广播外积 Add 的并行分段**（`Tensors/TensorOps.cs`）：SAM3 的相对位置外积
  `[1,200,72,1,8] + [1,200,1,72,8]` 以前掉进通用循环（每元素维护 5 个索引，5 ns/元素，8.3 M 元素 → 40 ms）。
  新增 `TryApplyStridedRun`：从最内维往外把「两侧都还是紧凑步长」的轴并成一段（这里段长 8），
  段内走 AVX、段外只推指针，并外层段并行 —— 1.2 ms（30×）。
  第一版并行块只算了起始偏移、没同步「掉电计数器」，除首块外第一次回绕就错位；已修，
  并补两个单测（`ElementwiseBroadcastTests`：串行段 / 跨并行阈值段）。
- **`Einsum`（mask 预测 `einsum('bqc,bchw->bqhw')`）**：原来按 query 逐个标量累加，
  200 条 query 每条都把 85 MB 的 `B` 完整重读一遍（17 GB 流量，780 ms）。
  `out[q][p] = Σ_c A[q][c]·B[c][p]` 本来就是一次 `[200,256]×[256,288²]` 的 GEMM，
  改成 `GemmKernel.Multiply` 后 40 ms 量级，decoder 整体 5.0 s → 2.9 s。

### 41.4 性能

同一会话交错测、取最小值。**16 线程**（默认 = 物理核数，§41 这批数据没有按 4 线程重测）。
注意这台机器（Ryzen 9 8945HX，16C/32T）本身有 ~50% 后台负载，绝对值波动 ±50%，只能看量级与相对比值。

| 模型 | OnnxSharp | onnxruntime | 比值 |
| --- | --- | --- | --- |
| image_encoder | 0.77–2.2 s | 0.44–1.4 s | 1.5–1.8× |
| text_encoder | 6.4 ms（20 轮稳态） | 2.9–4.2 ms | ~2× |
| decoder | 2.85–3.30 s | 2.30–3.04 s | 1.1–1.3× |

单算子微基准（同输入、单独建模型，比整网更能说明内核水平）：

| 层 | 形状 | OnnxSharp | onnxruntime |
| --- | --- | --- | --- |
| ConvTranspose 2×（dconv，1024→512） | `[1,1024,72,72]→[1,512,144,144]` | 121 ms | 41 ms |
| ConvTranspose 2×（dconv，512→256） | `[1,512,144,144]→[1,256,288,288]` | 77 ms | 108 ms |
| Conv 3×3 s1 p1（256→256） | `[1,256,288,288]` | 162 ms | 528 ms |

结论：3×3 卷积（Winograd）和通道多的 2× 上采样已经比 ORT 快；1024→512 这层的相位 GEMM 还差 ORT 3 倍
（`m = 512` 时 GEMM 走行切分，`B` 要按行块重复打包），是下一步最值得动的地方。

#### 41.4.1 交错 A/B 原始值（同一会话 4 轮，`.tmp/esam3_ab.py`）

每轮先让 ORT 跑 1 次，再起一个 CLI 进程跑 `bench --iters 5` 取 min，两家在同一轮里挨着测：

| 模型 | OnnxSharp 每轮 ms | 最佳 | onnxruntime 每轮 ms | 最佳 | 最佳值比值 |
| --- | --- | --- | --- | --- | --- |
| image_encoder | 1136 / 1528 / 1469 / 1517 | **1136** | 551 / 1191 / 1499 / 1735 | **551** | 2.06× |
| text_encoder | 13 / 37 / 8 / 12 | **8** | 4 / 5 / 4 / 3 | **3** | 2.3× |
| decoder | 3880 / 3761 / 5150 / 3880 | **3761** | 2983 / 4124 / 4250 / 4879 | **2983** | 1.26× |

image encoder 第 1 轮 1136 ms、第 4 轮 1517 ms，ORT 同一轮从 551 变 1735 —— 两家一起变慢，说明波动来自后台负载
而不是实现本身；所以「谁快」只看同一轮的相对关系，绝对毫秒不能当基线。

#### 41.4.2 profile 热点（share 按 sum-of-min）

`profile` 只拿来看**排序**：同一条命令两次跑，share 能差 3~4 个百分点（负载在变），绝对值更不能当基线。

```
# image_encoder，--iters 5（254 个 op；mean 2724.7 ms/iter，sum-of-min 2328.8 ms/iter）
Conv            81 ops   1091.7 ms   46.9%     # MBConv 的 3×3 / 1×1
ConvTranspose    3 ops   1060.2 ms   45.5%     # 三次 2× 上采样
HardSwish       30 ops    104.0 ms    4.5%     # 每个约 3.5 ms
Slice 35 / Pad 7 / Concat 7 / Transpose 7 / MatMul 14 / Add 25 / FusedEltwise 2  全部 ≤0.8%
单点：/convs.0/dconv_2x2_0/ConvTranspose 425.3 ms、/convs.0/conv_3x3/Conv 412.0 ms、/convs.1/dconv_2x2/ConvTranspose 320.7 ms
```

```
# decoder，--iters 3（2015 个 op；mean 5435.7 ms/iter，sum-of-min 4578.4 ms/iter）
MatMul                266 ops   2337.4 ms   51.1%
Conv                    4 ops    686.7 ms   15.0%
Softmax                36 ops    484.9 ms   10.6%
Gemm                   37 ops    290.3 ms    6.3%
Transpose             240 ops    264.4 ms    5.8%
InstanceNormalization   2 ops    119.6 ms    2.6%
Einsum                  1 op      52.8 ms    1.2%
单点：/segmentation_head/pixel_decoder/conv_layers.1/Conv 495.8 ms（10.8%）
      /segmentation_head/pixel_decoder/conv_layers.0/Conv 135.7 ms
      /encoder/layers.{1,2,4,5}/self_attn/MatMul_3  各 123–149 ms（6 层合计约 0.8 s）
```

- image encoder 92% 在卷积/反卷积，已经是本库最强的两条内核；剩下的 HardSwish 104 ms（30 个 op）
  可以并进前一层 Conv 的 epilogue。
- decoder 只有 4 个 Conv 却占 15%（`conv_layers.*` 是 256→256 @288²，正是微基准里我们比 ORT 快的那档）；
  MatMul 里约 1/3 是 6 层自注意力的 QK^T，其余是逐 query / 逐头的小矩阵乘；`Softmax` 485 ms 基本也在注意力里；
  mask 的 `Einsum` 已从 780 ms 降到 53 ms（§41.3）。
- profile 的 sum-of-min（4578 ms）比 `bench --iters 5` 的 min（3761 ms）大 20%：profile 把每轮 op 计时串行累加，
  还带计时开销，两个口径不能互相换算。

### 41.5 端到端：框提示分割

yolo26s 检出车辆框（归一化 `cxcywh`）→ 填进 `exemplar_boxes[0]`、`exemplar_labels[0] = 1`、`exemplar_mask[0] = 1`：

```
text    : "car"
presence: 0.9998
best query 157 logit 3.2821 score 0.9638
mask area 11.04%   bbox 1960 1073 2959 2159   (提示框 bbox IoU 0.955)
mask IoU (ours vs ORT, 288x288): 1.00000
prob max|d| = 2.480e-05  mean|d| = 1.194e-06
```

整条流水（image encoder → text encoder → decoder）全部用本库跑，decoder 吃的是**我们自己**的
`fpn_*` / `pos_2` / `language_*`，没有借 ORT 的中间结果；可视化 `.tmp/esam3/demo/ours_mask_vis.png`
掩码正好盖住下半辆车。

### 41.6 回归与复现

- `dotnet build OnnxSharp.slnx -c Release`：0 警告 0 错误。
- `dotnet run --project tests/OnnxSharp.Tests -c Release -f net10.0`（以及 `net11.0`）：
  `124/126 passed, 2 skipped, 0 failed`（新增的就是上面两个广播段用例）。
- golden 与 ORT 参考脚本都在 `.tmp/`（`esam3_ref.py`、`esam3_cmp.py`、`esam3_demo*.py`、
  `esam3_bench_ort.py`、`esam3_ab.py`），按需重跑；模型路径用 `F:\ai\models\esam3`。

```powershell
$cli = 'tools\OnnxSharp.Cli\bin\Release\net10.0\OnnxSharp.Cli.exe'
$P   = '.tmp\esam3\demo'

& $cli run F:\ai\models\esam3\efficientsam3_ev_m_image_encoder.onnx --input image=$P/image.f32 --dump $P/ours_img
& $cli run F:\ai\models\esam3\efficientsam3_ev_m_text_encoder.onnx  --input tokens=$P/tokens.i64 --dump $P/ours_txt

& $cli run F:\ai\models\esam3\efficientsam3_ev_m_decoder.onnx `
      --input fpn_0=$P/ours_img/out0.f32 --input fpn_1=$P/ours_img/out1.f32 --input fpn_2=$P/ours_img/out2.f32 `
      --input pos_2=$P/ours_img/out5.f32 `
      --input language_features=$P/ours_txt/out0.f32 --input language_padding=$P/ours_txt/out1.f32 `
      --input exemplar_boxes=$P/exemplar_boxes.f32 --input exemplar_labels=$P/exemplar_labels.i64 `
      --input exemplar_mask=$P/exemplar_mask.f32 --dump $P/ours_dec
```
