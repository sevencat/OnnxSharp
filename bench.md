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
