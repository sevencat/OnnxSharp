# OnnxSharp 优化策略 v2

> **文件分工**：`plan.md` 管「实现什么功能」，`bench.md` 管「现在多快」，本文件管
> **「接下来怎么更快、按什么顺序、怎么判定有效」**。
> 本文件取代 `plan.md §M4` 的「未完成 / 重新排序」清单 —— 那份基于 16 线程口径
> 且把差距归因给 GEMM，已被 `bench.md` 第八轮实测推翻。

- 立项：2026-11-15
- 口径：**4 线程**、FP32、batch=1、静态形状、只计时模型前向
- 基线：`bench.md` §8.3（4 线程全表）
- 进度：`progressv2.md`

---

## 1. 先把测量做对（这一节是前提，不是附录）

历史教训已四次出现「拿噪声当结论」，v2 的每一步都必须过这一关。

### 1.1 环境门控（新增，强制执行）

本机电源方案把 `PROCTHROTTLEMIN` 设为 100%，16 线程满载时 `% Processor Performance`
稳定掉到 **21%**，耗时随机涨 5~7 倍（详见 `bench.md §8.1`）。

```powershell
# 开跑前、跑完后各看一次；< 60% 本轮作废，60~80% 只信 min
(Get-Counter '\Processor Information(_Total)\% Processor Performance').CounterSamples.CookedValue
```

永久修复（需管理员，一次性）：`powercfg /setacvalueindex SCHEME_CURRENT SUB_PROCESSOR PROCTHROTTLEMIN 5`。

### 1.2 判定规则（沿用并收紧）

| 场景 | 方法 |
|---|---|
| 端到端改动 | 同会话交替 A/B，≥ 5 轮，取各自**最好 min**；同时报 median 与 p95 |
| 预期收益 < 2% | 只认 `min` 或逐算子 `profile`，median 不可用（±8% 抖动） |
| 任何改动 | 4 模型 `max|diff| = 0`（`run --dump` + `cmp`），FP32 必须逐位一致 |
| 独立微基准变快 | **不作为采纳依据**。历史三次「微基准 +45%、端到端 0」（`plan.md §9.1` 末三条） |

### 1.3 已否决、不要再做的事

- 整个 B 预打包给所有行块复用（304 → 234 GFLOPS）
- 微内核 k 方向展开、行块高度扫参、按流量改切分方向（端到端全为 0）
- 更聪明的逐元素融合（已贴 42 GB/s 带宽墙）
- GridSample 单算子再优化（端到端仅 0.6%）
- **对齐 ORT 的图融合**：实测 ORT 关掉全部图优化反而快 2.6%（`bench.md §8.5`），
  融合对 RF-DETR 是负收益，不是我们的欠账

---

## 2. 现状：差距已经从 GEMM 转移到 GEMM 之外

### 2.1 4 线程下的绝对位置（median，ms）

| 模型 | 我们 | ORT | 倍率 | 历史倍率（旧机 8845HS / 16 线程，**不可比**） |
|---|---:|---:|---:|---:|
| yolov8n | 46.0 | 28.3 | 1.63 | 1.18 |
| yolo11n | 39.1 | 23.6 | 1.66 | 1.17 |
| yolov8s | 121.0 | 76.7 | 1.58 | 1.32 |
| yolo11s | 111.7 | 65.1 | 1.71 | 1.37 |
| rfdetr-small | 262.5 | 189.4 | 1.39 | 1.18 |

**线程敏感性才是真问题**：16 → 4 线程，ORT 的 yolo11n 从 23.4 → 23.6（几乎无感，早已撞带宽墙），
我们从 27.4（实为 32 线程）→ 39.1。**我们的并行效率依赖线程数，ORT 不依赖。**

### 2.2 差距分解（rfdetr-small，4 线程；Conv/GEMM 为微基准实测，其余为折算）

| 类别 | 我们 | ORT | 差 | 数据来源 |
|---|---:|---:|---:|---|
| GEMM 类（MatMul+Gemm） | ~180 | 150 | **+30** | `gemmbench` 实测 6 类主形状 157 ms，占 86% |
| Conv（9 层） | **11.2** | 8 | **+3** | `convbench` **实测**（原折算 27，已废止） |
| 逐元素 + 激活（Mul/Add/Div/Relu/Silu） | ~32 | 4.5 | **+27** | `profile` 折算 |
| Transpose（111 个） | ~11 | 5 | **+6** | `profile` 折算 |
| Where（1 个） | ~5 | 0 | **+5** | `profile` 折算（ORT 已折叠掉） |
| GatherElements | ~2.5 | 0.6 | **+2** | `profile` 折算 |
| LayerNorm / Softmax / GridSample / Reshape 等 | ~20 | 14 | **+6** | `profile` 折算 |
| **合计** | **~262** | **189** | **+73** | |

⇒ **修正后的两条结论**（取代本节初版的「GEMM 已打平」）：

1. **GEMM 仍是最大单块（69%），但微内核已跑到 290~313 GFLOPS = 4 核 AVX2 峰值的 70~74%**，
   不是「小 GEMM 没优化」。剩下的 26% 只能靠 AVX-512 微内核或更大分块，属独立大项（P6'）。
   > 2026-11-15 更新：实测峰值是 427 GFLOPS（`flopsbench`），MR=6 之后未打包 75.6% /
   > 打包 **83.5%**，端到端 -5.0%。见 §4「P6'」与 `bench.md §12`。
2. **Conv 只有 11.2 ms（4.3%），与 ORT 的 8 ms 相差 3 ms** —— 初版把 `profile` 占比
   直接乘端到端，得出 27 ms，是错的（`profile` 回调开销按层数摊，MatMul 几百层 vs Conv 9 层）。
   已据此否决原 P1。

### 2.3 为什么这不是「再补一次预打包」的问题

ORT 快的根源（`bench.md §8.5` 已验证）：`MatMul::PrePack` 在 session 初始化时把常量权重
打包一次，与图优化等级无关；再加上 memory pattern 与常驻线程池。

**这三样我们都有，但第三项此前是坏的（2026-11-15 修复）**：

| ORT 机制 | 我们的等价物 | 位置 | 状态 |
|---|---|---|---|
| MatMul PrePack | `GemmKernel.PackedB` + `MultiplyPacked` + 按权重实例缓存 | `Tensors/GemmKernel.cs:192-330`、`Ops/DnnOps.cs:1868-1880, 2011-2023` | 等价 |
| Conv 权重打包 | `PackedConv.Cache`（`ConditionalWeakTable<Tensor, PackedWeights>`） | `Kernels/PackedConv.cs:43, 154` | 等价 |
| 常驻 intra-op 池 | `ParallelRange` 自旋领任务 | `Threading/ParallelRange.cs` | 等价 |
| memory pattern（内存复用规划） | `NetImpl._slots` + `Tensor.Fit(reuse:true)` | `Tensors/Tensor.cs:73-106` | **曾有 bug，已修** |

**第三项的实情**：复用机制本身是有的（槽位池 + `Fit` 容量判断），但 `Fit` 把
`DType != dtype` 也当成了重分配理由，而 `Buffer` 是 `byte[]`、`DType` 只是解释视图。结果是
共享槽位被 I64 小张量与 F32 大张量轮流使用时，小张量把 25 MB 缓冲缩成 2.4 KB，下一帧再重分配。
rfdetr-small 因此每帧白分配 **54.8 MB → 1.4 MB**、`Fit` 重分配 **19 次 → 0**、gen2 GC **5 → 0**
（详见 `bench.md §10`；输出逐字节等价，`gc pause` 仅 0.1~0.2 ms，故端到端在噪声内）。

⇒ v2 **仍不做**「预打包」课题；「内存复用」课题也**不再新开** —— 已在本轮关闭。
但它同样**不计入阶段 A 的收益**：这是消除 GC 抖动隐患，不是提速手段。

---

## 3. 目标（4 线程口径，2026-12 前）

初版目标按「Conv 27 ms」定的（补掉 Conv/逐元素/Transpose 各 60% → rfdetr 210）。
**Conv 实测只有 11.2 ms 后，那 60% 最多值 2 ms**，目标必须拆成两档：

| 模型 | 当前 | 阶段 A：不动 GEMM（P1'+P2+P3+P4） | 阶段 B：含 GEMM 微内核（P6'） |
|---|---:|---:|---:|
| rfdetr-small | 262.5 | **≤ 228**（-13%） | **≤ 200**（-24%） |
| yolo11n | 39.1 | ≤ 34（-13%） | **≤ 28**（-28%） |
| yolov8n | 46.0 | ≤ 40 | ≤ 34 |
| yolov8s | 121.0 | ≤ 106 | ≤ 93 |
| yolo11s | 111.7 | ≤ 98 | ≤ 85 |
| 对 OpenCV dnn | 0.80~0.87× | 保持领先 | 保持领先 |

阶段 A 的依据：逐元素 -20、Transpose -6、Where -5、Conv -2 ≈ -33 ms（rfdetr 262.5 → 229）。
阶段 B 的依据：GEMM 从 70~74% 效率提到 85%+，180 → 148 ms（-32）。
**YOLO 侧逐元素占比更低、GEMM 占比更高，所以阶段 B 才是 YOLO 的主要收益来源。**

---

## 4. 优化项（按 收益 ÷ 风险 排序）

### P0 测量基线冻结（先做，0.5 天）

- [x] `run_all.py` / `run_python_bench.py` / `ort_inspect.py` 默认线程改 4
- [x] 修 `run_all.py` 的 `env` 未传递 bug（`ONNXSHARP_THREADS` 从未生效）
- [ ] 每次基准在 `bench/results/latest.json` 里记 `perf%`（跑前/跑后各一次）
- [ ] 4 线程基线冻结到 `bench.md §8.3`，后续所有对比都对齐它
- [x] **补回 A/B 比对脚本**：新建 `bench/ab_compare.py`（放 `bench/` 下，不被 `results/` 的 ignore 波及）。
      现场跑两次 dump 逐元素比对，或 `--a/--b` 比对已有目录；输出 `max|diff|`、`max_rel`、首个差异下标，
      **exit code 0/1 可直接做提交前门禁**。自检（同二进制两跑）已验证 yolo11n 逐位一致
- [x] `run_all.py` 增加 SimdYolo 线程数断言：解析 JSON 里的 `CpuThreads`，与请求值不符直接抛错
      （防止再出现「参数没生效、静默按 32 线程跑」这类事故，onnxsharp 就栽过一次）
- [x] **SimdYolo 升到 0.11.4 并重测**（`824e574`，`dotnet build -c Release` 重建后验证 `Version=SimdYolo 0.11.4`、
      `CpuThreads=4`）。**收益在噪声内**：yolo11s 81.3 → 77.2（-5%），yolov8s 76.2 → 78.7（+3%），n 档持平。
      原以为「0.11.4 的 CPU 优化会推翻基线」，实际收益在噪声内；vs SimdYolo 倍率从 1.24~1.59 变成 **1.23~1.54**
- [x] **温度墙新发现（会影响后续所有 s 档测量）**：4 线程下 simdyolo 连续跑 `yolov8s`+`yolo11s` 约 40 s，
      perf% 从 138 掉到 **20.9**，median 虚高 6.8 倍。**s 档必须单模型跑 + 跑完即查 perf%**

### ~~P1 Conv 在低线程下的效率~~ **已否决（2026-11-15 实测）**

**原假设**：9 个 Conv 占 27 ms，16 线程时只占 3.6%、4 线程涨到 10.9%，判定为并行粒度失效，
预期优化到 15 ms。

**实测推翻了它**（`bench.md §9`，`convbench` 单层口径）：

| 项 | 原估算 | 实测 |
|---|---:|---:|
| Conv 总耗时 | 27 ms | **11.2 ms** |
| 占端到端 | 10.9% | **4.3%** |
| Winograd 吞吐（6 个 3×3） | — | **167 GMAC/s = 4 核 AVX2 峰值的 79%** |

27 ms 来自 `profile` 占比 × 端到端折算，而 `profile` 的回调开销**按层数摊**——
MatMul 几百层、Conv 只有 9 层，被放大倍数不同，占比必然失真。

三个候选根因逐条证伪：
1. `PackedConv` 面板数 = 64 ≥ 4×线程数 16 → 不空转；
2. Winograd 16 个 component / 4 线程已经接近线性（1→2 为 1.95×、2→4 为 1.97×）；
3. 没有任何层因 `ParallelThreshold` 退化为串行。

**结论**：Conv 的可优化上限只有 **-3~4%**（11.2 → 8 ms 已经接近 ORT 的 8 ms），
投入产出比远低于 GEMM 与逐元素。**转为 P5 级别的随手项，不占排期。**

### ~~P1' 逐元素算子为什么这么贵~~ **已关闭（2026-11-15 实测：它不贵）**

**原假设**：`Mul 8.6 + Add 7.8 + Relu 6.4 + Silu 6.0 ≈ 32 ms（12%）`，判定为「非带宽墙的效率问题」，
预期 -15~20 ms。

**实测推翻了它**（`bench.md §11`，新加 `eltwisebench` + `dump-graph --inputs` 流量统计）：

| 项 | 原估算 | 实测 |
|---|---:|---:|
| 逐元素合计 | 32 ms（12%） | **12 ms（4.7%）** |
| Mul（76 个） | 8.6 ms | **5.4 ms**，153 MB → 31 GB/s |
| Add（46 个） | 7.8 ms | **1.2 ms**，127 MB → 106 GB/s |
| Relu + Silu | 12.4 ms | **0.4 ms** |

32 ms 是陈旧口径（未预热 / 更旧的代码状态），同口径复现不出来。内核本身在 L3 常驻时能跑到
**100+ GB/s**（比 memmove 参考还高），图里 Mul 的 31 GB/s 是**缓存不命中**，不是算法缺陷。

**结论**：收益上限 ≈ 2.5 ms（1%），全部来自 `Where`（2.6 ms）与 `copy1` 式逐元素搬移。
**移出主线，并入 P5 随手项。**

### ~~P2 Conv+SiLU / Conv+ReLU 融合~~ **随 P1' 关闭（-2.8% → ~0）**

**原假设**：`Relu 6.4 ms + Silu 6.0 ms = 12 ms（4.8%）`，ORT 因 epilogue 融合而完全没有这两类。

**实测**：`Relu 8 个 0.2 ms + Silu 8 个 0.2 ms = 0.4 ms（0.2%）`。融合的收益上限只有 0.4 ms，
而三条 Conv 路径（`PackedConv`/`Winograd`/`Im2Col`）都要改 —— 投入产出比为负。
**不做。**

### P3 逐元素碎片与折叠漏网（**已完成**，Where 2.6 → 0.1 ms）

**事实**（`bench.md §11` 复测后）：`Where 1 个 2.6 ms`、`Div 7 个 0.7 ms`、`Mul 4.9 ms`（已贴带宽）、
`Add 1.2 ms`（106 GB/s，无可优化）。

- **`Where` 2.6 ms = 10 ns/元素**：`WhereLayer` 用 `OpUtil.CopyBytes(..., 1)` 逐元素搬 4 字节，
  `eltwisebench` 的 `copy1` 证明这个模式本身就要 0.8 ns/元素，加上掩码 `ReadBools` 与三套 stride 推进
  就是 10 ns。改法：按行 memcpy（`cond` 是 `[1,1024,1]`，每行一个布尔，两个分支分别是标量与整行）。
  顺带查常量折叠为什么没把这条链消掉（ORT 的 op 直方图里 `Where` = 0）。
- `Mul/Add` 已无空间：`Add` 106 GB/s 就是这台机器的带宽上限，`Mul` 的 31 GB/s 是缓存不命中导致，
  要改只能改调度顺序，收益不可预期。

**实测**（`bench.md §15.1`）：`WhereLayer.ForwardRows` —— 尾维「条件广播 + 源连续或广播」时，
外层走 `dims[0..^1]`、内层整段 memcpy 或按等宽整数 `Fill`。**2.6 → 0.1 ms**，逐位一致。
（常量折叠那条没查：收益已经在手，ORT 的 0 另有原因 —— 它把这条链整体优化掉了，属于另一个话题。）

**判定**：≤ 0.2 ms 达标。

### P4 Transpose 合并进 GEMM 的读面板（**已完成**，实测 rfdetr -4.5%，好于预期）

**事实**：111 个 Transpose 共 11 ms（0.1 ms/个），ORT 5 ms。已是 8×8 向量分块 + 同布局
memcpy 快路径，单算子再优化空间小。

**做法**（已落地，`bench.md §14`）：`GraphFusion.FuseTransposeIntoMatMul` 把 attention 的
`[B,H,T,D] ↔ [B,T,H,D]` 转置折成 GEMM 的读 stride（A 行距 `H*D`、B 行距 1 / 列距 `H*D`），
转置层直接删掉。外层的 `Mul(·, 1/√8)` **不折数值**（`(q·s)·(k·s) ≠ s²(q·k)`，会破坏逐位一致），
只把它的输入从「转置输出」改接到「转置的源」。

**实测**：Transpose **111 → 75**（−32.4%，12 层 × 3），命中 24 处；交替 A/B 有效 7 轮
中位 **−4.5%**（234.1 → 223.7 ms），最好读数 221.4 ms（−3.1%）；5 模型逐位一致。

**判定**：节点数 32.4% 未达 40%，但端到端 −4.5% 远超 −1.5% 的门槛，**通过**。
没到 40% 的原因：剩下的 36 个里 12 个是 PV 输出回 `[B,T,H,D]` 的转置（属于**写**面板，
不在本项范围），其余 63 个没有 MatMul 消费者。

**新识别出的余项**（可并入 P5 随手项）：PV 输出的 `Transpose[0,2,1,3] → Reshape`
可以折进 GEMM 的 **C 写面板**（12 个节点，约 1.4 ms，0.6%）。
yolo 系 attention 是 `[B,H,T,D]` 内的矩阵转置（`perm=[0,1,3,2]`），A 列主序，
需要内核支持 A 的双 stride —— 收益只有 0.5 ms，**不做**。

### P5 小算子批量化（**Reshape 已完成**，其余不做）

`bench.md §11` 复测后剩下的都是小项：

| 算子 | ops | ms（前） | ms（后） | 备注 |
|---|---:|---:|---:|---|
| Transpose | 111→75 | 10.5 | 7.5 | 见 P4，单算子 0.09 ms 已是分块 + memcpy 快路径 |
| Softmax | 18 | 7.9 | 7.9 | 36.3 M 元素 / 415 MB → 52 GB/s，已接近带宽 |
| Reshape | 119 | 4.4 | **0.2** | **已完成**：槽位复用，见 `bench.md §15.2` |
| FusedEltwise | 12 | 4.2 | 4.2 | 无对应 ORT 项，但已是按行算子的直写，空间不大 |
| GridSample | 3 | 3.5 | 3.5 | 双线性插值的 gather，改不动 |
| Where | 1 | 2.6 | **0.1** | 见 P3 |
| GatherElements | 24 | 1.2 | 1.2 | 76800 次 gather；idx 任意排列，没有向量化空间，**不做** |

**Reshape 的做法**（这是本项唯一值得做的）：`LayerInfo.InPlaceInput` + `BufferAllocator` 的槽位共享，
让 Reshape 的输出与输入是同一个 Tensor，Forward 直接 return —— 4.4 → 0.2 ms。
安全前提（输入是中间结果、本 op 是最后读者）见 `bench.md §15.2`。

剩下的 Softmax / FusedEltwise / GridSample / GatherElements 单项都 < 2%，且都已在带宽或访存模式的上限，
**不再做**。

### P6' GEMM 微内核（唯一大项，进行中）

**目标**：GEMM 类 186 ms → 150 ms，端到端 -13%。
**现状**：MR=12 之后 GEMM 约 168 ms（92% 打包 / 82% 动态），端到端 -9.8%（246.0 → 221.8）。
剩下的可行项只有 BLIS 式 A 打包，上限约 10 ms（端到端 ~4%）。

**第 1 步（`bench.md §12`）+ 第 2 步（`bench.md §13`）均已完成**：

- 先把峰值钉死：`flopsbench` 4 线程 = **427 GFLOPS**（= 2×FMA×8 lane×2×3.45 GHz），
  且 GEMM 满载期间 `perf%` 恒定 137.8 → **不存在降频**，可以直接算效率。
- 微内核 MR 4→6（`Multiply6Rows`，12 累加器，OpenBLAS Haswell 6×16 同款）：
  未打包 75.6% → **83.5%**（打包路径 356.6 GFLOPS），微基准 +6~9%，
  **端到端 246.0 → 234.3 ms（-5.0%）**。
- 微内核 MR 6→12（`Multiply12Rows`，24 累加器 = 27 ymm，按 `Avx512F.VL` 分派）：
  打包路径 **83.5% → 92.2%**（393.8 GFLOPS），微基准 +11.2%，
  **端到端中位 -5.4%，最好读数 221.8 ms**（累计 **-9.8%**）。
- 顺带确认「批量 GEMM 一次派发」（`MultiplyBatched`）端到端 **1.048×**，保留。
- 两步输出都与改动前**逐位一致**。

**剩余空间**（原本约 28 ms，端到端 -11%）：

1. **未打包路径的每帧重打包**（75.6% vs 83.5%，8 个点）。常量权重已走 `Pack` 缓存，
   剩下的只有注意力这类动态 B；`MultiplyBatched` 已把打包摊到列块任务里。
2. ~~微内核本身~~ **第 2 步已完成（同日）**：MR=12 的 24 累加器内核
   （`Multiply12Rows`，24 acc + 2 B + 1 broadcast = 27 ymm，按 `Avx512F.VL` 分派），
   打包路径 **83.5% → 92.2%**（3.079 ms = 393.8 GFLOPS / 峰值 427.1），未打包 81.7%。
   端到端 MR6→MR12 交替 A/B 中位 **-5.4%**，最好读数 **221.8 ms**
   （P6' 累计从 246.0 降到 221.8，**-9.8%**），逐位一致。
   踩坑记录：第一版把 12 个 bias 向量做成持久局部变量 → 24+12+3 > 32 溢出，只快 2%；
   改成初始化时就地广播（float 标量）才拿到 11%。
3. ~~剩下的 8 个点要靠 BLIS 式 A 打包 + k 分块，上限约 10 ms~~ **已实测否决**（`bench.md §18`）：
   A 打包成 `[k][6]` 面板后交替测，9 次里 8 次更慢（384 列时 −29% ~ −33%）。
   原路径的 6 条 stream 在 Zen4 的 L2 预取器能力（8~16 条）之内，逐字节流量又完全相同；
   面板还多占 1.58 MB 挤掉 B/C。**A 驻留的真实上限只有 4%**（m=1028 的 404 GFLOPS
   vs m=120 的 420），而 MC 分块要付 B 的重复读，净收益 < 2% —— 不值得。
   **P6' 关闭，打包路径定格 92.2%（393.8 / 427.1 GFLOPS）。**

**前置条件（本轮新增的纪律）**：`profile` 的 per-op 小算子在运行期间会漂
（同一个注意力 MatMul 测到 0.68 / 0.80 / 1.47 / 3.60 ms，见 `bench.md §12.6`）。
所以上面两项**必须先有可靠的效率读数**（微基准 + 关掉该路径的端到端 A/B），
否则容易对着一坨噪声优化。在拿到之前 P6' 不再往下推。

### ~~P6 并行粒度的线程数自适应~~ **前提已被证伪，降级为观察项**

**原假设**：所有分块策略按「块数 ≥ 线程数」判断，4 线程下小模型会退化成串行。

**实测**（`convbench`，128→128 @32² k3，Winograd）：

| 线程 | 1 | 2 | 4 | 8 |
|---|---:|---:|---:|---:|
| ms | 3.46 | 1.77 | 0.90 | 0.56 |
| 加速比 | — | 1.95× | 3.84× | 6.18× |

1→2 是 1.95×、2→4 是 1.97×，**接近线性，没有退化**。4→8 的 1.61× 是内存带宽开始起作用，
属正常。

**结论**：不做全局 `ParallelRange.Split` 重构（那是最容易把 32 线程性能改坏的改动）。
只在确实出现「块数 < 线程数」的具体 kernel 上单独处理，并且必须有微基准数据支撑。
**收益预期从「结构性」下调为 0~2%，移出主线。**

### P7 对照 ORT 1.32 源码（**参照清单，未排期**）

源码：`F:/code/onnxruntime-1.32.0/onnxruntime`（2026-11-16 通读 MLAS + CPU EP）。

#### P7.0 先纠正三个假设（否则会照着错的靶子优化）

| 曾经的猜测 | 实际 |
|---|---|
| ORT 的 Conv 靠 oneDNN | **CPU EP 里没有 oneDNN**：`core/providers/cpu` 全目录 `use_dnnl\|UseDnnl\|dnnl::` 0 命中；dnnl 是独立可选 EP（`core/providers/dnnl/`），要单独编译并显式挂载 |
| ORT 的 3×3 走 Winograd/FFT | **没有**：`core/mlas/lib` 里 `Winograd\|Fft\|Fourier` 0 命中 |
| ORT 用了 AMX 的 float 算力 | **没有**：`core/mlas/lib/platform.cpp:842` 只把 AMX 接到 `MlasGemmU8S8DispatchAmx`（int8 专用） |

结论：ORT 的 Conv 优势**全部来自布局与循环结构**，不是算法。

#### P7.1 它实际怎么做（对照我们的实现）

| 维度 | ORT | 我们 | 差距 |
|---|---|---|---|
| 布局 | **NCHWc(16)** 图级布局传播（`core/optimizer/nchwc_transformer.cc:83`，仅在收益为正时插 reorder） | NCHW | 见 §P7.4 否决 |
| 权重 | `OIHWBiBo`，**图优化期**重排成 initializer 并去重缓存（`nchwc_transformer.cc:448-469`，`MlasConvPrepare` 里 prepack） | `[block][k][4]` 4 行交错 + `ConditionalWeakTable` 缓存（`Kernels/PackedConv.cs:43`） | 基本持平 |
| im2col | **完全不物化**，直接从 NCHWc 取数 | `PackPanel` 物化 `K×16` 面板（K=576 时 36 KB，写一遍再读一遍，`PackedConv.cs:241/262`） | **差在这一步** |
| 微内核 | 6 像素 × 4 filter 块 × 16 oc = **24 个 zmm**（`core/mlas/lib/snchwc.cpp` 的 `MlasConvNchwcFloatKernelAvx512F`） | 16 像素 × 4 行 × 2 ymm = **8 累加器**（`PackedConv.cs:30` 的 `RowBlock=4`） | 我们窄 3 倍 |
| 循环顺序 | **输入通道块在外层**，整行输出反复累加（`ACCUMULATE_OUTPUT`，`snchwc.cpp:948`） | 面板内一次走完全 K | 输出行不复用 |
| 1×1 | 专用 `ConvPointwiseFloatKernel` | 走 GEMM + `PackedB` | 持平 |
| depthwise | 专用 + AVX-512 滑动窗口版 | 已有 `Kernels/DepthwiseConv.cs` | 持平 |

一条关键量化：ORT 的 3×3 内核里一个输入标量广播后服务 **4 filter 块 × 6 像素 = 24 条 FMA**，
我们的权重标量只服务 4 行 × 2 向量 = 8 条。**它的「每次 load 换到的 FMA 数」是我们的 3 倍** ——
这才是「87 个中等 conv 打不过」的直接原因，而 `PackedConv.cs:27-29` 那条「6 行慢 25%」的注释
是**纯 AVX2（16 个 YMM）时代的结论**，与 `GemmKernel` 后来按 `Avx512F.VL` 分派拿到 MR=12 并不矛盾。

#### P7.2 值得抄的四项（按性价比排序）

| # | 做法 | 改哪 | 与主线关系 | 预期 |
|---|---|---|---|---|
| A | **去掉面板物化**：`singleRow && strideW==1 && 面板在界内` 时，在内核 K 步里现取 16 像素（2 个 ymm），跳过 `PackPanel` | `Kernels/PackedConv.cs` | yolo 专用 | −10~20%（省 36 KB 写+读往返） |
| B | **行块 4→8**，用 AVX-512VL 的 ymm16~31（16 累加器 + 2 B + 广播 = 19 ymm），照搬 `GemmKernel.UseMr12` 的 ISA 分派 | `Kernels/PackedConv.cs` | yolo 专用 | −5~15% |
| C | **输入通道块外提 + 输出行累积**（一行输出常驻 L1，输入行被整条 K 复用） | `Kernels/PackedConv.cs` | yolo 专用 | 与 A 合并做才有意义 |
| D | **降低 GEMM 并行阈值**：MLAS 是 `M·N·K / 65536 + 1`，我们是 `GemmKernel.ParallelThreshold = 1<<22`（**高 64 倍**） | `Tensors/GemmKernel.cs` | **rfdetr 主线** | **已测：0，关闭**（`bench.md §17`） |

A+B+C 对 yolo11n 的上限：Conv 30.7 ms 若降到 23 ms（−25%），端到端 37.2 → 29.5，
对 ORT 从 1.59× 拉到 **1.26×**。**对 rfdetr 一点用都没有**（Conv 只占 5.1%，见 §P7.4）。

**A/B/C 已否决**（2026-11-16，`bench.md §19`）。A（去物化直读）与 B（8 行块）都实现并测过：
够得着的两层是赢的（`K=2304` 的 40² s1 **8.26 → 6.54 ms**、20² s1 2.08 → 1.97），
但**yolo11n 里一层也命不中** —— `K > 2048` 的层全在 20×20，选路顺序 `Winograd → PackedConv`
让 Winograd 先接走（同形状 4.74 ms / 199 GMAC/s，比放开准入后的打包还快 38%），
能走到 `PackedConv` 的只剩 s2 层（`K ≤ 1152`）。模型内交替 + 未变层归一化：中位 1.06（噪声内）。
代码已回退；C 以同一门槛为前提，命中面同样是空集，不启动。
**上表的 −10~20% / −5~15% 预估作废**；yolo 的 Conv 空间改记在 s2 层（见 §P7.5）。

**D 已关闭**（2026-11-16，`bench.md §17`）：阈值现在可配（`ONNXSHARP_GEMM_THRESHOLD`，
支持 `65536` / `1<<16` / `64K`），但 rfdetr 全部 139 个 MatMul 的 `m·n·k` 最小也有 19.6 M
（decoder `300×256×256`），QKV 151 M、QK 101 M、fc1 606 M —— **全都远高于 `1<<22`（4.19 M）**，
降阈值不改变任何一层的并行决策。端到端交替 A/B：217.8 vs 237.4（噪声内）。默认值不变。

#### P7.6 顺带查出来的新缺口：attention 的 QK/PV 在图里慢 1.8×

> **已落地为 P8 并判定（2026-11-16，`bench.md §20`）**：`FusedAttention`（QK→Softmax→PV 分块融合、
> 矩阵不落地）+ `ab` 同进程交替 A/B 工具。实测 **+2.2%**（一次 5.5%，配对中位，噪声地板 1.6%），
> 数值结构等价（与「只换内核」的对照差异同形态）→ **采纳，默认开启**（`ONNXSHARP_ATTN_FUSE_MAXSEQ=4096`）。
> 落在预估 −4% 的下限。
> **现在的新热点**：12 个 `FusedAttention` 各 7.2 ms，合计 **40%** 的端到端；
> 每层 804 MMAC → **112 GMAC/s**，只有 GEMM 峰值 197 的 57% —— 再往下是 flash-attention 式
> 寄存器分块 + 在线 softmax（不落地 S 块），属新立项。

用 `gemmbench --attn 1`（复刻真实形状 + `aRow/bCol` 视图 + `outer×inner` 派发）对比同参数 profile：

| 算子 | 图里 | 探针（数据全热） | 差距 |
|---|---:|---:|---:|
| QK^T（`257×64×257` × 24） | 1.73 ms / 117 GFLOPS | 0.95 ms / 213 GFLOPS | 1.8× |
| PV（`257×257×64` × 24） | 1.03 ms / 197 GFLOPS | 0.54 ms / 377 GFLOPS | 1.9× |

12 层合计 33.1 ms。不是打包也不是派发（probe 里 batch=6/12/24 都是 245~249 GFLOPS），
而是**每层 6.3 MB 的 attention 矩阵往返**（QK 写、Softmax 读+写、PV 再读，25 MB/层，
L2 只有 1 MB/核）。可行的只有「QK→Softmax→PV 分块融合、不落地矩阵」，
上限约 **−4%**（33.1 → 23 ms），工程量大于 P6'。**列为候选 P8，未排期。**

同时否决了一条「看起来该做」的微优化：把 `PackColumns` 的 `bColStride != 1` 分支改成
「按行优先」（cache line 利用率 6% → 100%）。微基准（`bColStride == k`）确实 5.314 → 0.728 ms，
但真实参数 `bColStride=384` 下两版完全一样（0.958 vs 0.952 ms）—— 原顺序的内层循环
已经在复用那 16 条 line。**已回退，不留代码。**

#### P7.3 已经持平或领先、不用抄

- **`beta==0` 不读 C**：我们的 bias/columnBias 直接做累加器初值（`GemmKernel.Multiply` 的
  `bias` / `columnBias` 参数），等价。
- **N 尾补齐到 16**：我们 `nb=16` 补零，等价于 MLAS 的掩码内核。
- **B 面板常驻**：我们有 `PackedB`（MatMul 常量权重）+ `PackedConv.Cache`（Conv 权重）。
  **反例：ORT 自己没做全** —— float `MatMul` 每次现 pack B（`core/providers/cpu/math/matmul.cc:436`
  直接 `MlasGemmBatch`），只有 `Gemm` 节点用了 `packed_b_`。这块**我们领先**，不要被「ORT 有预打包」误导。

#### P7.4 明确否决：NCHWc 全图布局传播

**不做。** 三条理由（第 1 条是决定性的）：

1. **rfdetr 的收益分母不在 Conv 上**。同口径 profile（`sum-of-min` 234 ms）：

   | type | ops | min ms | share |
   |---|---:|---:|---:|
   | MatMul | 139 | 180.3 | **77.1%** |
   | Conv | **9** | 12.0 | **5.1%** |
   | Softmax | 18 | 7.6 | 3.2% |

   NCHWc 只对「空间算子」（Conv / Pool / Concat / DWConv）有意义，对 2D 的 MatMul **零收益**。
   即使 Conv 快到 2×（12.0 → 6.0），端到端也只省 **2.6%**；而 NCHWc 本身最好情况也只是
   「Conv 提速 1.3~2×」，现实落在 **−1.3%~−2.6%**，连 3.5% 缺口都不够，且是高风险大改。

2. **reorder 成本会吃掉它**。NCHWc 是「传染式」布局：只有相邻节点的张量都在 c 布局时才免转换。
   rfdetr 的静态图里有 `Add 169 / Mul 114 / Reshape 122 / Transpose 113 / Slice 62`（`bench.md`
   的 2064 节点直方图），9 个 Conv 的输入/输出边界（约 18 处）都要么把邻接算子也 c 化、
   要么插一次全量 reorder。Conv 少的模型**恰恰是 reorder 摊不薄**的模型 —— ORT 自己在
   `nchwc_transformer.cc` 里也是按「收益/开销」逐节点判定的，不是无脑全转。

3. **我们的 Conv 本身已接近峰值**：`bench.md §9` 实测 Winograd 167 GMAC/s = 4 核 AVX2 峰值的 79%，
   剩余空间来自打包而非布局。真正想动 Conv 应该做 §P7.2 的 A/B/C（针对 yolo），
   而不是重写整个引擎的内存布局。

工程量参考：ORT 自己为此写了 `nchwc_transformer.cc`（1000+ 行图级 pass）+ 每个算子的 c 布局支持表
+ allocator 的形状改造。我们是 88 个算子注册、buffer 复用已按 NCHW 物理布局设计（`Engine/BufferAllocator.cs`），
这套改造的回归面覆盖全部 88 个算子 —— 对 3.5% 的目标而言**投入产出完全不成立**。

> **范围限定**：本条否决的是「rfdetr 主线上做**全图**布局传播」。yolo 系的下一站另行定为
> **NCHWc**（2026-11-16 定调，见 §P7.7），届时按局部化方案重估，不沿用本条结论。

#### P7.7 yolo 的下一站：NCHWc（**三个前置问题已量，全部通过 → 立项**）

2026-11-16 决策：yolo 系直接上 NCHWc，不再做 A/B/C 这一层的微调。
同日完成三个前置问题的实测（`bench.md §22`），结论：**成本账成立，可以立项**。

为什么是 NCHWc 而不是继续调打包：A/B/C 已实测命中 0 层（§P7.2），差距不在「少搬一趟」，
而在**通道访存布局**：s2 层每个输出像素要跨 `stride` 取样，空间局部性天然差，靠 im2col/packed
的手工分块已经到顶（`bench.md §19`）。

与 §P7.4 的区别（必须区分，否则会重复否决）：
- §P7.4 否决的是**全图传播**（88 个算子 + allocator 全面改造，rfdetr 只省 2.6%）。
- 这里做的是**局部 NCHWc Conv**：只在 conv 链上保持 c 布局，边界插一次 reorder。

**三个前置问题的答案（`bench.md §22`，实测）**：

1. **reorder 边界 = 5 处 / 10.5 MB / 0.336 ms（端到端 0.9%）**。不是先前估的 18 处 ——
   `Concat`（23 个）与 `Split`（11 个）的通道数都是 16 的倍数，块序列直接拼接即可；
   C2PSA 的 `Transpose`/`MatMul`/`Softmax` 前面本来就有一个 `Reshape` 打回 NCHW，
   整段 attention 保持 plain，不产生额外边界。即使阶段 1 不做 `MaxPool`/`Resize` 的 c 版，
   也只多 5 处、+0.13 ms。工具：`bench/layout_probe.py`。
2. **块宽 = 16**（正好 1 个 `zmm`），86/88 个 conv 的 `oc % 16 == 0`；块宽 32 只有 70/88，
   16 个 `oc=48/80/112` 要补一半。两个例外要排除在 c 之外：`ic=3` 的首层（读 NCHW、写 c，
   不用补到 16）与 `oc=8` 的 C3k2 瓶颈层（补到 16 是 2× 空转，保持 NCHW 更便宜）。
3. **`BufferAllocator` 零改动**：`Layout` 枚举 / `Tensor.Layout` / `Arg.Layout` /
   `LayerInfo.GetLayouts` 早已预留；槽只按「区间不重叠 + 字节够大」复用，与布局无关；
   `Tensor.Fit` 每次复位 `Layout = NCHW`、`Strides = null`，槽复用不会串布局。
   只需 `Fit` 加一个 layout 参数、`GetLayouts` 出时间表。

**收益推算（目标吞吐是假设，非实测）**：conv 的 31.78 ms 按家族分解为
3×3/s1 **13.44 ms（88.9 GMAC/s）** / 3×3 s2 8.15 ms（106.7）/ 1×1 10.19 ms（114.1）。
s2→170、1×1→150 保守给 **−5.1 ms**，加上 reorder 与（可选的）3×3/s1 中小层上行，
端到端 **−14% ~ −24%**（36.4 → 31.3 ~ 27.6 ms）。
旁证比推算硬：`§4` 里 OpenCV dnn（`NC8HW8`，块宽只有 8）实测 **28.6 ms**，比我们快 28%。

**顺带纠正 §19 的盲区**：§19 整节只盯 s2（因为那是 A/B/C 的命中面），而 conv 里最大的一块是
**3×3/s1 的 13.44 ms** —— Winograd 只接了空间 ≤4096 的层（199 GMAC/s），160²/80² 的十几个层
退回 im2col，把家族平均拉到 89。这一块此前从未进过视野。

**执行方式（先证一个数）**：
- **阶段 1（唯一承诺）**：只做 **s2 的 NCHWc 内核**（7 层、8.15 ms、占 conv 26%），
  验收 = 这 7 层「107 → ≥170 GMAC/s」，`convbench` 微基准 + 端到端 `ab` 双读数。
  **拿不到 1.5× 就停**，1×1 与 3×3 一律不做。
- 阶段 2（阶段 1 达标才谈）：~~1×1 的 c 布局内核 + `Concat`/`Split` 的 c 版本~~
  → **实际执行的是「内核泛化到 stride 1」**：3×3/s1 的层数远多于 s2，且不需要动 Concat/Split。
- 阶段 3（可选上行）：1×1 的 c 内核、`oc%32!=0` 的层、Winograd 选路阈值重估。

**门槛**：测量问题必须先解决（§19.5：同进程交替 A/B），否则 3~5% 的收益在本机判不出来。

**阶段 1 结果（2026-11-16 夜，已完成并采纳，`bench.md §23`）**：

| 判据 | 目标 | 实测 | 判定 |
|---|---|---|---|
| s2 7 层吞吐（MAC 加权） | ≥170 GMAC/s（1.5×） | **112.3 → 204.1 GMAC/s（1.82×）** | ✅ |
| 单层微基准 | — | 174~220 GMAC/s（对照 im2col/packed 88~138） | ✅ |
| 端到端 `ab` | −8.4%（推算） | **yolo11n −5.5%、yolov8n −3.9%**（配对中位，噪声地板 1.0%） | 采纳 |

落地物：`src/OnnxSharp/Kernels/NchwcConv.cs`（blocked 布局 + 16 zmm 累加器 + 8×8 转置
reorder 内核）、`ConvLayer` 的 `tempShapes` 与分派分支、`ab` 的 `nchwc=` 旋钮。

三条要记住的教训：

1. **`CanUse` 必须要求 `h`/`w` 为偶数**：blocked 输入只在左上留 1 像素 halo，奇数边长会多读
   一列/一行（`101×97` 实测 `max|d|=24`）。`2*oh==h && 2*ow==w` 也写进 `TryConv` 兜底。
2. **reorder 的朴素写法是 16× 写放大**：lane 外层逐元素写 64 字节块 → 320² 单层 3.72 ms。
   换成 8×8 转置内核（落盘 32 字节半行）后 0.19 ms，这才是 §22.1「reorder 不是拦路虎」的依据。
3. **端到端缺口全在「每层都付 reorder」**：阶段 1 未做布局传播，reorder 14 次而非 5 次，
   in situ 约 1.3~1.5 ms（微基准 0.47 ms 是滚烫缓冲的乐观值）。§22.1 的 0.336 ms 只能引用在
   **传播之后**。

**阶段 2 的入口因此明确**：先做图级布局传播（`Fit` 带 layout、`Concat`/`Split`/`MaxPool`/`Resize`
的 c 版、`GetLayouts` 时间表），把 reorder 压回 5 次（预期再补 −3%），再谈 1×1 的 c 内核
（45 层 / 1162.6 MMAC / 10.19 ms）与 3×3/s1 的选路重估。

**阶段 2 结果（2026-11-16 深夜，已完成并采纳，`bench.md §24`）：把内核泛化到 stride 1，
端到端 −21% ~ −28%。上面的「先做传播」被实测推翻。**

- 分阶段计时（新增 `ONNXSHARP_NCHWC_PROF=1`）先量了账：6 层命中时 reorder 只占 NCHWc 时间的
  18.9%、摊到整帧 **1.0 ms（2.9%）**。而 `dump-graph` 显示每个 s2 层的消费者是 3×3/s1 的 Conv
  和很后面的 Concat —— 要吃这 3% 得先把 Concat/Split/MaxPool 和邻居卷积全改成 c 布局。
  **收益 3%、成本数天，排序错了。**
- 改为同一内核加一个 `step` 参数（stride 1：四周各 1 像素 halo，无奇偶约束；指令数不变，
  吞吐同为 220 GMAC/s），3×3/s1 那批层从 100~160 GMAC/s 直接抬到 220。

| 判据 | 目标 | 实测 | 判定 |
|---|---|---|---|
| s1 单层吞吐 | — | **220.6~220.8 GMAC/s**（im2col 146~159、Winograd 102~184、packed 120~150） | ✅ |
| 端到端 | −5.5% 之上再拿一块 | **yolo11n −21.2%、yolov8n −25.1%、yolov11s −23.6%、yolov8s −27.0%、yolo11m −27.6%** | ✅ |
| 命中面 | — | **30 层/帧**，reorder 占 NCHWc 时间 9.5%（≈2.5 ms/帧） | — |

同期收掉一个隐患：`InferShapes`（申请 temp）与 `Forward`（分派内核）原本各写一份适用条件，
现在合并成 `NchwcConv.Dispatch()` 单点判据 —— 扩适用面时不会一侧改了一侧没改。

**下一块的候选**（按 24.5 的记录）：`oc % 32 != 0` 的层（放宽 `OcGroup` 到 1 要用微基准单独量）、
1×1 的 c 内核（45 层 / 10.19 ms，无 halo 天然衔接）、以及 1080p 级输入上 blocked 缓冲的分块流式化。

#### P7.5 一句话

**GEMM 侧我们已超过 ORT（B 预打包常驻）。Conv 侧的差距不在「物化 im2col + 窄行块」——
yolo 的 3×3/s1 早已被 Winograd 以 199 GMAC/s（~83% 峰值）接手，打包路径只剩 s2 层，
A/B/C 实测**命中 0 层**（§P7.2）。阈值（D）与 P6' 的 A 打包（§18）也都实测为 0 或负；
rfdetr 剩下的 3.5%：「少搬 6.3 MB 的 attention 矩阵」已做成 P8 并判定 **+2.2%**（§P7.6，采纳）；
下一个缺口回到 attention 内核本身（12 层 40% 的时间，只有 GEMM 峰值的 57%）。
**yolo 上局部 NCHWc（§P7.7）：已泛化到 stride 1 并采纳 —— 单层 220 GMAC/s（s2/s1 同速），
30 层/帧命中，端到端 −21% ~ −28%（yolo11n/v8n/v11s/v8s/11m 五个模型一致）。
原计划的「先做布局传播」被实测推翻（reorder 上限只占整帧 3~8%，且需要邻居算子全部 c 化）。**

### P9 attention 内核重写：**已执行 3/4，累计 +2.2%**（`bench.md §21`）

> **结论先行（2026-11-16）**：P9.1 命中「512 KB 已是最优」（默认不变）；P9.2 归一化后置
> = 端到端 +0.6%（保留，`max|d|=9.3e-5`、0 翻转）；P9.3 面板复用 = +1.0% 且**逐位相同**；
> 累计 **+2.2%**（`ab` 6 轮同向，跨度 0.7%，超过 1% 噪声地板）。
> P9.4 / P9.5 用实测天花板关闭，见 §P9.6。**并且 §P9.0 的前提被更正过了，先读它。**

#### P9.0 起点数据（先测，不猜）

`ONNXSHARP_ATTN_TRACE=1`，rfdetr-small，行块 127（`ScoreBudget` 512 KB），**只对大层**（T=1028）：

| 阶段 | 每块线程时间 | 占比 | 说明 |
|---|---:|---:|---|
| `qk` | 0.229 ms | **48%** | `127×1028×64` = 8.36 MMAC → 36.5 GMAC/s（含每块重打 K^T） |
| `pv` | 0.142 ms | 30% | 同规模 8.36 MMAC → 58.9 GMAC/s（含每块重打 V） |
| `soft` | 0.073 ms | 15% | max + exp/Σ + scale 三趟全落在 S 块上 |
| `copy` | 0.036 ms | 7.5% | Q 行块紧凑化 + 乘 scale |

**更正（这一条改变整节的结论）**：12 个 `FusedAttention` 不同构。只有 **3 个**是
`b=1, t=1028`（每层 811 MMAC，6.0~6.6 ms），另外 **9 个**是 `b=4, t=257`（窗口 16×16+1，
每层 203 MMAC，1.6~1.7 ms）。两族吞吐一致（134 vs 127 GMAC/s），**没有并行退化**。
所以 attention 的真实占比是 **16.4%（35.5 ms / 219）**，不是原先记的 40% —— 当时把 3 个大层的
数字当成了 12 层。**这直接把 flash 式重写的上限从 −16% 压到不到 −5%。**

**三条读数仍然成立，且决定计划排序**：

1. **PV 不要动** —— 同规模、同内核，PV 比 QK 快 1.6×，差在 K 长度（1028 vs 64）。
2. **QK 的 48% 来自 K=64 短归约**（只有 8 个 k 步，序言/尾声摊不掉），不是搬运。
3. **`soft + copy` 的 23% 是纯开销**，可以不动 GEMM 就拿掉一部分。

#### P9.1 行块预算参数化并扫（0 风险，先做）——**已做：默认不变**

`ScoreBudget` 是 `const`，改成 `ONNXSHARP_ATTN_SCORE_BUDGET`（默认 512 KB 不变），用 `ab` 扫
128 / 256 / 512 / 1024 KB。

依据：S 块 522 KB + Q 32 KB + K 平面 263 KB + V 平面 263 KB = **1.08 MB，已超 L2（1 MB/核）**，
理论上 256 KB 行块（块 263 KB、工作集 0.81 MB）应该更好 —— 但也可能是"块小了 K/V 重读次数翻倍"。
**这是纯经验问题，扫一次就有答案。**

**结果**：三组配对 0.991~0.993（128/256/1024 一致比 512 慢 0.7~0.9%），全部落在噪声地板内。
「超出 L2 会有悬崖」的猜想不成立 → 分块内核是算术受限、不是带宽受限（与 §21.6 一致）。
**默认保持 512，旋钮保留。**

#### P9.2 把 softmax 的归一化折进 PV 输出（精确等价）——**已做：+0.6%，保留**

现在 `SoftmaxRow` 把 S 就地归一化（`scale` 那趟读 522 KB + 写 522 KB/块），PV 再读。
改成：只算 `max` 与 `Σ`，**S 保留 exp 值**直接喂给 PV，`1/Σ` 改乘到输出行（每行只有 D=64 个值）。

数学等价：`Σ_j (e_j/Σ)·v_j` → `(1/Σ)·Σ_j e_j·v_j`，只差 1e-7 级重结合。

- 需要：`VectorMath` 补 `MaxRow`，`SumExp` 已有（返回 Σ）；`AttentionOps.ComputeBlocks` 末尾按行乘 `1/Σ`。
- 省掉：每块 522 KB 的读+写，以及 soft 那一趟的整段时间（三趟变两趟）。
- 验收：`ab`；数值用 §20.4 口径（`>1e-3` 计数应与 kernel-swap 对照同量级，不能更差）。
- 预期：attention 的 5 ~ 10%（端到端 2 ~ 4%）。**风险最低的一步，代码量也最小。**
- **结果**：端到端中位 1.007（5/6 轮同向，落在噪声内），段内 `soft+pv` 稳定 −3~4%；
  数值 `max|d| = 9.3e-5`、`>1e-3` 计数 **0/1200 与 0/900**。严格少一趟读写 → 保留（默认开）。

#### P9.3 打包跨行块复用（逐位相同）——**已做：+1.0%，逐位相同**

`K^T` 面板与 `V` 面板的尺寸都与行块无关（各 `T·D` = 263 KB），现在是**每个行块重打一遍**
（每层每头 9 次 → 只该 1 次）。`GemmKernel.MultiplyPacked(parallel: false)` 已支持读现成
`PackedB`，只需在 `ComputeBlocks` 外层按 `(b, h)` 持有面板缓冲。

- 验收：**输出必须逐位相同**（纯重排，无算法变化）—— 这是唯一一步能要求 `max|diff| = 0` 的，
  也给另外两步的数值判定提供一个干净基准。
- 代价：每线程多 526 KB 常驻（4 线程 2.1 MB），会挤 L2 —— 与 P9.1 的行块大小**互相影响**，
  必须放在 P9.1 之后测。
- 预期 2 ~ 6%。
- **结果**：中位 1.010（6 轮 1.008~1.013 全同向，跨度 0.5%），**`max|d| = 0` 两个输出都逐位相同**。
  实现比预期省事：并行区间在 `index = bh·rowBlocks + block` 上连续，**记住上一次的 `bh`**
  就能拿到几乎全部复用，不需要跨线程共享面板。顺带发现 `V` 复用也值 ~12 µs/块。
  新增 `GemmKernel.PackFromSource` / `PackedPanelSize`，开关 `ONNXSHARP_ATTN_PACK_REUSE`。

#### P9.4 QK 的短归约 ——**已量：天花板 2.5%，关闭**

K=64 时每个 k 步只用 8 次 FMA 就换 A 面板，复用不足。可选手段：把 Br 加到 256 以上（A 面板
32 KB → 64 KB，仍在 L1）、或对 K 双向展开。**先量后改**：`qk` 若已到单线程峰值的 75% 以上就别动。

- 验收：`ab`；数值应回到「`>1e-3` 计数不高于 kernel-swap 对照」。
- **实测天花板**：`gemmbench` 同 16.7 MFLOP，`127×64×1028` = 222 GFLOPS、`127×1028×64` = 334 GFLOPS
  → 纯 GEMM 就复现 QK/PV 的 1.5× 差距，与打包/softmax/S 落地无关。补齐只能靠**新写短 K 微内核**
  （MR=8/NB=32，AVX-512 下 32 累加器），收益 `qk 46% × 33%` ≈ 端到端 **2.5%**，
  而 P6' 已判定微内核侧「没有 >1% 的单点改善」。**不划算，关闭。**

#### P9.5 flash 式（在线 softmax，S 块不落地）——**关闭**

原设想的大改：按 `[Br × Bc]` 瓦片算 S，在线 max/Σ 并 rescale 累加器 O。它能省掉的是
「S 块落地 + 三趟 softmax」，但 **exp 的数量一个都省不掉**（`B·H·T²` 是硬成本），
而 P9.2 已经拿走了归一化那一趟的大头。

**触发条件**：P9.1 + P9.2 + P9.3 + P9.4 合计 < 5% 端到端，才考虑。否则收益抵不过
（在线 softmax 的 rescale + 边界处理 + 与 `ComputeRows`/`ComputeBlocks` 三路并存的维护成本）。

- **实测上限**：attention 只占 16.4%（§P9.0 更正），其中 `soft` 只剩 16%，S 落地往返也在
  P9.2 里被拿走一部分 → 上限 **< 2%**。**关闭，不做。**

#### P9.6 停止判定

- 目标 **rfdetr ≤ 210 ms**（起点 219.7，差 4.6%）；实际拿到 **+2.2%**（累计 `ab` 中位 1.022，
  跨度 0.7%，6 轮同向），绝对 min 220.7 / median 225.6（`bench --iters 8`）。
- 计划里「三步 ≥4% 就收工」的门槛**差一半没达到**，但剩下的 P9.4（2.5% 天花板 / 需动微内核）
  与 P9.5（<2% 天花板 / 大面积改动）都不划算 → **P9 到此关闭，rfdetr 的 attention 侧收工**。
- 要再往前走，方向不在 attention。但**「MatMul 占 62~64%」不等于「MatMul 有空间」**：
  同口径 `profile`（4 线程、3 iter、sum-of-min 199.8）拆开看，
  MatMul 115 个 / 124.2 ms，主体是 12 层 MLP —— `fc1` 3.2 ms×12 + `fc2` 2.91 ms×12 = **73.7 ms**，
  两者贡献各 1.213 GFLOP/层，合计 29.1 GFLOP / 73.7 ms = **395 GFLOPS = 4 线程峰值的 92.5%**
  （`fc1` 373 = 87%、`fc2` ≈417）。q/k/v 三个投影 36 个共 ~28.8 ms 也在同一档（~379 GFLOPS）。
  ⇒ **MatMul 剩余全部空间 ≈ 8%，约 9~10 ms = 端到端 4.7%，且要求打赢本机峰值**；
  而 P6' 已经把这条路判定为「微内核侧无 >1% 单点」。两件事合起来的结论是：
  **MatMul 不是可做的下一项，它是已经做完的那一项。**
- 顺带撤回一个错误引用：本行此前写「继续需要 NCHWc / 权重分块（§P7.7）」。NCHWc 对 2D MatMul
  **零收益**（§P7.4 第 1 条就是这个论证），§P7.7 的 NCHWc 是 **yolo 局部 conv** 的路线，与 MatMul 无关。
  另：yolo11n 里 MatMul 只有 **2 个**（C2PSA 内），MatMul 这条线是 rfdetr 专属，做它对 yolo 没有帮助。

#### P11（✅ 已实验：**负收益，结案不立项**，`bench.md §26`）
GEMM 的 K 分块 —— 照抄 MLAS 的 `StrideN`/`StrideK`

读 ORT `mlas/lib/sgemm.cpp`（`MlasSgemmOperation`）得到的、我们**确实缺**的一条：

```cpp
size_t StrideN = MLAS_SGEMM_STRIDEN;  size_t StrideK = MLAS_SGEMM_STRIDEK;
if (N >= K)   { while (StrideK / 2 >= K) { StrideN *= 2; StrideK /= 2; } }
else if (TransA == CblasNoTrans) { while (StrideN > 16 && StrideN / 2 >= N) { StrideK *= 2; StrideN /= 2; } }

for (size_t n = 0; n < N; n += CountN) {                  // 外层 N 段
    CountN = std::min(N - n, StrideN);
    size_t CountK;  bool ZeroMode = (beta == 0.0f);
    for (size_t k = 0; k < K; k += CountK) {              // 内层 K 分块
        CountK = std::min(K - k, StrideK);
        MlasSgemmCopyPackB(PanelB, B + n + k * ldb, ldb, CountN, CountK);   // 每块重打包
        MlasSgemmKernelLoop(A + k, PanelB, C + n, CountK, M, CountN, lda, ldc, alpha, ZeroMode);
        ZeroMode = false;
    }
}
```

我们（`GemmKernel.MultiplyFromSourceCore`）是 `panelSize = blockCount * k * blockWidth`，
**整条 k 打一次面板、整条 k 扫完**，没有 K 分块。面板因此随 k 线性膨胀
（k=6400 时 400 KB），而 MLAS 的面板恒为 `StrideN × StrideK`（栈上、留在 L1）。

支持这条的初步测量（`gemmbench`，4 线程，固定 m=128 / n=400）：

| k | 128 | 512 | 2048 | 8192 |
|---|---:|---:|---:|---:|
| GFLOPS | 40.5 | 43.2 | **28.1** | **28.2** |

k ≥ 2048 时掉 35%，正是面板离开 L1/L2 的形状。**但绝对值口径待复核**
（`bench.md §25.5`：同一形状三个仪表差 15 倍，机器当时满负荷），所以只当方向性证据。

动作（若立项）：不动微内核形状（MR=12 / nb=16 与 MLAS 一致），只把打包 + k 循环改成
「N 段 → K 块 → 重打包 → 累加」，`StrideK` 在 `gemmbench` 上扫 {128, 256, 512}；
累加态从「bias 进初值」改成 `ZeroMode` 首块写 / 后续 `+=`（与 MLAS 的 beta 切换等价）。
参照基线：`MlasSgemmOperation` 的 N 段宽度自适应规则（上面那段 `while`）。

**实验结果（`bench.md §26`）：已落地并 A/B，全面慢 12~22%，默认关闭（`ONNXSHARP_GEMM_KC=0`）。**

| m×k×n | KC=0 | KC=256 |
|---|---:|---:|
| 128×192×1600 | 0.212 ms / 371 GFLOPS | 0.243 / 324 |
| 1024×768×1536 | 6.686 / 361 | 8.184 / 295 |
| 1024×1280×1280 | 8.966 / 374 | 11.528 / 291 |

原因：MLAS 分块是为了用**定长栈缓冲**省掉分配，不是为带宽；我们从 `ArrayPool` 租缓冲，
前提不成立，而每块的 C 读改写成了纯额外开销。**这条不再追。**

顺带的收益：`gemmbench` 与模型内 `ConvProf` 现在对同一个热形状（0.212 vs 0.24 ms）自洽，
证明 1×1 GEMM 已在 4 线程峰值 87~90%，**GEMM 方向可以封板**。

#### P12（✅ 已实验：**前提否证，结案**）3×3 的 im2col —— 补了仪表后发现它不是浪费

原本的依据（`bench.md §27` 的减算法，yolo11n 4 线程）：

| 项 | ms/帧 | share |
|---|---:|---:|
| 1×1 GEMM | 9.57 | 33% |
| 1×1 融合 SiLU | 1.35 | 5% |
| **其余卷积（3×3，im2col）** | **13.3** | **46%** |
| 非卷积（Concat/MaxPool/Split/Resize…） | 4.7 | 16% |

四条路径全拆开后（`bench.md §29`）：

| 路径 | ms/帧 | 实测 |
|---|---:|---|
| 1×1 + SiLU | 9.83 | GEMM 136 GMAC/s |
| **NCHWc**（3×3 主路径） | **10.46** | 内核 9.44，reorder 仅 9.8% |
| Im2Col（兜底） | 2.02 | 打包 35% |
| PackedConv | 1.53 | 打包 64% |
| Winograd | 0.00 | **0 次命中** |
| 合计 | 23.84 | op 级 24.0 ✓ |

**结论：3×3 整体 150 GMAC/s = 4 线程峰值的 81%，全部打包加起来只有 1.7 ms（帧的 5.7%）。**
direct conv 的理论上限就是这 1.7 ms，**不值得做**。Winograd 现有实现的变换是标量散写，打不过 NCHWc。

顺带否证的还有：`§28.1`「6 ms 差距必然在 3×3」这条推理。

#### P13（进行中：第一步已落地 **张量 64B 对齐**，−1.2 ms；窄 m 待查）1×1 没跑满的原因

| 证据 | 数 |
|---|---|
| 1×1 在模型内（改前） | 124~136 GMAC/s |
| 同批形状在 `gemmbench` | 150~160 GMAC/s |
| 4 线程 GEMM 峰值 | 185 GMAC/s |

第一步扫真实窄 m 形状，扫出来的是**对齐**而不是级联：图里 A/B/C 的 64 字节对齐率只有 18% / 8% / 0%
（`bench.md §30`）。已修：`Tensor.Alignment = 64` + POH 分配 + `ByteOffset` 对齐，
同形状 124 → **138 GMAC/s**，yolo11n −1.2 ms（−4.4%）。

**剩余待查**（按价值排序）：

1. **B 只有 61% 对齐**：B 是激活，常为 `Concat`/`Split`/`Reshape` 的视图。要么让视图产出 64 对齐，
   要么在 GEMM 前复制（复制成本要用 `bench.md §30` 的方法先量）。估计收益 0.3~0.5 ms。
2. **窄 m 的级联**：对齐之后 `64×48×25600` 在微基准里本身也只有 119 GMAC/s（峰值 185），
   `32×32×25600` 只有 81 —— 这两个形状的算术强度低（k=32/48），要先分清是**内存屋顶**还是级联浪费，
   再决定是否动 `SelectRowBlock`。
3. 首层 `3→16`（`channels=3`，16 对齐失败）42.5 GMAC/s、1.04 ms（帧的 3.6%）：补 `channels<16` 专用内核可省 ~0.8 ms。



---

## 5. 执行顺序与依赖

```
P0（测量）──┬─→ P6'（GEMM 微内核）── MR=6（-5.0%）+ MR=12（-5.4%）已完成，累计 -9.8%
            │                        └→ 余项 A 打包：实测负收益，**已关闭**（bench.md §18）
            ├─→ P4（Transpose 折进 GEMM）**已完成 -4.5%**；余项：C 写面板 12 个转置（~0.6%）
            └─→ P3 / P5（Where 2.6→0.1 + Reshape 4.4→0.2）**已完成 -2.0%**
已关闭：P1（Conv，11.2 ms）、P1'（逐元素，12 ms）、P2（Relu/Silu 融合，0.4 ms）、
        P6（并行粒度，线性）、内存复用（Fit bug 已修）
参照（未排期）：P7 ─ 只有 D（GEMM 并行阈值 1<<22 → 1<<16 量级）落在 rfdetr 主线，
                    A/B/C（Conv 去物化 + 8 行块 + ic 块外提）是 yolo 专用
P8 已完成（attention 融合，+2.2%）──→ P9（attention 内核，§P9）**已完成 3/4，+2.2%**：
    P9.1 预算扫（默认不变）· P9.2 归一化后置 +0.6% · P9.3 面板复用 +1.0%（逐位相同）
    P9.4 短归约：天花板 2.5% / 需动微内核 → 关闭
    P9.5 flash 式：天花板 <2% → 关闭
    结论：attention 只占 16.4%，P9 收工
P10 = §P7.7 yolo 局部 NCHWc（本项，前置问题已量完）：
    q1 reorder 5 处 / 0.336 ms（0.9%）· q2 块宽 16（86/88 整除）· q3 allocator 零改动
    阶段 1：只做 s2 的 7 层，验收 107 → ≥170 GMAC/s，拿不到 1.5× 就停
            **已完成（bench.md §23）：112.3 → 204.1 GMAC/s（1.82×），端到端 −5.5% / −3.9%**
    阶段 2：内核泛化到 stride 1（同一 `step` 参数，指令数不变）
            **已完成（bench.md §24）：s1 单层 220 GMAC/s，30 层/帧命中，端到端 −21% ~ −28%**
            原计划的「先做图级布局传播」被实测推翻（reorder 上限只占整帧 3~8%）
    下一步候选：oc%32!=0 的层（放宽 OcGroup 需单独量）· 1×1 的 c 内核 · blocked 缓冲分块流式化
    推算（非实测）：保守 −14%（36.4 → 31.3 ms），上行 −24%；旁证 OpenCV NC8HW8 = 28.6 ms
```

**排期依据**：`sum-of-min` 现在与 `bench` min 对得上（252.5 vs 253.5），所以分解可以直接按绝对值读。
P6' 已把打包路径从 75.6% 推到 **92.2%**，端到端累计 **-9.8%（246.0 → 221.8）**；
余项（A 打包）已实测为负收益并关闭（`bench.md §18`）。**GEMM 侧不再有 >1% 的单点。**
P9 收工时纠正了一个记错的数：attention 不是 40%，是 **16.4%**（3 个 `T=1028` 大层 + 9 个
`b=4,T=257` 窗口层，见 §P9.0）。MatMul 占 62%（115 个 / 124.2 ms）看着最大，但
**12 层 MLP 已跑到 395 GFLOPS = 4 线程峰值的 92.5%**（§P9.6），剩余空间 ≈4.7% 且要求打赢峰值
→ **rfdetr 主线不再有「大项」**，只剩下合计 <2% 的碎块（Conv 79%→约 1.2%、格/逐元素 32 ms）。
要继续拿分只能是「多块各 0.5~1% 的磨」或者换目标，这一点必须在排新项前说清

P3 与 P1' 都动逐元素路径，**必须一起做**，否则两次回归两次重测。
P2 只改卷积 epilogue，与 Conv 内核本身无关（内核已 79% 效率，不值得再改），可以独立排。

## 6. 每项交付物（缺一不可）

1. 代码 + 单元/回归测试通过（现有 29/29 基线不得下降）
2. 交替 A/B ≥ 5 轮的 min 表（写进 `bench.md`）
3. 4 模型 `max|diff| = 0` 的比对结果
4. `progressv2.md` 追加一条日志
5. 若被否决，写进 `bench.md` 的「否决记录」并说明测量方式 —— 否决本身也是资产

---

## 7. 算子语义参考：`Operators.md`

根目录 `Operators.md`（1.4 MB / 48834 行）是 **ONNX 官方 `onnx/defs/gen_doc.py` 自动生成的算子
schema 全文镜像**，不是本项目写的文档。因此 **只看不改**：要改请回 onnx 仓库改 `defs/` 里的算子
定义后重新生成，否则下次同步就被覆盖。

覆盖范围：

| 域 | 算子数 | 与本项目的关系 |
|---|---:|---|
| `ai.onnx`（默认） | **210** | 唯一实际导入的域 |
| `ai.onnx.preview` | 2 | `FlexAttention`（experimental），不导入 |
| `ai.onnx.preview.training` | 8 | Adagrad / Adam / Gradient / Momentum 等，训练用，不导入 |

每个条目给出：输入输出（含可选输入与变长输入）、全部属性（名字 / 类型 / 默认值 / 允许值）、
类型约束（`T` 绑定的元素类型集合）、shape inference 说明，以及**逐 opset 的差异**
（`since_version` 列表 + `Changelog.md#Op-13` 形式的链接）。

### 7.1 三条使用纪律

1. **语义的唯一裁决者**。实现/修某个算子前先读它的条目：输入顺序、属性默认值、`axis` 的负数含义、
   广播规则、`ceil_mode`/`count_include_pad` 这类细节，**以本文档为准**，不去猜、也不把 ORT 的
   实现当规范（ORT 有偏离规范的历史 bug）。与本文档冲突的「优化」一律当成 bug。
2. **版本差异查表**。同一个算子在不同 opset 下语义会变，典型：`Slice` 10 起 `starts/ends/axes/steps`
   从属性变输入；`Squeeze`/`Unsqueeze` 13 起 `axes` 变输入；`Softmax`/`LogSoftmax` 13 起从
   「按 axis 展平后归一」改成「只沿 axis 归一」；`Resize` 11/13/18 三次改坐标变换；
   `LayerNormalization` 17 才进规范。导入器里每个 `IsSince(op, n)` 分支都应对着 changelog 写，
   不要凭印象。
3. **形状推断的对照物**。`dump-graph` 的 Arg shape 与 `onnx.shape_inference` 不一致时，先回本文档
   确认是规范允许的不确定性（`Reshape` 的 `-1`/`0`、`NonZero`/`Compress` 的动态维、`If` 分支形状）
   还是我们的 bug。

### 7.2 实现覆盖（210 → 80）

`LayerRegistry` 当前注册 **88** 个名字，与默认域规范表对比如下：

| 项 | 个数 | 说明 |
|---|---:|---|
| 命中规范表 | **80** | 真正实现了的 ONNX 算子 |
| 不在规范表内 | 8 | `Silu`（导出器别名，规范里叫 `Swish`）、`FusedEltwise` / `FusedAttention` / `DflSoftmax`（融合产物，无 ONNX 对应）、`DynamicQuantizeMatMul` / `QLinearAdd` / `QLinearSigmoid` / `MatMulArgMax`（ORT 贡献算子或融合标记） |
| 未实现 | 130 | 训练/序列/量化/字符串/随机/控制流为主：`GRU` `LSTM` `RNN` `Loop` `If` `Scan` `NonMaxSuppression` `QLinearMatMul` `StringSplit` `RandomUniform` `Sequence*` `Optional*` …；另有少量可拼出来的激活（`Gelu` `LeakyRelu` `HardSwish` `PRelu` `Elu` `Mish`） |

- **未实现算子的行为**：`LayerRegistry.Create` 直接抛 `op type 'X' is not registered`，fail-fast，
  不静默退化成一个「大概对」的参考实现。
- **为什么不追全 210 个**：主战场是检测/分割模型导出的算子集，剩下 130 个里绝大多数在推理图里
  根本不会出现；新增算子的触发条件永远是「有真实的 onnx 跑不通」，而不是「规范里还有 130 个没写」。
- **待办（不排期）**：用 `Operators.md` 的 `T` 类型约束自动生成 dtype 支持矩阵，比手写表可靠；
  也顺带能校验 `GetMemoryShapes` 的边界条件与规范一致。
