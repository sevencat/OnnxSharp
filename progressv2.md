# OnnxSharp 优化进度 v2

> 对应策略见 `planv2.md`，性能数字唯一事实源是 `bench.md`（本文件只记**状态与结论**，
> 不重复放全表，避免两处打架）。
> 口径：**4 线程**、FP32、batch=1、静态形状。动笔前先确认
> `(Get-Counter '\Processor Information(_Total)\% Processor Performance') > 80%`。

- 立项：2026-11-15
- 当前阶段：**P6' / P4 / P3 / P5 全部完成（rfdetr 246 → ~208 ms），下一个做 GEMM 的 C 写面板（0.6%）或测量工具**

---

## 1. 状态总览

| 项 | 内容 | 状态 | 预期收益（rfdetr） | 实测 |
|---|---|---|---|---|
| P0 | 测量基线冻结（4 线程 / 温度门控 / env bug） | **完成** | — | — |
| P1 | Conv 低线程效率 | **已否决**（实测 11.2 ms） | ~~-6~7%~~ | 上限 -3~4% |
| P1' | 逐元素 32 ms 为什么贵 | **已关闭**（实测 12 ms） | ~~-8%~~ | 上限 -1% |
| P2 | Conv+SiLU/ReLU 融合 | **已否决**（激活只有 0.4 ms） | ~~-2.8%~~ | 0 |
| P3 | `Where` 逐元素搬移 + 折叠漏网 | **完成** | -1% | **2.6 → 0.1 ms**（按行选择） |
| P4 | Transpose 折进 GEMM 读面板 | **完成** | -2.4% | **-4.5%**（234.1→223.7），Transpose 111→75 |
| P5 | Reshape / GatherElements / 小算子 | **完成**（只做 Reshape） | -1% | **Reshape 4.4 → 0.2 ms**（槽位复用）；GatherElements 1.2 不做 |
| P6 | 并行粒度按线程数自适应 | **前提证伪**，降为观察项 | ~~结构性~~ | 1→2→4 近线性 |
| P6' | GEMM 微内核 70~74% → 85% | **完成**（MR=6 + MR=12） | ~~-13%~~ 上限 -10% | 打包路径 **92.2%**，端到端 **-9.8%**（246.0→221.8） |
| — | 内存复用（ORT memory pattern 对应项） | **完成**（修 `Fit` 缩容 bug） | 0（隐患消除） | 54.8→1.4 MB/帧 |

目标线（4 线程 median）：rfdetr-small **262.5 → ≤ 210**、yolo11n **39.1 → ≤ 27**。

当前（2026-11-16 四引擎重测，median）：rfdetr-small **217.4 ms**（262.5 → **−17.2%**）、
yolo11n **37.2 ms**（39.1 → −4.9%）。阶段 A 目标 ≤228 已达成；最终目标 ≤210 还差 **3.5%**。
（profile 的 sum-of-min 是 208.5 ms，比 median 低 3~5% 属正常，两者不矛盾。）
5% 以下的收益判定只看 profile 的逐算子 min 求和，见 `bench.md §15.3 / §16.2`。

---

## 2. 基线（2026-11-15，冻结）

`python bench/run_all.py --iters 20 --rounds 2`，同会话，median（min）：

| 模型 | onnxsharp | onnxruntime-cpu | opencv-dnn | simdyolo（0.11.4） |
|---|---:|---:|---:|---:|
| yolov8n | 46.0（45.6） | 28.3（27.9） | 54.5（48.5） | 31.4（29.9） |
| yolo11n | 39.1（37.5） | 23.6（23.5） | 48.6（46.8） | 31.8（29.2） |
| yolov8s | 121.0（120.0） | 76.7（76.1） | 138.8（135.5） | 78.7（76.6） |
| yolo11s | 111.7（103.0） | 65.1（63.8） | 133.4（127.6） | 77.2（76.7） |
| rfdetr-small | 262.5（252.4） | 189.4（187.9） | 326.3（317.8） | N/A |

**可复现性**：n/s 档单模型跑两轮差异 < 1%。**但 s 档连续跑会撞墙**（见 `bench.md §8.1` 补充：
`yolov8s`+`yolo11s` 连跑约 40 s，perf% 138 → 20.9，`yolo11s` median 虚高到 526 ms）。

### 2.1 当前成绩（2026-11-16，四引擎同会话重测）

median（ms），4 线程，静态 batch=1，FP32，纯前向：

| 模型 | onnxsharp | vs 基线 | onnxruntime-cpu | opencv-dnn | simdyolo | vs ORT |
|---|---:|---:|---:|---:|---:|---:|
| yolov8n | **43.0** | −6.5% | 27.9 | 50.3 | 33.0 | 1.54× |
| yolo11n | **37.2** | −4.9% | 23.4 | 46.3 | 30.4 | 1.59× |
| yolov8s | **116.2** | −4.0% | 77.0 | 138.5 | 82.1 | 1.51× |
| yolo11s | **108.6** | −2.8% | 67.3 | 133.2 | 77.9 | 1.61× |
| rfdetr-small | **217.4** | **−17.2%** | 188.8 | 326.0 | N/A | **1.15×** |

- 对 OpenCV dnn 全面领先（0.67× ~ 0.85×）；对 SimdYolo 仍慢 22%~42%。
- 与 ORT 的差距从 1.39×~1.72× 收窄到 **1.15×~1.61×**，rfdetr 已进入 15% 以内。
- 目标线复核（median）：rfdetr-small **262.5 → ≤210**，现 217.4，**差 3.5%**；
  yolo11n **39.1 → ≤27**，现 37.2，差距仍在（yolo 的瓶颈是 GEMM 打包，见 `bench.md §12/§13`）。
- 测量口径与作废规则见 `bench.md §16.2`。
- **新发现（`bench.md §16.3`）**：yolo11n 的 Conv 占 **84.4%（30.7/36.4 ms）**，87 个 conv 最大单个
  只有 1.94 ms —— yolo 系的瓶颈是「大量中等尺寸 Conv」，不是 GEMM。`planv2` 里 P1 的否决
  （「Conv 只有 11.2 ms」）用的是 **rfdetr** 的形状，对 yolo 系不成立。yolo 要提速必须重开 Conv，
  需按 yolo 的形状分布重做阈值/打包分析。**未排期**（当前目标是 rfdetr ≤210）。

---

## 3. 差距分解（rfdetr-small，4 线程；Conv/GEMM 已改为微基准实测）

| 类别 | 我们 | ORT | 差 | 归属 |
|---|---:|---:|---:|---|
| GEMM 类 | ~180（实测 6 类 157） | 150 | +30 | **P6'（微内核 70~74% → 85%）** |
| Conv | **11.2（实测）** | 8 | +3 | 原 P1 **已否决**，只剩 epilogue 融合（P2） |
| 逐元素 + 激活 | ~32 | 4.5 | +27 | **P1'**、P2、P3 |
| Transpose | ~11 → **7.5**（111 → 75 个） | 5 | +6 → +2.5 | P4 **已完成**；余项在 C 写面板 |
| Where | **0.1** | 0 | — | P3 **已完成** |
| GatherElements | ~2.5 | 0.6 | +2 | P5 判定不做 |
| LN/Softmax/GridSample/Reshape | ~16（Reshape 4.4 → **0.2**） | 14 | +2 | Reshape **已完成**，其余暂不动 |
| 合计 | ~245 | 189 | +56 | |

结论一句话：**GEMM 占 69% 且只跑到 70~74% 峰值；Conv 只占 4.3%（原 27 ms 是折算错误）。
真正的低成本机会在逐元素 32 ms，真正的上限在 GEMM 微内核。**

> 本表是 P0 期（11-15）的分解快照，逐元素那 32 ms 后来证明只有 12 ms（P1'）。
> GEMM 已从 ~180 降到 ~168（MR=12 之后），Transpose 从 11 降到 7.5，Where/Reshape 已归零。
> **当前实测成绩见 §2.1。**

---

## 4. 已完成工作（P0）

### 4.1 温度墙定位（2026-11-15）

16 线程同代码同模型 min 在 108 ↔ 780 ms 之间跳（7 倍）。满载期间采样
`% Processor Performance` = **20.96 / 21.05 / 21.00**（15 个点极稳），空闲 138%。
`138 / 21 ≈ 6.6` 对得上 7 倍。

根因：电源方案 `QuickCpuScheme` 的 `PROCTHROTTLEMIN = 100%`，CPU 常驻最高频，满载即撞墙。

处置：
- 默认线程数 16 → **4**（`run_all.py` / `run_python_bench.py` / `ort_inspect.py` 三处）
- `bench.md` 顶部口径表加注：11-07 ~ 11-11 的历史段落是 16 线程，且 onnxsharp 列实为 32 线程
- 永久修复建议（未执行，需管理员）：`powercfg /setacvalueindex SCHEME_CURRENT SUB_PROCESSOR PROCTHROTTLEMIN 5`

### 4.2 `ONNXSHARP_THREADS` 从未生效（bug，已修）

`bench/run_all.py`：

```python
env = dict(os.environ, ONNXSHARP_THREADS=str(threads))          # 算出来了
code, out, err = run([...])                                      # 没传进去
```

`run()` 签名原为 `run(cmd, cwd=ROOT)`，`subprocess.run` 拿不到 `env`。
后果：`--threads N` 对自研引擎完全无效，一直跑 `Environment.ProcessorCount`（本机 32）。

修复：`run(cmd, cwd=ROOT, env=None)` + 调用处 `env=env`。

**影响范围**：`bench.md` 11-07 起所有 onnxsharp 的绝对值都是 32 线程跑出来的。
同会话内的**相对倍率**仍有效，绝对值以 4 线程重测为准。

### 4.3 ORT 侧诊断工具（`bench/ort_inspect.py`，新增）

```powershell
python bench/ort_inspect.py --names rfdetr-small --no-dump          # 计时（单 session、单进程）
python bench/ort_inspect.py --names rfdetr-small --profile          # 逐算子占比
python bench/ort_inspect.py --names rfdetr-small --compare-levels   # 四档图优化对比
```

顺手修掉的三个坑：

1. 带 `optimized_model_filepath` 的 session 不能用来计时（实测 300~780 vs 148 ms）→ 拆成独立 dump session，用完即删。
2. `--compare-levels` 同时驻留 4 个 session（各有线程池 + 110 MB packed 权重）会互相抢核 → 改逐档独立 session、`gc` 后重建、多轮取最好 min。
3. 导图那段会把 110 MB 的 `ort-opt.onnx` 写进 `bench/results/` → 用完删除，不进仓库。

### 4.4 关键结论（已写进 `bench.md §8.4 / §8.5 / §9`）

- ~~**GEMM 打平**：我们 154 ms（占比 61.2%）vs ORT 150 ms（占比 79.2%）~~
  **已被 §9 微基准修正**：`gemmbench` 实测 6 类主形状 157 ms，加小 GEMM 约 180 ms；
  微内核 290~313 GFLOPS = 4 核 AVX2 峰值的 70~74%，还有 26% 没吃到。
- **Conv 只有 11.2 ms（4.3%）**：`convbench` 实测 9 层，Winograd 167 GMAC/s（79% 峰值）。
  §8.4 折算的 27 ms 是错的（`profile` 回调按层数摊导致占比失真）→ 原 P1 否决。
- **并行扩展性接近线性**：1→2 为 1.95×、2→4 为 1.97×、4→8 为 1.61× → 原 P6 前提不成立。
- **ORT 的图优化对 RF-DETR 是负收益**：`disable` 182.07 / 182.77 vs `all` 186.87 / 187.14，
  两次复现。原因是 `MatMul::PrePack` 属于 kernel 级（session 初始化时对常量权重做），
  与 `graph_optimization_level` 无关；而优化引入的 253 个 `Reshape` + NCHWc Reorder 反而要付费。
- **4 线程下 OpenCV 退化最严重**（yolo11n 28.6 → 48.6），我们反而领先 16~20%。
  「谁快」是线程数决定的，引用任何对比都必须带线程数。

### 4.5 内存复用（ORT memory pattern 的对应项）已闭环

此前的判断是「ORT 的 PrePack / memory pattern / 常驻线程池三样我们都已等价实现」。
本轮拿这句去做验证，发现**第三项是真缺陷**：

- **病灶**：`Tensor.Fit` 把 `DType != dtype` 也当作重分配理由。`Buffer` 是 `byte[]`，
  `DType` 只是解释视图。共享槽位被 `[1,128,32,32] I64 (2400 B)` 与
  `[1,6,64,1028] F32 (25 MB)` 轮流使用，I64 那一步把 25 MB 缓冲缩成 2.4 KB，
  下一帧 F32 再重新分配 → **每帧都在摧毁自己刚建好的复用容量**。
- **效果**：每帧分配 **54 823 → 1 397 KB**、`Fit` 重分配 **19 次 → 0**、gen2 GC **5 → 0**。
- **等价性**：`git stash` 前后双 dump 逐字节比对，`rfdetr out0/out1`、`yolo11n out0` 全部 IDENTICAL。
- **但不是提速项**：`gc pause` 实测仅 0.1~0.2 ms，端到端 252.6 → 254.2（噪声内），
  回归 yolo11n 38.0/38.9、yolov8n 46.3/46.9、rfdetr 254.0/261.9 全过。
  定位为**消除 GC 抖动隐患**，不计入阶段 A 的收益。
- 顺带修掉一个失效判断：`Interpreter.Run` 里 `signatureChanged` 依赖 `ShapeCacheValid`，
  动态形状层永不写该标志 → 每帧重跑 `Finalize`。已改为按 in-shapes/in-dtypes 比对
  （改完分配数字不变，说明它不是大头，但逻辑上必须对）。

详见 `bench.md §10`。

### 4.6 逐元素「32 ms」被推翻，P1'/P2/P3 三项作废

先把 `profile` 的分解校准好：加了预热后 `sum-of-min` 与 `bench` min 只差 0.4%（252.5 vs 253.5），
于是可以直接按绝对值读，不必再用份额折算。然后：

- **逐元素合计 12 ms（4.7%）**，不是 32 ms。`Mul 5.4`、`FusedEltwise 4.2`、`Add 1.2`、
  `Div 0.7`、`Relu+Silu 0.4`。
- 新加 `dump-graph --inputs` 算真实流量：`Mul` 153 MB → **31 GB/s**，`Add` 127 MB → **106 GB/s**
  （就是这台机器的带宽上限）。
- 新加 `eltwisebench`：连续内核在 L3 常驻时 **100+ GB/s**（比 memmove 参考还高），
  所以内核没有算法缺陷，图里 `Mul` 慢是缓存不命中。
- **结论**：`P2`（Conv epilogue 激活融合）上限 0.4 ms → 否决；`P1'` 上限 ≈ 2.5 ms → 关闭；
  `P3` 收窄成「`Where` 一个 2.6 ms」+「`Reshape` 4.4 ms 待查」。

顺带做了两件事：`ApplyBroadcast` 的分段 switch 提到内核外（`mul-tail` 0.71 → 0.53 ms，
图中 `Mul` 5.4 → 4.9 ms，输出逐字节一致）；`dump-graph --inputs` 保留下来做流量分析。

详见 `bench.md §11`。

### 4.7 P6' 第 1 步：微内核 MR 4→6，端到端 -5.0%

先把「峰值」钉死再动手：`flopsbench`（纯 FMA 环）给出 1 线程 111.0 / 4 线程 **427.1 GFLOPS**，
GEMM 满载 6 s 内 `% Processor Performance` 恒定 137.8（≈3.45 GHz）→ **不存在降频**。
以前说的「70~74%」是拿 290 GFLOPS 比一个含糊峰值，现在可以精确算：
`1028×384×1536` 打包路径 3.400 ms = 356.6 GFLOPS = **83.5%**，未打包 3.759 ms = 75.6%。

- **内核**：新增 `Multiply6Rows`（6×16 = 12 累加器），原来 4×16 每 k 只有 `2+4=6` 条 load 对 8 FMA
  （25% 余量），现在 `2+6=8` 条 load 对 12 FMA（50% 余量），每行 B 流量降到 2/3，
  与 OpenBLAS Haswell sgemm 同款分块。三处主循环（打包/面板/非打包）都改成 `6 → 4 → 1` 级联收尾。
- **微基准**：`1028,384,1536` 295.8 → 321.1 GFLOPS，`1028,1536,384` 297.8 → 323.4，
  `300,256,2048` 310 → 339，其它形状 +6~9%。
- **端到端**（交替 A/B、冷却后取 min of 3）：**246.0 → 234.3 ms（-5.0%）**，极差 < 1.2%。
  `yolo11n` 37.9 ms 不变（卷积模型，MatMul 没几个）。
- **批量 GEMM 合并保留**：`MultiplyBatched` 把 `batch × 列块` 一次派发，注意力 QK^T/PV 实测
  端到端 **1.048×**（238.1 vs 249.6），此前只是「收益有限」的猜测。
- **两个猜测被推翻**：①「大 k（n 小）掉一半」是降频假象（同命令复测 47 → 297.8 GFLOPS，
  同一进程连跑 4 个形状会撞墙，必须每形状单独起进程）；②`--misalign 1` 对
  `257,64,257×24` 无影响，4 字节对齐的 `byte[]` 切片不是问题。
- **等价性**：`bench/ab/` 前后 dump 逐位 IDENTICAL（rfdetr out0/out1、yolo11n out0），
  MR6 vs MR4（`ONNXSHARP_GEMM_MR=4`）输出 MD5 亦相同。
- **遗留口径问题**：同一个注意力 MatMul（203 MFLOP）在 `gemmbench`/`MM_TRACE`/`profile` 下
  分别读出 0.68 / 0.80 / 1.47~3.60 ms，且预热越多数字越大 → **per-op 小算子的 min 不可单独引用**，
  判断一条路径值不值得做只能靠「关掉它」的端到端 A/B。

详见 `bench.md §12`。

### 4.8 P6' 第 2 步：微内核 MR 6→12，打包路径到 92%

6 行版每个 k 是 8 条 load 对 12 条 FMA（load 占 67%），12 行时 14 条对 24 条（58%），
顺带把 B 面板被重流的次数减半。24 个累加器 + 2 B + 1 广播 = 27 个 ymm，只有 AVX-512 的
32 个向量寄存器装得下，所以 `UseMr12` 按 `Avx512F.VL.IsSupported` 运行时分派。

- **踩坑**：第一版照抄 6 行内核写成 `var b0v = Vector256.Create(bias[0]) … b11v`，
  24+12+3 > 32 直接溢出，只快 2% 且方差 6.8%。改成「初始化时就地广播 + float 标量」
  后持久向量降到 27 个，性能跳到 11%。
- **微基准**（交替 A/B、冷却 20 s、两次复现）：`1028,384,1536` 打包
  **3.424 → 3.079 ms**（356.6 → **393.8 GFLOPS = 峰值 427 的 92.2%**），未打包 318 → 349。
- **端到端**（rfdetr，6 对交替 A/B，含反向顺序）：5/6 对 MR=12 更快，中位 **-5.4%**；
  最好读数 **221.8 ms**（相对 MR=4 时代的 246.0 累计 **-9.8%**）。唯一反例伴随整机发热。
- `yolo11n` 同协议 37.9 → **36.4 ms**。
- **等价性**：MR=12 与 MR=6 的 dump 逐位 IDENTICAL（行块变宽不改变 k 递增的求和顺序）。

详见 `bench.md §13`。

### 4.9 P4：Transpose 折进 GEMM 的读面板，端到端 -4.5%

先探针确认结构（原来想当然的 `Transpose[0,2,1,3] → MatMul` 并不存在）：

```
q[B,T,H,D] ─T[0,2,1,3]→ [B,H,T,D] ─Mul(·,s)→ A ┐
k[B,T,H,D] ─T[0,2,3,1]→ [B,H,D,T] ─Mul(·,s)→ B ┴→ MatMul      (QK^T)
P[B,H,T,T]（Softmax）──────────────────────────→ A ┐
v[B,T,H,D] ─T[0,2,1,3]→ [B,H,T,D] ─────────────→ B ┴→ MatMul  (PV)
```

- **做法**：`FuseTransposeIntoMatMul` 把转置换成 GEMM 的读 stride（A 行距 `H*D`、B 行距 1 或列距 `H*D`），
  `gemm_view` 12 个整数驱动 `MatMulLayer` 的视图分支。
- **`Mul(·, 1/√8)` 不折数值**：`s` 不是 2 的幂，`(q·s)·(k·s) ≠ s²(q·k)`，折进 B 打包就破坏逐位一致；
  正确做法是把 `Mul` 的**输入**从转置输出改接到转置的源（乘标量与转置可交换，元素值逐个不变）。
- **踩坑**：B 侧是裸 Transpose（PV）时，`matmul.Inputs[1]` 指向已被 drop 的层输出 → rfdetr 输出全错
  （max|diff| 0.99），而 QK^T 单独折却完全正确、yolo 系 0 命中看起来「也一致」。
  二分办法：临时用环境变量分别只折 `bTransposed==true/false`。修法是
  `Inputs.Add(bScale?.Outputs[0] ?? bSource)`。
- **结果**：Transpose 111 → 75（-36），命中 24 处；交替 A/B 有效 7 轮中位 **-4.5%**（234.1 → 223.7 ms），
  最好 221.4 ms；5 模型 dump 逐位一致；yolo11n 持平（0 命中）。
- **判定**：节点数 -32.4%（未到 40% 的初版线，剩下的在写面板/无 MatMul 消费者），端到端 -4.5% → **通过**。
- **pass 顺序**：必须排在 `FuseAttention` 之后，否则合成 MHA 图被提前拆散
  （单测 `attention: fused MHA matches onnxruntime` 直接失败）—— rfdetr 上两者不冲突（它 0 命中整块 MHA）。
- **余项**：PV 输出 `Transpose[0,2,1,3] → Reshape` 可折进 **C 写面板**（12 个，约 1.4 ms）；
  yolo 系的 attention 是 `[B,H,T,D]` 内的矩阵转置（A 列主序），要 A 双 stride，收益 0.5 ms，不做。

详见 `bench.md §14`。

### 4.10 P3/P5：Where 按行选择 + Reshape 零拷贝，合计 -2.0%

两项都是「本来不该花这个时间」的开销：

- **Where 2.6 → 0.1 ms**。形状 `cond[1,1024,1] + 标量 + [1,1024,256]`，每行 256 个元素共享一个
  布尔值，原实现却逐元素 `CopyBytes`（每元素 3 次 `Span` 构造，10 ns/元素）。
  新增 `ForwardRows`：尾维「条件广播 + 源连续或广播」时按行整段 memcpy / 按等宽整数 `Fill`。
  填充用 `Span<short/int/long>.Fill` 而不是按 dtype 分派 —— float 的位模式按 int 填完全一样。
- **Reshape 4.4 → 0.2 ms**（119 个，平均每次 3.6 MB 的整块 memcpy，纯浪费）。
  `LayerInfo.InPlaceInput`（默认 -1）+ `BufferAllocator` 槽位共享：Reshape 的输出 arg 与输入 arg
  共用一个槽（= 同一个 `Tensor` 对象），`Forward` 里 `ReferenceEquals` 直接 return。
  两条安全前提缺一不可 —— **输入必须是中间结果**（图输入/常量改形状会污染调用方）、
  **本 op 必须是该输入的最后读者**（否则下游读到被改过的形状）。`ONNXSHARP_NO_INPLACE=1` 可关闭。

**测量教训（重要）**：端到端 bench 已经测不出这个量级的改动了。同一配置下「后跑的那个进程」
平均慢 **11 ms（4.7%）**，比本轮收益还大；交替顺序反演（new 先跑 3 对 −15.6 ms、base 先跑 3 对
+6.4 ms）得 Δ ≈ −4.6 ms（−2.0%），与单项 profile 的 −2.5…−6.6 ms 吻合。
**以后端到端只认 profile 的 sum-of-min（211.0 → 208.5 ms）**，bench 当粗筛。

详见 `bench.md §15`。

---

## 5. 下一步（按 planv2 §4 执行）

0. ~~补回 A/B 比对工具~~ **已完成**：`bench/ab_compare.py`（现场双 dump 或 `--a/--b` 比目录，
   exit code 0/1 可直接做门禁）。`run_all.py` 也加了 SimdYolo 线程数断言。
   SimdYolo 已升到 **0.11.4** 并重测（收益在噪声内），基线表已更新。
1. ~~P1 先测后改（Conv）~~ **已测完并否决**：Conv 实测 11.2 ms（4.3%），Winograd 已 79% 峰值，
   三个候选根因全部证伪。详见 `bench.md §9`。
2. ~~P1'（逐元素为什么贵）~~ **已关闭**：`eltwisebench` + `dump-graph --inputs` 证明逐元素只有
   **12 ms（4.7%）**，`Add` 106 GB/s、`Mul` 31 GB/s（缓存不命中）、`Relu+Silu` 0.4 ms。
   原「32 ms」是陈旧口径。详见 `bench.md §11`。
3. ~~P2（Conv epilogue 激活融合）~~ **已否决**：激活只剩 0.4 ms，融合上限 0.4 ms。
4. ~~P6'（GEMM 微内核）~~ **已完成两步**：MR=6（752 → 83.5%）与 MR=12（→ **92.2%**），
   端到端累计 **-9.8%**（246.0 → 221.8），逐位一致。**剩余上限约 4%**，只有 BLIS 式
   A 打包 + k 分块能把未打包路径（81.7%）拉平，暂不启动。
5. ~~P4（Transpose 折进 GEMM）~~ **已完成**：`bench.md §14`，rfdetr -4.5%（中位 234.1 → 223.7），
   Transpose 111 → 75，5 模型逐位一致。
6. ~~P3 / P5（Where + Reshape）~~ **已完成**：`bench.md §15`，Where 2.6 → 0.1 ms、Reshape 4.4 → 0.2 ms，
   profile sum-of-min 211.0 → 208.5 ms（−2.0%），6 模型逐位一致。GatherElements（1.2 ms）
   与 Softmax / FusedEltwise / GridSample 都已在访存上限，不做。
7. **下一步候选**（按性价比排）：
   - **P4 余项：C 写面板**。PV 输出的 `Transpose[0,2,1,3] → Reshape` 属于「写侧布局」，
     让 GEMM 直接把 C 写到 `[B,T,H,D]` 布局，12 个节点约 1.4 ms（0.6%）。
     代价是所有 MR×NR 微内核的 C 写要支持 stride，需新增一组写路径。
   - **P6' 余项（BLIS 式 A 打包 + k 分块）**：上限约 4%，工程量最大，最后考虑。
   - **yolo 系的 Conv（新开，未排期）**：Conv 占 yolo11n 的 84.4%（30.7 ms），比 ORT 全模型还慢；
     87 个 conv 最大单个 1.94 ms → 是「中等尺寸 conv 的效率」问题，要按 yolo 的形状分布重做
     Winograd/Im2Col 阈值与打包开销分析。**收益上限 ~25%（对 ORT 从 1.59× 拉到 1.2× 量级），
     但工作量大且与 rfdetr 主线无关**，等 rfdetr ≤210 达成后再评估。
   - **`Operators.md` 的 dtype 矩阵自动生成**：把规范的 `T` 约束接进 `GetMemoryShapes` 的校验，
     不做性能、只减少「某 dtype 悄悄走错分支」这类 bug（不排期）。
   - **测量工具**：给 `bench` 加同进程交替 A/B（两次 `Bind` 换配置跑同一份输入），
     否则 5% 以下的收益在本机永远测不准。

---

## 6. 日志

### 2026-11-16

- **P7-D 验证：GEMM 并行阈值不是瓶颈（`bench.md §17`）**。阈值改为可配
  （`ONNXSHARP_GEMM_THRESHOLD` / `gemmbench --threshold`，默认仍是 `1<<22`），
  但 rfdetr 的 139 个 MatMul 最小 `m·n·k` = 19.6 M ≫ 4.19 M，降阈值不改变任何一层的
  并行决策，端到端交替 A/B 217.8 vs 237.4（噪声内）→ **D 关闭，默认值不变**。
  顺带加了 `gemmbench --attn 1`（复刻 QK/PV 的真实形状+视图+派发序列）和
  `--btrans`（K^T 视图 B）、`MMv` trace 补 stride。
- **新缺口：QK/PV 在图里比同参数探针慢 1.8×**（1.73 vs 0.95 ms、1.03 vs 0.54 ms，
  12 层 33.1 ms）。不是打包/派发，而是每层 6.3 MB attention 矩阵往返（25 MB/层，L2 1 MB）。
  候选 P8「QK→Softmax→PV 分块融合」，上限 −4%，未排期。
- **否决并用回退的微优化**：`PackColumns` 的 `bColStride != 1` 分支改「按行优先」，
  微基准 5.314 → 0.728 ms（`bColStride==k`），真实 `bColStride=384` 下 0.958 vs 0.952 无差 ——
  已回退，`--btrans` 留作探针。回归：单测 116/118（2 skipped，0 失败）、rfdetr 端到端 min 212.2。
- **通读 ORT 1.32 源码，产出参照清单（`planv2.md §P7`）**：三个假设被证伪 —— CPU EP 里
  **没有 oneDNN**、MLAS 里**没有 Winograd/FFT**、AMX 只接 int8。ORT 的 Conv 优势全在
  「NCHWc(16) + 图优化期权重 `OIHWBiBo` 重排 + **不物化 im2col** + 输入通道块外提 + 24 累加器
  微内核」，我们对应位置是 `PackedConv`（面板 16 像素 × 4 行，`RowBlock=4`）。
  排序后的待办：**D（GEMM 并行阈值 `1<<22` vs MLAS 的 `M·N·K/65536`，高 64 倍）落在 rfdetr 主线**；
  A/B/C（去物化、行块 4→8 走 AVX-512VL、ic 块外提）只对 yolo 有用（Conv 占 84.4%）。
  **NCHWc 全图布局传播明确否决**：rfdetr 同口径 `sum-of-min` 是 MatMul 139 个 180.3 ms（77.1%）
  vs **Conv 9 个 12.0 ms（5.1%）**，NCHWc 对 2D MatMul 零收益，Conv 就算 2× 也只省 2.6%，
  还要在 18 处 Conv/逐元素边界付 reorder。
- **全引擎对比重测（`bench.md §16`）**：rfdetr-small **262.5 → 217.4 ms（−17.2%）**，
  vs ORT 从 1.39× 收窄到 **1.15×**；yolo 四档 −2.8%…−6.5%（vs ORT 1.51×~1.61×）；
  对 OpenCV dnn 全面领先 0.67×~0.85×，对 SimdYolo 仍慢 22%~42%。
  测量教训：多引擎同会话时「后跑的被余热拖慢」，本轮出现 ORT rfdetr median 254.8 / min 189.9、
  ours 337.9 / 243.1 这种 min–median 差 100 ms 的自相矛盾数据 → **该轮作废**，
  改为每引擎独立会话 + 冷却到 perf 137% + 单引擎最好值。

### 2026-11-15

- **P3/P5：Where 按行选择 + Reshape 槽位复用**。Where（`cond[1,1024,1]` + 标量 + `[1,1024,256]`）
  逐元素搬移 10 ns/元素 → 按行整段 memcpy / 等宽整数 `Fill`，**2.6 → 0.1 ms**；
  Reshape 119 个整块 memcpy → `LayerInfo.InPlaceInput` + allocator 槽位共享（输出与输入同一个
  Tensor，Forward 直接 return），**4.4 → 0.2 ms**。前提：输入是中间结果 + 本 op 是最后读者。
  合计 profile sum-of-min 211.0 → 208.5（−2.0%）、yolo11n 36.9 → 36.5，6 模型逐位一致。
  **教训**：本机端到端 A/B 的顺序偏差达 11 ms（后跑者慢 4.7%），5% 以下的收益只能靠
  profile 的逐算子 min 求和判断。
- **`Operators.md` 纳入文档体系**（`planv2.md §7`）。它是 ONNX 官方 `gen_doc.py` 生成的算子 schema
  全文（默认域 210 个 + preview 2 + preview.training 8），**只读镜像、不手改**。
  写清三条纪律：算子的语义/属性默认值/负数 axis 以它为准（不拿 ORT 当规范）、opset 差异查它的
  changelog 段（`Slice`/`Squeeze`/`Softmax`/`Resize` 都改过语义）、形状推断差异先回它确认是否规范允许。
  附实现覆盖：`LayerRegistry` 注册 88 个，命中规范表 **80** 个（8 个是别名/融合产物/ORT 贡献算子），
  未实现 130 个以训练·序列·量化·控制流为主 —— 不做，等真实模型跑不通再说。
- **P4：Transpose 折进 GEMM 读面板**。探针先推翻原设想（attention 的 A/B 各自还套着一个
  `Mul(·, 1/√8)`），改为「折转置、留乘法、只把 Mul 的输入改接到源张量」；
  `gemm_view` 12 元组 + `PackColumns(srcColStride)` 吃转置读。
  Transpose **111 → 75**，命中 24 处，交替 A/B 中位 **-4.5%**（234.1 → 223.7 ms，7 轮有效），
  最好 221.4 ms；5 模型 dump 逐位一致；yolo11n 0 命中（`[B,H,T,D]` 内的矩阵转置，需 A 双 stride，收益 0.5 ms 不做）。
  教训：B 侧是裸 Transpose 时必须把输入换成源张量，否则视图层读到已删层留下的空槽 ——
  **且这个 bug 只在 PV 上现形，QK^T 和 yolo 全是对的**，靠环境变量二分才定位到。
- **P6' 第 2 步：微内核 MR 6→12**。新增 `Multiply12Rows`（24 累加器 = 27 ymm，按 `Avx512F.VL` 分派），
  打包路径 356.6 → **393.8 GFLOPS（92.2%）**，微基准 +11.2%；端到端 6 对交替 A/B 中位 **-5.4%**，
  最好 **221.8 ms**；与 MR=6 逐位一致。教训：12 个 bias 向量做成持久局部变量会溢出（只快 2%），
  改成初始化时就地广播才拿到 11%。
- **P6' 第 1 步：微内核 MR 4→6**。新增 `Multiply6Rows`（12 累加器，load 余量 25%→50%），
  打包路径 356.6 GFLOPS = 83.5%（峰值 427.1 由 `flopsbench` 实测），
  端到端交替 A/B **246.0 → 234.3 ms（-5.0%）**。MR6 vs MR4 及改动前后 dump 全部逐位一致。
- **`MultiplyBatched` 合并批量 GEMM**：一次派发 `batch × 列块`，注意力 QK^T/PV 端到端 **1.048×**。
- **口径纪律**：`profile` 的 per-op min 在小算子上会随运行时间变大（同一 MatMul 0.68/0.80/1.47/3.60 ms），
  且同一进程连跑多个形状会撞降频墙 → 单形状单独起进程 + 冷却 10~20 s + 取 min；
  路径价值一律用「关掉该路径」的端到端 A/B 判定（`bench.md §12.6`）。
- **`--misalign` 结论**：4 字节对齐的 `byte[]` 切片对 32 字节向量 load 无影响。
- **新工具 `eltwisebench`**：逐元素单算子微基准，每项给 ms / GB/s / ns·元素，并自带 `copy`（memmove）
  作带宽参考；另加 `copy1`（逐元素 4 字节 `CopyTo`）复刻 `Where`/`GatherElements` 的搬移方式。
  默认形状取 rfdetr 的真实分布（`[4,257,384]` 24 次为最大宗，`[1,300,256]` 刚好卡在
  `ParallelElementThreshold` 之上）。
- **`dump-graph --inputs`**：同时打印输入形状，用来区分标量广播（2 趟）/ 尾广播（≈2 趟）/ 全长（3 趟），
  从而算出真实字节数。`Mul` 153 MB、`Add` 127 MB、`Relu` 17 MB、`Silu` 10 MB。
- **推翻「逐元素 32 ms」**：同口径实测 12 ms（4.7%）。`Add` 106 GB/s（带宽上限）、`Relu+Silu` 0.4 ms、
  `Mul` 31 GB/s（缓存不命中，非内核缺陷：L3 常驻时内核能跑 100+ GB/s）。
  → `P2` 否决、`P1'` 关闭、`P3` 收窄。
- **`ApplyBroadcast` 分段 switch 外提**：`mul-tail` 0.71 → 0.53 ms（-28%），图中 `Mul` 5.4 → 4.9 ms。
  遗留：仍比连续内核慢 13×，且与段大小无关（inner 384→4112 恒定 1.3~1.7 ns/元素），
  怀疑 4K 别名但未定位；图中节点实际 0.13 ms/个，未继续追。
- **等价性**：改动前后 `rfdetr out0/out1`、`yolo11n out0` MD5 全部一致。
- **分配归因工具**：`Interpreter.Run` 增 `allocs` 参数（`GC.GetTotalAllocatedBytes(precise:true)` 逐层差值），
  CLI 加 `profile --alloc` / `bench --alloc`。探针必须包住「形状推断 + `BindAll` + `PrepackWeights` +
  `Finalize` + `Forward`」，只包 `Forward` 会漏掉 97% 的分配。
- **抓到 `Tensor.Fit` 的真 bug**：`DType != dtype` 被当作重分配理由，共享槽位在
  `I64 [1,128,32,32] (2400 B)` 与 `F32 [1,6,64,1028] (25 MB)` 间切换时把 25 MB 缓冲缩成 2.4 KB。
  rfdetr-small 每帧白分配 **54.8 MB**、`Fit` 重分配 19 次/20 帧、gen2 GC 5 次。
  修复（去掉 dtype 判断 + 诊断计数器 `FitReallocCount/Bytes/ReallocLog`）后：**1.4 MB / 0 次 / 0 次**。
- **等价性与收益口径**：双 dump 逐字节 IDENTICAL（4800/3600/2822400 字节）；端到端在噪声内
  （`gc pause` 仅 0.1~0.2 ms）→ 记为隐患消除，不记为提速。
- **修 `signatureChanged` 误判**：原依赖 `ShapeCacheValid`，动态形状层永不写该标志 → 每帧重跑
  `Finalize`。改为按 in-shapes/in-dtypes 比对（分配数字未变，说明不是分配大头，但逻辑必须对）。
- 新增 `bench/ab_compare.py`：现场跑两次 `run --dump` 逐元素比对，量出 `max|diff|` / `max_rel` /
  首个差异下标，exit code 0/1 可做提交前门禁。自检（同二进制两跑）yolo11n 逐位一致。
- `run_all.py` 解析 SimdYolo JSON 的 `CpuThreads` 并断言与请求值一致 —— 已实测确认 SimdYolo 侧
  4 线程真实生效（yolo11n min 31.5 / median 36.8，与 `bench.md` 记录一致）。
- **发现 SimdYolo 本地版本落后**：源码 0.11.3，落后 `origin/main` 7 个 commit，远程最新
  `v0.11.4`（`Release 0.11.4 with CPU and Vulkan inference optimizations`）。升级后 simdyolo 列必须重测。
- **SimdYolo 0.11.4 重测完成**（`git pull` 后源码已是 0.11.4，但 `bin` 仍是 11-10 的旧 dll ——
  重新 `dotnet build -c Release` 后 JSON 自报 `Version=SimdYolo 0.11.4`、`CpuThreads=4`）。
  结果：yolo11s 81.3 → **77.2**（-5%），yolov8s 76.2 → **78.7**（+3%），n 档持平。
  **0.11.4 的 CPU 优化对 YOLO 检测基本没收益**，P1 基线不必重排；vs SimdYolo 倍率 1.23~1.54。
- **温度墙新证据**：4 线程下 `yolov8s`+`yolo11s` 连跑约 40 s，perf% 从 138 打到 20.9，
  `yolo11s` median 526 ms（正常 77 ms）。冷却后复测正常 → **s 档必须单模型跑并逐个查 perf%**。
- **修掉 `convbench --shape` 的死参数**：`ParseOptions` 把它收进 `CliOptions.Shapes` 列表，
  而 `ConvBench` 用 `GetString("--shape")` 从字典取 → 永远 null，一直在跑 yolo11n 默认层。
  改为读 `options.Shapes` 并支持重复传多个形状。
- **§8.4 的「Conv 27 ms」被实测推翻**：`convbench` 跑 RF-DETR 真实 9 层（4 线程）合计 **11.2 ms**，
  Winograd 167 GMAC/s（4 核 AVX2 峰值 79%）；`PackedConv` 面板 64 ≥ 16 不空转、
  线程 1→2→4→8 为 1.95×/1.97×/1.61× 接近线性。**原 P1 与 P6 的前提同时被证伪。**
- **GEMM 真实成本**：全模型 27.1 GMAC（Conv 仅 1.78），6 类主形状 `gemmbench` 实测 157 ms、
  290~313 GFLOPS，占端到端约 69%，效率 70~74%。**这才是剩下的大头，且 YOLO 侧占比更高。**
- `planv2` 据此重排：目标拆成阶段 A（不动 GEMM，rfdetr 262.5 → ≤228）与阶段 B（含 GEMM
  微内核，→ ≤200）；P1'（逐元素 32 ms）升为第一优先。
- 定位温度墙（perf% 21%），基准口径 16 → 4 线程，两轮重跑差异 < 0.5%。
- 修 `run_all.py` env 未传递 bug，确认历史 onnxsharp 数据实为 32 线程。
- 新增 `bench/ort_inspect.py`（导图 diff / 逐算子 profile / 四档优化对比 / `--opt-level` / `--no-dump`）。
- 4 线程全表重测（5 模型 × 4 引擎），rfdetr 262.5 / ORT 189.4。
- 得出「GEMM 打平、差距在 GEMM 之外」与「ORT 图优化负收益 -2.6%」两条结论。
- 建立 v2 策略（`planv2.md`）与本进度文件。
