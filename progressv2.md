# OnnxSharp 优化进度 v2

> 对应策略见 `planv2.md`，性能数字唯一事实源是 `bench.md`（本文件只记**状态与结论**，
> 不重复放全表，避免两处打架）。
> 口径：**4 线程**、FP32、batch=1、静态形状。动笔前先确认
> `(Get-Counter '\Processor Information(_Total)\% Processor Performance') > 80%`。

- 立项：2026-11-15
- 当前阶段：**2026-11-17 收工**（`bench.md §37`）。rfdetr 目标线**达成**（262.5 → **198.0**，目标 ≤210）；
  yolo11n 30.4（目标 ≤27，差 12.6%）。本轮改动全在 Conv 之外：池化 / Resize 行切分 + `EliminateIdentity`
  + 默认线程改物理核 + OCR 输入张量复用。**收益只在池化/Resize 占比高的图**：OCR 端到端 −29%、
  YuNet int8 −11%；**yolo / rfdetr 持平（噪声内）**，四引擎同会话复测也证明这不是环境漂移。
  下一步若要继续 yolo：只剩微内核深度与 NCHWc 覆盖面两条路（`§26~§34` 已把 `1×1` 的账算干）。
- 补测（`bench.md §38`）：与 **MLAS 单算子**对照后，落后只落在 **F32 GEMM 单线程**（−20~30%）；
  16 线程 SGEMM 多数形状反超 ORT 1.1~2.1×、int8 打平、卷积 NCHWc 更好 ⇒ yolo 的 1.28× 不在内核。

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
| P6' | GEMM 微内核 70~74% → 85% | **完成并关闭**（MR=6 + MR=12；余项 A 打包实测负收益） | ~~-13%~~ 上限 -10% | 打包路径 **92.2%**，端到端 **-9.8%**（246.0→221.8） |
| — | 内存复用（ORT memory pattern 对应项） | **完成**（修 `Fit` 缩容 bug） | 0（隐患消除） | 54.8→1.4 MB/帧 |
| P8 | QK→Softmax→PV 分块融合 | **完成并采纳**（`bench.md §20`） | **+2.2%** | 配对中位 1.022，噪声地板 1.6% |
| — | `ab` 同进程交替 A/B 工具 | **完成**（`bench.md §20.1/20.2`） | — | 自检同配置 1.0% 跨度，可辨 ≥1% 信号 |
| P9 | attention 内核（4 步 + flash 兜底） | **完成 3/4 并关闭**（`bench.md §21`） | 目标 −4.6% | **+2.2%**（P9.1 中性 · P9.2 +0.6% · P9.3 +1.0% 逐位相同）；P9.4/9.5 按天花板关闭 |
| P7-A/B | yolo Conv 去物化 + 行块 4→8 | **已否决并回退**（命中 0 层，`bench.md §19`） | ~~−3%~~ | 够得着的两层 −21%，但 yolo 里被 Winograd 全接走 |
| P7-C | yolo ic 块外提 | **不启动**（前提同 A/B，命中面为空集） | — | — |
| P7.7 | yolo 改走 NCHWc | **阶段 1+2 完成并采纳**（`bench.md §22` 立项 / `§23` s2 / `§24` s1） | yolo −14% ~ −24% | **30 层/帧命中，单层 220 GMAC/s**；端到端 **−21.2% / −25.1% / −23.6% / −27.0% / −27.6%**（11n/8n/11s/8s/11m） |
| P10 | yolo 的 NCHWc 内核（s2 + s1） | **完成**：泛化到 stride 1，指令数不变 | — | s2 204、s1 220 GMAC/s（对照 88~184）；reorder 占 9.5% |

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

### 2.1 当前成绩（2026-11-17，四引擎同会话重测，`bench.md §37.3`）

median（ms），4 线程，静态 batch=1，FP32，纯前向：

| 模型 | onnxsharp | vs 最初基线 | onnxruntime-cpu | opencv-dnn | simdyolo | vs ORT | 上轮（§28） |
|---|---:|---:|---:|---:|---:|---:|---:|
| yolov8n | **35.8** | −22.2% | 28.2 | 50.8 | 31.3 | 1.27× | 34.4 |
| yolo11n | **30.4** | −22.2% | 23.8 | 48.8 | 31.2 | 1.28× | 29.6 |
| yolov8s | **90.2** | −25.5% | 79.5 | 151.8 | 83.2 | 1.13× | 87.4 |
| yolo11s | **76.6** | −31.4% | 63.5 | 129.7 | 77.8 | 1.21× | 77.2 |
| rfdetr-small | **198.0** | **−24.6%** | 187.7 | 314.8 | N/A | **1.05×** | 200.0 |

- 对 OpenCV dnn 全面领先（0.59× ~ 0.70×）；对 SimdYolo n 档反超（0.97× / 1.14×），s 档持平（1.08× / 0.98×）。
- 与 ORT 的差距从 1.39×~1.72× 收窄到 **1.05×~1.28×**。
- **本轮（11-17）对 yolo / rfdetr 的贡献 ≈ 0**：与上轮差 +2.7/+4.1/+3.2/−0.8/−1.0%，
  同期 ORT 与 SimdYolo 也同向偏慢 2~4% ⇒ 机器状态，不是代码（`bench.md §37.3`）。
- 目标线复核（median）：rfdetr-small **262.5 → ≤210**，现 198.0 —— **达成**；
  yolo11n **39.1 → ≤27**，现 30.4，**差 12.6%**。
- 测量口径与作废规则见 `bench.md §16.2`。
- **新发现（`bench.md §16.3`）**：yolo11n 的 Conv 占 **84.4%（30.7/36.4 ms）**，87 个 conv 最大单个
  只有 1.94 ms —— yolo 系的瓶颈是「大量中等尺寸 Conv」，不是 GEMM。`planv2` 里 P1 的否决
  （「Conv 只有 11.2 ms」）用的是 **rfdetr** 的形状，对 yolo 系不成立。yolo 要提速必须重开 Conv，
  需按 yolo 的形状分布重做阈值/打包分析。**仍未排期**（rfdetr 目标已达成，见上）。

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
   - **yolo 系的 Conv（暂缓，路线已定）**：Conv 占 yolo11n 的 84.4%（30.7 ms），比 ORT 全模型还慢；
     87 个 conv 最大单个 1.94 ms → 是「中等尺寸 conv 的效率」问题。
     **A/B（去物化 + 行块 8）已测死**（`bench.md §19`：命中 0 层，3×3/s1 全在 Winograd 手里，
     且 Winograd 199 GMAC/s 比放开准入后的打包快 38%）；C 同前提，不启动。
     2026-11-16 定调：**yolo 现在直接上 NCHWc**（`planv2 §P7.7`：局部 c 布局 Conv，
     只覆盖 s2 层与 1×1，边界插 reorder；与 §P7.4 否决的「全图传播」不是一回事）。
     收益上限 ~25%（对 ORT 从 1.59× 拉到 1.2× 量级）。
     **阶段 1（s2 的 7 层）已完成并采纳**：内核 1.82×、端到端 −5.5% / −3.9%（`bench.md §23`）。
     **阶段 2（泛化到 stride 1）已完成并采纳**：30 层/帧命中、单层 220 GMAC/s、
     端到端 **−21% ~ −28%**（`bench.md §24`）。图级布局传播经实测降级为可选项（上限 3~8%）。
     下一步候选：1×1 的 c 内核（45 层 / 10.19 ms）、`oc%32!=0` 的层、blocked 缓冲按行带流式化。
   - **`Operators.md` 的 dtype 矩阵自动生成**：把规范的 `T` 约束接进 `GetMemoryShapes` 的校验，
     不做性能、只减少「某 dtype 悄悄走错分支」这类 bug（不排期）。
   - **测量工具**：给 `bench` 加同进程交替 A/B（两次 `Bind` 换配置跑同一份输入），
     否则 5% 以下的收益在本机永远测不准。

---

## 6. 日志

### 2026-11-18（车牌专用 OCR + yolo26s 支持，`bench.md §40`）

- **车牌特化不是换模型**：`Ocr/Plate/` 新增 7 个文件 —— 把通用 CTC 在 6906 类上的 beam 压到车牌字符集
  （31 省 + 24 字母 + 10 数字 + 6 后缀 ≈ 70 类，前缀打包进 `long`：7 位 rank + 高 8 位长度），
  再加逐位约束（`PlateCharset.AllowedAt`：省/字母/数字/后缀，字母与数字排除 `I`/`O`）
  与形态校验（`PlateNumber`：7 位燃油 / 8 位新能源 / 学警挂港澳 / 使领）。
- **置信度换成对齐口径**：对最终序列做 CTC Viterbi（2L+1 状态）取逐字后验再几何平均，
  比「beam 路径分」更能指出弱字（实测第一张 `浙 91% … 8 94%`，第二张 `新 17% A 44%`）。
- **流水线不复制预处理**：`PaddleOcrPipeline` 拆出 `DetectBoxes`（返回已缩放到源图的框）与
  `RecognizeLogits`，`PlateReader` 只做「宽高比筛 → 逐框解码 → 排序限数」。
- **实测**（3200×1800，`--limit 960`）：`浙AES8381 97.6% NewEnergy`、`新A3M15X 62.9% Fuel`，
  整图 825 ms（检测主导），load 7 ms；单框 rec best 3.6~5.1 ms —— beam 8 与 greedy 的差
  落在降频噪声里（`avg` 17~30 ms 抖动），**不据此下结论**，要判需给 `plate` 加进程内交替计时。
- **yolo26s 落地**：480×640 非方形输入、stride 8/16/32 ⇒ 6300 锚点，`detect` 的 `yoloFields`
  改为首次 `Forward` 后按输出形状推断（84 字段自动匹配 `CocoLabels`，`--labels coco` 变可选）；
  4 线程 forward 55.3 ms / decode 0.9 ms。新增 ORT golden 用例
  `yolo26s: end-to-end matches onnxruntime golden`（容差 2e-2），`YoloGoldenTests.Matches`
  参数化后 yolo11 / yolo26 共用同一套对比逻辑。
- **修命名冲突**：`plate --box`（单框）与 `ocr --box`（DB 框分数阈值）撞名，单框入口改 `--crop`。
- **修编码**：`DbPostprocess.cs` / `PaddleOcrModel.cs` 注释是 GBK 字节被当 UTF-8 读出的乱码
  （原字符不可逆），按语义重写为 UTF-8；代码一行未动。
- **`--direct` 入口**（「上游已抠出车牌」的场景正式化）：`PlateReader.ReadWhole(...)` +
  `plate --direct` —— 整张图就是车牌时跳过 DB，直接 rec + 受限解码。
  `仅车牌1.png` **100.0%**（best 4.45 ms）、`仅车牌2.png` **99.6%**（7.45 ms），
  比同图走 DB 的 99.9% / 99.3% 还高一点（DB 框会裁掉号牌边缘）。反向验证：把 3840×2160 的整图
  喂给 `--direct` 返回 **0 张**，形态校验 + 分数下限挡住了误用。
- **一张图验证三件事**（`tests/Image/20260828092321383_121.jpg`，3840×2160）：yolo26s 出 2×`car`
  （forward 44.9 ms / decode 0.8 ms）；车牌 `浙ABM770` 83.5% / `浙ACP5872` 82.7%；
  普通 OCR 出 `浙A BM770` 与 **`H浙ACP5872`** —— 多出的 `H` 被车牌的受限解码剪掉，
  这是「字符集 + 逐位规则」最直接的一次收益。车牌分偏低的原因被逐字后验定位到省字
  （33% / 24%），是 DB 框把 3.1:1 的号牌压成 258×110 所致，**不是解码问题**。
- **README / 版本**：包版本 `0.4.0` → **`0.5.0`**（`dotnet pack` 产出 `Sevencat.OnnxSharp.0.5.0.nupkg`）。
  README 中英双份补齐：车牌三入口（`DetectAndRead` / `Read` / `ReadWhole`，对应 CLI
  `默认` / `--crop` / `--direct`）、`plate` 子命令与参数表、以及新小节「OCR 模型从哪来」——
  四条获取路径（仓库脚本解包 / 直读 `Sdcb.SimdPaddleOCR.Models.*` 的嵌入资源、纯托管无 native /
  Sdcb 引擎共用同一份 ONNX / `paddle2onnx` 自导），其中「直读嵌入资源」那段代码在
  `.tmp/readme-sample` 里真实编译运行过（`6906 类，cls True`）。
- 回归：`OnnxSharp.slnx -c Release` 0 警告 0 错误；两套 TFM `122/124 passed, 2 skipped, 0 failed`
  （新增 `PlateOcrTests` 5 例）。

### 2026-11-18（首层 `ic=3` 直接卷积 AVX-512 版：1.6×，并修 `convbench` 计时顺序偏置，`bench.md §39`）

- §38.4 里唯一输给 MLAS 的层是 `3→16 640² k3s2`（1T 16.8 vs 18.0，16T 20.5 vs 123.4）。
  形状本身不喂通用路径：`K=27`、`M=16`，im2col 要物化 11 MB col 缓冲。
  新开 `Kernels/FirstConv.cs`：向量沿输出宽（16 像素/zmm）× 4 通道 × 4 像素块，
  一条 `vbroadcastss` 喂 4 条 FMA（1:4 而非 1:1），步长 2 用一次 32 元素读 + `vpermt2ps` 抽偶数位，
  面板 27×64 手工对齐到 64 字节。
- 读数（`convbench --iters 200`，与 im2col 交替计时）：1T 2.15 → **1.35 ms**、
  4T 0.56 → **0.34**、16T 0.23 → **0.14**，即 **1.6× 全线程一致**；
  对单核峰值 32.8/55.4 = 59%，1→16 线程 9.6× 扩展。`max|d| = 0.0E+000`。
- **方法学教训（第三次）**：`ConvBench` 先连测 im2col 再连测 firstc，本机功耗受限，
  被烤热的那一段（firstc，512 位 FMA 功耗最高）读数**直接砍半** —— 0.70 而不是 0.34。
  据此先后否掉了两种布局，结论全是假的。改成两个内核**轮流各跑一次**（`BestRound`）后对得上。
  凡同进程比两个内核，必须交替计时。
- 端到端 A/B（`ab --a firstconv=0 --b firstconv=1`）：`max|d|=0`，min 22.1 vs 22.2 ms，
  配对中位 1.012，噪声地板 31.8% ⇒ 判不出；按上式这层 4 线程只省 0.22 ms ≈ 1%，本就在噪声以下。
- 回归：`OnnxSharp.slnx` 零警告；两套 TFM 单测 116/118（2 skipped）、0 failed；
  `yolo11n: end-to-end matches onnxruntime golden` 通过。`AbFlags` 新增 `firstconv=`。

### 2026-11-17（与 MLAS 单算子对照：GEMM / int8 / Conv，`bench.md §38`）

- 把 `F:\code\mlas`（ORT 内部的 GEMM/Conv 库）编的 dll 直接调起来，与我们的实现对拍（同会话、同线程口径）。
  踩坑记录在 §38.1：修饰名导出、`ThreadPool=nullptr` ⇒ 单线程、packed 缓冲需 64 字节对齐、
  `MlasConv` 的 `InputChannels`/`FilterCount` 是**每组**值、`MlasActivation` 不接受空指针。
- **唯一系统性落后的是 F32 GEMM 单线程**：MLAS 84~106 GFLOPS（单核 FMA 峰值的 76~96%），
  我们 68~87（62~79%），yolo 形状落后 19~30%；三处断崖 `n=100`(0.38×)、GEMV(0.24×)、`512×1024×1024`(0.53×)。
- **16 线程我们不落后**：SGEMM 多数形状 1.1~2.1× 于 ORT MatMul（ORT 自己在大宽形状只吃到峰值 21%）；
  int8 `512×1024×1024` 双方 **1067.1 vs 1066.3 GMAC/s**（0.1% 内）；卷积 NCHWc 16T 中位 437 vs MLAS NCHW 210。
- **yolo 的 1.28× 不在内核**：同一层裸 Conv 比，我们 NCHWc 单线程与 MLAS NCHW 同档、16T 更好 —— 与 §37.5 同向。
- 口径边界：ORT 图内走未导出的 `MlasNchwcConv`，`MlasConv` 是 fallback，§38.4 不能换算成整图性能。
- 顺带修 `convbench --threads` 不生效（只有 `GemmBench` 设过 `MaxDegreeOfParallelism`）。

### 2026-11-17（收工：Conv 之外的零钱，OCR −29% / YuNet int8 −11%，yolo 持平，`bench.md §37`）

- 不再动 GEMM / Conv。改三块「剩下的零钱」+ 一个默认值纠错：
  - `MaxPool` 两个专用内核（2×2/s1 滑窗、3×3/s2），`Resize` 全路径按行切分 + 最近邻 2× 表驱动；
  - 池化 / Resize 的并行粒度「每平面」→「平面 × 行块」——16 个平面铺不满 16 线程时等于串行；
  - `GlobalAveragePool` 整平面快速路径（两路 AVX + 行切分，部分和数组归并而非 CAS 原子加）：SE 0.24 → 0.10 ms；
  - `EliminateIdentity`：rec 图删 58 个节点；
  - `ParallelRange` 默认线程 `ProcessorCount`(32) → **物理核**(16)：负载 35% 时 OCR 端到端中位数 184 → 35 ms，
    安静窗口 min 不变 ⇒ 治尾巴不治均值；
  - OCR 输入张量按形状复用（`SetInput` 只引用不拷贝），每帧省 5 MB 进 LOH。
- OCR（4 线程，min/median）：det 17.89/26.12 → **13.18/14.56**；cls 1.86/2.77 → 1.55/1.77；
  rec48×384 4.36/5.32 → 2.90/3.57；端到端 min 34.3 → **24.2**（vs SimdPaddleOCR 1.66× → **1.25×**）。
  识别文本与置信度（0.957/0.982）逐位不变。
- 四引擎（4 线程、rounds 4、每轮轮转、各取最好一轮）：yolo11n **30.4 / 23.8 / 48.8 / 31.2**、
  yolov8n 35.8 / 28.2 / 50.8 / 31.3、yolov8s 90.2 / 79.5 / 151.8 / 83.2、yolo11s 76.6 / 63.5 / 129.7 / 77.8、
  rfdetr 198.0 / 187.7 / 314.8 / N/A。
- **对 yolo / rfdetr 是持平**：对 `§28` 差 +2.7/+4.1/+3.2/−0.8/−1.0%，同期 ORT 与 SimdYolo 也同向偏慢
  2~4% ⇒ 机器状态而非代码。**一度误把 `§8.3`（11-11）当基线，算出 −22% ~ −31%，已更正**（中间夹着
  NCHWc、64 字节对齐、整数 GEMM 三轮）。教训：跨会话对照必须用**最近一次同口径**的会话，不是「基线」。
- **rfdetr 目标线达成**（262.5 → ≤210，现 198.0，`§28` 那轮就到了）；yolo11n 30.4，距 ≤27 差 12.6%。
- YuNet 复测：fp32 5.7/5.9（无变化）、int8 22.95/23.44 → **20.3/20.6**（−11%，仍是 fp32 的 3.5×）；
  score 0.9135 / 0.8944 与 §36 逐位一致。
- **诚实标注**：OCR 的 −29% 是「同一段代码改前/改后同会话」的对照，可信；但它没有开关式单项剥离，
  池化 / Resize / Identity / 输入复用各占多少不知道。yolo 那 −22% 的说法已作废（见上）。
- 回归：`OnnxSharp.slnx` 零警告；测试 116/118（2 skipped）、0 failed。改动未提交。

- 前提纠正：§32.4 的「.NET 拿不到 VNNI」只在 `net10.0` 成立；`net11.0` 下本机 Zen4 的
  `AvxVnni.IsSupported == true`，256 位 `vpdpbusd` 可用。**§32「decoder 3× 不可消除」作废。**
- 形状照 MLAS 的 6 行 × 48 列（`QgemmU8X8KernelAvx512Core.asm`，18 累加 zmm + 3 B + 1 A）等比缩到
  **4 行 × 24 列**（12 + 3 + 1 = 16，正好占满 16 个 ymm）：MAC/指令 14.2 → **20.2**。
  B 对 4 行复用、A 广播对 3 列组复用；12 条独立累加链已够盖延迟，**不需要 K 展开**。
- 配套：epilogue 向量化（`StoreVector`，旧版逐列 `GetElement`）、列尾巴 8 列向量路径
  （n=320/1280 在 24 列块后正好剩 8）、`MultiplyVnniDispatch` 按 4 行切 `ForEach`。
- 吞吐（`tier_probe`，4 线程）：16×320×320 96.7 → 173.4；16×320×1280 364 → **736.4**；
  76×320×1280 314 → **732.5**。单核 183 GFLOPS = `20.2 MAC/cycle × 4.5 GHz × 2`，即 256 位理论天花板。
- 图内：`MatMulInteger` 12 个 1.3 → **0.7 ms**（−46%）；decoder 纯 forward 10.6 → **6.0 ms**，
  ÷ORT 由 2.97× 收到 **1.67×**；端到端 `dec` 段 7 → **6 ms**，文本不变。
- encoder 未变（30.3 vs 29.9，噪声内）：它的 160 个 DQM 走 F32 路径，不属于本次改动。
- **DQM 换整数内核被实测否证**：拆解显示其成本就是 F32 GEMM 本身（≈512 GFLOPS），
  换过去还多一趟 `int32 → float` 输出 pass，收益为负 ⇒ 维持现状。
- 回归：`net11.0` 单测 116/118（2 skipped）、0 failed，数值逐位不变。

- 起因是 P13 第一步「用 gemmbench 扫真实窄 m 形状」：发现**同一批形状**在图内比微基准慢 12~15%，
  而 `--misalign 4`（起点推 16 字节）能精确复现这个跌幅 ⇒ 怀疑指针对齐。
- 新增诊断 `ONNXSHARP_GEMM_ALIGN=1`，量出图里 GEMM 的 64 字节对齐率 **A 18% / B 8% / C 0%**。
  根因：`Tensor.Fit` 的 `new byte[need]` 只保证 8 字节对齐。
- 修复：`Tensor.Alignment = 64`，分配 `need+64` 并放 **POH**（GC 不搬），`ByteOffset` 推到对齐边界；
  新增 `Bytes()`（含偏移的字节视图），改掉 5 处「假设偏移为 0」的裸写（`OnnxFile` 三条解码路径、
  `SafeTensorFormat`、`Paraformer` 三处 `Buffer.BlockCopy`、`Tensor.Clone`）与 5 个测试用例的同类写法。
- 过程中踩坑并定位：第一版改动后有 5 个数值用例失败（quant / argmax / attention / yunet-int8 / asr），
  `Alignment=1` 复跑全恢复 ⇒ 确认是「写端假设偏移 0」而不是内核算错。
- 收益（同会话交替 A/B，min ms）：yolo11n 29.9/29.2/29.9 → **28.4/28.5/28.6**；yolov11s 77.2 → 75.9；
  rfdetr-small 221.5 → 219.7。逐路径只动 1×1 GEMM（9.34 → 8.44），Im2Col/PackedConv 不变（它们本来就对齐）。
- 四引擎刷新：yolo11n **28.9 / 23.5 / 45.9 / 29.3**，与 ORT 的差距 1.26× → **1.23×**。
- 回归：116/118（2 skipped）、0 failed；`alloc` 仍 116 KB/iter（缓冲复用，没变多）。
- 残留写进 P13：B 仍只有 61% 对齐（激活多是视图张量）；窄 m 的真实瓶颈要先分清内存屋顶与级联。

### 2026-11-16（拆开 4 条 conv 路径：**P12 前提否证**，肉在 1×1 的窄 m，`bench.md §29`）

- 把 `PackedConv` / `Im2Col` / `Winograd` 补上「打包 vs 内核」+ 墙钟，复用手写的 `NchwcConv`
  三段计时（`ONNXSHARP_NCHWC_PROF`）。四条路径首次与 op 级 `Conv` 对上：**9.83 + 10.46 + 2.02 + 1.53
  = 23.84 vs 24.0（差 0.7%）**。
- **关键发现：3×3 不是 im2col 打包浪费**。14.0 ms 里 75% 走 NCHWc，内核 9.44 ms、reorder 仅 9.8%；
  整体 **150 GMAC/s = 4 线程峰值 81%**。全部打包合计 **1.7 ms（5.7% 帧）** → direct conv 上限就这么点，结案。
- Winograd **0 次命中**（3×3 s1 的空间都 > `MaxSpatialPixels`=4096），现有实现是标量变换，打不过 NCHWc。
- 真正的差口：**1×1 = 136 GMAC/s（模型内）vs 150~160（gemmbench 同形状）vs 185（峰值）**，
  窄 m（16/32/64 行）＋ epilogue 吃掉 ~2.2 ms；另有首层 `3→16` 只 42.5 GMAC/s、独占 1.04 ms（3.6%）。
  ⇒ 新候选 **P13**。
- 方法学补记：逐任务计时在多线程下是「CPU 累加 ≈ 墙钟 × 占用核数」，所以 `ConvProf` 现在同时打
  墙钟和任务累计两个数；`NchwcConv` 的 profile 也改成预热后清零并按帧归一。
- 零仪表开销校验：`ONNXSHARP_CONV_PROF=0` 时 bench min 30.3 / median 30.8（同机 29.6 / 30.2，噪声内）；
  测试 116/118（2 skipped）。
- 顺手修了 `bench.md` 的章节顺序（§28 原本被插到 §27 前面）。

### 2026-11-16（四引擎同会话复测：ORT 领先 7~26%，差口在 3×3，`bench.md §28`）

- PATH 上的 `python` 是失效 shim（exit 49），改用 `E:\misedir\installs\python\3.11.17\python.exe`
  （ort 1.32.0 / cv2 5.0.0）才跑通 `bench/run_all.py`。
- 4 线程、iters 30 / warmup 10 / rounds 2（median，min 差 <1%）：
  yolo11n **29.6** vs ORT **23.5** vs OpenCV 46.4 vs SimdYolo 29.9；yolov8n 34.4/27.8；
  yolov8s 87.4/76.4；yolo11s 77.2/63.5；rfdetr-small 200.0/187.2。
  ⇒ **ORT 全面领先 7~26%**（1.07~1.26×），我们与 SimdYolo 打平。
- OpenCV 4 线程崩到 46.4（16 线程历史是 26.5）—— 两套口径不可混比。
- 差距定位：`bench min 29.8 ↔ sum-of-min 29.4` 闭合，Conv 24.2（84%），非卷积 4.7；
  1×1 已封板（10.9 ms、87~90% 峰值），**6 ms 的差只能出在 3×3 的 13.3 ms 上** → P12 依据成立。
- 记一条方法学警告：`ort_inspect.py --profile` 的逐算子合计 37.25 ms vs 同机 bench 23.5（1.58×），
  ORT 的 profiling 开销极重，只能看相对占比，不能跨引擎加总对比。

### 2026-11-16（修仪表：`ConvProf` 三处 bug，帧预算重算，`bench.md §27`）

- 定位到 `§25.5`「三个仪表差 15 倍」的真凶不在内核，在仪表本身：
  ① `ProfileAll` 先跑 5 帧预热但 `ConvProf` 不清零，除数只有 `iters`(12) → 每帧次数虚高 1.42×；
  ② 表头「pointwise 整段」打印的是**全进程累计值**（217 ms 是 17 帧之和），压根没除帧数；
  ③ `Capacity=4096` 溢出静默丢弃。
- 修法：`ProfileAll` 加 `onMeasurementStart` 回调（`profile` 传 `ConvProf.Reset`）；
  `Take(frames)` 内部除帧数；`Capacity` → 16384 + `dropped` 警告。
- **修正后 yolo11n 帧预算（4 线程，sum-of-min 28.9 ms / bench 中位 29.9 ms，两口径差 3.5%）**：
  1×1 GEMM **9.57**（33%）+ 融合 SiLU 1.35（5%）+ 3×3（im2col）**13.3**（46%）+ 非卷积 4.7（16%）。
  `§25.1` 的「1×1 = 15.5 ms / 49%」作废 —— 实际 10.9 ms / 38%。
- 口径咬合：单形状 `128×192×1600` 表内 0.25 ms ↔ `gemmbench` 0.212 ms（86%）；`pointwise 9.57`
  vs `GEMM 9.56` → 非 GEMM 开销 0.01 ms（1×1 的账全在 GEMM 里）。
- 结论：**GEMM 封板，第一块肉变成 3×3 的 im2col（13.3 ms / 46%），而它现在没有分阶段计时**。
  下一步先给 im2col 补「打包 vs GEMM」两段计时，再决定 direct conv / NCHWc。
- 回归：测试 116/118（2 skipped，基线一致）。

### 2026-11-16（动手：K 分块照 MLAS 落地 → **负收益，默认关**，`bench.md §26`）

- 用户拍板「ORT 这么做的就照做」，于是把 `MlasSgemmOperation` 的循环形状（N 段 → K 块 →
  重打 PanelB → `ZeroMode` 累加）落进 `GemmKernel`：`MultiplyColumnRange` / `MultiplyRowBlock` /
  `MultiplyPacked.Block` 三处驱动 + 四个内核加 `accumulate` 形参，`ONNXSHARP_GEMM_KC` 控制块高。
- **A/B 全面变慢 12~22%**（128×192×1600：371 → 324 GFLOPS；1024×1280×1280：374 → 291）。
  判因：MLAS 分块是为「定长栈缓冲免分配」，我们用 `ArrayPool`，前提不成立；每块的 C 读改写
  是纯额外开销。**P11 结案，不立项，开关默认 0。**
- 连带把 `§25.5` 的仪表矛盾解决了一半：`gemmbench` 现在对热形状报 0.212 ms / 371 GFLOPS，
  与模型内 `ConvProf` 的 0.24 ms 自洽 → **1×1 GEMM 已在 4 线程峰值 87~90%，GEMM 方向封板**；
  此前 1.39 ms / 28 GMAC/s 的读数是坏数据。
- 仍待修：`ConvProf` 的「每帧贡献」归一化除错了帧数（累计值/迭代数对不上），所以
  `§25.1` 的「1×1 = 15.5 ms」不可引用；按 185 GMAC/s 反推实际约 5 ms/帧。帧预算暂时只认
  `profile --top` 的 sum-of-min。
- 回归：测试 116/118（2 skipped，基线一致，含 yolo11n 与 ORT golden 的端到端比对）；
  模型 `bench` 中位 29.9 ms / min 29.4 ms（与 §24 的 30.1/31.7 同档，无回退）。

### 2026-11-16（更晚：1×1 的账，`bench.md §25`）

- 3×3 收完后先算「剩下的钱在谁身上」，新增仪表 `ONNXSHARP_CONV_PROF=1`（`ConvProf`：
  pointwise 整段 / 其中的 GEMM / 卷积后融合激活，外加逐形状「每帧贡献」聚合）。
- 帧预算（yolo11n 31.7 ms sum-of-min）：**1×1 GEMM 15.5 ms（49%）**、3×3 ~10.5 ms（33%）、
  融合 SiLU 3.7 ms（12%）、非卷积 ~6 ms。pointwise 的非 GEMM 开销只有 0.13 ms（0.4%）
  —— 包装不是问题，GEMM 就是全部。
- 大形状 1×1 的 GEMM 已经在 136~161 GMAC/s（峰值 213 的 65~77%），没什么可榨；
  `64×128×400` 这类小形状因为低于 `ParallelThreshold` 走串行只有 40 GMAC/s。
- 上一轮「1×1 = 10.19 ms / 45 层」作废（逐层 min 手工归属偏小），改引用 15.5 ms。
- 两个异常挂着没下结论：`32×32×25600` 就地 2.40 ms vs 独立 0.309 ms（且同层 profile 只报
  0.40 ms，两个仪表打架）；降阈值的 A/B 被 CodeBuddy 占 30% CPU 搞成噪声地板 48.6%、判不出。
- 回归：测试 116/118（与基线一致），CLI 零警告。
- 下一步两个候选：(A) 1×1 走 blocked 布局（与 3×3 同布局，链上免 reorder）；
  (B) 把 SiLU 熔进 GEMM 的 C 写回（省一趟读+写，约 2 ms = 6%，但要动全库最精细的代码）。

### 2026-11-16（定调：**参照 ORT**，放弃 B、降级 A，`bench.md §25.4`）

- 决定：**(B) SiLU 熔进 GEMM 的 C 写回 —— 放弃**；(A) 1×1 blocked —— 降级为「ORT 都没试过」的
  实验，不再当作下一步主线。改为主张「优先对齐 ORT 的做法」。
- 依据（读 ORT main 的 `mlas/lib/{convolve,activate,sgemm}.cpp`）：
  - 1×1 conv：`MlasConvPrepare` → `MlasConvAlgorithmGemmDirect`，`WorkingBufferSize = 0`，
    直接 `MlasGemm(oc × pixels × ic)` —— 和我们 `Pointwise` 一模一样。
  - 激活：GEMM 之后单独一趟 `MlasActivation(Activation, Output, Bias, oc, pixels, ldc)`，
    微内核里没有任何激活；融合清单只有 Identity/Relu/LeakyRelu/Tanh/Logistic/Clip/
    HardSigmoid/HardSwish，**没有 SiLU**。
  - 微内核：`MLAS_SGEMM_TRANSA_ROWS = 12`、B 面板 16 列一块打包、每线程栈上面板、K 方向按
    `STRIDEK` 滚动 —— 我们 `Multiply12Rows` + `SelectColumnBlock`=16 + 每线程打包，形状参数一致。
  - 并行：`Complexity = oc × OutputSize × K` 对比 `MLAS_SGEMM_THREAD_COMPLEXITY`，再按
    `STRIDEN_THREAD_ALIGN` 对齐切 N 段 —— 与我们的 colBlock=16 + `ParallelThreshold` 同思路。
- 推论：**架构层面我们已经和 ORT 同构**，快慢差在微内核深度（k 展开、提前 load 的软件流水 +
  K 分块滚动让面板留在 L1），不在「有没有某个专用内核」。下一轮实验应放在
  `GemmKernel` 的 K 分块与 k 展开上，目标把 1×1 的 136~161 GMAC/s 推向 190+。
  `MLAS_SGEMM_STRIDEK` 的具体值不在 `sgemm.cpp` 里，KC 取值得自己在 `gemmbench` 上扫。
- **前置阻塞（`bench.md §25.5`）**：动手前复核产能时发现三个仪表对不上 —— 同一形状模型内
  `ConvProf` 0.25 ms / `gemmbench` 1.39 ms / `convbench` 1.94 ms，`profile --top` 与 `ConvProf`
  也差 15 倍；当次 `profile` 的 mean 193.3 ms 是 sum-of-min 38.8 ms 的 5 倍（机器满负荷）。
  在安静窗口把「单层 in-model 计时 ↔ gemmbench 同形状」对齐之前，**不基于 §25.1 的
  「1×1 = 15.5 ms / 49%」立项**。


### 2026-11-16（深夜：P7.7 阶段 2 —— 泛化到 stride 1，端到端 −21% ~ −28%）

- 先给 `TryConv` 加了分阶段计时（`ONNXSHARP_NCHWC_PROF=1`，`profile` 末尾打印每次调用与
  reorder 占比），量出**阶段 1 只有 6 层命中、reorder 占 NCHWc 时间 18.9%、摊到整帧 1.0 ms（2.9%）**。
- 按这个账，`§23.7` 原定「下一步做图级布局传播」被推翻：`dump-graph` 显示每个 s2 层的消费者是
  3×3/s1 的 Conv 加远处的 Concat，要吃掉那 3% 得把 Concat/Split/MaxPool 和邻居卷积全 c 化。
  **收益 3%、成本数天 —— 排序错了。** 改为「先把内核覆盖面扩到 stride 1」。
- 实现：`HaloPad(stride)`（stride2 → 左上 1 像素，stride1 → 四周各 1 像素）+ `PackInput` 多清
  最底一行/最右一列 + 内核把硬编码的 `2` 换成 `step`（`pix = step*Block`）。**指令数不变**，
  所以 stride1 与 stride2 同速。同时把 `InferShapes` 与 `Forward` 的两份手写条件合并成
  `NchwcConv.Dispatch()` 单点判据（申请 temp 与分派内核不能各写一份）。
- 微基准（s1）：64→64 160² **220.6**、128→128 80² **220.8**、256→256 40² **220.8**、
  256→256 20² 179.3（对照 im2col 106~159、Winograd 102~184、真模型里的 packed 120~150）。
  奇数边长 101×97 例外：`ow=101` 有 5 个像素走尾路径，实测 110.7（比 im2col 慢 18%），真模型不常见。
- 端到端 `ab`（配对中位）：yolo11n 38.6→31.8（**1.212**）、yolov8n 45.9→36.6（**1.251**）、
  yolov11s 1.236、yolov8s 1.270、yolo11m 1.276、yolov5su **1.298**；`max|d|` 相对 1.6e-5，
  `alloc 129 KB/iter`。yunet（`oc=8/16`，一层的 `oc%32` 都不满足）输出**逐位相同**、判定噪声内 —— 没命中就没风险。
- 代价记账：blocked 输入缓冲 160² 单块 16.8 MB（temp 池按索引复用，峰值 ≈ 2 块）；
  1080p 级输入会到 100 MB 量级 → 将来要么按行带流式化，要么给 `Tensor` 加 blocked 视图。
- 回归：测试 116/118（1 个 argmax 计时用例受机器负载偶发失败，复跑通过）。

### 2026-11-16（夜：P7.7 阶段 1 完成）

- **s2 的 NCHWc16 内核已落地并采纳**（细节 `bench.md §23`）。新文件
  `src/OnnxSharp/Kernels/NchwcConv.cs`：输入 `[C/16][H+1][W+1][16]`（左上 halo）、输出
  `[OC/16][OH][OW][16]`、权重 `[OC/16][IC/16][9][icLane][ocLane]`（弱表缓存），2 oc 块 ×
  8 像素 = **16 个 `zmm`** 累加器。接入点是 `ConvLayer.GetMemoryShapes` 加两块 temp +
  `Forward` 里排在 Winograd 之前的分派分支。
- **微基准（MAC 加权，yolo11n 的 7 个 s2 层）**：112.3 → **204.1 GMAC/s（1.82×）**，
  单层 174~220（对照 im2col/packed 88~138）。判据「≥170 / 1.5×」通过。
- **端到端 `ab`**：yolo11n 36.4 → 34.5 ms（配对中位 **1.055**，范围 1.047~1.057）；
  yolov8n 42.5 → 40.8（**1.039**）。等价性 `max|d|` 相对 2.2e-5（累加顺序）。
- **三个坑，都记进 `bench.md §23`**：
  1. `CanUse` 必须要求 `h`/`w` 偶数 —— 只有左上 halo，奇数边长会多读一列/一行（`101×97`
     的 `max|d|=24`）。`2*oh==h && 2*ow==w` 同时写进 `TryConv` 兜底。
  2. reorder 第一版是「lane 外层逐元素写 64 字节块」→ 16× 写放大，320² 单层 **3.72 ms**
     （比 conv 还贵）。换成 8×8 转置内核（`UnpackLow/High + Shuffle + Permute2x128`，
     打包与解包共用）后 **0.19 ms**。
  3. **进程里第一个 bench 配置永远不可信**：同一形状排第一 14.4 GMAC/s、排第二 92.5，
     与 `DOTNET_TieredCompilation/TieredPGO/ReadyToRun` 全关无关，与形状无关。
     `convbench` 已内置「首个配置读数丢弃」（自动插一个 `128→128 20²` 打头），现在裸跑即稳。
- **端到端只有 −5.5% 而非推算 −8.4% 的根因**：阶段 1 没做布局传播，reorder 是「每层进、每层出」
  共 14 次，in situ 约 1.3~1.5 ms（微基准 0.47 ms 是缓冲滚烫的乐观值）。`§22.1` 的 0.336 ms
  只能引用在**传播之后**。→ 阶段 2 先做传播（`Fit` 带 layout、`Concat`/`Split`/`MaxPool`/`Resize`
  的 c 版、`GetLayouts` 时间表），再谈 1×1 的 c 内核。

### 2026-11-16（下午续）

- **yolo 局部 NCHWc 的三个前置问题全部量完，结论是立项**（细节 `bench.md §22`）。
  1. **reorder 边界 = 5 处 / 10.5 MB / 0.336 ms（端到端 0.9%）**，不是先前估的 18 处。
     在 ONNX 图上做布局传播（新增常驻工具 `bench/layout_probe.py`）：`Concat`×23 与 `Split`×11
     的通道数都是 16 的倍数，块序列直接拼接，免 reorder；C2PSA 的 `Transpose`/`MatMul`/`Softmax`
     前面本来就有一个 `Reshape` 打回 NCHW，attention 整段保持 plain，不额外制造边界。
     5 处 reorder **全是 `Reshape`**（C2PSA 入口 400 KB + head 的 P3/P4/P5 + DFL）。
     就算阶段 1 不做 `MaxPool`/`Resize` 的 c 版，也只多 5 处、+0.13 ms。
  2. **块宽 = 16**（1 个 `zmm`），86/88 个 conv 的 `oc % 16 == 0`；块宽 32 只有 70/88，
     16 个 `oc=48/80/112` 要补一半 → 否决。两个例外排除在 c 之外：`ic=3` 首层（读 NCHW、写 c，
     不必补到 16）与 `oc=8` 的 C3k2 瓶颈层（补到 16 是 2× 空转，保持 NCHW 更便宜）。
     注意两个会误导人的平均值：按 conv 算术平均填充 26%、按 MAC 加权 117% —— 后者 95% 来自
     首层的 `ic=3`，而首层根本不需要 c 输入。
  3. **`BufferAllocator` 零改动**：`Layout` 枚举 / `Tensor.Layout` / `Arg.Layout` /
     `LayerInfo.GetLayouts` 早已预留；槽只按「区间不重叠 + 字节够大」复用，与布局无关；
     `Tensor.Fit` 每次复位 `Layout = NCHW`、`Strides = null`，槽复用不会串布局。
  收益推算（目标吞吐是假设）：conv 31.78 ms = 3×3/s1 **13.44**（88.9 GMAC/s）/ s2 8.15（106.7）/
  1×1 10.19（114.1）。s2→170 + 1×1→150 保守 −5.1 ms，端到端 **−14% ~ −24%**。
  旁证比推算硬：OpenCV dnn 走 `NC8HW8`（块宽只有 8）实测 **28.6 ms**，比我们快 28%。

- **顺带补了 §19 的盲区**：§19 整节只盯 s2（因为那是 A/B/C 的命中面），但 conv 里最大的一块是
  **3×3/s1 的 13.44 ms** —— Winograd 只接了空间 ≤4096 的层（199 GMAC/s），160²/80² 的十几个层
  退回 im2col，把家族平均拉到 89。新增工具 `bench/conv_family.py` 做这个分解。
  **教训**：算「某家族的效率」不能只看该家族里最大的那层，要把没进优化路径的中小层一起算。

### 2026-11-16

- **新增 `ab` 子命令：同进程交替 A/B（`bench.md §20`）**。两个引擎放同一进程，逐轮交替、
  每轮交换先后顺序，每轮取 min 后看**配对比值的中位数**，噪声地板取 `max(1%, 3×MAD)`；
  另带图算子直方图 diff（确认开关真改了图）与逐输出 max/mean/p99/>1e-3 计数 + 最差下标。
  自检（两侧同配置，6 轮）：比值 1.001 ~ 1.011，判定「噪声内」→ **工具能分辨的最小信号约 1%**。
  为此把 `GraphFusion.AttentionFusionMaxSeq` 与 `AttentionOps.BlockKernelMinSeq` 改成可写属性
  （后者由层在构造时捕获，否则第二个引擎会覆盖第一个）。
- **P8 判定：融合比不融合快 2.2%，采纳**。`attn_fuse=0` vs `4096`，6 轮交替：
  A min 225.5 / B min 220.8 ms，配对中位 **1.022**，跨度 2.9%，噪声地板 1.6%（另一次跑出 1.055，
  取保守值）。图上 `FusedAttention +12`、`MatMul −24`、`Softmax −12`（664 → 556 算子）。
- **P8 数值差异归因（重要）**：融合前后 `dets` 有 108/1200 个值变化 >1e-3、max 0.5。
  关键对照是「同为融合、只把块内核换成逐行内核」→ 60/1200、max 0.47，**同量级同形态**
  （相邻 query 的 det/label 成对翻转）。结论：12 层 transformer 把 1e-7 的求和顺序差放大到 1e-3，
  top-k 近邻换序，**不是 P8 的 bug**。故默认开启。
- **新热点**：12 个 `FusedAttention` 各 7.2 ms = **端到端 40%**，804 MMAC/层 →
  **112 GMAC/s**（同机 GEMM 峰值 197 的 57%）。下一步是 flash-attention 式（Q 分块 + 在线
  softmax、S 块不落地），上限 `7.2 → 4.2 ms` ≈ 端到端 −16%，属新立项。
  **↑ 这条当天就被推翻了，见下面 P9 的三条日志（attention 实为 16.4%，不是 40%）。**
- **yolo A/B（Conv 去物化 + 行块 4→8）否决并回退（`bench.md §19`）**。A 做出 `Multiply8Direct`
  + `TryTapOffsets`（面板完全不物化，跨行/贴边回退打包），B 做出 `Multiply8`（19 个 YMM）
  与按层解析的行块，并把准入从 `K ≤ 2048` 放开到 `K ≤ 4096`。单层交替 5 轮取 min：
  够得着的两层确实是赢的（ic=256 的 40² s1 **8.26 → 6.54 ms**、20² s1 2.08 → 1.97）。
  **但 yolo11n 里一层也命不中**：`K > 2048` 的层全在 20×20，选路顺序让 **Winograd 先接走**
  （同形状 winograd 4.74 ms / 199 GMAC/s，比放开准入后的打包 6.54 还快 38%）；能走到
  `PackedConv` 的只剩 s2 层（`K ≤ 1152`）。模型内交替 4 轮 + 未变层归一化：new/old 中位 1.06（噪声内）。
  **代码全部回退**；C（ic 块外提）前提相同、命中面同样是空集，不启动。
- 副产物两条，已写进 `bench.md §19.5`：① 8 行在**打包**路径上无收益（oc=128/256 打平），
  oc 小时明显更慢（640²→320² 1.23→3.28、320²→160² 1.19→5.88）；② **直读只在冷 cache 口径下领先**，
  模型内上游刚写过的输入还在 L2，省一趟搬运测不出来 → 以后此类优化用 `profile` 逐算子 min 判。
- **本机噪声再降级**：同一形状同一配置的 `convbench` min 在 0.85 ~ 1.30、1.19 ~ 5.88 ms 之间跳，
  yolo11n `sum of min` 在 33.9 ~ 158 ms 之间跳。**低于 5% 的差异只能用同进程交替 + 对照层归一化。**
- **yolo 暂停，路线定为 NCHWc**（`planv2 §P7.7`）：不再做打包/行块这一层的微调；
  将来只对 conv 链做**局部 c 布局**（覆盖 s2 与 1×1，边界插 reorder），
  与 §P7.4 否决的「全图布局传播」区分开。立项门槛：先解决 <5% 的测量问题。
- **P6' 余项验证：BLIS 式 A 打包是负收益，P6' 关闭（`bench.md §18`）**。把 A 的 6 行一组
  打包成 `[k][6]` 面板（MR=6，因为 MR=12 需要 AVX-512VL 的 ymm16~31，混进来会测到 spill），
  内核与 `Multiply6Rows` 逐指令同构、逐位一致；`gemmbench --apack 1` **交替**测两条路径、
  各自取 min（IDE 抢核导致分开测的绝对值在 3~20 ms 间跳）。9 次交替 8 次更慢，
  384 列时稳定 −29%~−33%。原因：原路径 6 条 stream 在 Zen4 预取器能力内、流量完全相同、
  面板多占 1.58 MB。**A 驻留的真实上限只有 4%**（m=1028 的 404 vs m=120 的 420 GFLOPS），
  MC 分块要付 B 的重复读，净收益 <2%。实验代码已全部回退。
- **P7-D 验证：GEMM 并行阈值不是瓶颈（`bench.md §17`）**。阈值改为可配
  （`ONNXSHARP_GEMM_THRESHOLD` / `gemmbench --threshold`，默认仍是 `1<<22`），
  但 rfdetr 的 139 个 MatMul 最小 `m·n·k` = 19.6 M ≫ 4.19 M，降阈值不改变任何一层的
  并行决策，端到端交替 A/B 217.8 vs 237.4（噪声内）→ **D 关闭，默认值不变**。
  顺带加了 `gemmbench --attn 1`（复刻 QK/PV 的真实形状+视图+派发序列）和
  `--btrans`（K^T 视图 B）、`MMv` trace 补 stride。
- **新缺口：QK/PV 在图里比同参数探针慢 1.8×**（1.73 vs 0.95 ms、1.03 vs 0.54 ms，
  12 层 33.1 ms）。不是打包/派发，而是每层 6.3 MB attention 矩阵往返（25 MB/层，L2 1 MB）。
  候选 P8「QK→Softmax→PV 分块融合」，上限 −4%，未排期。→ **2026-11-16 已做成并判定 +2.2%（§20）。**
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

- **P9 执行（3/4 步，累计 +2.2%，`bench.md §21`）**。分步计划见 `planv2 §P9`，四步里做完三步：
  **P9.1** 把 `ScoreBudget` 旋钮化（`ONNXSHARP_ATTN_SCORE_BUDGET`）后扫 128/256/512/1024 KB，
  三组配对 0.991~0.993 → **512 已是最优，默认不变**（「超出 L2 会有悬崖」的猜想不成立：分块内核
  是算术受限，不是带宽受限）。**P9.2** softmax 的 `1/Σ` 折到 PV 之后（S 只存 exp 值），
  端到端中位 1.007、段内 `soft+pv` 稳定 −3~4%，数值 `max|d|=9.3e-5`、0 翻转 → 保留。
  **P9.3** `K^T`/`V` 面板按 `(b,h)` 只打一次（并行区间连续，记住上次的 `bh` 即可，无需共享），
  中位 **1.010** 且 **`max|d| = 0` 逐位相同**；顺带发现 `V` 复用也值 ~12 µs/块。
  累计 `ab`（两项全关 vs 全开）**1.022，跨度 0.7%，6 轮同向**。
- **口径纠正（重要）：attention 只占 16.4%，不是 40%**。`ONNXSHARP_ATTN_TRACE=1` 新增 `[attnthr]`
  探针后发现 12 个 `FusedAttention` **不同构**：只有 3 个是 `b=1,t=1028`（811 MMAC / 6.0~6.6 ms /
  134 GMAC/s），另 9 个是 `b=4,t=257`（窗口 16×16+1，203 MMAC / 1.6~1.7 ms / 127 GMAC/s）。
  两族吞吐一致，**没有并行退化**——「12 层各 7.2 ms」是把 3 个大层当成 12 层。
  `profile` sum-of-min 里 attention 合计 35.5 ms / 219 = **16.4%**。
  后果：flash 式重写的收益上限从 −16% 修正为 **不到 −5%**。
- **P9.4 / P9.5 按天花板关闭**。`gemmbench` 同 16.7 MFLOP：`127×64×1028`（QK 形状）222 GFLOPS、
  `127×1028×64`（PV 形状）334 GFLOPS → **纯 GEMM 就复现了 QK/PV 的 1.5× 差距**，与打包/softmax/
  S 落地无关，是 k=64 时序言/尾声摊不掉。补齐要新写短 K 微内核，收益上限 2.5%，
  而 P6' 已判定微内核侧无 >1% 单点改善 → 关闭。flash 式上限 <2% → 关闭。
  **rfdetr 的 attention 侧到此收工。**
- **「下一个大项是 MatMul（64%）」这句是错的，当天撤回**。同口径 `profile`（4 线程 3 iter、
  sum-of-min 199.8）：MatMul 115 个 / 124.2 ms（62%），但主体是 12 层 MLP ——
  `fc1` 3.2 ms×12 + `fc2` 2.91 ms×12 = **73.7 ms / 29.1 GFLOP = 395 GFLOPS = 4 线程峰值的 92.5%**，
  q/k/v 的 36 个投影也在同一档（~379）。**剩余 8% 且要求打赢峰值 → 不是「可做」，是「已做完」。**
  同时撤回「要靠 NCHWc」：NCHWc 对 2D MatMul 零收益（§P7.4 自己论证过），§P7.7 的 NCHWc
  是 **yolo 局部 conv** 的路线，与 MatMul 无关；**yolo11n 只有 2 个 MatMul（C2PSA 内）**，
  MatMul 这条线是 rfdetr 专属，做了对 yolo 一点帮助都没有。
  含义：rfdetr 主线**不再有可立项的大项**，剩下的是 Conv 79%→1.2%、attention 63%→5%（已判定不划算）、
  其余 ~32 ms 的逐元素/格/LN 碎块。
- **收工清理**：删掉一次性 `[attnthr]` 探针（`TraceBlocks` 门控下的 `ConcurrentDictionary` +
  `Stopwatch`）。复测：单测 **116/118**（同基线）、`bench` 4 线程 min 222.7 / median 228.7 ms、
  累计 `ab` 6 轮同向中位 **1.016**（1.011~1.021，地板 1.0%，`out1 max|d|=3.86e-5`、0/900 翻转）
  → 与 §21.5 的 1.022 同向，幅度随机器状态浮动，结论不变。
  坑记一笔：`ab`/`bench` **不显式设 `ONNXSHARP_THREADS=4`** 会用满核，绝对耗时掉到 140 ms 量级、
  配对跨度 7.8%，直接误判成「噪声内」。

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
