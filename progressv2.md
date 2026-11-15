# OnnxSharp 优化进度 v2

> 对应策略见 `planv2.md`，性能数字唯一事实源是 `bench.md`（本文件只记**状态与结论**，
> 不重复放全表，避免两处打架）。
> 口径：**4 线程**、FP32、batch=1、静态形状。动笔前先确认
> `(Get-Counter '\Processor Information(_Total)\% Processor Performance') > 80%`。

- 立项：2026-11-15
- 当前阶段：**测量与逐元素/内存两轮清理完成，下一个做 P6'（GEMM 微内核）**

---

## 1. 状态总览

| 项 | 内容 | 状态 | 预期收益（rfdetr） | 实测 |
|---|---|---|---|---|
| P0 | 测量基线冻结（4 线程 / 温度门控 / env bug） | **完成** | — | — |
| P1 | Conv 低线程效率 | **已否决**（实测 11.2 ms） | ~~-6~7%~~ | 上限 -3~4% |
| P1' | 逐元素 32 ms 为什么贵 | **已关闭**（实测 12 ms） | ~~-8%~~ | 上限 -1% |
| P2 | Conv+SiLU/ReLU 融合 | **已否决**（激活只有 0.4 ms） | ~~-2.8%~~ | 0 |
| P3 | `Where` 逐元素搬移 + 折叠漏网 | 待启动（随手项） | -1% | 2.6 ms → ≤0.2 |
| P4 | Transpose 折进 GEMM A 面板 | 待启动 | -2.4% | 10.5 ms |
| P5 | Reshape / GatherElements / 小算子 | 随手做 | -1% | Reshape 4.4 ms 可疑 |
| P6 | 并行粒度按线程数自适应 | **前提证伪**，降为观察项 | ~~结构性~~ | 1→2→4 近线性 |
| P6' | GEMM 微内核 70~74% → 85% | **完成**（MR=6 + MR=12） | ~~-13%~~ 上限 -10% | 打包路径 **92.2%**，端到端 **-9.8%**（246.0→221.8） |
| — | 内存复用（ORT memory pattern 对应项） | **完成**（修 `Fit` 缩容 bug） | 0（隐患消除） | 54.8→1.4 MB/帧 |

目标线（4 线程 median）：rfdetr-small **262.5 → ≤ 210**、yolo11n **39.1 → ≤ 27**。

当前（同一套交替 A/B 口径，min）：rfdetr-small **221.8 ms**（-9.8% vs P6' 起点 246.0）、
yolo11n **36.4 ms**（-4%）。距离目标线还差约 5%。

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

---

## 3. 差距分解（rfdetr-small，4 线程；Conv/GEMM 已改为微基准实测）

| 类别 | 我们 | ORT | 差 | 归属 |
|---|---:|---:|---:|---|
| GEMM 类 | ~180（实测 6 类 157） | 150 | +30 | **P6'（微内核 70~74% → 85%）** |
| Conv | **11.2（实测）** | 8 | +3 | 原 P1 **已否决**，只剩 epilogue 融合（P2） |
| 逐元素 + 激活 | ~32 | 4.5 | +27 | **P1'**、P2、P3 |
| Transpose | ~11 | 5 | +6 | P4 |
| Where | ~5 | 0 | +5 | P3 |
| GatherElements | ~2.5 | 0.6 | +2 | P5 |
| LN/Softmax/GridSample/Reshape | ~20 | 14 | +6 | 暂不动 |
| 合计 | ~262 | 189 | +73 | |

结论一句话：**GEMM 占 69% 且只跑到 70~74% 峰值；Conv 只占 4.3%（原 27 ms 是折算错误）。
真正的低成本机会在逐元素 32 ms，真正的上限在 GEMM 微内核。**

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
5. **P3 / P5（随手项）**：`Where` 2.6 ms（逐元素 4 字节搬移 → 按行 memcpy）、
   `Reshape` 4.4 ms（119 个 reshape 若是 view 应归零，先查为什么不是）。

---

## 6. 日志

### 2026-11-15

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
