# OnnxSharp 进度 v3

> 计划见 `planv3.md`；CPU 性能数字仍以 `bench.md` 为准，Vulkan 数字进 `benchv3.md`。
> 本文件只记**状态与结论**，不重复放全表。

- 立项：2026-11-18
- 当前阶段：**P0 完成（0.6.0）**，下一步 P1（Vulkan 骨架）；基线见 §0
- 口径：CPU 4 线程 / FP32 / batch=1 / 静态形状

---

## 0. 基线（2026-11-18，v3 起点）

代码 `d5b2acc`（未提交改动仅为 README/plan/csproj 文案）。`bench` 40 iters / warmup 5 / 4 线程 / 每模型独立进程，四轮交替：

| 模型 | 四轮 median | 取用值 | 备注 |
|---|---|---|---|
| yolo11n 1×3×640×640 | 28.1 / 29.1 / 29.1 / 29.0 | **28.1 ms** | 跨度 3.5% |
| rfdetr-small 1×3×512×512 | 216.8 / 194.1 / 193.7 / 216.5 | **193.7 ms** | **双稳态 11.6%**，非顺序可解释 |

- 机器：Ryzen 9 8945HX（16C/32T APU），`% Processor Performance` 137.6–137.8，Win10 IoT LTSC 19044。
- `profile` 校验：sum-of-min 28.2 / 192.4，与 bench 误差 <0.7%。
- 正确性：两模型 `run` 均正常输出（yolo `[1,84,8400]`、rfdetr `dets[1,300,4]` + `labels[1,300,3]`）。
- 可 GPU 化占比：yolo **96.2%**（Conv 85.9%）、rfdetr **91.4%**（MatMul 64.0% + Attention 17.3%）。
- 由此定的判定门槛：yolo A/B ≥4%、rfdetr A/B **≥12%** 或强制同进程交替。

---

## 1. 状态总览

| ID | 任务 | 状态 | 结论/实测 |
|---|---|---|---|
| P0 | `LayerPlan` 收敛（10 字段 → 1 对象） | **完成** | 0.6.0，`Layer.Plan` 单一签名 |
| P0-1 | 新增 `Core/LayerPlan.cs` + 单测 | **完成** | `LayerPlanTests` 6 例全通过 |
| P0-2 | `Layer` 加 `internal LayerPlan Plan`；3 个 public 字段收 internal | **完成** | §9-A 选 1（直接收，升 0.6.0） |
| P0-3 | `Interpreter.Run` 5 分支 → 3 段 | **完成** | 签名比较 2 次 → 1 次；删 `SameShapes`/`SameTypes` |
| P0-4 | 删 10 个旧字段，`grep` 归零 | **完成** | 全仓 0 处残留 |
| P0-5 | 4 模型 golden 逐位回归 | **完成** | 134/136 通过、0 失败；yolo11n max\|diff\|=2.319e-3、yolo26s 2.441e-3（与 0.5.0 同值） |
| P0-6 | 记录分配量前后对比 | **完成（结论修正）** | 见下：52 MB bug 实际在 0.5.0 已修，P0 是防复发 |
| P1 | Vulkan 骨架（`OnnxSharp.Vulkan`） | **进行中** | 已接入解决方案，P1 期间不发包 |
| P1-1 | 建项目/目录/接线 | **完成** | `src/OnnxSharp.Vulkan` + `src/Directory.Build.props`（版本单一来源） |
| P1-2 | `Vk.cs` 手写绑定（~500 行） | **完成** | 511 行移植 + `VulkanLoader.TryProbe`；实测 `vulkan-1.dll`、2 设备 |
| P1-3 | `VkDevice`（instance/选卡/队列/内存/命令池/pipeline） | **完成** | 精简移植 ~440 行；双 GPU 往返 OK、50 次提交无 lost；pipeline 待 P1-4 出 shader 后验证 |
| P1-4 | shader 工具链 + `EmbeddedResource` | **完成** | `build-shaders.ps1`（glslc/glslang）+ `VkShaders.Load` + 缺件防线；双 GPU 真实 dispatch 通过 |
| P1-5 | 通用 pipeline 模板 | **完成** | 默认走 push descriptor；SPIR-V 反射 local_size/binding；`VkTimer` 拆出总线开销（约占小 kernel 95%） |
| P1-6 | `VkSegment`（binding/arena/dispatch 列表） | **完成** | arena 256 B 对齐 + push descriptor 切片 + 依赖屏障；16 dispatch 一次提交比逐算子快 **10.5×/5.0×** |
| P1-7 | `VkPartitionLayer`（stage in/submit/stage out） | **完成（2026-11-18）** | `Tensor`(byte[]+ByteOffset) → 槽位 stage；权重走常驻 arena；端到端 maxErr 1.0e-6，400 KB 往返 234/249 us |
| P1-8 | 分区 pass（最长段 + `Graph.Reset`） | **完成（2026-11-18）** | 规划器 + 段物化（`VkGraphLayer` 做段内延迟形状推断）+ `Graph.Reset` 拼接 + 模型级共享权重；真图 7 算子 → 3（GPU/CPU/GPU），数值与 CPU 一致 |
| P1-9 | 生命周期（LRU shape 缓存/Dispose/设备丢失） | **完成（2026-11-18）** | 落成缓冲池而非段缓存（段重建 499 → **1 us**）；丢失即弃池 + 快速失败 |
| P2 | 算子覆盖与分区策略 | **进行中（2026-11-18）** | 逐元素族 + 2-D Conv 已上 GPU 并端到端跑通；朴素卷积 dGPU 3.5x 快、iGPU 3x 慢，性能特化（P2-2/P2-3）与成本模型（P2-6）是下一步 |
| P2-B1 | Conv/Gemm/Act/Add-Mul/Pool | **部分完成** | `Add`/`Sub`/`Mul`/`Div`/`Max`/`Min`（F32 + 右对齐广播）与 **2-D `Conv`**（F32，groups/strides/dilations/pads/auto_pad/bias；1×1 走专用核）已可发射；Gemm/Act/Pool 未做 |
| P2-B2 | Softmax/LayerNorm/Resize/Concat/Transpose/Slice | 未开始 | rfdetr attention |
| P2-3 | 段内 NHWC + 向量化载入 + shared memory 分块 | **部分完成（1×1 与 3×3 均特化）** | 1×1：TP=4/MT=8 二维分块，端到端无变化（415 us 里 kernel 仅 ~200 us）。3×3：`conv3x3.comp` 8 通道 × 4 连续像素 + 滑动窗口复用，单卷积 608 → **427 us**（0.36 → 0.58 TFLOP/s），yolo 6op 395 → 340 us。剩余空间在「3 行 halo 未跨线程复用」（共享内存分块），且该尺寸下即便计算归零端到端也只有 1.6× 上限 |
| P2-6 | 分区成本启发式 + 探针 env | **完成（设备门 + 段级成本判据）** | 两道门：`VkDevicePolicy` 按设备类别过滤（核显拒发 `Conv`）；`VkCostModel` 按实测成本预测（`t ≈ 120us/段 + 4us/dispatch + 字节/10.9GBps + 3×3快路径 FLOPs/0.58T + 通用卷积/0.36T + 点卷积/1.47T`）。真规模对账：yolo 6op（64×64×32）dGPU 1.02×（打平），预测误差 3%；iGPU 把 0.25× 拦回纯 CPU。`ONNXSHARP_VK_COST=0` 可关，`CostReport` 出段表 |
| P3 | 性能与收尾 | 未开始 | — |
| P3-1 | fp16 存储（运算全 fp32） | 未开始 | 目标 ≥1.5× 带宽算子 |
| P3-5 | 部署验证（Win10 1903+/Mesa/核显） | 未开始 | — |
| P3-6 | `Sevencat.OnnxSharp.All` 单 dll 全包 | **项目已建、机制已验证**（2026-11-18） | `src/OnnxSharp.All` 链接两份源码；零依赖、单 `OnnxSharp.dll`；与主包互斥 |

---

## 2. 里程碑

| 里程碑 | 内容 | 状态 |
|---|---|---|
| V3-M0 | P0 完成，golden 逐位不变 | **完成（2026-11-18）** |
| V3-M1 | P1 完成，yolo11n 数值正确 | 未开始 |
| V3-M2 | P2 完成，B1 算子 + 启发式 | 未开始 |
| V3-M3 | P3 完成，fp16 存储 + 部署验证 | 未开始 |

---

## 3. 变更日志

| 日期 | 项 | 结论 |
|---|---|---|
| 2026-11-18 | v3 立项 | 定位：P0 收敛状态机 + Vulkan 独立包；不做通用 ExecPlan（收益 ≤1%） |
| 2026-11-18 | 现状核查 | 10 个状态字段仅被 `Interpreter.cs`(28) 与 `LayerInfo.cs`(10) 引用，重构局部可逆 |
| 2026-11-18 | 集成形态定案 | `VkPartitionLayer : Layer` + `Graph.Reset`，**不改 `Interpreter`**；所需 API 全 public |
| 2026-11-18 | 基线实测 | yolo11n **28.1 ms**、rfdetr-small **193.7 ms**（4 轮取最小）；rfdetr 双稳态 11.6% |
| 2026-11-18 | 热点归因 | yolo Conv 85.9%、可 GPU 化 96.2%；rfdetr MatMul 64.0% + Attention 17.3%、可 GPU 化 91.4% |
| 2026-11-18 | §9-A 拍板 | 三个 public 状态字段**直接收 internal**，版本 0.5.0 → **0.6.0**（不做 `[Obsolete]` 过渡） |
| 2026-11-18 | P0 完成 | `LayerPlan`（`State` + 唯一签名 + 4 组形状/类型列表）；`Interpreter` 5 分支 → 3 段；新增 6 个测试 |
| 2026-11-18 | P0 回归 | 134/136 通过 0 失败，yolo golden 与 0.5.0 同误差；性能 min 27.8 / 192.6 ms，与基线持平 |
| 2026-11-18 | 结论修正 | 「rfdetr 每帧白分配 52 MB」在 0.5.0 已修（`LastIn*` 那套签名已在用）；P0 的价值是**结构上防复发**，不是修 bug |
| 2026-11-18 | 包形态定案 | 两个包：主包保持纯托管 + `Sevencat.OnnxSharp.Vulkan`；P1 只走 `ProjectReference` 不发包；P3-6 再出 `.All` 聚合包 |
| 2026-11-18 | 版本单一来源 | `<Version>` 移到 `src/Directory.Build.props`，主包 csproj 不再各自持有 |
| 2026-11-18 | P1-1/P1-2 | Vulkan 项目接入 `slnx`；`Vk.cs` 511 行移植 + `VulkanLoader.TryProbe`；实测 `vulkan-1.dll`、2 个设备 |
| 2026-11-18 | P1-3 | `VkDevice` 完成：instance / 排序选卡 / 计算队列 / 内存类型 / 命令池+fence / buffer 往返 / pipeline 骨架；5060 与 610M 均 OK，50 次提交无 lost |
| 2026-11-18 | P1-4 | shader 工具链落地：`build-shaders.ps1`（glslc 优先、glslang 兜底）、`.spv` 入库 + 嵌入资源、`VkShaders.Load`；双 GPU 真实 dispatch 100k 元素误差 9.5e-7，pipeline/描述符/push constant 路径同时得到验证 |
| 2026-11-18 | P1-5 | 模板 `VkPipelineTemplate`：默认 push descriptor（两台设备均支持，省掉 set 分配/更新）、SPIR-V 反射 `local_size` 与 binding（声明不符在创建期报错）、`VkTimer` 时间戳；实测 kernel 2.4 us / 墙钟 96 us，**小 kernel 95% 是提交税** → P1-6 必须批量提交 |
| 2026-11-18 | P1-6 | `VkSegment`：一个 arena（256 B 对齐切槽）+ push descriptor 切片绑定 + 依赖感知 compute 屏障 + 单次提交/复用 fence 与命令缓冲；2 算子依赖链结果正确（maxErr 3.3e-6），16 dispatch 批量提交比逐算子快 **10.5×（N 卡）/ 4.3–5.0×（A 卡）** |
| 2026-11-18 | 硬件更正 | 开发机实为 **RTX 5060 Laptop + Radeon 610M**，非计划中的 Arc B580；恰好覆盖独显/核显与 N/A 两家驱动 |
| 2026-11-18 | P1-7 | `VkPartitionLayer : Layer` + `VkWeightArena`：权重登记一次常驻显存、激活按槽位 stage in/out、中间结果不出显存；26 → 一个 submit。端到端 `Tensor` 往返 maxErr 1.0e-6，forward 234 us（N）/ 249 us（A），并挡住非紧凑视图与形状不符 |
| 2026-11-18 | P1-7 性能坑 | stage out 曾要 2.2 ms：`preferHost` 选到的 host 内存是写合并（WC），CPU 读 400 KB 只有 0.2 GB/s。改 `MemoryKind`（上传 WC / 回读 HOST_CACHED）后回读 33 us（**13×**）；批量 stage 相对逐张量往返由 1.3× 升到 2.0× |
| 2026-11-18 | P1-7 顺带修 | `Upload`/`Download` 改复用 staging（一次 `vkAllocateMemory` ≈200 us > 400 KB 拷贝本身）；`Flush`/`Invalidate` 按 `nonCoherentAtomSize` 外扩；`StageOut.Bytes` 支持只读回 payload 不读对齐填充 |
| 2026-11-18 | P1-9 实测 | 重建一个分区 **499/512 us**，其中 1171 KB 裸分配就 **646/570 us** —— 瓶颈全在 `vkAllocateMemory`，不在建段逻辑 |
| 2026-11-18 | P1-9 定案 | **不做「按 shape 缓存 VkSegment」**，改为设备级 `VkBufferPool`：键是 (用途, 申请字节数)，与布局无关，没有跨对象别名风险。段缓存能省的只有 arena 分配，池化同样省掉，却不必处理「段被两个 shape 共享」的生命周期 |
| 2026-11-18 | P1-9 完成 | 池化后段重建 **499/512 → 1 us（≈500×）**，完整分区含权重 **1301/1422 → 691/619 us**；命中率 75%。权重常驻区**不进池**（长生命周期，复用会让旧 `VkBind` 静默指向新数据） |
| 2026-11-18 | P1-9 防线 | `MemKind` 用「归还即入池」+`Released` 标志：Dispose 后再 `Map`/`Flush` 抛 `ObjectDisposedException`，而不是静默读写别人的数据。设备丢失时 `MarkLost()` 清空池、后续分配快速失败 |
| 2026-11-18 | 待办（P1-8 带入） | 剩下 619 us 是**权重重新上传**（390 KB device-local 分配 + 一次提交）；应改成模型级共享 `VkWeightArena`，让段重建不碰权重 |
| 2026-11-18 | P1-8 规划器 | `VkPartitionPlanner`：最长连续可发射段 + live-in/live-out 判定 + `SelfCheck` 自检；纯图逻辑，无设备即可验证（4 项边界用例全过）。`MinOpsPerSegment` 默认 **3**：1~2 个算子成段是净亏（单次往返 234 us > 单算子 CPU 计算） |
| 2026-11-18 | P1-8 接缝 | `IVkOpEmitter` + `VkOpRegistry`：P2 每实现一个 GPU 算子注册一次即可参与选段。**注册表为空时规划结果为空**——P1 阶段不会误把 CPU 算子搬上 GPU |
| 2026-11-18 | P1-8 共享权重 | `VkPartitionLayer` 可挂模型级 `VkWeightArena`（`OwnsWeights=false`，不负责 Commit/Dispose）。3 段一起重建 **4.6/4.3 us**（独占权重时单段 619/691 us，几乎全是权重重传），三段各绑自己切片数值正确 |
| 2026-11-18 | P1-8 段物化 | `VkGraphLayer`：段内**延迟形状推断**（自己的 `GetMemoryShapes` 把子算子按拓扑序推一遍），`Finalize` 时按结果开槽位、叫 `IVkOpEmitter` 追加 dispatch、Seal。形状未知是图改写期的固有约束——建段只能晚到 `Finalize` |
| 2026-11-18 | P1-8 拼接 | `VkPartitionSet.Apply(net, graph)`：规划 → 段内 blob/常量登记进模型级权重区并**统一上传一次** → `Graph.Reset` 把一段换成一个层。`Interpreter` 一行未改 |
| 2026-11-18 | P1-8 接缝取舍 | 段内子算子不是 `prog` 成员，`PrepareForGraph` 不给它们绑 `Net`，而 `Reshape/Slice` 的形状推断要读常量值。试过 `InternalsVisibleTo("OnnxSharp.Vulkan")`，但探针把后端源码链进自己的程序集（`vkdev`）仍然编不过；最终加一条公开接缝 `NetImpl.Attach(op)`，行为与 `PrepareForGraph` 内部一致，且不污染 friend 列表 |
| 2026-11-18 | P1-8 端到端 | 真图 7 算子（3 GPU + 1 CPU + 3 GPU）→ `[VkSegment, CpuScale, VkSegment]`，槽位 8、dispatch 6，`maxErr` 与纯 CPU **完全相同（9.54e-7）**；换输入重跑 4.77e-7，段复用不残留上一帧数据 |
| 2026-11-18 | P2 首个真实算子 | `binary_bcast.comp` + `VkElementwiseEmitter`：`Add/Sub/Mul/Div/Max/Min`，F32，ONNX 右对齐广播用**零步幅**表达（广播维步幅 0），一份 shader 覆盖同形 / `[1,C,1,1]` / 标量三种形态，无需按形状特化 |
| 2026-11-18 | P2 常量归位 | 常量输入**不再算 live-in**：规划器加 `isConst` 谓词把它们排除，发射器绑成权重切片。否则 BN 的 gamma/beta 每帧被重拷一遍（正是 P1-8 权重常驻要解决的问题又漏回来） |
| 2026-11-18 | P2 能力守卫 | `VkOpRegistry.CanEmit` 之前只查注册表、不问发射器，结果 **int64 的 Add 也被规划进段**（探针实测 segs=1）；改成委托给 `IVkOpEmitter.CanEmit` 后 segs=0。dtype 在加载期就已知，不必等建段才炸 |
| 2026-11-18 | P2 端到端 | 真 `NaryEltwiseLayer` 组成的 7 算子图（含 `Pow` 作 CPU 硬边界）→ `[VkSegment, Pow, VkSegment]`，槽位 10、dispatch 6，**maxRelErr = 0**（与 CPU 逐位相同） |
| 2026-11-18 | P2 成本数据 | 同图 16K 元素：CPU 230 us / GPU 373 us（VK-NV），1186 / 2034 us（VK-iGPU）——3 算子纯逐元素段是**净亏**。逐元素算子的价值在于把 Conv 之间填满、让段长够，不是自己上 GPU |
| 2026-11-18 | P2 Conv | `conv2d.comp` + `VkConvEmitter`：朴素直接卷积（NCHW、一线程一输出、c→kh→kw），支持 groups/strides/dilations/pads/`auto_pad`(SAME_UPPER/SAME_LOWER/VALID)/bias。pad 用**反推自检**：用 pad 重算输出尺寸，与形状推断不符就抛，而不是静默算错边界 |
| 2026-11-18 | P2 Conv 端到端 | yolo 风格 8 算子图（conv+bias → Add → Mul → **Pow(CPU)** → 1×1 分组 stride2 → Max → 3×3 dilation2 → Sub）→ 2 段、9 槽、7 dispatch，**maxRelErr = 0**；`auto_pad` 三连（SAME_UPPER/SAME_LOWER/SAME_UPPER，9×11 奇数尺寸）maxRelErr 5.96e-8 |
| 2026-11-18 | **CPU bug（探针暴露）** | `kernel_shape` 在 ONNX 里是可选的，`Forward` 的 `Resolve` 缺省成 **1**，而形状推断用的是权重尺寸。于是「有 auto_pad、没有 kernel_shape」的合法模型按 pad=0 计算：数值错，且末尾会读到输出行外（越界读）。修法：`Resolve` 增加 weight 形参，缺省时取 `weight.Dims[2+i]` |
| 2026-11-18 | 回归测试 | 新增 `conv: 2-D SAME_UPPER equals explicit pads without kernel_shape`，与显式 pads 逐元素比对。**已验证**：把修复退回去该用例失败（134/137），恢复后通过（135/137） |
| 2026-11-18 | P2 Conv 成本 | 单层 64→64 3×3 @32×32（75 MFLOP）：dGPU **1.2x–4.0x 快**、iGPU **0.05x–0.33x（慢 3–20 倍）**（预热后多轮区间）。朴素卷积只在独显上可能回本，核显必须靠 P2-2（1×1 特化）/P2-3（NHWC 向量化）或 P2-6 的设备感知成本模型 |
| 2026-11-18 | P2 接缝 | `LayerRegistry.Create` 现在把 `parameters` 落到 `LayerInfo.Parameters`：属性是算子定义的一部分，GPU 发射器读 stride/pad/group 全靠它，不能再依赖「恰好由导入器创建」 |
| 2026-11-18 | **P2-2 1×1 特化** | `conv2d_1x1.comp`：沿输出通道做寄存器分块（MT=8），`v` 载入一次复用 8 次，x 载入量降到 1/8。条件：`KH=KW=1`、`D=1`、`pad=0`、`OutPerGroup % 8 == 0`，否则回退朴素核。正确性（group=2 + bias + 12×10 奇数尺寸）`maxRelErr = 0` |
| 2026-11-18 | P2-2 收益 | 同形状 A/B（M=256 命中新核 vs M=255 回退朴素核，FLOPs 差 0.4%）：**dGPU 1.4–2.0x、iGPU 5.7–10.0x**。但绝对水平仍低（dGPU ~300 GFLOPS），1×1 在 CPU 的分块实现面前并不占优 |
| 2026-11-18 | **并行度坑** | 1×1 首版按「一线程一像素、M 全在核内」实现，dGPU 上比朴素核**慢 14%**：256→256@32×32 只有 1024 线程，独显 SM 填不满。改为「像素 × 通道块」二维分派（块号在外、像素连续以保合并）后反超。同一 bug 在 iGPU 上完全看不出来——两卡必须都测 |
| 2026-11-18 | **测量方法修正** | CPU 卷积在约 30 次调用处因 tiered-JIT 提升从 1.3 ms 掉到 0.17 ms，不预热会把 8 倍波动算进基线。探针 `Best()` 强制 `warmup=40`，此前记下的「3.5x」等单值已改为区间 |
| 2026-11-18 | **P2-6 设备门** | 新增 `VkDevicePolicy`，把「能不能发射」（`CanEmit`：dtype/形状/属性）与「值不值得发射」（设备类别）拆开，在 `VkPartitionSet.Apply` 里串联。核显拒发 `Conv`；**逐元素不拦**——它们是段的主体且在核显上仍赚。理由是实测而非猜测：核显 Conv 慢 3–20 倍、1×1 特化后仍慢 5–7 倍 |
| 2026-11-18 | 设备门验证 | 双卡探针：核显上 8 算子图 → **0 段、prog 保持 8 个算子**（整段回退到 CPU），而逐元素图仍是 2 段；独显不变（2 段、3.65x）。拒因随 `Allows(out reason)` 打出来，设备门不是黑盒 |
| 2026-11-18 | **P2-3 1×1 向量化分块** | `conv2d_1x1.comp` 改为「像素 × 通道」二维分块（TP=4、MT=8）：一线程算 4 个连续像素 × 8 个输出通道，算术强度 4.0 → **16.0 FLOP/B**。像素组沿 OW 向上取整（一组必须同一行，否则输出跨行、合并性归零），尾组（OW%4 = 1/2/3）逐像素判断，探针覆盖 **OW=12/10/13** 三种尾巴，`maxRelErr ≤ 5.96e-8` |
| 2026-11-18 | **P2-3 收益 = 0（重要）** | 端到端 408–426 us vs TP=1 的 435 us，**在噪声边缘**；把 c 循环展开 4 次同样无变化（401–426 us）。GPU 侧时间本身很稳（±3%，来自 5 次重复），所以不是测不准。假设「带宽受限」被推翻，转而查时间去向 |
| 2026-11-18 | **搬运主导（关键发现）** | 尺寸扫描 16/32/64 → 228/414/1266 us，拟合 `time ≈ 155 us + FLOPs/486 GFLOPS`。判别实验（FLOPs 同为 33.5M、字节 0.52 vs 2.10 MB）→ 227 vs 372 us，**1.58 MB 换 145 us ≈ 10 GB/s（PCIe 量级）**。结论：32×32 的 415 us 里 kernel 仅 ~200 us，stage 往返 ~210 us。1×1 即使 kernel 无限快也只能到 210 us，而 CPU 基线 250–380 us ⇒ 最好打平 |
| 2026-11-18 | 优先级调整 | NHWC 重排（P2-3 后半）的端到端收益上限约 **1.4×**，不再是性能突破的唯一路径，优先级下调。搬运侧只有三条路：fp16 存储减半（P3-1）、少断段（每个 CPU-only 算子都强制一次往返）、跨帧复用（推理省不掉）。ReBAR 已可用但**救不了这一项**：数据每帧仍要过一次 PCIe |
| 2026-11-18 | **模型纠错（重要）** | 尺寸扫描里字节与 FLOPs 完全共线，第一版把搬运成本误算进算力（"486 GFLOP/s"）。用字节实验解耦后重算：`t ≈ 120us/dispatch + 字节/10.9GBps + FLOPs/1.47T`，五个数据点（含 3 算子逐元素段）全部落在 ±16%，其中逐元素段 372 vs 实测 373。教训：**共线的两个自变量无法靠扫描分离，必须构造单一变量实验** |
| 2026-11-18 | **P2-6 成本判据** | 新增 `VkCostModel`（段级成本预测）与 `VkPartitionSet.Apply(..., costGate)`。CPU 侧必须按算子类别取速率（1×1 ≈400 GFLOP/s vs 3×3 ≈60 GFLOP/s，差一个数量级），混用会让判据在 1×1 上完全失效 |
| 2026-11-18 | **模型二次纠错（重要）** | 上一版把「120 us」当成**每 dispatch** 摊。缩放实验（1/2/4/8 个串联算子，同张量同 submit）证明是**4 us/dispatch + 120 us 每段**：6 个算子拼一段 = 144 us，拆 6 段 = 720 us，差五倍。顺带翻案：早先「3 算子逐元素 16K：CPU 230 / GPU 373 us 必亏」其实是切成了**三段**，拼成一段只要 ~132 us，比 CPU 快。**真凶是断段** |
| 2026-11-18 | **真规模对账** | yolo 风格 6 算子块（中间层 64×64×32，83.8 MFLOP）：dGPU **395 us vs CPU 402 us = 1.02×**（打平），判据预测 gpu 389/cpu 407，误差 3%，err 5.4e-7。iGPU 判据把 0.25× 的亏损拦回纯 CPU。**首测曾报 7.3×，是 CPU 基线没热透（20 次预热）的假象**——真实图景是卷积块只能打平，因为 GPU 3×3 核 0.44 TFLOP/s 对 CPU 0.2–0.4 TFLOP/s 只差 1–2 倍，再扣 120 us 固定成本 + 搬运就没了 |
| 2026-11-18 | **基线纪律（重要）** | CPU 基线必须预热 ≥80 次：tiered-JIT 约第 30 次调用才提 tier1，20 次预热把 yolo 基线从 420 虚报到 2807 us（6.7 倍）。同一个坑还污染了早先的「3×3 CPU 1246 us」→ `CpuConvGflops` 被低估 3 倍（60 应为 200）。**这个错误方向上等于白扔 PCIe 往返** |
| 2026-11-18 | 判据细化 | 卷积算力按核类别分开：点卷积 1.47 TFLOP/s、通用 3×3 卷积 0.44 TFLOP/s。卷积不再叠加每算子开销（已折进实测算力）。`VkPartitionSet.CostReport` 输出段表，供真模型对账「判据估错」与「kernel 慢」 |
| 2026-11-18 | 判据验证 | 探针四个已知答案：1×1@32×32 拒（gpu 408/cpu 396）、3×3@32×32 发（305/1107）、1×1@64×64 发（1259/1342）、3 算子逐元素 16K **发**（~132/246，一段）。切片类用例显式 `costGate: false`——它们验的是拼接正确性，张量只有几百字节，开着判据就变成在验判据 |
| 2026-11-18 | **下一步优先级重排** | 按新模型拆 yolo 6op 的 395 us：固定 120 + dispatch 20 + 搬运 63 + **3×3 算力 192**。fp16 只能动 63（省 31 us = 8%），而 3×3 核从 0.44 TFLOP/s 提到 1.5 T 可省 130 us（33%）。**先做 3×3 分块核，fp16（P3-1）后置**——这与我上一轮的判断相反，依据是那条 192 us 的实际占比 |
| 2026-11-18 | **3×3 专用核** | 新增 `conv3x3.comp`：线程算 8 通道 × 4 连续像素，每 (c, kh) 只载 6 个连续输入供 96 次 MAC 复用。单卷积 32→32@64×64：608 → **427 us**，有效算力 0.36 → **0.58 TFLOP/s**，err 1.6e-6 不变；yolo 6op 395 → 340 us。只覆盖 stride=1/dilation=1（stride≠1 时连续窗口不成立，硬用会毁掉唯一优势），`ONNXSHARP_VK_CONV3=0` 可回退 |
| 2026-11-18 | 判据三档计价 | 卷积按 `ConvKind` 分档：点卷积 1.47、3×3 快路径 0.58、通用 0.36 TFLOP/s。类别判定与发射侧**共用 `VkConvEmitter.IsConv3x3Fast` 一个谓词**——判据以为快路径、实际发射慢路径，会把亏的段判成赚的 |
| 2026-11-18 | **fence 等待实测** | 8K 单算子（GPU 干活 <5 us）同进程 200 次：阻塞 min 367/p50 606，自旋 200us 为 124/434，自旋 5ms 为 **104/177**。阻塞的线程挂起/唤醒就是每段 120 us 的大头。但真实负载（yolo ~350 us）只有 **9%** 收益，故默认 200 us 预算（`ONNXSHARP_VK_SPIN`），大预算需显式开。坑：`vkGetFenceStatus` 的 `NotReady` 是成功码不是错误 |
| 2026-11-18 | **测量纪律** | 同一份二进制同一天，yolo CPU 基线在 402/1194/2094/3799 us 间跳，GPU 340/475/798/1140 同步跳。原因：tiered-JIT（<30 次调用不提 tier1，曾把 1.02× 报成 7.3×）+ IDE 索引抢核。规矩：跨进程加速比只在空闲窗口采信，可疑时改**同进程 A/B**（同图同权重只切一个开关） |
| 2026-11-18 | 全包形态修正 | 不做 meta-package（两个 dll）；新建 `src/OnnxSharp.All` 链接两份源码，产出**单个 `OnnxSharp.dll`**、nuspec 零依赖；消费方单包引用已实测可用 |
