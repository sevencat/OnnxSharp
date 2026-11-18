# OnnxSharp 实施计划 v3（LayerPlan 收敛 + Vulkan 后端）

> **文件分工**：`plan.md` 管功能，`planv2.md` 管 CPU 优化，`bench.md` 管现状数字，
> **本文件管 v3：先把 `LayerInfo` 的运行期状态收敛成 `LayerPlan`，再接入 Vulkan 计算后端**。
> 进度见 `progressv3.md`，Vulkan 数字见 `benchv3.md`。

- 立项：2026-11-18
- 前置：v2 收工（`bench.md §37/§38`）：rfdetr-small 198.0 ms、yolo11n 30.4 ms、OCR e2e −29%
- **v3 基线已重测**（2026-11-18，`benchv3.md §5.0`）：yolo11n **28.1 ms**、rfdetr-small **193.7 ms**
  （4 轮取最小）；rfdetr 存在 11.6% 的功耗档位双稳态，A/B 门槛相应提高
- 口径：CPU 侧沿用 4 线程 / FP32 / batch=1 / 静态形状

---

## 0. 定位

v2 已把 CPU 侧「够得着的大项」吃完（差距只剩 F32 GEMM 单线程与 NCHWc 覆盖面）。v3 转向：

1. **P0** 收敛 `LayerInfo` 上 10 个运行期状态字段 → 一个 `LayerPlan`。**行为不变的内部重构**，
   目标是消掉一类已踩过的 bug（`Interpreter.cs:153` 记录的每帧白分配 52 MB）。
2. **P1~P3** Vulkan 计算后端，以独立可选包提供，面向工控机与跨厂商 GPU；主包保持零依赖。

**不新建通用 ExecPlan**：`ConstFolder` + `GraphFusion`(12 pass) + `BufferAllocator`(生命周期着色) +
形状缓存已等价于 SimdOnnx 的 ExecPlan 的 80%；缺的「解释循环→步骤数组」实测收益 ≤1%
（yolo 0.3–0.7% / OCR 0.7–1.8% / rfdetr 0.3–0.8%）。Vulkan 需要的那部分由 `VkSegment` 自己长。

---

## 1. 目标与非目标

| 编号 | 目标 | 判定 |
|---|---|---|
| G1 | `LayerPlan` 收敛 | 字段 10 → 1 对象；签名比较 2 次 → 1 次；golden **逐位不变** |
| G2 | Vulkan 骨架跑通 | yolo11n 640² 出正确结果 |
| G3 | 分区式集成 | 不改 `Interpreter`，用 `VkPartitionLayer : Layer` + `Graph.Reset` |
| G4 | 性能可用 | yolo11n ≤12 ms（目标 ≤8 ms），GPU 段覆盖 ≥80% conv/gemm FLOPs |
| G5 | 部署轻 | 主包 `PackageReference` 仍为 0；Vulkan 包 ≤5 MB |

**非目标**：CUDA/TensorRT/OpenCL/DirectML/WebGPU/ROCm；全局 NHWC 布局规划（会推翻 1800 行
NCHWc/Winograd，CPU 负收益）；fp16 运算（只做 fp16 存储）；多后端 `IBackend` 抽象层；
控制流子图；int8 GPU 内核。

---

## 2. 现状核查（P0 依据）

| 字段 | 可见性 | 语义 |
|---|---|---|
| `FinalizedOnce` | **public** | 跑过 `Finalize` |
| `LastInShapes` / `LastInTypes` | **public readonly** | 上次真正 Finalize 的输入签名 |
| `CachedInShapes` / `CachedInTypes` | internal | 缓存判定用签名 |
| `CachedOutShapes` / `CachedOutTypes` | internal | 输出形状/类型 |
| `CachedTempShapes` / `CachedTempTypes` | internal | 临时张量形状/类型 |
| `ShapeCacheValid` | internal | 缓存可信 |

3 组语义 / 10 个字段 / **2 套签名**。全仓引用面：`Interpreter.cs` 28 处、`LayerInfo.cs` 10 处（声明）、
**其余 0 处** ⇒ `GraphFusion`/`ConstFolder`/`NetImpl` 不依赖，重构局部可逆。

三个问题：
1. **两套签名会不同步**：`sameSignature` 依赖 `ShapeCacheValid`（动态层从不写），
   `signatureChanged` 依赖 `LastIn*` → 动态层每帧重跑 `PrepackWeights+Finalize`（52 MB/帧）。
2. 状态散在 `Interpreter` 的 5 个分支（`:78/:84/:122/:158/:162`），读不懂。
3. `Layer` 公开面泄漏实现（3 个 public 可写字段）。

---

## 3. P0 — LayerPlan 收敛（先做，2~3 天）

目标形态（`Core/LayerPlan.cs`，internal）：

```csharp
internal enum PlanState { None, Shaped, Finalized }

internal sealed class LayerPlan
{
    public PlanState State;
    public readonly List<TensorShape> InShapes = new();   // 唯一一份签名
    public readonly List<DType>       InTypes  = new();
    public readonly List<TensorShape> OutShapes = new(), TempShapes = new();
    public readonly List<DType>       OutTypes  = new(), TempTypes  = new();
    public bool Matches(IReadOnlyList<TensorShape> s, IReadOnlyList<DType> t);
    public void Invalidate() => State = PlanState.None;   // 不清列表，保容量
}
```

`Interpreter` 主循环 5 分支 → 3 段：`if (!plan.Matches(...)) { 推断 + 写签名; State=Shaped; }` →
`BindAll/FitTemps` → `if (State != Finalized) { PrepackWeights(); Finalize(); State=Finalized; }`。
`Shaped` 与 `Finalized` **共用一份签名**，52 MB 那类 bug 从「靠注释提醒」变成「结构上不可能」。

| ID | 任务 |
|---|---|
| P0-1 | 新增 `LayerPlan`（含 `Matches` 短路径与逐项比较）+ 单测 |
| P0-2 | `Layer` 加 `internal LayerPlan Plan`；3 个 public 字段收为 internal（见 §9-A） |
| P0-3 | 改 `Interpreter.Run` 为 3 段，删重复的 `SameShapes`/`SameTypes` 调用 |
| P0-4 | 删 10 个旧字段，`grep` 归零 |
| P0-5 | 回归：4 模型 golden **逐位**比对 + `dotnet test` |
| P0-6 | `progressv3.md` 记结论（含 rfdetr 每帧分配前后） |

验收：旧字段 `grep` 为 0；golden 输出 `float` 哈希一致；rfdetr 每帧分配不高于改造前；
`Interpreter.cs` 行数下降；API 变化仅 §9-A 一处并写 CHANGELOG。
回退：改动集中在 2 文件 + 1 新文件，单 commit revert。

---

## 4. P1 — Vulkan 骨架（3~4 周）

**包边界**：`src/OnnxSharp.Vulkan/`（引用主包；主包 `PackageReference` 保持 0）。
P1 期间**只用 `ProjectReference`，不发 NuGet**：管理成本与同包持平，但主包「纯托管」定位不动；
版本号抽到 `src/Directory.Build.props` 做单一来源（注意别放仓库根，`.tmp/` 下二十余个探测项目会继承）。
绑定**手写 `[LibraryImport]`**（SimdOnnx `Vk.cs` 511 行同款，AOT 友好、无 NuGet），不引 Silk.NET。
shader 用 `build-shaders.ps1` 生成：优先 `glslc -O --target-env=vulkan1.1`，没有 glslc 时退到
`glslang -V -Os --target-env vulkan1.1`（便携版放仓库根 `bin/`，已在 `.gitignore`）。
`.spv` **入库** + `<EmbeddedResource LogicalName="OnnxSharp.Vulkan.Shaders.<name>.spv">`，
运行时由 `VkShaders.Load(name)` 按名取，**build 与 CI 不需要 Vulkan SDK**；只有改了 `.comp`
的机器才需要编译器。缺 `.spv` 时 MSBuild 在 `CoreCompile` 前直接报错（`VerifyShaderArtifacts`），
不会拖到运行时才炸。新增 shader 时记得同步 `src/OnnxSharp.All` 里的同名 `EmbeddedResource`。

**集成形态**：`VkPartitionLayer : Layer` 包住一段连续算子，`Forward` 里 `StageIn → Submit → StageOut`；
分区 pass 用 `Graph.Reset` 替换 `prog`。已核实的三个红利：
- `Layer`/`Forward`/`GetMemoryShapes`/`DynamicOutputShapes`/`PrepackWeights` 全 `public virtual`；
- `Graph.Reset`/`BufferAllocator.Assign`/`GetArgTensor`/`GetArgData`/`BindArgTensor` 全 public；
- 段内 op 移出 `prog` 后，中间 Arg 因 `firstWrite==NeverWritten` **自动不占 host 槽位**。

| ID | 任务 | 规模 |
|---|---|---|
| P1-1 | 建项目、目录、解决方案接线 | 0.5d |
| P1-2 | `Vk.cs` 绑定 + `SetDllImportResolver`（win/linux/android） | 2d |
| P1-3 | `VkDevice`：instance/选卡/计算队列/内存类型/命令池/fence/buffer/pipeline | 4d |
| P1-4 | shader 工具链：`build-shaders.ps1` + `VkShaders.Load` | 1d |
| P1-5 | 通用 pipeline 模板（N storage buffer + push constant） | 2d |
| P1-6 | `VkSegment`：binding 映射、arena 偏移、dispatch 列表、push constant 打包 | 5d |
| P1-7 | `VkPartitionLayer`：stage in/out（`Tensor` 是 `byte[]`+`ByteOffset`） | 2d |
| P1-8 | 分区 pass：`EndOfGpuRunnable` + 最长段 + `MinOpsPerSegment` | 3d |
| P1-9 | 生命周期：按 shape 缓存 `VkSegment`(LRU)、`Dispose`、设备丢失重试 | 2d |

验收：yolo11n 全图（或 ≥80% FLOPs 段）跑通，与 CPU 余弦相似度 ≥0.9999；连续 50 次无设备丢失；
主包依赖仍为 0；无 Vulkan SDK 的机器 build 成功。

---

## 5. P2 — 算子与分区（4~6 周）

| 批次 | 算子 |
|---|---|
| B1 | `Conv`(3×3/1×1/depthwise)、`Gemm`/`MatMul`、`Relu`/`Silu`/`Sigmoid`、`Add`/`Mul`(broadcast)、`MaxPool`/`AveragePool`/`GlobalAveragePool` |
| B2 | `Softmax`、`LayerNormalization`、`Resize`、`Concat`、`Transpose`、`Slice` |
| 不做 | 元数据类（`Shape`/`Slice` 等 4 处动态 op 留 CPU）、控制流、int8 |

分区约束：`MinOpsPerSegment ≥ 8`（段碎会被 stage 吃掉）；动态 op 作硬边界；段内 NHWC、
边界由 stage 顺带 relayout（不新增图节点）；不支持即断开，不做「部分 GPU」。
成本模型：`段 FLOPs / GPU 吞吐 > 2 × (段输入+输出字节) / 有效带宽` 才值得 GPU 化；
探针用 `ONNXSHARP_VK_MIN_OPS` / `ONNXSHARP_VK_DISABLE` 做 A/B。

> 实测（2026-11-18）：`MinOpsPerSegment` 暂取 **3**。16K 元素的 3 算子纯逐元素段仍是净亏
> （CPU 230 us / GPU 373 us，VK-NV），说明「算子个数」只是粗代理；真正的判据是上面的
> FLOPs/带宽式——见 `benchv3.md` §5.2。所以 `≥8` 不是硬规则，而是「没有成本模型时的经验值」，
> P2-6 落地成本模型后应改为按段估算。

---

## 6. P3 — 性能与收尾（2~3 周）

fp16 **存储**（load 即 widen，运算全 fp32）→ 带宽敏感算子 ≥1.5×；`VK_KHR_push_descriptor` 优先；
pipeline cache 落盘；timestamp query 接 `--profile`；部署验证（Win10 1903+ / Mesa / 核显）；
README 加 Vulkan 包小节 + `benchv3.md` 结果。

**P3-6 单 dll 全包**（不是 meta-package）：`src/OnnxSharp.All/OnnxSharp.All.csproj` 用
`<Compile Include="..\OnnxSharp\**\*.cs">` + `..\OnnxSharp.Vulkan\**\*.cs`（各自排除 `obj`/`bin`）
把两份源码链进同一项目，`AssemblyName` 仍为 `OnnxSharp`，`dotnet pack` 一次产出
**只含 `lib/net*/OnnxSharp.dll` 一个程序集、nuspec 零依赖**的 `Sevencat.OnnxSharp.All`。

- **与主包互斥**：装配体标识同为 `OnnxSharp`，同一项目不能同时引用 `.All` 与 `Sevencat.OnnxSharp`
  （CS0433）。全包面向「一台工控机一个包」的部署形态，不与生态共存。
- **为什么不用 `-p:IncludeVulkan=true` 开关主包**：实测有增量陷阱 —— 默认构建之后直接
  `dotnet pack -p:IncludeVulkan=true`，MSBuild 复用 `bin/` 里的旧 dll，打出的包 537,600 B、
  不含任何 Vulkan 类型（假全包）。独立项目有自己的 `obj/`，两次构造成果稳定。
- **同步点**：`InternalsVisibleTo`、`PackageIcon`、将来的 `EmbeddedResource`（`.spv`）需在
  All 项目重复一份；Vulkan 侧新增资源时必须同步补。
- **已验证（2026-11-18）**：`Sevencat.OnnxSharp.All.0.6.0.nupkg` 710 KB，nuspec 依赖数 **0**，
  三个 TFM 的 dll 均 558,080 B；`.tmp/allprobe` 只引用该包即可同时拿到
  `OnnxSharp.Core.Interpreter` 与 `OnnxSharp.Vulkan.VulkanLoader`，并枚举到 2 个设备。

---

## 7. 里程碑

| 里程碑 | 内容 | 工期 |
|---|---|---|
| V3-M0 | P0 完成，全部 golden 逐位不变 | 2~3 天 |
| V3-M1 | P1 完成，yolo 全图数值正确 | 3~4 周 |
| V3-M2 | P2 完成，B1 算子 + 分区启发式 | +4~6 周 |
| V3-M3 | P3 完成，fp16 存储 + 部署验证 | +2~3 周 |

合计 **10~14 周**（单人，MVP = M1 约 5 周）。

---

## 8. 风险

| 风险 | 应对 |
|---|---|
| 段切太碎，stage 吃掉收益 | `MinOpsPerSegment` 起步 8，用 A/B 调；必要时段间保留 device buffer 驻留表 |
| 工控机核显驱动行为差异（N/A/I 三家） | 只用 Vulkan 1.1 core + 可选扩展探测，不做 vendor 分支；`ONNXSHARP_VK_*` 提供降级开关 |
| 无 ROCm/CUDA 的机器性能不如预期 | 先测 B580 + 一台 Intel/AMD 核显，低于阈值就只发「可用」不发「加速」 |
| P0 破坏 public API | 升 0.6.0；如需兼容保留 `[Obsolete]` 转发一版 |

---

## 9. 待确认

- **A（P0 阻塞）→ 已定（2026-11-18）**：三个 public 状态字段**直接收为 internal**，
  版本升 **0.6.0**，不做 `[Obsolete]` 转发过渡。理由：语义上本就不该公开，
  且全仓（含 tests/tools）无任何外部读取方，过渡期只会让错误用法多活一个版本。
- **B**：Vulkan 包名用 `OnnxSharp.Vulkan` 还是 `OnnxSharp.Backends.Vulkan`？
- **C**：首批验收机型（是否需要 AMD 独显/核显各一台）？
- **D（已定 2026-11-18）**：分两个包发布（主包保持纯托管、Vulkan 独立可选包）；
  **将来**另出 `Sevencat.OnnxSharp.All` 聚合包（meta-package，只含依赖）。
  P1 阶段不发 NuGet，仅 `ProjectReference`；聚合包排在 P3-6。
