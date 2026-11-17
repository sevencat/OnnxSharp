# OnnxSharp 0.5.0：车牌能认、SAM3 能分割、YOLO 会报名字

`Sevencat.OnnxSharp` **0.5.0** 已经推到 NuGet（上一版是 0.4.0）。版本号只跳了一格，但这一轮的四条改动
份量不一样：一条是新的业务门面（车牌），一条是把「提示式分割」这种三模型串联跑通（EfficientSAM3 EV-M），
一条是把 YOLO 检测结果从 `class0` 变成 `car`，还有一条是给首层卷积单开内核。

本文按「为什么做 → 怎么做 → 实测多少 → 坑在哪」写，所有数字都能在 `bench.md` §37~§41 找到原始记录与复现命令。
先给结论：

| 线 | 一句话 | 实测 |
|---|---|---|
| 车牌门面 `OnnxSharp.Ocr.Plate` | 不加模型、不加训练，纯解码侧特化 | 清晰裁剪图 100.0% / 99.6%；3200×1800 整图 825 ms 出 2 张 |
| YOLO 类别名 | 从 ONNX metadata 的 `names` 直接读 | `detect` 输出 `car 0.9103`，不再要 `--labels` |
| EfficientSAM3 EV-M | 三个 ONNX 全部用本库跑，逐张量与 ORT 对齐 | 掩码 IoU **1.00000**；`pred_masks` `max|d|` 2.0e-3（值域 ±32） |
| 首层 `ic=3` 卷积 | 单开 AVX-512 直接卷积内核 | 同形状 1.6×（16T 0.23 → 0.14 ms），端到端落在噪声里 |

---

## 一、车牌：不用新模型，也没打算停在「特化」这一步

先说清定位：**这一版的车牌通路只是把「通用 OCR 会犯的错」挡在解码阶段，它不是车牌识别的终点。**

通用 PP-OCRv6 的 rec 输出 6906 类，车牌字符全在里面，所以这条路不需要专门模型 —— 但也正因为如此，
模型那 6900 个跟车牌无关的类还占着容量，`S`/`5`、`B`/`8`、`0`/`O` 这类同形竞争会白吃 beam 宽度。
这一版做的是把噪声先砍掉：

| 层 | 做法 |
|---|---|
| 字符集 | 31 省 + 24 字母 + 10 数字 + 6 后缀 ≈ 70 类，字母数字里排除 `I` / `O` |
| 逐位规则 | `PlateCharset.AllowedAt(position, length)`：首位省、次位发牌机关字母、其余字母/数字、尾位可 学·警·挂·港·澳 |
| 解码 | CTC 前缀 beam（log 域），前缀打包进 `long`（每字符 7 位 rank + 高 8 位长度），非法长度直接剪 |
| 置信度 | 对最终序列做 CTC Viterbi 对齐（2L+1 状态）取**逐字后验**，几何平均 |
| 形态校验 | `PlateNumber.TryParse`：7 位燃油 / 8 位新能源 / 尾字特殊 / 首字 使·领 |

### 收益是真实的，但边界也很清楚

同一套 det/rec，只换解码逻辑（`plate` 子命令 vs `ocr` 子命令）：

| 通路 | 输出 | 分数 |
|---|---|---|
| 通用 `ocr` | `浙A BM770`、**`H浙ACP5872`** | 0.857 / 0.834 |
| 车牌 `plate` | `浙ABM770`、`浙ACP5872` | 83.5% / 82.7% |

`H` 那个字是 DB 框把号牌左侧的贴字/边框带了进来，车牌路径的逐位规则直接把它剪掉 —— 这是「字符集 + 位置约束」
最直接的一次收益。

清晰裁剪图（`--direct`，即上游已经把牌抠好）才是这条通路该有的水平：

| 图 | 输出 | 分数 | 类型 | rec 稳态 |
|---|---|---:|---|---:|
| 135×46 | `浙ABM770` | **100.0%** | 燃油 | 4.45 ms |
| 175×49 | `浙ACP5872` | **99.6%** | 新能源 | 7.45 ms |

整图（3200×1800，走 DB，`--limit 960`）825 ms 出 2 张：`浙AES8381` 97.6% 新能源、`新A3M15X` 62.9% 燃油。

### 这只是开始：真正要提升，得靠样本 + 微调

逐字后验把「哪一步不行」直接摊开了，两种失败要分开看：

```
# 3200×1800 整图，两块牌
浙  91% A 100% E 100% S  98% 8 100% 3 100% 8  94% 1 100%    ← 97.6%，可信
新  17% A  44% 3 100% M  94% 1  68% 5 100% X  81%           ← 62.9%，肉眼也糊

# 3840×2160 整图：省字只剩 33% / 24%（后 7 位仍 90~100%）
# DB 给的框是 258×110，真实号牌约 3.1:1 —— 是检出框把省字压窄了，不是解码错
```

- 3840 宽图那两个 83.5% / 82.7% 属于**检出框问题**：换成上游给框（`--crop`）或裁剪图（`--direct`）就回到 99%+。
- 3200×1800 里那块 62.9% 才是 **rec 本身的能力边界** —— 模糊、遮挡、污损的牌，它认不出来就是认不出来。

第二类问题靠再写多少约束都盖不住，所以下一阶段的方向很明确：

1. **攒样本**：不同省份、蓝牌/绿牌/黄牌（含双层）、旧式/新能源、白天/夜间/逆光/雨天、倾斜与透视、
   污损与遮挡、卡口与手机拍摄、不同分辨率与被压窄的号牌 —— 现在手里只有几张图，覆盖远远不够。
2. **微调 rec**：把输出头收窄到车牌字符集（~70 类），用真实样本微调；现在这套解码器（字符集 / 逐位规则 /
   形态校验）正好可以作为微调后的约束层，两者是叠加关系而不是替代关系。
3. **样本回来以后重跑这套指标**：逐字后验分布、`--direct` 准确率、整图召回，作为是否真的变好的判据。

换句话说：**0.5.0 交付的是「不会瞎报一个假号牌」的骨架，准确率要靠数据把它喂起来。**
本库不会内置样本、不会内置权重，也不会为了跑分去调阈值 —— 阈值和规则都在 `PlateReaderOptions` 里，可关可调
（`--no-structure` / `--loose` / `--score`）。

反向验证也留着：把 3840×2160 的整图硬塞给 `--direct`（输入本该是一块牌），结果是 `result : 0 张` ——
形态校验 + 逐字几何平均分会拦住它，不会因为用户选错入口就吐出一个假号牌。

```csharp
using OnnxSharp.Ocr.Plate;

var pipeline = new PaddleOcrPipeline(model);      // 与通用 OCR 同一个流水线
var reader = new PlateReader(pipeline, new PlateReaderOptions
{
    MinScore = 0.5f,                              // 逐字几何平均分下限
    MinAspect = 1.6f, MaxAspect = 6.5f,           // 只对「像车牌」的框解码
    MaxPlates = 4,
});

reader.DetectAndRead(pixels, width, height, OcrPixelFormat.Bgra32, stride);   // 整图，自己找牌
reader.Read(pixels, width, height, OcrPixelFormat.Bgra32, stride, x, y, w, h); // 上游给了框
reader.ReadWhole(pixels, width, height, OcrPixelFormat.Bgra32, stride);        // 输入就是一块牌
```

`PlateResult.Decoding.CharScores` 就是上面那张逐字后验，用来定位「哪个字不可信」。

---

## 二、YOLO：检测结果终于有名字了

以前 `Detection` 只有 `ClassId`，CLI 打出来是 `class0`，谁都不知道那是什么。而 ultralytics 导出的 ONNX
本来就把类别表写在 metadata 里：

```
names = "{0: 'person', 1: 'bicycle', …}"     # Python dict 字面量，不是 JSON
imgsz = "[640, 640]"
```

所以 0.5.0 直接解析它，坏格式 / 缺号都不抛：

```csharp
var detector = new YoloDetector("yolo11n.onnx");

detector.ClassNames              // 80 条，下标即类别 id；模型没带 names 时是空表
detector.NameOf(0)               // "person"；缺号或没表时为 null
detector.Metadata.InputHeight    // imgsz 的高/宽，int?（yolo11n 是 640 / 640；没写时 null）

foreach (var d in detector.Detect(pixels, width, height))
    Console.WriteLine($"{d.Label} {d.Score:F3}");   // car 0.432
```

`Detect` / `Decode` 返回的每个 `Detection` 都填好了 `Name`（缺号时 `null`）与 `Label`（退回 `class{id}`）。
自己跑前向的可以手动传表：`YoloDecoder.Decode(output, transform, options, names)`；
只想要 metadata 的：`YoloMetadata.Parse(net.Metadata)`。

名字来源的优先级分两层，别混：

- **库**（`YoloDetector`）**不猜**：模型没带 `names` 就留空，要 COCO 名字请显式传 `CocoLabels.Names`。
- **CLI `detect`** 会兜底：`--labels` 文件 > 模型 `names` > COCO（仅当检测头字段数 == 84），并把来源打在 `labels :` 那行：

```
$ OnnxSharp.Cli detect yolo11n.onnx photo.jpg --conf 0.25 --iters 40
model      : yolo11n.onnx
image      : photo.jpg 3840x2160
family     : yolo
labels     : 80 classes  (model metadata)
input      : [1, 3, 640, 640]  (3840x2160 -> 640x640 (scale 0.1667x0.1667, pad 0,140))
detections : 1
  #0   car              0.4323  [1847.3, 11.9, 951.6 x 978.3]
```

顺带解决的是**非方形输入**：yolo26s 是 480×640（yolo11 是 640×640），锚点变成 `60×80 + 30×40 + 15×20 = 6300`，
letterbox 的 pad 是 `(0, 60)`，坐标反算仍然准；`84 = 4 + 80` 由字段数反推 COCO，`--labels coco` 变成可选。
golden 从 ORT 生成，逐元素容差 2e-2。

---

## 三、EfficientSAM3 EV-M：三个模型跑通，四个 bug 落地

官方 EV-M 导出把分割拆成三个 ONNX，和仓库里原有的 SAM3 门面不是同一套（没有 geometry-encoder，提示走
`exemplar_*` 槽位 + CLIP 文本）。本库没给它做门面，直接用 `Net` 串，先把契约摆清：

| 模型 | 输入 | 输出 |
|---|---|---|
| `image_encoder` | `image[1,3,1008,1008]`（拉伸 + ImageNet 归一化） | `fpn_0[1,256,288,288]` / `fpn_1` / `fpn_2` + `pos_0/1/2` |
| `text_encoder` | `tokens[1,16]`（int64，BOS 49406 / EOS 49407） | `language_features[16,1,256]` / `language_padding[1,16]` |
| `decoder` | `fpn_*` + `pos_2` + `language_*` + `exemplar_boxes[8,1,4]` / `labels` / `mask` | `pred_logits[1,200,1]` / `pred_boxes` / `pred_masks[1,200,288,288]` / `presence_logit` |

串联只有两个必须记住的点：**输出张量指向引擎缓冲槽，跨模型必须 `Clone()`**；**掩码是 logits，`> 0` 即前景**。
后处理是 `score = sigmoid(pred_logits) × sigmoid(presence_logit)`，取最高分的 query。

### 四个 bug，全是「形状一换就露馅」的类型

| bug | 症状 | 根因 |
|---|---|---|
| `FirstConv` 末块写穿行尾 | image encoder 输出对不上 ORT | 1008/2 = 504 = 7×64 + 56，末块只剩 56 像素但仍走满 4 次 16 宽 Store，污染下一行头 8 个像素 |
| `ConvTranspose` 相位 GEMM 散写 | `fpn_0` **87%** 元素错、`max|d|` 到 7.3 | `dstCo = yb + co*outPlane + kh` 把行偏移当列偏移加了，少乘 `outW` |
| 广播外积 `Add` 掉通用循环 | 5 ns/元素 × 8.3 M = 40 ms | 相对位置外积 `[1,200,72,1,8] + [1,200,1,72,8]`，每元素维护 5 个索引 |
| `Einsum('bqc,bchw->bqhw')` 逐 query 标量累加 | mask 一步 780 ms，decoder 5.0 s | 200 条 query 每条把 85 MB 的 `B` 全读一遍 = 17 GB 流量 |

修法都不是「调参」：末块判据改成 `remaining >= Tile`；`kh` 乘 `outW`；新增 `TryApplyStridedRun`
把「两侧都还是紧凑步长」的轴并成一段走 AVX（40 → **1.2 ms**，30×）；`Einsum` 认成一次
`[200,256]×[256,288²]` 的 GEMM（780 → **53 ms**，decoder 5.0 → 2.9 s）。

修完的数值对齐（同输入 vs onnxruntime）：

| 模型 | 输出 | `max|d|` |
|---|---|---|
| image_encoder | `fpn_0/1/2` | 6.3e-7 / 2.1e-5 / 2.3e-5 |
| text_encoder | `language_features` / `padding` / `embeds` | 4.1e-6 / 0 / 0 |
| decoder | `pred_logits` / `pred_masks` | 6.2e-5 / 2.0e-3（值域 ±32） |

端到端（提示框 = yolo26s 检出的一辆车）：`presence 0.9998`、best query 157（`score ≈ 0.964`）、
前景 11.04%，掩码与 ORT 的 **IoU = 1.00000**（概率图 `max|d|` 2.5e-5）。

### 两个「别照着直觉写」的坑

- **掩码坐标**：预处理是整图拉伸（不是 letterbox），所以 288² 掩码直接覆盖原图，放大按 `x * width / 288` 即可，
  不需要反 letterbox。
- **别用 `pred_boxes` 当框**：这一版导出里它和同 query 的掩码对不上（实测只有掩码 bbox 的 1/2~1/7），
  乘原图宽高会得到错框 —— **要框就从掩码算 bbox**。

### 性能：两档比 ORT 快，一档差 3 倍

16 线程、同会话交错测（机器有 ~50% 后台负载，绝对值波动可达 2×，只比值参考）：

| 模型 | OnnxSharp | onnxruntime | 比值 |
|---|---|---|---|
| image_encoder | 0.77–2.0 s | 0.44–1.4 s | 1.5–2.6× |
| text_encoder | 6.4–11 ms | 2.9–4.3 ms | 2.0–2.6× |
| decoder | 2.85–4.16 s | 2.30–3.71 s | 1.1–1.3× |

单算子微基准更能说明内核水平：

| 层 | OnnxSharp | onnxruntime |
|---|---:|---:|
| Conv 3×3 s1（256→256 @288²） | **162 ms** | 528 ms |
| ConvTranspose 2×（512→256） | **77 ms** | 108 ms |
| ConvTranspose 2×（1024→512） | 121 ms | **41 ms** |

第三行是已知短板：`m = 512` 时 GEMM 走行切分，`B` 要按行块重复打包，差 ORT 约 3 倍，是下一步最值得动的地方。

---

## 四、首层 `ic=3` 卷积：单开一个内核，外加一次方法学纠错

`3→16 640² k3s2` 这一层以前是全表唯一「我们 1T 比 MLAS 慢、16T 差 6 倍」的形状：
`K = 27`、`M = 16`，im2col 要为 102400 个像素物化 ~11 MB 的 col 缓冲（写一遍读一遍 = 22 MB 搬运），
面板式也省不掉。所以单开一个内核（`Kernels/FirstConv.cs`）：沿输出宽 16 像素 = 一个 `zmm`，4 通道 × 4 像素块
= 16 个累加器，一次 `vbroadcastss` 喂 4 条 FMA（1:4），面板 64 字节对齐常驻 L1，`stride=2` 用一条 `vpermt2ps`
抽偶数位。

| 线程 | im2col | **firstc** | 比值 |
|---|---:|---:|---:|
| 1 | 2.15 ms (20.6 GMAC/s) | **1.35 (32.8)** | 0.63 |
| 4 | 0.56 (78.3) | **0.34 (131.9)** | 0.61 |
| 16 | 0.23 (192.9) | **0.14 (314.2)** | 0.61 |

对拍 `max|d| = 0.0`（逐位一致）；16T 用到峰值算力的 72%（im2col 37%）。**但端到端判不出**：
yolo11n 22.1 vs 22.2 ms，配对中位 1.012，而当场噪声地板 31.8% —— 按账这层 4 线程省 0.22 ms / 22 ms ≈ 1%，
本来就在噪声地板以下。

**顺带修的是一个更值钱的错**：`convbench` 原来是「im2col 连测 150 次 → firstc 连测 150 次」。这台 APU 功耗受限，
firstc 用满 512 位 FMA、降频最狠，等它上机时 CPU 已经被 im2col 烤热，150 次全落在低频区 —— `min` 也取不到凉态样本，
**恰好把这个内核的读数砍半**（0.70 vs 真实 0.34 ms）。我据此先否掉了两种布局，结论全是假的。
改成交替各跑一次后读数才正确。同一条教训在 §19.5 / §33 已经吃过一次：**同进程比两个内核，必须交替计时。**

---

## 五、顺手捡的零钱：OCR 端到端 −29%

这一轮还把「Conv 之外那 15%」里最便宜的三块换掉了：`MaxPool` 的 2×2/s1 与 3×3/s2 滑窗内核、`Resize` 全路径行切分、
`GlobalAveragePool` 整平面快速路径；另外删掉了 rec 图里 58 个 `Identity`，并把**默认线程数从逻辑核改成物理核**
（32 → 16）。

最后那条是纠错不是提速：超线程对这些 FP 内核没有吞吐收益，多出来的 worker 只跟别的进程抢时间片。
效果在**尾巴**上：负载 35% 时 OCR 端到端中位数从 **184 ms 回到 35 ms**，而安静窗口下的 min 几乎不变。

| 段（4 线程 min / median） | 优化前 | 现在 |
|---|---:|---:|
| det 1×3×288×896 | 17.89 / 26.12 | **13.18 / 14.56** |
| cls 1×3×80×160 | 1.86 / 2.77 | **1.55 / 1.77** |
| rec 1×3×48×384 | 4.36 / 5.32 | **2.90 / 3.57** |
| 端到端（min） | 34.3 | **24.2**（−29%） |
| vs `Sdcb.SimdPaddleOCR` | 1.66× | **1.25×** |

yolo / rfdetr 吃不到这轮收益（回合在噪声内）：它们的池化只有 SPPF 三个 5×5、Resize 只有一次 2× upsample，
这几处合计不到 1 ms。受益的是池化/Resize 占比高的图 —— OCR det 与 YuNet。

---

## 六、数据口径（读性能表前请先看这段）

- 这台机器是 16C/32T 的 APU，常驻 ~50% 后台负载，**绝对值波动可达 2×**，只有同一轮内的相对关系可信。
- 比值只写到「image encoder 约 2×、decoder 约 1.2×」这个粒度；再精确就是噪声了。
- `profile` 的 sum-of-min 比 `bench` 的 min 大 ~20%（每轮 op 计时串行累加 + 计时开销），只能看排序与占比，两个口径不能互换。
- `convbench` / `ab` 这类同进程比较已全部改成**交替计时**（见第四节那个假结论）。
- 四引擎全表（yolo11n 30.4 / yolov8n 35.8 / yolov8s 90.2 / yolo11s 76.6 / rfdetr-small 198.0 ms，4 线程 median）
  与 ORT 的差距仍是 1.05~1.28×，本版没有改变这个格局。

---

## 七、升级到 0.5.0 要注意什么

- **目标框架**：`net8.0` / `net10.0` / `net11.0`（`net11.0` 才有 AVX-VNNI 的 int8 快路径，其余自动回退 F32）。
- **默认线程数变了**：逻辑核 → 物理核。重负载下延迟更稳，压满机器的吞吐会降；需要旧行为就设 `ONNXSHARP_THREADS`。
- **CLI 有改名**：单框入口 `--box` → **`--crop x,y,w,h`**，`--box` 恢复为 DB 框分数阈值语义。
- **新增 API**：`Detection.Name` / `Detection.Label`、`YoloDetector.ClassNames` / `NameOf` / `Metadata`、
  `YoloMetadata.Parse`、`OnnxSharp.Ocr.Plate.*`、`PlateReader.ReadWhole`、`plate` 子命令。
- **新算子**：`HardSwish`、`ScatterND`、`ScatterElements`（EV-M 与后续模型要用）。
- **回归**：`dotnet build OnnxSharp.slnx -c Release` 0 警告 0 错误；单测两套 TFM 全过（车牌那轮 122/124、
  EV-M 那轮 124/126，均为 2 skipped / 0 failed），并新增 `PlateOcrTests` 与广播段用例。

## 最后

0.5.0 的定位很清楚：**把「能跑」扩到更多真实模型（EV-M / yolo26 / 车牌），把「能读」扩到类别名，
把「能信」扩到逐个张量与 ORT 对齐**。车牌是这套能力的第一个业务出口，但它现在只走到了「规则层面不会瞎报」，
准确率的下一跳必须靠样本积累 + rec 微调 —— 那部分工作不在这个版本里，也不打算用调阈值糊过去。

- 仓库：[github.com/sevencat/OnnxSharp](https://github.com/sevencat/OnnxSharp)
- NuGet：[`Sevencat.OnnxSharp` 0.5.0](https://www.nuget.org/packages/Sevencat.OnnxSharp)
- 性能事实源与复现命令：[bench.md](https://github.com/sevencat/OnnxSharp/blob/master/bench.md)（§37~§41）
