"""局部 NCHWc 布局传播探针（planv2 §P7.7 前置问题 1/2，结果见 bench.md §22）。

用途：给定 onnx，算出「若只把 conv 链做成 c 布局，需要几处 reorder、共多少字节」，
以及给定块宽下 oc/ic 的填充开销。回答「reorder 成本能不能接受」和「块宽选 16 还是 32」。

模型：某张量记为 c-layout，当且仅当它的生产者/消费者都落在 c 支持集
（Conv / 逐元素 / Concat·Split·Slice（通道维且对齐块宽）/ MaxPool·Resize·Pad）。
一条「c → plain」的边就需要一次 reorder（读 NCHW + 写 NCHWc = 2× 字节）。

用法: python bench/layout_probe.py <model.onnx> [blockWidth=16]
"""
import sys
from collections import Counter, defaultdict, deque

import onnx

path = sys.argv[1] if len(sys.argv) > 1 else "F:/ai/models/yolo/yolo11n.onnx"
BW = int(sys.argv[2]) if len(sys.argv) > 2 else 16

m = onnx.load(path)
g = m.graph

shapes = {}
for vi in list(g.value_info) + list(g.input) + list(g.output):
    t = vi.type.tensor_type
    shapes[vi.name] = [d.dim_value for d in t.shape.dim]
for i in g.initializer:
    shapes[i.name] = list(i.dims)

inits = {i.name for i in g.initializer}
nodes = list(g.node)

# consumer map: tensor -> [node]
cons = defaultdict(list)
for n in nodes:
    for x in n.input:
        if x and x not in inits:
            cons[x].append(n)

OPS = Counter(n.op_type for n in nodes)

# elementwise / layout-neutral: work per element, independent of channel blocking
NEUTRAL = {"Mul", "Add", "Sub", "Div", "Sigmoid", "Relu", "Clip", "BatchNormalization",
           "HardSigmoid", "HardSwish", "Exp", "Pow"}
# channel-dim ops: can keep c-layout if the split points align to the block width
CHANNEL = {"Concat", "Split", "Slice"}
# spatial ops: need a dedicated c-layout kernel (per-block reuse is still correct)
SPATIAL = {"MaxPool", "AveragePool", "Resize", "Upsample", "Pad"}
# hard boundaries: semantics depend on contiguous NCHW
HARD = {"Transpose", "Reshape", "MatMul", "Softmax", "Gemm", "Gather", "GatherElements",
        "TopK", "ArgMax", "Expand", "Tile", "ReduceSum", "ReduceMax", "Sin", "Cos"}


def nbytes(name):
    s = shapes.get(name)
    if not s or any(d <= 0 for d in s):
        return 0
    v = 1
    for d in s:
        v *= d
    return v * 4


def align_ok(node):
    """Channel-dim op: do its split/slice points stay on block boundaries?"""
    if node.op_type == "Split":
        axis = 1
        for a in node.attribute:
            if a.name == "axis":
                axis = a.i
        if axis not in (1, -3):
            return False
        return (shapes.get(node.input[0], [0, 0])[1] % BW) == 0
    if node.op_type == "Slice":
        return False  # conservative: assume not aligned
    if node.op_type == "Concat":
        for a in node.attribute:
            if a.name == "axis" and a.i not in (1, -3):
                return False
        for x in node.input:
            s = shapes.get(x)
            if s and len(s) == 4 and s[1] % BW:
                return False
        return True
    return True


# ---- forward propagation: which tensors can stay in c-layout -----------------
# seed: every Conv output is c-layout
cl = set()
work = deque()
for n in nodes:
    if n.op_type == "Conv":
        for o in n.output:
            if o and o not in cl:
                cl.add(o)
                work.append(o)

while work:
    t = work.popleft()
    for c in cons[t]:
        if any(x not in cl for x in c.input if x and x not in inits):
            continue  # not all inputs are c yet
        if c.op_type == "Conv":
            # a Conv can consume/emit c-layout; its other inputs are weights
            pass
        elif c.op_type in NEUTRAL:
            pass
        elif c.op_type in CHANNEL:
            if not align_ok(c):
                continue
        elif c.op_type in SPATIAL:
            pass
        else:
            continue
        for o in c.output:
            if o and o not in cl:
                cl.add(o)
                work.append(o)

# ---- reorder edges: tensor read by a consumer that is not in the c set -------
need = {}
for t in list(cl):
    for c in cons[t]:
        if c.output and c.output[0] in cl:
            continue
        if c.op_type in HARD or (c.op_type in CHANNEL and not align_ok(c)):
            need[t] = c

print(f"model={path}  nodes={len(nodes)} convs={OPS['Conv']} blockWidth={BW}")
print()
print("op histogram:")
for k, v in OPS.most_common():
    tag = "neutral" if k in NEUTRAL else "channel" if k in CHANNEL else \
          "spatial" if k in SPATIAL else "HARD" if k in HARD else "conv" if k == "Conv" else "other"
    print(f"  {v:4d}  {k:<18} {tag}")
print()
print(f"tensors in c-layout: {len(cl)}")
print(f"reorder edges (c -> plain): {len(need)}")
tot = 0
by_op = Counter()
for t, c in need.items():
    b = nbytes(t)
    tot += b
    by_op[c.op_type] += 1
    print(f"  {b/1024:9.1f} KB  {shapes.get(t)}  -> {c.op_type}  {c.name[:50]}")
print(f"reorder traffic: {tot/1024:.1f} KB one-way, {2*tot/1024:.1f} KB with write-back")

# 32 GB/s calibrated from bench.md §P4 (111 transposes = 11 ms on ~1.6 MB tensors)
print(f"estimated reorder cost @32 GB/s: {2*tot/32e9*1e3:.3f} ms per forward")
for k, v in by_op.items():
    print(f"    {k}: {v}")

# ---- block-width utilisation -------------------------------------------------
print()
print(f"--- oc/ic utilisation at blockWidth={BW} ---")
waste = 0.0
wcount = 0
macs = 0
fam = defaultdict(lambda: [0, 0])
for n in nodes:
    if n.op_type != "Conv":
        continue
    w = shapes.get(n.input[1])
    if not w or len(w) != 4:
        continue
    oc, ick, kh, kw = w
    gr = 1
    s = 1
    for a in n.attribute:
        if a.name == "group":
            gr = a.i
        if a.name == "strides":
            s = a.ints[0]
    ish = shapes.get(n.input[0], [0, 0, 0, 0])
    osh = shapes.get(n.output[0], [0, 0, 0, 0])
    icg = ick // gr
    if oc <= 0 or icg <= 0:
        continue
    m = osh[1] * osh[2] * osh[3] * icg * kh * kw if len(osh) == 4 else 0
    macs += m
    pb_oc = (oc + BW - 1) // BW * BW
    pb_ic = (icg + BW - 1) // BW * BW
    waste += (pb_oc * pb_ic) / (oc * icg) - 1
    wcount += 1
    fam[(tuple(w[2:]), s)][0] += 1
    fam[(tuple(w[2:]), s)][1] += m
print(f"total MACs: {macs/1e6:.1f} MMAC over {wcount} convs")
print(f"avg padding overhead (per conv): {waste/wcount*100:.2f}%")

mac_pad = 0
for n in nodes:
    if n.op_type != "Conv":
        continue
    w = shapes.get(n.input[1])
    osh = shapes.get(n.output[0], [])
    if not w or len(w) != 4 or len(osh) != 4:
        continue
    oc, ick, kh, kw = w
    gr = 1
    for a in n.attribute:
        if a.name == "group":
            gr = a.i
    icg = ick // gr
    if oc <= 0 or icg <= 0:
        continue
    pb_oc = (oc + BW - 1) // BW * BW
    pb_ic = (icg + BW - 1) // BW * BW
    mac_pad += (pb_oc * pb_ic - oc * icg) * osh[1] * osh[2] * osh[3] * kh * kw
print(f"MAC-weighted padding overhead: {mac_pad/macs*100:.2f}%")
print(f"oc % {BW} distribution: {dict(Counter((shapes.get(n.input[1]) or [0])[0] % BW for n in nodes if n.op_type=='Conv'))}")
print()
print("family (kernel,stride): count / MMAC / share")
for k, (c, mm) in sorted(fam.items(), key=lambda x: -x[1][1]):
    print(f"  k={k[0]} s={k[1]}: {c:3d} convs  {mm/1e6:7.1f} MMAC  {mm/macs*100:5.1f}%")
