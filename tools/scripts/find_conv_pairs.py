"""统计 yolo 图里「Conv1×1 → SiLU → Conv3×3」可融合的链，以及能省多少 DRAM 往返。"""

import collections
import sys

import onnx
from onnx import shape_inference

model = sys.argv[1] if len(sys.argv) > 1 else r"F:\ai\models\yolo\yolo11n.onnx"
m = shape_inference.infer_shapes(onnx.load(model, load_external_data=False))
g = m.graph
sh = {vi.name: [d.dim_value for d in vi.type.tensor_type.shape.dim]
      for vi in list(g.value_info) + list(g.output)}
inits = {i.name: list(i.dims) for i in g.initializer}
prod = {o: n for n in g.node for o in n.output}
cons = collections.defaultdict(list)
for n in g.node:
    for v in n.input:
        cons[v].append(n)


def single(name, op):
    """name 的唯一消费者，且是 op 类型。"""
    cs = cons.get(name, [])
    if len(cs) != 1 or cs[0].op_type != op:
        return None
    return cs[0]


rows = []
for conv in g.node:
    if conv.op_type != "Conv":
        continue
    w1 = inits.get(conv.input[1])
    out = sh.get(conv.output[0])
    if not w1 or not out or len(out) != 4:
        continue

    # Conv → Sigmoid → Mul → Conv  （SiLU 在 ONNX 里没有算子：Mul 同时吃 conv 输出和 Sigmoid 输出）
    consumers = cons.get(conv.output[0], [])
    sigmoid = next((n for n in consumers if n.op_type == "Sigmoid"), None)
    mul = next((n for n in consumers if n.op_type == "Mul"), None)
    if sigmoid is None or mul is None:
        continue
    if sigmoid.output[0] not in mul.input or conv.output[0] not in mul.input:
        continue

    nxt = single(mul.output[0], "Conv")
    if nxt is None:
        continue

    w2 = inits.get(nxt.input[1])
    if not w2:
        continue

    megabytes = out[1] * out[2] * out[3] * 4 / 1048576.0
    rows.append((megabytes, (w1[2], w1[3]), (w2[2], w2[3]), nxt.name))

rows.sort(key=lambda r: -r[0])
total = sum(r[0] for r in rows)
print(f"可融合链 {len(rows)} 条，中间张量合计 {total:.1f} MB/帧 "
      f"→ 省掉一写一读 ≈ {total * 2:.1f} MB/帧")
print(f"{'MB':>7s}  kernel1  kernel2  下游")
for megabytes, k1, k2, name in rows[:12]:
    print(f"{megabytes:7.2f}  {str(k1):8s} {str(k2):8s} {name}")
