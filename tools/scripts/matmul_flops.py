"""把 profile 的逐层耗时与 ONNX 的 MatMul 形状对齐，按类别算有效 GFLOPS。

用法：
    OnnxSharp.Cli profile <model> --input <f32> --iters 6 --top 200 > prof.txt
    python tools/scripts/matmul_flops.py prof.txt <model.onnx>
"""

import collections
import re
import sys

import onnx
from onnx import shape_inference

prof_path, model_path = sys.argv[1], sys.argv[2]

raw = open(prof_path, "rb").read()
text = raw.decode("utf-16") if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else raw.decode("utf-8", "replace")

times = {}
for line in text.splitlines():
    m = re.match(r"\s+\d+\s+(\w+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)%\s+(\S+)", line)
    if m:
        times[m.group(5)] = float(m.group(2))

model = shape_inference.infer_shapes(onnx.load(model_path, load_external_data=False))
graph = model.graph
shapes = {}
for vi in list(graph.value_info) + list(graph.input) + list(graph.output):
    shapes[vi.name] = [d.dim_value for d in vi.type.tensor_type.shape.dim]
inits = {i.name: list(i.dims) for i in graph.initializer}
prod = {o: n for n in graph.node for o in n.output}

rows = []
for node in graph.node:
    if node.op_type != "MatMul":
        continue
    a = shapes.get(node.input[0])
    b = inits.get(node.input[1]) or shapes.get(node.input[1])
    if not a or not b or len(a) < 2 or len(b) != 2:
        continue
    m, k = a[-2], a[-1]
    n = b[1]
    if k != b[0]:
        continue

    # 按节点名分类更可靠：rf-detr 的导出名里带 mlp / attn / query / key / value
    name = node.name.lower()
    if "mlp" in name:
        kind = "MLP"
    elif "attn" in name or "attention" in name:
        if m == n and k != m:
            kind = "attention QK^T"
        elif m == k and n != m:
            kind = "attention PV"
        else:
            kind = "attention 投影"
    elif m == n and k != m:
        kind = "方阵 QK^T (?)"
    elif k > 4 * max(1, min(m, n)):
        kind = "投影 (K 很大?)"
    else:
        kind = "投影 / 其他"

    flops = 2.0 * m * n * k
    ms = times.get(node.name)
    if ms is None:
        continue
    rows.append((kind, flops, ms, m, k, n, node.name))

total_ms = sum(r[2] for r in rows) or 1e-9
total_flops = sum(r[1] for r in rows)
print(f"MatMul {len(rows)} 个  合计 {total_ms:.1f} ms  {total_flops / 1e9:.1f} GFLOP  "
      f"整体 {total_flops / (total_ms / 1000) / 1e9:.0f} GFLOPS\n")

print(f"{'类别':22s} {'#':>3s} {'ms':>7s} {'GFLOP':>7s} {'GFLOPS':>7s} {'占比':>6s}")
per_class = collections.defaultdict(lambda: [0, 0.0, 0.0])
for kind, flops, ms, *_ in rows:
    acc = per_class[kind]
    acc[0] += 1
    acc[1] += ms
    acc[2] += flops

for kind, (count, ms, flops) in sorted(per_class.items(), key=lambda kv: -kv[1][1]):
    print(f"{kind:22s} {count:3d} {ms:7.2f} {flops / 1e9:7.2f} {flops / (ms / 1000) / 1e9:7.0f} "
          f"{ms / total_ms * 100:5.1f}%")

print("\n最慢的 12 个：")
print(f"{'ms':>7s} {'GFLOPS':>7s}  {'m':>5s} {'k':>5s} {'n':>5s}  类别 / 名字")
for kind, flops, ms, m, k, n, name in sorted(rows, key=lambda r: -r[2])[:12]:
    print(f"{ms:7.2f} {flops / (ms / 1000) / 1e9:7.0f}  {m:5d} {k:5d} {n:5d}  {kind:20s} {name}")
