"""把 profile 的逐层耗时和 ONNX 里的卷积形状对起来，按「卷积类别」算有效 GFLOPS。

用法：
    OnnxSharp.Cli profile <model> --input <f32> --iters 8 --top 100 > prof.txt
    python tools/scripts/conv_flops.py prof.txt <model.onnx>
"""

import collections
import re
import sys

import onnx
from onnx import shape_inference

prof_path, model_path = sys.argv[1], sys.argv[2]

# PowerShell 的 `*>` 重定向默认写 UTF-16LE，这里按 BOM 判断，避免读成乱码
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


def classify(kh, kw, sh, sw, groups, cin, cout):
    if groups > 1:
        return f"depthwise {kh}x{kw} s{sh}" if groups == cin else f"grouped({groups}) {kh}x{kw} s{sh}"
    if kh == 1 and kw == 1:
        return f"1x1 s{sh}"
    if kh == 3 and kw == 3:
        return f"3x3 s{sh}"
    return f"{kh}x{kw} s{sh}"


rows = []
for node in graph.node:
    if node.op_type != "Conv":
        continue
    out_name = node.output[0]
    out = shapes.get(out_name)
    weight = inits.get(node.input[1])
    if not out or not weight or len(out) != 4:
        continue

    attrs = {a.name: a for a in node.attribute}
    groups = attrs["group"].i if "group" in attrs else 1
    strides = list(attrs["strides"].ints) if "strides" in attrs else [1, 1]
    cin, cout = weight[1] * groups, weight[0]
    kh, kw = weight[2], weight[3]
    n, _, oh, ow = out

    flops = 2.0 * n * cout * oh * ow * cin * kh * kw
    ms = times.get(node.name)
    if ms is None:
        continue
    rows.append((node.name, classify(kh, kw, strides[0], strides[1], groups, cin, cout),
                 flops, ms, cin, cout, oh * ow))

total_ms = sum(r[3] for r in rows)
total_flops = sum(r[2] for r in rows)
print(f"conv 总数 {len(rows)}  合计 {total_ms:.1f} ms  合计 {total_flops / 1e9:.2f} GFLOP  "
      f"整体 {total_flops / (total_ms / 1000) / 1e9:.0f} GFLOPS\n")

print(f"{'类别':22s} {'#':>3s} {'ms':>7s} {'GFLOP':>7s} {'GFLOPS':>7s} {'占比':>6s}")
per_class = collections.defaultdict(lambda: [0, 0.0, 0.0])
for _, cls, flops, ms, *_ in rows:
    acc = per_class[cls]
    acc[0] += 1
    acc[1] += ms
    acc[2] += flops

for cls, (count, ms, flops) in sorted(per_class.items(), key=lambda kv: -kv[1][1]):
    print(f"{cls:22s} {count:3d} {ms:7.2f} {flops / 1e9:7.2f} {flops / (ms / 1000) / 1e9:7.0f} "
          f"{ms / total_ms * 100:5.1f}%")

print("\n最慢的 8 个卷积：")
print(f"{'ms':>7s} {'GFLOPS':>7s}  {'Cin':>4s} {'Cout':>5s} {'HxW':>7s}  类别 / 名字")
for name, cls, flops, ms, cin, cout, spatial in sorted(rows, key=lambda r: -r[3])[:8]:
    print(f"{ms:7.2f} {flops / (ms / 1000) / 1e9:7.0f}  {cin:4d} {cout:5d} {spatial:7d}  {cls:16s} {name}")
