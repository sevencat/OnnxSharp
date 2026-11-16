"""把 profile 的逐算子耗时与 conv 权重形状对齐，按 (kernel, stride) 家族聚合时间
（planv2 §P7.7 收益推算，结果见 bench.md §22.4）。

回答的问题：conv 的 31.8 ms 里，s2 / 1×1 / 3×3-s1 各占多少、各自跑多少 GMAC/s。
只看 `--top N` 的几层会误判——§19 就是因为只盯 s2，漏掉了 3×3/s1 的 13.4 ms。

用法: python bench/conv_family.py <model.onnx> <profile.log>
      （profile.log 由 `OnnxSharp.Cli profile <model> --top 300 > log` 生成，UTF-16/UTF-8 均可）
"""
import re
import sys
from collections import defaultdict

import onnx

path, plog = sys.argv[1], sys.argv[2]
g = onnx.load(path).graph

shapes = {}
for vi in list(g.value_info) + list(g.input) + list(g.output):
    t = vi.type.tensor_type
    shapes[vi.name] = [d.dim_value for d in t.shape.dim]
for i in g.initializer:
    shapes[i.name] = list(i.dims)

# parse profile lines: "#  Conv   1.96  2.00  5.4%  name"
raw = open(plog, "rb").read()
for enc in ("utf-16", "utf-8", "gbk"):
    try:
        text = raw.decode(enc)
        if "Conv" in text:
            break
    except Exception:
        continue
else:
    text = raw.decode("utf-8", "ignore")

timing = {}
for line in text.splitlines():
    m = re.match(r"\s*\d+\s+(\w+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)%\s+(\S+)\s*$", line)
    if m:
        timing[m.group(5)] = float(m.group(2))

fam = defaultdict(lambda: [0, 0.0, 0])  # macs, ms, count
missing = 0
rows = []
for n in g.node:
    if n.op_type != "Conv":
        continue
    w = shapes.get(n.input[1])
    osh = shapes.get(n.output[0], [])
    if not w or len(w) != 4 or len(osh) != 4:
        missing += 1
        continue
    oc, ick, kh, kw = w
    gr = 1
    s = 1
    for a in n.attribute:
        if a.name == "group":
            gr = a.i
        if a.name == "strides":
            s = a.ints[0]
    macs = osh[1] * osh[2] * osh[3] * (ick // gr) * kh * kw
    ms = timing.get(n.name)
    if ms is None:
        continue
    key = (kh, kw, s)
    fam[key][0] += macs
    fam[key][1] += ms
    fam[key][2] += 1
    rows.append((ms, macs, key, n.name))

tot_mac = sum(v[0] for v in fam.values())
tot_ms = sum(v[1] for v in fam.values())
print(f"matched convs: {len(rows)}  missing shape: {missing}")
print(f"total {tot_mac/1e6:.1f} MMAC in {tot_ms:.2f} ms -> {tot_mac/1e6/tot_ms:.1f} GMAC/s")
print()
print(f"{'family':<14}{'n':>4}{'MMAC':>9}{'ms':>8}{'share':>8}{'GMAC/s':>9}")
for k, (mm, ms, c) in sorted(fam.items(), key=lambda x: -x[1][1]):
    print(f"k={k[0]}x{k[1]} s={k[2]:<4}{c:>4}{mm/1e6:9.1f}{ms:8.2f}{ms/tot_ms*100:7.1f}%{mm/1e6/ms:9.1f}")

print()
print("--- what-if: raise each family to a target GMAC/s ---")
targets = {(3, 3, 2): 200, (1, 1, 1): 220, (3, 3, 1): 199}
for k, (mm, ms, c) in sorted(fam.items(), key=lambda x: -x[1][1]):
    cur = mm / 1e6 / ms
    tgt = targets.get(k, cur)
    new = mm / 1e6 / tgt
    print(f"k={k[0]}x{k[1]} s={k[2]}: {ms:5.2f} -> {new:5.2f} ms  save {ms-new:5.2f} ms "
          f"({(ms-new)/36.4*100:4.1f}% of 36.4 ms e2e)  [{cur:.0f} -> {tgt} GMAC/s]")

print()
print("--- s2 layers detail ---")
for ms, mm, k, name in sorted([r for r in rows if r[2][2] == 2], key=lambda x: -x[0]):
    print(f"  {ms:5.2f} ms  {mm/1e6:6.1f} MMAC  {mm/1e6/ms:5.1f} GMAC/s  {name}")
