"""打印 encoder 图里一条 `DQL → MatMulInteger → ...` 链的真实样子（用于对齐融合匹配条件）。"""

import collections
import sys

import onnx

m = onnx.load(sys.argv[1] if len(sys.argv) > 1 else "encoder.onnx", load_external_data=False)
g = m.graph
prod = {o: n for n in g.node for o in n.output}
cons = collections.defaultdict(list)
for n in g.node:
    for v in n.input:
        cons[v].append(n)


def show(name, depth=0, seen=None):
    seen = seen or set()
    if name in seen:
        return
    seen.add(name)
    node = prod.get(name)
    if node is None:
        print("  " * depth + f"{name[:60]}  <- (initializer/input)")
        return
    print("  " * depth + f"{node.op_type:22s} {node.name[:52]:52s} out={node.output[0][:40]}")
    for value in node.input:
        if value:
            show(value, depth + 1, seen)


# 找第一个 A 由 DQL 产出的 MatMulInteger
for mm in g.node:
    if mm.op_type != "MatMulInteger":
        continue
    src = prod.get(mm.input[0])
    if src is None or src.op_type != "DynamicQuantizeLinear":
        continue

    print("=== MatMulInteger ===")
    print("  inputs :", [v[:70] for v in mm.input])
    print("  outputs:", mm.output)
    print("=== A 的 DQL ===")
    print("  inputs :", [v[:60] for v in src.input])
    print("  outputs:", [v[:60] for v in src.output])

    print("=== MatMulInteger 输出的消费者链 ===")
    frontier = [mm.output[0]]
    for _ in range(4):
        nxt = []
        for value in frontier:
            for node in cons.get(value, []):
                print(f"  {node.op_type:14s} {node.name[:50]:50s} in={[x[:36] for x in node.input]}")
                nxt.extend(node.output)
        frontier = nxt

    print("=== DQL scale 的消费者 ===")
    for node in cons.get(src.output[1], []):
        print(f"  {node.op_type:14s} {node.name[:50]:50s} in={[x[:40] for x in node.input]}")
    break
