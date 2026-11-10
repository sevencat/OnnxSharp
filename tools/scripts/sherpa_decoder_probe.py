"""体检 sherpa Paraformer 的 decoder 段：① 打印第 N 个 Loop 的 body；② 从 `logits` 反向 BFS，
找出「decoder 子图」的算子集合与它依赖的外部张量（CIF 输出 / 图输入）。

用法：python tools/scripts/sherpa_decoder_probe.py <model.onnx> [loopIndex=1]
"""

import collections
import sys

import onnx
from onnx import AttributeProto, numpy_helper


def attrs(node) -> str:
    parts = []
    for attribute in node.attribute:
        kind = attribute.type
        if kind == AttributeProto.INT:
            parts.append(f"{attribute.name}={attribute.i}")
        elif kind == AttributeProto.FLOAT:
            parts.append(f"{attribute.name}={attribute.f:.6g}")
        elif kind == AttributeProto.STRING:
            parts.append(f"{attribute.name}={attribute.s.decode()}")
        elif kind == AttributeProto.INTS:
            parts.append(f"{attribute.name}={list(attribute.ints)}")
        elif kind == AttributeProto.TENSOR:
            array = numpy_helper.to_array(attribute.t)
            parts.append(f"{attribute.name}=tensor{array.shape}")
    return " ".join(parts)


def dump_loop(model, index: int) -> None:
    loops = [n for n in model.graph.node if n.op_type == "Loop"]
    if index >= len(loops):
        print(f"no loop #{index} (only {len(loops)})")
        return

    loop = loops[index]
    body = next(a.g for a in loop.attribute if a.name == "body")
    print(f"=== loop #{index}: {loop.name}  (body {len(body.node)} nodes) ===")
    print(f"  inputs : {[i.name for i in body.input]}")
    print(f"  outputs: {[o.name for o in body.output]}")
    print(f"  carried: {[v[:40] for v in loop.input]}")
    for node in body.node:
        print(f"  {node.op_type:18s} {node.name[:34]:34s} "
              f"({', '.join(v[:24] for v in node.input)}) -> ({', '.join(v[:20] for v in node.output)}) {attrs(node)}")


def decoder_subgraph(model, target: str) -> None:
    nodes = model.graph.node
    producer = {out: node for node in nodes for out in node.output}
    initializers = {init.name for init in model.graph.initializer}
    graph_inputs = {i.name for i in model.graph.input}

    needed = {}
    frontier = set()
    stack = [target]
    seen = set()

    while stack:
        name = stack.pop()
        if name in seen:
            continue
        seen.add(name)

        node = producer.get(name)
        if node is None:
            if name not in initializers and name not in graph_inputs:
                frontier.add(name)
            continue

        needed[id(node)] = node
        for value in node.input:
            if value:
                stack.append(value)

    ordered = [n for n in nodes if id(n) in needed]
    histogram = collections.Counter(n.op_type for n in ordered)
    print(f"=== decoder subgraph ending at '{target}': {len(ordered)} nodes, {len(histogram)} op types ===")
    for op, count in histogram.most_common():
        print(f"  {op:22s} {count}")
    print(f"  frontier (外部依赖, {len(frontier)}): {sorted(v[:48] for v in frontier)[:12]}")


def main(path: str, loop_index: int) -> None:
    model = onnx.load(path, load_external_data=False)
    outputs = [o.name for o in model.graph.output]
    print(f"graph outputs: {outputs}")

    dump_loop(model, loop_index)
    print()
    for name in outputs:
        if name == "logits":
            decoder_subgraph(model, name)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(2)
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 1)
