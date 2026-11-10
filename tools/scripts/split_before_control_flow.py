"""把带控制流的量化模型在第一个 Loop / If 之前切开，导出「encoder 段」子图。

用途：验证 OnnxSharp 能否跑 sherpa Paraformer 的 encoder（int8 量化图）。
注意：sherpa 的导出**不是拓扑序**（ORT 容忍），所以这里自己做一次拓扑排序，
否则引擎的 importer 会因为「生产者还没定义」直接报错。

用法：
    python tools/scripts/split_before_control_flow.py <in.onnx> <out.onnx>
"""

import os
import sys
from collections import defaultdict, deque

import onnx
from onnx import helper

SEQUENCE_OPS = {
    "SequenceEmpty",
    "SequenceInsert",
    "SequenceAt",
    "SequenceConstruct",
    "ConcatFromSequence",
    "SplitToSequence",
    "SequenceLength",
    "SequenceErase",
}


def topological_sort(nodes):
    producers = {out: node for node in nodes for out in node.output}
    index = {id(node): node for node in nodes}
    indegree = {id(node): 0 for node in nodes}
    successors = defaultdict(list)

    for node in nodes:
        for value in node.input:
            producer = producers.get(value)
            if producer is None or producer is node:
                continue
            successors[id(producer)].append(node)
            indegree[id(node)] += 1

    queue = deque(node for node in nodes if indegree[id(node)] == 0)
    ordered = []
    while queue:
        node = queue.popleft()
        ordered.append(node)
        for successor in successors[id(node)]:
            indegree[id(successor)] -= 1
            if indegree[id(successor)] == 0:
                queue.append(successor)

    if len(ordered) != len(nodes):
        print(f"  warning: cycle detected, kept {len(ordered)}/{len(nodes)} nodes")
    return ordered


def split(path_in: str, path_out: str, explicit_outputs: list[str] | None = None) -> None:
    model = onnx.load(path_in, load_external_data=True)
    graph = model.graph

    control = [n for n in graph.node if n.op_type in ("Loop", "If", "Scan")]
    if not control:
        print("no control-flow node found; nothing to split")
        return

    target = control[0]
    producer = {out: node for node in graph.node for out in node.output}
    initializers = {init.name: init for init in graph.initializer}
    graph_inputs = {info.name: info for info in graph.input}
    value_infos = {info.name: info for info in graph.value_info}

    def is_sequence(name: str) -> bool:
        node = producer.get(name)
        return node is not None and node.op_type in SEQUENCE_OPS

    boundary = [name for name in target.input if name and not is_sequence(name)]
    if explicit_outputs:
        boundary = list(dict.fromkeys(boundary + explicit_outputs))

    def is_constant_like(name: str) -> bool:
        if name.startswith("onnx::"):
            return True
        node = producer.get(name)
        return node is not None and node.op_type == "Constant"

    # 反向 BFS：只保留产出 boundary 所需的节点
    kept = {}
    stack = list(boundary)
    seen = set()
    while stack:
        name = stack.pop()
        if name in seen:
            continue
        seen.add(name)

        node = producer.get(name)
        if node is None:
            continue
        kept[id(node)] = node
        for value in node.input:
            if value:
                stack.append(value)

    ordered = topological_sort(list(kept.values()))
    produced = {out for node in ordered for out in node.output}
    outputs = [name for name in boundary if name in produced and not is_constant_like(name)]
    used = {value for node in ordered for value in node.input}
    used_initializers = [initializers[name] for name in initializers if name in used]
    inputs = [info for name, info in graph_inputs.items() if name in used]

    output_infos = []
    for name in outputs:
        if name in value_infos:
            output_infos.append(value_infos[name])
        else:
            shape = None
            for node in ordered:
                if name in node.output and "Constant" in node.op_type:
                    shape = None
            output_infos.append(onnx.ValueInfoProto(name=name))

    new_graph = helper.make_graph(ordered, "encoder", inputs, output_infos, used_initializers)
    new_model = helper.make_model(new_graph, opset_imports=model.opset_import, producer_name="onnxsharp-split")
    new_model.ir_version = 10
    # 保留原模型的 metadata（lfr_window_size / neg_mean / inv_stddev 等前端参数都在里面）
    del new_model.metadata_props[:]
    new_model.metadata_props.extend(model.metadata_props)

    # 用 shape_inference 补齐缺类型的新输出（checker 不跑：原图本身就不满足拓扑序检查）
    try:
        new_model = onnx.shape_inference.infer_shapes(new_model, strict_mode=False)
    except Exception as error:                     # noqa: BLE001
        print(f"  shape inference skipped: {error}")

    os.makedirs(os.path.dirname(os.path.abspath(path_out)), exist_ok=True)
    onnx.save(new_model, path_out)

    print(f"cut at {target.op_type} '{target.name}'")
    print(f"  boundary: {boundary}")
    print(f"  nodes {len(graph.node)} -> {len(ordered)}, initializers {len(graph.initializer)} -> {len(used_initializers)}")
    print(f"  inputs: {[i.name for i in inputs]}")
    print(f"  outputs: {outputs}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        raise SystemExit(2)

    extra = sys.argv[3].split(",") if len(sys.argv) > 3 else None
    split(sys.argv[1], sys.argv[2], extra)
