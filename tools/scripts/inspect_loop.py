"""打印模型里第一个 Loop 节点的输入/输出/属性，帮助定位 encoder 段的真实输出张量。"""

import onnx
import sys


def main(path: str) -> None:
    model = onnx.load(path, load_external_data=False)
    graph = model.graph
    producer = {out: node for node in graph.node for out in node.output}
    infos = {v.name: v for v in list(graph.value_info) + list(graph.output) + list(graph.input)}

    def describe(name: str) -> str:
        node = producer.get(name)
        info = infos.get(name)
        shape = "-"
        if info is not None and info.type.HasField("tensor_type"):
            dims = [d.dim_value if d.HasField("dim_value") else (d.dim_param or "?")
                    for d in info.type.tensor_type.shape.dim]
            shape = str(dims)
        return f"{node.op_type if node else '-':12s} {shape}"

    loops = [n for n in graph.node if n.op_type == "Loop"]
    print(f"loops: {len(loops)}")
    target = loops[0]
    print(f"first loop: {target.name}")
    for index, name in enumerate(target.input):
        print(f"  in[{index:2d}] {name:60s} {describe(name)}")
    for index, name in enumerate(target.output):
        print(f"  out[{index:2d}] {name:60s} {describe(name)}")

    # 子图里引用了、但子图内部没有产出的张量 = 外层作用域捕获（encoder 的真实输出藏在里面）
    regions = [(a.name, a.g) for a in target.attribute if a.type == onnx.AttributeProto.GRAPH]
    for attr_name, body in regions:
        local = {out for node in body.node for out in node.output}
        local.update(i.name for i in body.input)
        local.update(i.name for i in body.initializer)
        captured = []
        for node in body.node:
            for name in node.input:
                if name and name not in local and name not in captured:
                    captured.append(name)
        print(f"  body[{attr_name}] nodes={len(body.node)} captured={len(captured)}")
        for name in captured:
            print(f"    capture {name:60s} {describe(name)}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "model.int8.onnx")
