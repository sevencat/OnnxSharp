"""打印模型里第一个 Loop 的 body（节点、属性、输入输出），用于把 CIF 手工移植成 C#。

用法：python tools/scripts/dump_loop_body.py <model.onnx>
"""

import sys

import onnx
from onnx import AttributeProto, numpy_helper


def attrs(node) -> str:
    parts = []
    for attribute in node.attribute:
        if attribute.type == AttributeProto.INT:
            parts.append(f"{attribute.name}={attribute.i}")
        elif attribute.type == AttributeProto.FLOAT:
            parts.append(f"{attribute.name}={attribute.f:.6g}")
        elif attribute.type == AttributeProto.STRING:
            parts.append(f"{attribute.name}={attribute.s.decode()}")
        elif attribute.type == AttributeProto.INTS:
            parts.append(f"{attribute.name}={list(attribute.ints)}")
        elif attribute.type == AttributeProto.TENSOR:
            array = numpy_helper.to_array(attribute.t)
            parts.append(f"{attribute.name}=tensor{array.shape}{array.flatten()[:4]}")
        else:
            parts.append(f"{attribute.name}=<{attribute.type}>")
    return " ".join(parts)


def main(path: str) -> None:
    model = onnx.load(path, load_external_data=False)
    loops = [n for n in model.graph.node if n.op_type == "Loop"]
    body = next(a.g for a in loops[0].attribute if a.name == "body")

    print(f"=== body inputs ({len(body.input)}) ===")
    for index, value in enumerate(body.input):
        print(f"  in[{index}] {value.name}")
    print(f"=== body outputs ({len(body.output)}) ===")
    for index, value in enumerate(body.output):
        print(f"  out[{index}] {value.name}")
    print(f"=== body initializers ({len(body.initializer)}) ===")
    for init in body.initializer:
        array = numpy_helper.to_array(init)
        print(f"  {init.name}: {array.dtype}{array.shape} = {array.flatten()[:6]}")

    print(f"=== nodes ({len(body.node)}) ===")
    for index, node in enumerate(body.node):
        a = attrs(node)
        print(f"{index:3d} {node.op_type:16s} {node.name[:44]:44s} "
              f"({', '.join(node.input)}) -> ({', '.join(node.output)}) {a}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "model.int8.onnx")
