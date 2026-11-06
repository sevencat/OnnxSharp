"""用 onnx.shape_inference 生成形状 oracle，供 C# 侧逐 Arg 比对（plan §8.1）。

输出格式（TSV，一行一个张量，保持拓扑序）：

    <node_index> <node_name> <op_type> <tensor_name> <onnx_dtype> <dim0,dim1,...>

动态维输出为 -1。节点行与新增张量行交替出现，便于定位第一个偏离点。

用法：
    python tools/scripts/dump_shapes.py F:\ai\models\yolo\yolo11n.onnx
    python tools/scripts/dump_shapes.py ...onnx --out tests/data/yolo11n.shapes.tsv
"""

import argparse
import sys
from pathlib import Path

import onnx


def dims_of(tensor_type):
    shape = tensor_type.shape
    dims = []
    for dim in shape.dim:
        if dim.HasField("dim_value"):
            dims.append(dim.dim_value)
        else:
            # dim_param 或完全未知
            dims.append(-1)
    return dims


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    model_path = Path(args.model)
    model = onnx.load(str(model_path), load_external_data=False)
    model = onnx.shape_inference.infer_shapes(model, strict_mode=False, check_type=False)
    graph = model.graph

    value_info = {}
    for vi in list(graph.value_info) + list(graph.input) + list(graph.output):
        value_info[vi.name] = vi

    for init in graph.initializer:
        if init.name not in value_info:
            value_info[init.name] = None  # initializer 用自身信息

    lines = []
    unknown = 0

    def describe(name):
        nonlocal unknown
        vi = value_info.get(name)
        if vi is None:
            if name in initializer_map:
                init = initializer_map[name]
                return init.data_type, list(init.dims)
            unknown += 1
            return -1, []
        tensor_type = vi.type.tensor_type
        return tensor_type.elem_type, dims_of(tensor_type)

    initializer_map = {init.name: init for init in graph.initializer}

    for index, node in enumerate(graph.node):
        lines.append(f"{index}\tNODE\t{node.op_type}\t{node.name or '-'}\t-\t-")
        for tensor_name in node.output:
            if not tensor_name:
                continue
            dtype, dims = describe(tensor_name)
            dim_text = ",".join(str(d) for d in dims) if dims else ""
            lines.append(f"{index}\tTENSOR\t{node.op_type}\t{tensor_name}\t{dtype}\t{dim_text}")

    for vi in graph.input:
        dtype = vi.type.tensor_type.elem_type
        dims = dims_of(vi.type.tensor_type)
        lines.append(f"-1\tGRAPH_INPUT\t-\t{vi.name}\t{dtype}\t{','.join(str(d) for d in dims)}")

    for vi in graph.output:
        dtype = vi.type.tensor_type.elem_type
        dims = dims_of(vi.type.tensor_type)
        lines.append(f"-1\tGRAPH_OUTPUT\t-\t{vi.name}\t{dtype}\t{','.join(str(d) for d in dims)}")

    text = "\n".join(lines) + "\n"
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
        print(f"wrote {len(lines)} lines to {out_path} (unknown tensors: {unknown})")
    else:
        sys.stdout.write(text)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
