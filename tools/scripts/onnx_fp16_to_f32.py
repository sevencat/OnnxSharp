"""把 FP16 导出的 ONNX 改写成 FP32（纯 CPU 引擎用）。

为什么需要：SAM3（`wkentaro/sam3-onnx`）的四个模型都是 FP16 导出 —— 权重是 F16，
图内还有 `Cast → F16 ... Cast → F32` 的边界。OnnxSharp 是 FP32-only 引擎，
`Cast` 连 F16 源都不支持，所以直接导入会报 `Cast: unsupported source dtype F16`。

这个脚本做两件等价变换（不改变数值语义，只会更准）：

1. 所有 F16 的 initializer / Constant 张量 → F32；
2. 只用于 FP16 边界的 `Cast` 节点 → `Identity`：
   - `to = FLOAT16`（进入 FP16 计算域）
   - `to = FLOAT` 且输入本来就是 F16（离开 FP16 计算域）
   其它 Cast（例如 `int64 → bool`、`int64 → float`）原样保留。

权重写成同目录的外部数据文件（`weights.bin`），因为 FP32 化之后模型通常超过 protobuf 的 2 GB 限制；
OnnxSharp 支持外部数据（`data_location=EXTERNAL`，带路径逃逸校验）。

用法：
    python tools/scripts/onnx_fp16_to_f32.py <in.onnx> <out.onnx>

实测（2026-11-10，F:\\ai\\models\\sam3）：
    vision-encoder.onnx   543 个 F16 张量 → 1.84 GB 外部权重；改完可导入并跑通前向
    text-encoder.onnx     396 个 F16 张量
    geometry-encoder.onnx 101 个 F16 张量
    decoder.onnx          460 个 F16 张量
"""

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, numpy_helper
from onnx.external_data_helper import convert_model_to_external_data


def _to_f32(tensor: onnx.TensorProto) -> onnx.TensorProto:
    array = numpy_helper.to_array(tensor).astype(np.float32)
    return numpy_helper.from_array(array, tensor.name)


def convert(path_in: str, path_out: str, weights: str | None = None) -> None:
    # 每个模型用**独立**的外部权重文件：onnx.save 是 append 写外部数据，
    # 同名文件会让第二、三个模型读到第一段数据（offset 从 0 重新算）。
    if weights is None:
        weights = Path(path_out).stem + ".weights.bin"
    model = onnx.load(path_in, load_external_data=True)
    graph = model.graph

    converted = 0

    # 1) initializer
    initializers = []
    for tensor in graph.initializer:
        if tensor.data_type == TensorProto.FLOAT16:
            initializers.append(_to_f32(tensor))
            converted += 1
        else:
            initializers.append(tensor)
    graph.ClearField("initializer")
    graph.initializer.extend(initializers)

    # 2) 任意节点上的 TENSOR 属性（Constant 的 value、ConstantOfShape 的 value 等）
    for node in graph.node:
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.TENSOR and attribute.t.data_type == TensorProto.FLOAT16:
                attribute.t.CopyFrom(_to_f32(attribute.t))
                converted += 1

    # 3) Cast → Identity（只去掉 FP16 边界那一层）
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=False)
    types = {}
    for collection in (inferred.graph.input, inferred.graph.value_info, inferred.graph.output):
        for value_info in collection:
            types[value_info.name] = value_info.type.tensor_type.elem_type

    stripped = 0
    for node in graph.node:
        if node.op_type != "Cast" or not node.input:
            continue
        target = next((a.i for a in node.attribute if a.name == "to"), None)
        source = types.get(node.input[0])
        if target == TensorProto.FLOAT16 or (target == TensorProto.FLOAT and source == TensorProto.FLOAT16):
            node.op_type = "Identity"
            for attribute in list(node.attribute):
                node.attribute.remove(attribute)
            stripped += 1

    onnx.checker.check_model(model)

    # 4) value_info / 输入输出里仍然写着 F16 的条目改成 F32
    #    （ORT 会按 value_info 做类型校验，残留的 F16 会让模型直接加载失败）
    for collection in (graph.value_info, graph.input, graph.output):
        for value_info in collection:
            tensor_type = value_info.type.tensor_type
            if tensor_type.elem_type == TensorProto.FLOAT16:
                tensor_type.elem_type = TensorProto.FLOAT

    # 外部权重文件必须从零开始写：onnx.save 是 append，重复转换会让文件翻倍（offset 又从 0 算）。
    weights_path = Path(path_out).parent / weights
    if weights_path.exists():
        weights_path.unlink()

    convert_model_to_external_data(
        model,
        all_tensors_to_one_file=True,
        location=weights,
        size_threshold=1024,
        convert_attribute=True,
    )
    onnx.save(model, path_out)

    print(f"{path_in} -> {path_out}: f16 tensors {converted}, cast->identity {stripped}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        raise SystemExit(2)
    convert(sys.argv[1], sys.argv[2])
