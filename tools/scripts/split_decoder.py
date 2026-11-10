"""切出 Paraformer 的 **decoder 段**：去掉两个 `Loop` / `Sequence*`，把 loop 的输出提升成图输入，
只保留从 `logits`/`token_num` 可达的节点（编码器仍保留在图里，先把链路验证通，之后再切深）。

用法：python tools/scripts/split_decoder.py <model.onnx> <out.onnx>
产出模型输入：speech, speech_lengths, memory(onnx::Slice_6785), max_label_len(max_label_len.28)
"""

import collections
import sys

import onnx
from onnx import TensorProto, helper

# 只保留真正要用的输出：`token_num` 由 C# 的 CIF 给出，若保留它会把 alphas/encoder 支路整个拉回来
KEEP_OUTPUTS = ["logits"]

CONTROL_OPS = {"Loop", "If", "Scan", "SequenceEmpty", "SequenceInsert", "SequenceAt",
               "SequenceConstruct", "ConcatFromSequence", "SplitToSequence", "SequenceLength"}

# 提升成图输入的张量：名字 → (ONNX 类型, 形状)。
#   max_label_len.28 / onnx::Slice_6785 / token_num 来自 CIF 那个 Loop（由 C# 的 CIF 提供）；
#   后面三个是 **encoder 的量化输出**（decoder 的 cross-attention K/V 用它），
#   由 encoder 图直接吐出来，这样 decoder 图里就不用再跑一遍 encoder。
PROMOTED = {
    "max_label_len.28": (TensorProto.INT64, []),
    "onnx::Slice_6785": (TensorProto.FLOAT, [1, "max_len", 320]),
    "token_num": (TensorProto.INT32, [1]),
    "/encoder/after_norm/Add_1_output_0_quantized": (TensorProto.UINT8, [1, "enc_len", 320]),
    "/encoder/after_norm/Add_1_output_0_scale": (TensorProto.FLOAT, []),
    "/encoder/after_norm/Add_1_output_0_zero_point": (TensorProto.UINT8, []),
}

# 这些名字不一定出现在 Loop 的输出里，但同样要提升成图输入
EXTRA_INPUTS = [
    "token_num",
    "/encoder/after_norm/Add_1_output_0_quantized",
    "/encoder/after_norm/Add_1_output_0_scale",
    "/encoder/after_norm/Add_1_output_0_zero_point",
]


def main(path_in: str, path_out: str) -> None:
    model = onnx.load(path_in, load_external_data=False)
    graph = model.graph

    loops = [n for n in graph.node if n.op_type == "Loop"]
    if len(loops) != 2:
        raise SystemExit(f"expected 2 Loop nodes, found {len(loops)}")

    # CIF loop 的输出 → 输入（memory / max_label_len 由 C# 提供）
    decoder_loop = loops[1]
    promoted = [name for name in decoder_loop.output if name] + EXTRA_INPUTS
    print(f"promoting to inputs: {promoted}")
    producer = {out: node for node in graph.node for out in node.output}
    outputs = [o for o in graph.output if o.name in KEEP_OUTPUTS]
    if not outputs:
        raise SystemExit(f"none of {KEEP_OUTPUTS} is a graph output")
    print(f"keeping outputs: {[o.name for o in outputs]}")

    # 剪枝：从图输出反向可达，跳过控制流节点与被提升的张量（后者的生产者整棵子树都可以丢掉）
    stop = CONTROL_OPS | set(EXTRA_INPUTS)
    needed = {}
    stack = [o.name for o in outputs]
    seen = set()
    while stack:
        name = stack.pop()
        if name in seen:
            continue
        seen.add(name)

        node = producer.get(name)
        if node is None or node.op_type in CONTROL_OPS or name in EXTRA_INPUTS:
            continue

        needed[node.name] = node
        for value in node.input:
            if value:
                stack.append(value)

    kept_nodes = [node for node in graph.node if node.name in needed]
    used = {value for node in kept_nodes for value in node.input}
    initializers = [init for init in graph.initializer if init.name in used]

    # 输入：原图输入里仍被用到的 + 提升上来的 loop 输出
    inputs = [info for info in graph.input if info.name in used]
    for name in promoted:
        if name not in used:
            continue

        dtype, dims = PROMOTED.get(name, (TensorProto.FLOAT, None))
        inputs.append(helper.make_tensor_value_info(name, dtype, dims))

    new_graph = helper.make_graph(
        kept_nodes, "decoder", inputs, outputs, initializers)

    new_model = helper.make_model(new_graph, opset_imports=model.opset_import, producer_name="onnxsharp-split")
    new_model.ir_version = 10
    # 保留 metadata（前端参数 / 模型描述都靠它）
    del new_model.metadata_props[:]
    new_model.metadata_props.extend(model.metadata_props)
    onnx.save(new_model, path_out)

    print(f"nodes {len(graph.node)} -> {len(kept_nodes)}, initializers {len(graph.initializer)} -> {len(initializers)}")
    print(f"inputs : {[i.name for i in inputs]}")
    print(f"outputs: {[o.name for o in new_graph.output]}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        raise SystemExit(2)
    main(sys.argv[1], sys.argv[2])
