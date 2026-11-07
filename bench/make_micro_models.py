"""生成单算子/少算子微模型，用来给 GEMM 内核做可复现的定点测量。

微模型的意义：`convbench` / `gemmbench` 用人工分配的缓冲区测内核，而真实模型里
A/B/C 来自引擎的 buffer slot。两者对不上时（DETR 就是这样被发现「内核没问题、
并行扩展性有问题」的）需要把形状搬进一个真模型里再测一遍。

    python bench/make_micro_models.py
    tools/OnnxSharp.Cli/bin/Release/net10.0/OnnxSharp.Cli.exe bench bench/models/attn-6x1028x64.onnx --input bench/inputs/i1x6x1028x64.f32 --iters 60 --warmup 10

产物（都在 .gitignore 里，随时可重建）：
    bench/models/attn-6x1028x64.onnx    6 个连续 MatMul，形状取自 RF-DETR 主干 attention
    bench/models/matmul-1025x384x1536.onnx   单 MatMul，形状取自 DETR 的 MLP
"""

import os

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(HERE, "models")
INPUTS = os.path.join(HERE, "inputs")


def save(model, name):
    onnx.save(model, os.path.join(MODELS, name + ".onnx"))
    print("model:", name)


def save_input(shape):
    os.makedirs(INPUTS, exist_ok=True)
    path = os.path.join(INPUTS, "i" + "x".join(str(d) for d in shape) + ".f32")
    if not os.path.exists(path):
        np.full(int(np.prod(shape)), 0.5, dtype=np.float32).tofile(path)
    print("input:", os.path.basename(path))


def build_attention_chain(batch, tokens, head_dim, reps):
    """reps 组 (qkᵀ, av)，全部静态 shape，对应 DETR 主干里连续 2*reps 个 MatMul。"""
    inp = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, batch, tokens, head_dim])
    out = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, batch, tokens, head_dim])

    nodes, inits, cur = [], [], "x"
    for rep in range(reps):
        for tag, weight in (("qk", (batch, head_dim, tokens)), ("av", (batch, tokens, head_dim))):
            name = f"w_{rep}_{tag}"
            inits.append(numpy_helper.from_array(np.full(weight, 0.01, dtype=np.float32), name))
            is_last = rep == reps - 1 and tag == "av"
            dst = "y" if is_last else name + "_out"
            nodes.append(helper.make_node("MatMul", [cur, name], [dst], name=name + "_mm"))
            cur = dst

    graph = helper.make_graph(nodes, "attn", [inp], [out], inits)
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], producer_name="bench")


def build_single_matmul(m, k, n):
    inp = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, m, k])
    out = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, m, n])
    weight = numpy_helper.from_array(np.full((k, n), 0.01, dtype=np.float32), "w")
    node = helper.make_node("MatMul", ["x", "w"], ["y"], name="mm")
    graph = helper.make_graph([node], "matmul", [inp], [out], [weight])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], producer_name="bench")


def main():
    os.makedirs(MODELS, exist_ok=True)
    save(build_attention_chain(6, 1028, 64, 3), "attn-6x1028x64")
    save_input((1, 6, 1028, 64))

    save(build_single_matmul(1025, 384, 1536), "matmul-1025x384x1536")
    save_input((1, 1025, 384))


if __name__ == "__main__":
    main()
