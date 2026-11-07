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


def build_mlp_chain(layers, tokens, d, hidden, residual=True):
    """layers 层 MLP 串联：每层 (fc1 + fc2 [+ Add])，形状完全取自 rfdetr-small。

    用途：真模型里 fc1/fc2 单个要 5 ms，而 gemmbench 跑同形状只要 0.9 ms。
    这个微模型把「多个 MatMul 串在一个图里」单独拎出来，判断差异是不是
    「前后算子把缓存挤掉 / 线程池反复休眠」造成的。
    """
    inp = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, tokens, d])
    out = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, tokens, d])

    nodes, inits, cur = [], [], "x"
    for i in range(layers):
        for tag, k, n in (("fc1", d, hidden), ("fc2", hidden, d)):
            w = f"w_{i}_{tag}"
            inits.append(numpy_helper.from_array(
                np.full((k, n), 0.01, dtype=np.float32), w))
            nodes.append(helper.make_node("MatMul", [cur, w], [f"{w}_out"], name=w))
            cur = f"{w}_out"

        if residual:
            dst = "y" if i == layers - 1 else f"add_{i}"
            nodes.append(helper.make_node("Add", [cur, "x" if i == 0 else f"add_{i-1}"], [dst],
                                          name=f"add_{i}"))
            cur = dst

    graph = helper.make_graph(nodes, "mlp", [inp], [out], inits)
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], producer_name="bench")


def build_single_matmul(m, k, n):
    inp = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, m, k])
    out = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, m, n])
    weight = numpy_helper.from_array(np.full((k, n), 0.01, dtype=np.float32), "w")
    node = helper.make_node("MatMul", ["x", "w"], ["y"], name="mm")
    graph = helper.make_graph([node], "matmul", [inp], [out], [weight])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], producer_name="bench")


def build_elementwise(op, shape, second_input, reps):
    """reps 个连续二元逐元素算子，用来量「纯带宽算子」的天花板。

    second_input=False 时第二操作数是常量，走 ApplyScalar 快路径；True 时是张量，走 ApplyContiguous。
    """
    inp = helper.make_tensor_value_info("x", TensorProto.FLOAT, list(shape))
    out = helper.make_tensor_value_info("y", TensorProto.FLOAT, list(shape))
    nodes, inputs, inits, cur = [], ["x"], [], "x"

    for rep in range(reps):
        is_last = rep == reps - 1
        dst = "y" if is_last else f"t{rep}"
        if second_input:
            other = f"b{rep}"
            inits.append(numpy_helper.from_array(np.full(shape, 0.5, dtype=np.float32), other))
            nodes.append(helper.make_node(op, [cur, other], [dst], name=f"{op.lower()}{rep}"))
        else:
            nodes.append(helper.make_node(op, [cur, "k"], [dst], name=f"{op.lower()}{rep}"))
        cur = dst

    if not second_input:
        kshape = [1] if op != "Add" else [1]
        inits.append(numpy_helper.from_array(np.full(kshape, 0.5, dtype=np.float32), "k"))

    graph = helper.make_graph(nodes, "eltwise", [inp], [out], inits)
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], producer_name="bench")


def main():
    os.makedirs(MODELS, exist_ok=True)
    save(build_attention_chain(6, 1028, 64, 3), "attn-6x1028x64")
    save_input((1, 6, 1028, 64))

    save(build_single_matmul(1025, 384, 1536), "matmul-1025x384x1536")
    save_input((1, 1025, 384))

    # 纯带宽对照：1M 元素（4 MB/张量），8 个算子串联
    # 每次 Add 的流量 = 读 4 MB + 读 4 MB + 写 4 MB = 12 MB，除以耗时就是有效带宽
    save(build_elementwise("Add", (1, 1024, 1024), second_input=True, reps=8), "add-1024x1024")
    save_input((1, 1024, 1024))

    # 标量广播版（走 ApplyScalar），用来分清「广播开销」和「带宽上限」
    save(build_elementwise("Mul", (1, 1024, 1024), second_input=False, reps=8), "mulscalar-1024x1024")

    # 24 个链式 MLP MatMul（12 层 fc1+fc2），形状取自 rfdetr-small 主干
    # 裸内核同形状 0.88~0.97 ms/个，若这个微模型也是 0.9 那就是「真模型特有的干扰」
    save(build_mlp_chain(12, 257, 384, 1536), "mlp-12x257x384x1536")
    save_input((1, 257, 384))


if __name__ == "__main__":
    main()
