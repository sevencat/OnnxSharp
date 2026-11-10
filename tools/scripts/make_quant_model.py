"""生成一个最小的量化模型（DynamicQuantizeLinear + MatMulInteger）并用 ORT 录金标准。

用于验证 OnnxSharp 的两个新算子：
    x(f32) --DQL--> y(u8), scale, zp --MatMulInteger(W u8, zp, wzp)--> i32 --Cast--> f32 --Mul(scale)--> out

用法：python tools/make_quant_model.py <out dir>
产物：quant.onnx / quant_x.f32 / quant_golden.f32
"""

import os
import sys

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

M, K, N = 2, 4, 3
A_ZP = 121
W_ZP = 133


def build(path: str) -> None:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [M, K])
    out = helper.make_tensor_value_info("out", TensorProto.FLOAT, [M, N])
    y = helper.make_tensor_value_info("y", TensorProto.UINT8, [M, K])

    weights = (np.arange(K * N, dtype=np.uint8).reshape(K, N) + 100)
    initializers = [
        numpy_helper.from_array(weights, "W"),
        numpy_helper.from_array(np.array(W_ZP, dtype=np.uint8), "w_zp"),
    ]

    nodes = [
        helper.make_node("DynamicQuantizeLinear", ["x"], ["y", "a_scale", "a_zp"], name="dql"),
        helper.make_node("MatMulInteger", ["y", "W", "a_zp", "w_zp"], ["acc"], name="mmi"),
        helper.make_node("Cast", ["acc"], ["acc_f32"], to=TensorProto.FLOAT, name="cast"),
        helper.make_node("Mul", ["acc_f32", "a_scale"], ["out"], name="scale"),
    ]

    graph = helper.make_graph(nodes, "quant", [x], [out], initializers, value_info=[y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], producer_name="onnxsharp-tests")
    model.ir_version = 10          # 老版本 ORT 只认到 13，用 10 兼容性最好
    onnx.checker.check_model(model)
    onnx.save(model, path)


def build_signed(path: str) -> None:
    """s8 权重 + per-column zero point：Paraformer 量化后的真实形态。"""
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [M, K])
    out = helper.make_tensor_value_info("out", TensorProto.FLOAT, [M, N])

    weights = (np.arange(K * N, dtype=np.int16).reshape(K, N) - 10).astype(np.int8)
    weight_zp = np.array([0, 3, -5], dtype=np.int8)          # per-column

    initializers = [
        numpy_helper.from_array(weights, "W"),
        numpy_helper.from_array(weight_zp, "w_zp"),
    ]

    nodes = [
        helper.make_node("DynamicQuantizeLinear", ["x"], ["y", "a_scale", "a_zp"], name="dql"),
        helper.make_node("MatMulInteger", ["y", "W", "a_zp", "w_zp"], ["acc"], name="mmi"),
        helper.make_node("Cast", ["acc"], ["acc_f32"], to=TensorProto.FLOAT, name="cast"),
        helper.make_node("Mul", ["acc_f32", "a_scale"], ["out"], name="scale"),
    ]

    graph = helper.make_graph(nodes, "quant_s8", [x], [out], initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], producer_name="onnxsharp-tests")
    model.ir_version = 10
    onnx.checker.check_model(model)
    onnx.save(model, path)


def main(out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, "quant.onnx")
    build(model_path)

    rng = np.random.RandomState(3)
    x = (rng.rand(M, K) * 6 - 3).astype(np.float32)
    x.tofile(os.path.join(out_dir, "quant_x.f32"))

    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    (result,) = session.run(None, {"x": x})
    result.astype(np.float32).tofile(os.path.join(out_dir, "quant_golden.f32"))

    signed_path = os.path.join(out_dir, "quant_s8.onnx")
    build_signed(signed_path)
    (signed_result,) = ort.InferenceSession(signed_path, providers=["CPUExecutionProvider"]).run(None, {"x": x})
    signed_result.astype(np.float32).tofile(os.path.join(out_dir, "quant_s8_golden.f32"))

    print("ort out (u8)  =\n", result)
    print("ort out (s8)  =\n", signed_result)
    print(f"wrote {model_path} and {signed_path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
