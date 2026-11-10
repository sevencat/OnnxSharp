"""生成 attention 融合的对拍夹具（静态形状的 MHA 子图）+ ORT 参考输出。

用法：python tools/scripts/make_attention_model.py .tmp/attention

图结构故意写成「教科书式」的 HF/ONNX 导出形态，用来验证 `GraphFusion.FuseAttention`
在**能静态读出 head 数**时走的那条路（连 Reshape+Transpose 一起折掉）：

    x ─┬─ MatMul(Wq) → Reshape[B,T,H,D] → Transpose[0,2,1,3] ─┐
       ├─ MatMul(Wk) → Reshape[B,T,H,D] → Transpose[0,2,3,1] ─┤
       └─ MatMul(Wv) → Reshape[B,T,H,D] → Transpose[0,2,1,3] ─┤
                                                            S = MatMul(Mul(Qh, scale), Kt)
                                                            S = Add(S, mask)      ← 带 mask 路径
                                                            P = Softmax(S, axis=-1)
                                                            O = MatMul(P, Vh)
                                                              → Transpose[0,2,1,3]
                                                              → Reshape[B,T,C] = y
"""

import os
import sys

import numpy as np
import onnx
from onnx import TensorProto as TP, helper as h, numpy_helper as nh

B, T, H, D = 1, 8, 2, 4
C = H * D

rng = np.random.default_rng(7)
Wq = rng.normal(0, 0.4, (C, C)).astype(np.float32)
Wk = rng.normal(0, 0.4, (C, C)).astype(np.float32)
Wv = rng.normal(0, 0.4, (C, C)).astype(np.float32)
x = rng.normal(0, 1, (B, T, C)).astype(np.float32)

mask = np.zeros((B, 1, 1, T), np.float32)
mask[0, 0, 0, -1] = -1e4                      # 屏蔽最后一个位置，覆盖 mask 分支
scale = np.float32(1.0 / np.sqrt(D))

nodes = [
    h.make_node("MatMul", ["x", "Wq"], ["q"], name="proj_q"),
    h.make_node("MatMul", ["x", "Wk"], ["k"], name="proj_k"),
    h.make_node("MatMul", ["x", "Wv"], ["v"], name="proj_v"),
    h.make_node("Reshape", ["q", "shape_bt_hd"], ["qr"], name="reshape_q"),
    h.make_node("Transpose", ["qr"], ["qh"], perm=[0, 2, 1, 3], name="transpose_q"),
    h.make_node("Reshape", ["k", "shape_bt_hd"], ["kr"], name="reshape_k"),
    h.make_node("Transpose", ["kr"], ["kt"], perm=[0, 2, 3, 1], name="transpose_k"),
    h.make_node("Reshape", ["v", "shape_bt_hd"], ["vr"], name="reshape_v"),
    h.make_node("Transpose", ["vr"], ["vh"], perm=[0, 2, 1, 3], name="transpose_v"),
    h.make_node("Mul", ["qh", "scale"], ["qs"], name="scale_q"),
    h.make_node("MatMul", ["qs", "kt"], ["scores"], name="scores"),
    h.make_node("Add", ["scores", "mask"], ["masked"], name="add_mask"),
    h.make_node("Softmax", ["masked"], ["probs"], axis=-1, name="softmax"),
    h.make_node("MatMul", ["probs", "vh"], ["attended"], name="attend"),
    h.make_node("Transpose", ["attended"], ["attended_t"], perm=[0, 2, 1, 3], name="transpose_out"),
    h.make_node("Reshape", ["attended_t", "shape_bt_c"], ["y"], name="reshape_out"),
]

initializers = [
    nh.from_array(Wq, "Wq"),
    nh.from_array(Wk, "Wk"),
    nh.from_array(Wv, "Wv"),
    nh.from_array(np.array([B, T, H, D], np.int64), "shape_bt_hd"),
    nh.from_array(np.array([B, T, C], np.int64), "shape_bt_c"),
    nh.from_array(np.array(scale, np.float32), "scale"),
    nh.from_array(mask, "mask"),
]

graph = h.make_graph(
    nodes,
    "attention",
    [h.make_tensor_value_info("x", TP.FLOAT, [B, T, C])],
    [h.make_tensor_value_info("y", TP.FLOAT, [B, T, C])],
    initializers,
)

model = h.make_model(graph, opset_imports=[h.make_opsetid("", 17)])
model.ir_version = 10
model = onnx.shape_inference.infer_shapes(model)     # 让 head 数从 value_info 读得出来
onnx.checker.check_model(model)

out = sys.argv[1] if len(sys.argv) > 1 else ".tmp/attention"
os.makedirs(out, exist_ok=True)
onnx.save(model, os.path.join(out, "attention.onnx"))
x.tofile(os.path.join(out, "attention_x.f32"))

import onnxruntime as ort

session = ort.InferenceSession(os.path.join(out, "attention.onnx"), providers=["CPUExecutionProvider"])
y = session.run(None, {"x": x})[0]
y.astype(np.float32).tofile(os.path.join(out, "attention_y.f32"))

print(f"attention.onnx  y={y.shape}  head_dim={D}  heads={H}")
print("y[0,0,:6] =", np.array2string(y[0, 0, :6], precision=6))
