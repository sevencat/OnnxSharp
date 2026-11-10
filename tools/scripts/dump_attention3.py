"""列出 encoders0.0 的 self_attn 块所有真实节点（去掉 Constant），附 perm/axis。"""

import sys

import onnx
from onnx import numpy_helper

m = onnx.load(sys.argv[1], load_external_data=False)
g = m.graph
inits = {i.name: numpy_helper.to_array(i) for i in g.initializer}

KEEP_PREFIX = ("/encoder/encoders0/encoders0.0/self_attn/",)
KEEP_NAMES = ("/encoder/Mul_1", "/encoder/Sub", "/encoder/Unsqueeze_1", "/encoder/Unsqueeze_2",
              "/encoder/make_pad_mask/Cast_3")


def attrs(node):
    out = {}
    for a in node.attribute:
        if a.type == onnx.AttributeProto.INTS:
            out[a.name] = list(a.ints)
        elif a.type == onnx.AttributeProto.INT:
            out[a.name] = a.i
        elif a.type == onnx.AttributeProto.FLOAT:
            out[a.name] = round(a.f, 6)
    return out


def label(name):
    if not name:
        return "-"
    if name in inits:
        a = inits[name]
        return f"const{list(a.shape)}{a.flatten()[:3].tolist()}"
    return name.split("/")[-1]


for n in g.node:
    if n.op_type == "Constant":
        continue
    if not (n.name.startswith(KEEP_PREFIX) or n.name in KEEP_NAMES):
        continue
    print(f"{n.op_type:12s} {n.name.replace('/encoder/encoders0/encoders0.0/self_attn/', ''):18s} "
          f"{str(attrs(n)):24s} {[label(v) for v in n.input]}  -> {[v.split('/')[-1] for v in n.output]}")
