"""给 SAM3 decoder 建一个「probe」模型：把若干中间张量追加进 graph.output，
分别用 ORT 和 OnnxSharp 跑，再逐值对比，用于定位数值 bug。

用法：
    python tools/sam3_decoder_probe.py make  <model dir> <probe.onnx> [names...]
    python tools/sam3_decoder_probe.py run   <model dir> <probe.onnx> <inputs dir> <out dir>
    python tools/sam3_decoder_probe.py diff  <ref dir> <got dir>
"""

import os
import sys

import numpy as np
import onnx
import onnxruntime as ort

DEFAULT_NAMES = [
    "prompt_features_fp16",
    "/dot_product_scoring/Cast_output_0",
    "/dot_product_scoring/text_mlp/layer1/MatMul_output_0",
    "/dot_product_scoring/text_mlp/Relu_output_0",
    "/dot_product_scoring/ReduceSum_75736c73_output_0",
    "/dot_product_scoring/Clip_75736c73_output_0",
    "/dot_product_scoring/Div_75736c73_output_0",
    "/dot_product_scoring/text_proj/Gemm_75736c73_output_0",
]

FEEDS = {
    "fpn_feat_0": (1, 256, 288, 288),
    "fpn_feat_1": (1, 256, 144, 144),
    "fpn_feat_2": (1, 256, 72, 72),
    "fpn_pos_2": (1, 256, 72, 72),
    "prompt_features": (1, 2, 256),
    "prompt_mask": (1, 2),
}


def make(model_dir: str, probe_path: str, names):
    model = onnx.load(f"{model_dir}/decoder.onnx")
    existing = {output.name for output in model.graph.output}
    unknown = {node_output for node in model.graph.node for node_output in node.output}
    for name in names:
        if name not in existing and name in unknown:
            model.graph.output.append(onnx.ValueInfoProto(name=name))
    onnx.save(model, probe_path)
    print(f"probe model -> {probe_path}")


def run(model_dir: str, probe_path: str, inputs_dir: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    feeds = {}
    for name, shape in FEEDS.items():
        path = os.path.join(inputs_dir, name + (".bin" if name == "prompt_mask" else ".f32"))
        dtype = np.bool_ if name == "prompt_mask" else np.float32
        feeds[name] = np.fromfile(path, dtype=dtype).reshape(shape)

    session = ort.InferenceSession(probe_path, providers=["CPUExecutionProvider"])
    output_names = [output.name for output in session.get_outputs()]
    values = session.run(None, feeds)
    for name, value in zip(output_names, values):
        safe = name.replace("/", "_")
        value.astype(np.float32).tofile(os.path.join(out_dir, safe + ".f32"))
        print(f"{name:60} {value.shape} min {value.min():.5f} max {value.max():.5f} nan {int(np.isnan(value).sum())}")


def diff(ref_dir: str, got_dir: str):
    for name in sorted(os.listdir(ref_dir)):
        ref_path = os.path.join(ref_dir, name)
        got_path = os.path.join(got_dir, name)
        if not os.path.exists(got_path):
            print(f"{name:60} MISSING")
            continue
        ref = np.fromfile(ref_path, dtype=np.float32)
        got = np.fromfile(got_path, dtype=np.float32)
        if ref.shape != got.shape:
            print(f"{name:60} shape {ref.shape} vs {got.shape}")
            continue
        difference = np.abs(ref - got)
        print(f"{name:60} max|diff| {difference.max():.6f} mean|diff| {difference.mean():.6f}")


if __name__ == "__main__":
    command = sys.argv[1]
    if command == "make":
        make(sys.argv[2], sys.argv[3], sys.argv[4:] or DEFAULT_NAMES)
    elif command == "run":
        run(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
    else:
        diff(sys.argv[2], sys.argv[3])
