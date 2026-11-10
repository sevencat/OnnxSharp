"""用 ORT 跑完整的 sherpa Paraformer 模型，但把 **CIF 段的中间张量**加成输出，
为「用 C# 重写 CIF」录金标准。

加进去的输出：
  /predictor/Concat_5_output_0     encoder 输出  [B, T, D]
  /predictor/Add_output_0          CIF 的 alphas [B, T]
  ConcatFromSequence(6682/6683)    每步的 frame / integrate 序列
  frame.7 / integrate.7            Loop 的标量输出

用法：python tools/scripts/sherpa_cif_probe.py <model.onnx> <out dir> [frames=96]
"""

import json
import os
import sys

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto

WANTED = [
    "/predictor/Concat_5_output_0",
    "/predictor/Add_output_0",
    "frame.7",
    "integrate.7",
]


def main(model_path: str, out_dir: str, frames: int) -> None:
    os.makedirs(out_dir, exist_ok=True)

    model = onnx.load(model_path, load_external_data=False)
    infos = {v.name: v for v in list(model.graph.value_info) + list(model.graph.output)}

    # Loop 的两个序列输出是主图里 ConcatFromSequence 的产物，名字以 onnx:: 开头
    loop_outputs = [n for n in model.graph.node if n.op_type == "Loop"][0].output
    sequence_names = [
        node.output[0] for node in model.graph.node
        if node.op_type == "ConcatFromSequence" and any(value in loop_outputs for value in node.input)
    ]

    existing = {o.name for o in model.graph.output}
    for name in WANTED + sequence_names:
        if name in existing:
            continue
        if name in infos:
            model.graph.output.append(infos[name])
        else:
            model.graph.output.append(onnx.ValueInfoProto(name=name))

    probe_path = os.path.join(out_dir, "cif_probe.onnx")
    onnx.save(model, probe_path)

    rng = np.random.RandomState(7)
    speech = (rng.rand(1, frames, 560).astype(np.float32) - 0.5) * 2.0
    lengths = np.array([frames], dtype=np.int32)
    speech.tofile(os.path.join(out_dir, "cif_speech.f32"))

    session = ort.InferenceSession(probe_path, providers=["CPUExecutionProvider"])
    results = session.run(None, {"speech": speech, "speech_lengths": lengths})

    names = [o.name for o in session.get_outputs()]
    shapes = {}
    for name, value in zip(names, results):
        safe = name.replace("/", "_").replace("::", "_").replace(".", "_")
        array = value.astype(np.float32)
        array.tofile(os.path.join(out_dir, f"ort_{safe}.f32"))
        shapes[name] = {"shape": list(value.shape), "file": f"ort_{safe}.f32"}
        print(f"{name:40s} {value.shape}  min={array.min():+.4f} max={array.max():+.4f}")

    with open(os.path.join(out_dir, "cif_shapes.json"), "w", encoding="utf-8") as handle:
        json.dump(shapes, handle, indent=2)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        raise SystemExit(2)
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 96)
