"""在切好的子图上做逐层对拍：把一批中间张量加进输出，ORT / OnnxSharp 各跑一次，
打印每个张量的 max|diff|，用来二分定位数值偏差来源。

用法：
    python tools/scripts/probe_tensors.py <model.onnx> <work dir> <cli.exe> <stride> [start]
"""

import os
import subprocess
import sys

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto


def load_with_inference(path: str):
    model = onnx.load(path)
    try:
        model = onnx.shape_inference.infer_shapes(model, strict_mode=False)
    except Exception as error:                       # noqa: BLE001
        print(f"shape inference skipped: {error}")
    return model


def main(model_path: str, work_dir: str, cli: str, stride: int, start: int = 0) -> None:
    os.makedirs(work_dir, exist_ok=True)
    model = load_with_inference(model_path)

    infos = {v.name: v for v in list(model.graph.value_info) + list(model.graph.output) + list(model.graph.input)}
    nodes = [n for i, n in enumerate(model.graph.node) if i >= start and (i - start) % stride == 0]

    probes = []
    for node in nodes:
        for out in node.output:
            info = infos.get(out)
            if info is None or not info.type.HasField("tensor_type"):
                continue
            if info.type.tensor_type.elem_type not in (TensorProto.FLOAT, TensorProto.UNDEFINED):
                break
            probes.append(info)
            break

    print(f"probes: {len(probes)}")
    if not probes:
        return

    model.graph.output.extend(probes)
    probe_model = os.path.join(work_dir, "encoder_probe.onnx")
    onnx.save(model, probe_model)

    session = ort.InferenceSession(probe_model, providers=["CPUExecutionProvider"])
    folder = os.path.dirname(model_path)
    speech = np.fromfile(os.path.join(folder, "speech.f32"), dtype=np.float32).reshape(1, 96, 560)
    lengths = np.fromfile(os.path.join(folder, "speech_lengths.i32"), dtype=np.int32)
    results = session.run(None, {"speech": speech, "speech_lengths": lengths})

    goldens = {}
    for name, value in zip([o.name for o in session.get_outputs()], results):
        path = os.path.join(work_dir, f"ort_{len(goldens)}.f32")
        value.astype(np.float32).tofile(path)
        goldens[name] = path

    dump_dir = os.path.join(work_dir, "ours")
    os.makedirs(dump_dir, exist_ok=True)
    subprocess.run(
        [cli, "run", probe_model,
         "--input", f"speech={os.path.join(folder, 'speech.f32')}",
         "--input", f"speech_lengths={os.path.join(folder, 'speech_lengths.i32')}",
         "--shape", "speech=1,96,560", "--shape", "speech_lengths=1",
         "--dump", dump_dir],
        check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    print(f"{'tensor':62s} {'size':>7s} {'max|d|':>10s} {'mean|d|':>10s}")
    for index, name in enumerate(goldens):
        golden = np.fromfile(goldens[name], dtype=np.float32)
        ours_path = os.path.join(dump_dir, f"out{index}.f32")
        if not os.path.exists(ours_path):
            print(f"{name[:62]:62s} {golden.size:7d}  (missing)")
            continue
        ours = np.fromfile(ours_path, dtype=np.float32)
        if ours.size != golden.size:
            print(f"{name[:62]:62s} {golden.size:7d}  size {ours.size}")
            continue
        diff = np.abs(golden - ours)
        print(f"{name[:62]:62s} {golden.size:7d} {diff.max():10.3e} {diff.mean():10.3e}")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(__doc__)
        raise SystemExit(2)
    main(sys.argv[1], sys.argv[2], sys.argv[3],
         int(sys.argv[4]) if len(sys.argv) > 4 else 300,
         int(sys.argv[5]) if len(sys.argv) > 5 else 0)
