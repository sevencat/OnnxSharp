"""用 ONNXRuntime 跑 `yunet_int8.onnx`（QOperator 量化图）录金标准，供 OnnxSharp 对拍。

用法：python tools/scripts/yunet_int8_probe.py [模型= F:\\ai\\models\\yunet\\yunet_int8.onnx] [输入=.tmp/yunet/yunet_input.f32]

输出到 `.tmp/yunet/int8_<name>.f32`，并打印每个输出的 min/max/mean（先看数量级对不对）。
"""

import os
import sys

import numpy as np
import onnxruntime as ort

MODEL = sys.argv[1] if len(sys.argv) > 1 else r"F:\ai\models\yunet\yunet_int8.onnx"
INPUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(".tmp", "yunet", "yunet_input.f32")
OUT_DIR = os.path.join(".tmp", "yunet")


def main():
    options = ort.SessionOptions()
    options.intra_op_num_threads = 16
    session = ort.InferenceSession(MODEL, options, providers=["CPUExecutionProvider"])

    data = np.fromfile(INPUT, dtype=np.float32).reshape(1, 3, 640, 640)
    feeds = {session.get_inputs()[0].name: data}

    for _ in range(3):
        outputs = session.run(None, feeds)

    timings = []
    import time

    for _ in range(10):
        start = time.perf_counter()
        outputs = session.run(None, feeds)
        timings.append((time.perf_counter() - start) * 1e3)

    print(f"ORT yunet_int8: min {min(timings):.1f} ms  median {sorted(timings)[5]:.1f} ms")

    for info, values in zip(session.get_outputs(), outputs):
        flat = np.asarray(values, dtype=np.float32).reshape(-1)
        flat.tofile(os.path.join(OUT_DIR, f"int8_{info.name}.f32"))
        print(f"  {info.name:8} shape {np.asarray(values).shape}  "
              f"min {flat.min():.6f}  max {flat.max():.6f}  mean {flat.mean():.6f}")


if __name__ == "__main__":
    main()
