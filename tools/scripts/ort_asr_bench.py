"""用 ONNXRuntime 测 sherpa 的 encoder 段与**整模型**（与 OnnxSharp 交替跑，各自取最好值）。

和 `OnnxSharp.Cli profile` / `asr --repeat` 用的是同一批输入：
  - encoder 段：`.tmp/sherpa/encoder.onnx` + `.tmp/sherpa/speech.f32`（1×96×560）
  - 整模型：   `F:\\ai\\sherpa\\model.int8.onnx` + `.tmp/sherpa/e2e_feats.f32`（1×76×560）

机器上有后台负载时均值会被重尾拉高，所以输出 **min / median**，比较时只看 min。

用法：python tools/scripts/ort_asr_bench.py [iters=12] [threads=16]
"""

import os
import statistics
import sys
import time

import numpy as np
import onnxruntime as ort

ENCODER = os.path.join(".tmp", "sherpa", "encoder.onnx")
ENCODER_INPUT = os.path.join(".tmp", "sherpa", "speech.f32")
ENCODER_LENGTHS = os.path.join(".tmp", "sherpa", "speech_lengths.i32")
FULL = r"F:\ai\sherpa\model.int8.onnx"
FULL_INPUT = os.path.join(".tmp", "sherpa", "e2e_feats.f32")
FULL_LENGTHS = os.path.join(".tmp", "sherpa", "e2e_lengths.i32")


def bench(session, feeds, iters):
    for _ in range(4):
        session.run(None, feeds)

    timings = []
    for _ in range(iters):
        start = time.perf_counter()
        session.run(None, feeds)
        timings.append((time.perf_counter() - start) * 1e3)

    return min(timings), statistics.median(timings)


def main(iters, threads):
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    encoder = ort.InferenceSession(ENCODER, options, providers=["CPUExecutionProvider"])
    speech = np.fromfile(ENCODER_INPUT, dtype=np.float32).reshape(1, 96, 560)
    lengths = np.fromfile(ENCODER_LENGTHS, dtype=np.int32)
    encoder_min, encoder_median = bench(encoder, {"speech": speech, "speech_lengths": lengths}, iters)

    full = ort.InferenceSession(FULL, options, providers=["CPUExecutionProvider"])
    feats = np.fromfile(FULL_INPUT, dtype=np.float32).reshape(1, -1, 560)
    full_lengths = np.fromfile(FULL_LENGTHS, dtype=np.int32)
    full_min, full_median = bench(full, {"speech": feats, "speech_lengths": full_lengths}, iters)

    print(f"threads       : {threads}   iters: {iters}")
    print(f"ORT encoder   : min {encoder_min:7.1f} ms   median {encoder_median:7.1f} ms   (1x96x560)")
    print(f"ORT full model: min {full_min:7.1f} ms   median {full_median:7.1f} ms   (1x{feats.shape[1]}x560)")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 12, int(sys.argv[2]) if len(sys.argv) > 2 else 16)
