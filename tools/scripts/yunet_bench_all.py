"""YuNet 单模型推理三方对比：OnnxSharp（另跑 CLI）/ ONNXRuntime / OpenCV 5.0 dnn。

同一份输入 `(.tmp/yunet/yunet_input.f32`，1×3×640×640 预算好的 blob），
fp32 与 int8 两个模型各测一遍，输出 min / median（ms）。

用法：python tools/scripts/yunet_bench_all.py [iters=20]
"""

import statistics
import sys
import time

import cv2
import numpy as np
import onnxruntime as ort

FP32 = r".tmp\yunet\yunet.onnx"
INT8 = r"F:\ai\models\yunet\yunet_int8.onnx"
INPUT = r".tmp\yunet\yunet_input.f32"


def load_input():
    return np.fromfile(INPUT, dtype=np.float32).reshape(1, 3, 640, 640)


def bench_ort(model, data, iters, threads=16):
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    session = ort.InferenceSession(model, options, providers=["CPUExecutionProvider"])
    feeds = {session.get_inputs()[0].name: data}

    for _ in range(3):
        session.run(None, feeds)

    timings = []
    for _ in range(iters):
        start = time.perf_counter()
        session.run(None, feeds)
        timings.append((time.perf_counter() - start) * 1e3)

    return min(timings), statistics.median(timings)


def bench_cv2(model, data, iters, threads=16):
    cv2.setNumThreads(threads)
    net = cv2.dnn.readNetFromONNX(model)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)

    for _ in range(3):
        net.setInput(data)
        net.forward(net.getUnconnectedOutLayersNames())

    timings = []
    for _ in range(iters):
        start = time.perf_counter()
        net.setInput(data)
        net.forward(net.getUnconnectedOutLayersNames())
        timings.append((time.perf_counter() - start) * 1e3)

    return min(timings), statistics.median(timings)


def main(iters):
    data = load_input()
    rows = []

    for title, model in (("fp32", FP32), ("int8", INT8)):
        for engine, runner in (("ORT", bench_ort), ("OpenCV5-dnn", bench_cv2)):
            try:
                low, median = runner(model, data, iters)
                rows.append((title, engine, f"{low:7.1f}", f"{median:7.1f}"))
            except Exception as error:      # 引擎不支持这个图时把原因打出来
                rows.append((title, engine, "  n/a", str(error).replace("\n", " ")[:60]))

    print(f"iters = {iters}, threads = 16, input 1x3x640x640")
    print(f"{'model':6} {'engine':12} {'min ms':>8} {'median ms':>10}")
    for title, engine, low, median in rows:
        print(f"{title:6} {engine:12} {low:>8} {median:>10}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 20)
