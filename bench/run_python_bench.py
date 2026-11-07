"""OnnxSharp 对外对比基准：ONNXRuntime CPU / OpenCV dnn。

只测「纯推理」耗时（不含前处理/NMS），与 OnnxSharp.Cli bench、SimdYolo bench 口径一致：
统一常量输入 0.5、静态 batch=1、FP32、预热若干帧后取 N 次样本的中位数。

用法：
    python bench/run_python_bench.py --runtime ort  --names yolo11n,rfdetr-small
    python bench/run_python_bench.py --runtime cv   --names yolo11n --threads 16

输出：每个模型一行 JSON（附 "runtime"/"error" 字段），方便直接喂给 bench.md。
"""

import argparse
import json
import os
import statistics
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def load_models(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def static_shape(model_path, default_hw=640):
    """返回 (N, C, H, W)；动态维按 default_hw 补齐。"""
    import onnx

    proto = onnx.load(model_path, load_external_data=False)
    dims = proto.graph.input[0].type.tensor_type.shape.dim
    out = []
    for i, d in enumerate(dims):
        out.append(d.dim_value if d.dim_value > 0 else (1 if i == 0 else default_hw))
    while len(out) < 4:
        out.append(1)
    return tuple(out[:4])


def samples_to_stats(ms):
    ms = sorted(ms)
    n = len(ms)

    def pct(p):
        idx = p * (n - 1)
        lo, hi = int(idx), min(int(idx) + 1, n - 1)
        return ms[lo] + (ms[hi] - ms[lo]) * (idx - lo)

    return {
        "iters": n,
        "min_ms": round(ms[0], 2),
        "median_ms": round(pct(0.5), 2),
        "p95_ms": round(pct(0.95), 2),
        "mean_ms": round(statistics.fmean(ms), 2),
    }


def bench_ort(model_path, threads, iters, warmup):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(model_path, opts, providers=["CPUExecutionProvider"])

    feeds = {}
    for inp in sess.get_inputs():
        shape = [d if isinstance(d, int) and d > 0 else 640 for d in inp.shape]
        feeds[inp.name] = np.full(shape, 0.5, dtype=np.float32)

    outs = [o.name for o in sess.get_outputs()]
    for _ in range(warmup):
        sess.run(outs, feeds)

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        sess.run(outs, feeds)
        times.append((time.perf_counter() - t0) * 1000.0)
    return samples_to_stats(times)


def bench_cv(model_path, threads, iters, warmup):
    import cv2

    cv2.setNumThreads(threads)
    net = cv2.dnn.readNetFromONNX(model_path)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    n, c, h, w = static_shape(model_path)
    blob = np.full((n, c, h, w), 0.5, dtype=np.float32)
    net.setInput(blob)

    for _ in range(warmup):
        net.forward()

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        net.forward()
        times.append((time.perf_counter() - t0) * 1000.0)
    return samples_to_stats(times)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runtime", required=True, choices=["ort", "cv"])
    ap.add_argument("--models", default=os.path.join(HERE, "models.json"))
    ap.add_argument("--names", default="")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    args = ap.parse_args()

    models = load_models(args.models)
    wanted = [s.strip() for s in args.names.split(",") if s.strip()]
    if wanted:
        models = [m for m in models if m["name"] in wanted]

    label = "onnxruntime-cpu" if args.runtime == "ort" else "opencv-dnn"
    if args.runtime == "ort":
        import onnxruntime as ort

        version = ort.__version__
    else:
        import cv2

        version = cv2.__version__

    for m in models:
        record = {
            "runtime": label,
            "version": version,
            "model": m["name"],
            "threads": args.threads,
        }
        if not os.path.exists(m["path"]):
            record["error"] = "model not found"
            print(json.dumps(record, ensure_ascii=False), flush=True)
            continue
        try:
            run = bench_ort if args.runtime == "ort" else bench_cv
            record.update(run(m["path"], args.threads, args.iters, args.warmup))
        except Exception as exc:  # noqa: BLE001 - 不支持就如实记录
            record["error"] = f"{type(exc).__name__}: {exc}".split("\n")[0][:300]
        print(json.dumps(record, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    sys.exit(main())
