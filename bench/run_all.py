"""统一基准编排：OnnxSharp / ONNXRuntime CPU / OpenCV dnn / SimdYolo。

四个引擎都在同一台机器、同一会话里依次跑，口径一致：
静态 batch=1、FP32、输入全 0.5、预热后取 N 次样本，只统计纯推理前向（不含前处理与 NMS）。

    python bench/run_all.py                  # 默认 iters=40 warmup=5 threads=16 rounds=2
    python bench/run_all.py --iters 20 --rounds 1 --names yolo11n,rfdetr-small

结果写 bench/results/latest.json，并打印可直接粘进 bench.md 的 Markdown 表。
"""

import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(HERE, "results")
INPUTS = os.path.join(HERE, "inputs")

SIMDYOLO_CLI = r"F:\code\SimdYolo\src\SimdYolo.Cli\bin\Release\net10.0\SimdYolo.Cli.exe"
OURS_CLI = os.path.join(ROOT, "tools", "OnnxSharp.Cli", "bin", "Release", "net10.0", "OnnxSharp.Cli.exe")

RUNTIMES = ["onnxsharp", "onnxruntime-cpu", "opencv-dnn", "simdyolo"]


def run(cmd, cwd=ROOT):
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    return proc.returncode, proc.stdout, proc.stderr


def model_shape(path):
    import onnx

    proto = onnx.load(path, load_external_data=False)
    dims = [d.dim_value if d.dim_value > 0 else 1 for d in proto.graph.input[0].type.tensor_type.shape.dim]
    while len(dims) < 4:
        dims.append(1)
    return dims[:4]


def ensure_input(model_path):
    """按模型静态输入形状生成全 0.5 的 FP32 输入文件（缓存复用）。"""
    import numpy as np

    n, c, h, w = model_shape(model_path)
    os.makedirs(INPUTS, exist_ok=True)
    path = os.path.join(INPUTS, f"i{n}x{c}x{h}x{w}.f32")
    if not os.path.exists(path):
        np.full(n * c * h * w, 0.5, dtype=np.float32).tofile(path)
    return path, h


# ---------------- 各引擎的采集函数：返回 (median_ms, min_ms) ----------------

def bench_onnxsharp(model, iters, warmup, threads):
    inp, _ = ensure_input(model["path"])
    env = dict(os.environ, ONNXSHARP_THREADS=str(threads))
    code, out, err = run([OURS_CLI, "bench", model["path"], "--input", inp,
                          "--iters", str(iters), "--warmup", str(warmup)])
    if code != 0:
        raise RuntimeError((err or out).strip().splitlines()[-1][:200] if (err or out).strip() else "exit %d" % code)
    vals = {}
    for line in out.splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            key = key.strip()
            if key in ("median", "min"):
                vals[key] = float(value.strip().split()[0])
    return vals["median"], vals["min"]


def bench_python(runtime, model, iters, warmup, threads):
    code, out, err = run([sys.executable, os.path.join(HERE, "run_python_bench.py"),
                          "--runtime", runtime, "--names", model["name"],
                          "--threads", str(threads), "--iters", str(iters), "--warmup", str(warmup)])
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("{"):
            record = json.loads(line)
            if "error" in record:
                raise RuntimeError(record["error"])
            return record["median_ms"], record["min_ms"]
    raise RuntimeError((err or out).strip().splitlines()[-1][:200] if (err or out).strip() else "no output")


def bench_simdyolo(model, iters, warmup, threads):
    _, height = ensure_input(model["path"])
    code, out, err = run([SIMDYOLO_CLI, "bench", model["path"], "cpu", str(height),
                          str(iters), str(threads), str(warmup)])
    if code != 0:
        msg = (err or out).strip().splitlines()
        raise RuntimeError(msg[-1][:200] if msg else "exit %d" % code)
    start = out.find("{")
    if start < 0:
        raise RuntimeError("no JSON in output")
    record = json.loads(out[start:])
    return record["MedianMs"], record["MinMs"]


COLLECTORS = {
    "onnxsharp": bench_onnxsharp,
    "onnxruntime-cpu": lambda m, i, w, t: bench_python("ort", m, i, w, t),
    "opencv-dnn": lambda m, i, w, t: bench_python("cv", m, i, w, t),
    "simdyolo": bench_simdyolo,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=os.path.join(HERE, "models.json"))
    ap.add_argument("--names", default="")
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--runtimes", default=",".join(RUNTIMES))
    args = ap.parse_args()

    with open(args.models, "r", encoding="utf-8") as f:
        models = json.load(f)
    wanted = [s.strip() for s in args.names.split(",") if s.strip()]
    if wanted:
        models = [m for m in models if m["name"] in wanted]
    runtimes = [s.strip() for s in args.runtimes.split(",") if s.strip()]

    table = {}
    for model in models:
        for runtime in runtimes:
            key = (model["name"], runtime)
            if not os.path.exists(model["path"]):
                table[key] = {"error": "model not found"}
                continue
            best = None
            errors = []
            for _ in range(max(1, args.rounds)):
                try:
                    median, minimum = COLLECTORS[runtime](model, args.iters, args.warmup, args.threads)
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"{type(exc).__name__}: {exc}")
                    continue
                if best is None or median < best[0]:
                    best = (median, minimum)
            table[key] = ({"median_ms": round(best[0], 2), "min_ms": round(best[1], 2)}
                          if best else {"error": errors[-1] if errors else "failed"})
            print(f"{model['name']:<14}{runtime:<18}{table[key]}", flush=True)

    os.makedirs(RESULTS, exist_ok=True)
    payload = {
        "iters": args.iters,
        "warmup": args.warmup,
        "threads": args.threads,
        "rounds": args.rounds,
        "results": [{"model": m, "runtime": r, **v} for (m, r), v in table.items()],
    }
    out_path = os.path.join(RESULTS, "latest.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print("\n| 模型 | " + " | ".join(runtimes) + " |")
    print("| --- | " + " | ".join("---:" for _ in runtimes) + " |")
    for model in models:
        cells = []
        for runtime in runtimes:
            cell = table.get((model["name"], runtime), {})
            cells.append(f"{cell['median_ms']:.1f}" if "median_ms" in cell else "N/A")
        print(f"| {model['name']} | " + " | ".join(cells) + " |")
    print(f"\nwritten: {out_path}")


if __name__ == "__main__":
    sys.exit(main())
