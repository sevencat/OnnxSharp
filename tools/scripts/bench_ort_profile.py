"""用 ORT 的 profiler 看它**实际**用的是哪些 kernel、每个花多少时间。

这条是用来回答「同样没有 VNNI，为什么 ORT 快」——ORT 的 QOperator 优化器会把
`DynamicQuantizeLinear + MatMulInteger` 融合/改写成 `DynamicQuantizeMatMul`（或直接反量化成 FP32 GEMM），
profile 里能看到真实 kernel 名与耗时。

用法：python tools/scripts/bench_ort_profile.py <model.onnx> <fixture dir> [iters]
"""

import json
import os
import sys
from collections import defaultdict

import numpy as np
import onnxruntime as ort


def main(model_path: str, fixture_dir: str, iters: int) -> None:
    feats = np.fromfile(os.path.join(fixture_dir, "e2e_feats.f32"), dtype=np.float32).reshape(1, -1, 560)
    lengths = np.fromfile(os.path.join(fixture_dir, "e2e_lengths.i32"), dtype=np.int32)

    options = ort.SessionOptions()
    options.enable_profiling = True
    session = ort.InferenceSession(model_path, options, providers=["CPUExecutionProvider"])

    for _ in range(3):
        session.run(None, {"speech": feats, "speech_lengths": lengths})

    for _ in range(iters):
        session.run(None, {"speech": feats, "speech_lengths": lengths})

    profile_path = session.end_profiling()
    with open(profile_path, encoding="utf-8") as handle:
        events = json.load(handle)

    per_op = defaultdict(lambda: [0, 0.0])     # name -> [count, total_us]
    for event in events:
        if event.get("cat") != "Node" or event.get("dur") is None:
            continue
        args = event.get("args", {})
        name = args.get("op_name") or event.get("name", "?")
        entry = per_op[name]
        entry[0] += 1
        entry[1] += float(event["dur"])

    total = sum(v[1] for v in per_op.values()) / 1000.0
    print(f"profile: {len(events)} events, node time total {total:.1f} ms over {iters + 3} runs")
    print(f"{'op':28s} {'count':>6s} {'us/run':>10s}")
    for name, (count, micro) in sorted(per_op.items(), key=lambda kv: -kv[1][1])[:15]:
        print(f"{name:28s} {count:6d} {micro / (iters + 3):10.1f}")

    os.remove(profile_path)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        raise SystemExit(2)
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 5)
