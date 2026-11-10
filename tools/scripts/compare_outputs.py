"""比较 OnnxSharp `run --dump` 的输出与 ORT 金标准。

用法：python tools/scripts/compare_outputs.py <golden dir> <ours dir>
"""

import json
import os
import sys

import numpy as np


def main(golden_dir: str, ours_dir: str) -> None:
    with open(os.path.join(golden_dir, "golden_shapes.json"), encoding="utf-8") as handle:
        shapes = json.load(handle)["shapes"]

    for index, (name, shape) in enumerate(shapes.items()):
        golden_path = os.path.join(golden_dir, f"golden_{name.replace('/', '_')}.f32")
        ours_path = os.path.join(ours_dir, f"{name.replace('/', '_')}.f32")
        if not os.path.exists(ours_path):
            # CLI 的 --dump 按输出顺序写 out0.f32 / out1.f32 ...
            ours_path = os.path.join(ours_dir, f"out{index}.f32")
        if not os.path.exists(ours_path):
            print(f"{name}: missing ours ({ours_path})")
            continue

        golden = np.fromfile(golden_path, dtype=np.float32)
        ours = np.fromfile(ours_path, dtype=np.float32)
        if golden.size != ours.size:
            print(f"{name}: size mismatch {golden.size} vs {ours.size}")
            continue

        diff = np.abs(golden - ours)
        scale = np.maximum(np.abs(golden), 1e-6)
        print(f"{name:28s} n={golden.size:6d}  max|d|={diff.max():.3e}  "
              f"mean|d|={diff.mean():.3e}  max rel={np.max(diff / scale):.3e}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        raise SystemExit(2)
    main(sys.argv[1], sys.argv[2])
