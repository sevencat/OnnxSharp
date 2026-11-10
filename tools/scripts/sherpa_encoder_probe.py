"""用 ORT 跑切出来的 sherpa encoder 子图，录输入与金标准输出（供 OnnxSharp 对拍）。

用法：python tools/scripts/sherpa_encoder_probe.py <encoder.onnx> <out dir> [feats=96]
"""

import json
import os
import sys

import numpy as np
import onnxruntime as ort

FEATS = 560


def main(model_path: str, out_dir: str, frames: int) -> None:
    os.makedirs(out_dir, exist_ok=True)

    rng = np.random.RandomState(7)
    speech = (rng.rand(1, frames, FEATS).astype(np.float32) - 0.5) * 2.0
    lengths = np.array([frames], dtype=np.int32)

    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    feeds = {"speech": speech, "speech_lengths": lengths}
    results = session.run(None, feeds)

    speech.tofile(os.path.join(out_dir, "speech.f32"))
    lengths.tofile(os.path.join(out_dir, "speech_lengths.i32"))

    names = [o.name for o in session.get_outputs()]
    shapes = {}
    for name, value in zip(names, results):
        value.astype(np.float32).tofile(os.path.join(out_dir, f"golden_{name.replace('/', '_')}.f32"))
        shapes[name] = list(value.shape)
        print(f"{name:12s} {value.shape}  min={value.min():.6f} max={value.max():.6f} mean={value.mean():.6f}")

    with open(os.path.join(out_dir, "golden_shapes.json"), "w", encoding="utf-8") as handle:
        json.dump({"frames": frames, "shapes": shapes}, handle, indent=2)

    print(f"wrote inputs + {len(results)} goldens to {out_dir}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        raise SystemExit(2)
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 96)
