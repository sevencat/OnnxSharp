"""验证切出来的 decoder.onnx 是否忠实：用 ORT 跑「全模型」时录下的 frames/fires 还原 memory，
喂给 decoder.onnx，比较 `logits` 是否与全模型的 `logits` 一致。

顺带产出 C# 侧需要的 fixture：
  decoder_speech.f32 / decoder_lengths.i32 / decoder_memory.f32 / decoder_max_len.i32
  decoder_logits_golden.f32

用法：python tools/scripts/sherpa_decoder_split_check.py <decoder.onnx> <cif fixture dir>
"""

import json
import os
import sys

import numpy as np
import onnxruntime as ort


def load(directory: str, name: str) -> np.ndarray:
    return np.fromfile(os.path.join(directory, name), dtype=np.float32)


def main(model_path: str, fixture_dir: str) -> None:
    with open(os.path.join(fixture_dir, "cif_shapes.json"), encoding="utf-8") as handle:
        shapes = json.load(handle)

    frames = load(fixture_dir, "ort_frames.f32").reshape(shapes["frames"]["shape"])
    fires = load(fixture_dir, "ort_fires.f32").reshape(shapes["fires"]["shape"])
    token_num = int(load(fixture_dir, "ort_token_num.f32")[0])
    speech = load(fixture_dir, "cif_speech.f32").reshape(1, -1, 560)

    # memory = 触发帧（fires >= 1）对应的候选帧；单条音频，B = 1
    mask = fires[0] >= 1.0
    memory = frames[0][mask][None, ...].astype(np.float32)
    print(f"token_num {token_num}, memory {memory.shape}, selected {int(mask.sum())}")

    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    feeds = {
        "speech": speech,
        "speech_lengths": np.array([speech.shape[1]], dtype=np.int32),
        "onnx::Slice_6785": memory,
        "max_label_len.28": np.array(token_num, dtype=np.int64),
    }
    logits, tok = session.run(None, feeds)

    golden = load(fixture_dir, "ort_logits.f32")
    diff = np.abs(logits.reshape(-1) - golden)
    print(f"logits {logits.shape} vs golden {golden.shape}: max|d| = {diff.max():.3e}, mean|d| = {diff.mean():.3e}")
    print(f"token_num: cut model {int(tok.reshape(-1)[0])} vs cif fixture {token_num}")
    print(f"decoder argmax head: {np.argmax(logits[0, 0])}, tail: {np.argmax(logits[0, -1])}")

    out = fixture_dir
    speech.tofile(os.path.join(out, "decoder_speech.f32"))
    np.array([speech.shape[1]], dtype=np.int32).tofile(os.path.join(out, "decoder_lengths.i32"))
    memory.tofile(os.path.join(out, "decoder_memory.f32"))
    np.array(token_num, dtype=np.int64).tofile(os.path.join(out, "decoder_max_len.i64"))
    print(f"wrote decoder fixture to {out}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        raise SystemExit(2)
    main(sys.argv[1], sys.argv[2])
