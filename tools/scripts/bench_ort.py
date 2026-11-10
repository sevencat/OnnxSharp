"""测 ORT 各段的热态耗时，用来和 OnnxSharp 的 C# 链路对比。

用法：python tools/scripts/bench_ort.py <model.onnx> <decoder.onnx> <asr fixture dir> [iters]
fixture dir 里需要：e2e_feats.f32 / e2e_lengths.i32（sherpa_e2e_reference.py 产物）
"""

import os
import sys
import time

import numpy as np
import onnxruntime as ort

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def bench(run, iters: int, warmup: int = 3) -> float:
    for _ in range(warmup):
        run()

    best = float("inf")
    for _ in range(iters):
        start = time.perf_counter()
        run()
        best = min(best, (time.perf_counter() - start) * 1000)
    return best


def main(model_path: str, decoder_path: str, fixture_dir: str, iters: int) -> None:
    feats = np.fromfile(os.path.join(fixture_dir, "e2e_feats.f32"), dtype=np.float32).reshape(1, -1, 560)
    lengths = np.fromfile(os.path.join(fixture_dir, "e2e_lengths.i32"), dtype=np.int32)

    # 1) 特征（kaldi-native-fbank）
    import wave
    from fbank_probe import apply_lfr_and_cmvn, compute_fbank, load_metadata

    with wave.open(os.path.join(fixture_dir, "huihui.wav"), "rb") as handle:
        samples = np.frombuffer(handle.readframes(handle.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0

    meta = load_metadata(model_path)
    fbank_ms = bench(lambda: compute_fbank(samples, 16000), iters)

    # 2) 完整模型（encoder + CIF + decoder 一条龙）
    full = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    full_ms = bench(lambda: full.run(None, {"speech": feats, "speech_lengths": lengths}), iters)

    # 3) decoder 段（深切版，不含 encoder）：fixture 可能放在另一个目录（.tmp/sherpa）
    def find(name: str) -> str:
        for directory in (fixture_dir, os.path.join(os.path.dirname(fixture_dir.rstrip("\\/")), "sherpa")):
            path = os.path.join(directory, name)
            if os.path.exists(path):
                return path
        raise SystemExit(f"fixture {name} not found (looked in {fixture_dir})")

    decoder = ort.InferenceSession(decoder_path, providers=["CPUExecutionProvider"])
    memory = np.fromfile(find("dec2_memory.f32"), dtype=np.float32)
    enc_u8 = np.fromfile(find("dec2_enc_u8.bin"), dtype=np.uint8)
    enc_len, dim = enc_u8.size // 320, 320

    feeds = {
        "speech_lengths": np.fromfile(find("dec2_speech_lengths.i32"), dtype=np.int32),
        "onnx::Slice_6785": memory.reshape(1, -1, 320),
        "max_label_len.28": np.fromfile(find("dec2_max_len.i64"), dtype=np.int64).reshape(()),
        "token_num": np.fromfile(find("dec2_token_num.i32"), dtype=np.int32),
        "/encoder/after_norm/Add_1_output_0_quantized": enc_u8.reshape(1, enc_len, dim),
        "/encoder/after_norm/Add_1_output_0_scale": np.fromfile(find("dec2_enc_scale.f32"), dtype=np.float32).reshape(()),
        "/encoder/after_norm/Add_1_output_0_zero_point": np.fromfile(find("dec2_enc_zp.u8"), dtype=np.uint8).reshape(()),
    }
    decoder_ms = bench(lambda: decoder.run(None, feeds), iters)

    print(f"ORT fbank+LFR+CMVN : {fbank_ms:8.2f} ms")
    print(f"ORT full model     : {full_ms:8.2f} ms")
    print(f"ORT decoder graph  : {decoder_ms:8.2f} ms  (含内部重跑 encoder 段)")
    print(f"feats {feats.shape}, meta lfr {meta['lfr_window_size']},{meta['lfr_window_shift']}")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(__doc__)
        raise SystemExit(2)
    main(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 10)
