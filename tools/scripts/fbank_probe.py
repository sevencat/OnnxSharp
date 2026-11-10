"""用 kaldi-native-fbank（sherpa-onnx 内部用的同一个库）给 C# 的 fbank 前端录金标准，
并按 funasr 的 WavFrontend 语义算 LFR(7,6) + CMVN（参数取自模型 metadata）。

用法：python tools/scripts/fbank_probe.py <out dir> [seconds] [sample_rate]
产物：
  fbank.wav            16 kHz PCM16 合成语音（chirp + 噪声）
  fbank_golden.f32     [T, 80]      kaldi fbank
  lfr_golden.f32       [T_lfr, 560] LFR(7,6) + CMVN 之后
  fbank_shapes.json
"""

import json
import math
import os
import sys
import wave

import numpy as np


def make_wav(path: str, seconds: float, sample_rate: int) -> np.ndarray:
    n = int(seconds * sample_rate)
    t = np.arange(n, dtype=np.float32) / sample_rate

    # 扫频（200 Hz → 3.4 kHz）+ 带限噪声，尽量覆盖整个梅尔频带
    sweep = 0.35 * np.sin(2 * np.pi * (200.0 * t + (3400.0 - 200.0) * t * t / (2 * seconds)))
    rng = np.random.RandomState(3)
    noise = (rng.rand(n).astype(np.float32) - 0.5) * 0.05
    signal = np.clip(sweep + noise, -1.0, 1.0)

    pcm = (signal * 32767.0).astype(np.int16)
    with wave.open(path, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())

    return pcm.astype(np.float32) / 32768.0


def compute_fbank(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    import kaldi_native_fbank as knf

    opts = knf.FbankOptions()
    opts.frame_opts.samp_freq = sample_rate
    opts.frame_opts.dither = 0
    opts.frame_opts.preemph_coeff = 0.97
    opts.frame_opts.remove_dc_offset = True
    opts.frame_opts.window_type = "povey"
    opts.frame_opts.snip_edges = True
    opts.mel_opts.num_bins = 80
    opts.mel_opts.low_freq = 20
    opts.mel_opts.high_freq = -400
    opts.use_energy = False
    opts.use_log_fbank = True

    extractor = knf.OnlineFbank(opts)
    extractor.accept_waveform(sample_rate, samples)
    extractor.input_finished()

    frames = extractor.num_frames_ready
    return np.stack([extractor.get_frame(i) for i in range(frames)]).astype(np.float32)


def apply_lfr_and_cmvn(fbank: np.ndarray, window: int, shift: int,
                       neg_mean: np.ndarray, inv_stddev: np.ndarray) -> np.ndarray:
    """funasr WavFrontend：T_lfr = ceil(T/shift)，左侧补 (window-1)//2 帧，右侧重复最后一帧。"""
    total = fbank.shape[0]
    out_frames = int(math.ceil(total / shift))
    left = (window - 1) // 2

    rows = np.empty((out_frames, window * fbank.shape[1]), dtype=np.float32)
    for i in range(out_frames):
        parts = []
        for j in range(window):
            index = i * shift + j - left
            index = min(max(index, 0), total - 1)
            parts.append(fbank[index])
        rows[i] = np.concatenate(parts)

    return (rows + neg_mean) * inv_stddev


def load_metadata(model_path: str) -> dict:
    import onnx

    model = onnx.load(model_path, load_external_data=False)
    meta = {p.key: p.value for p in model.metadata_props}

    def parse(name):
        return np.array([float(v) for v in meta[name].split(",")], dtype=np.float32)

    return {
        "lfr_window_size": int(meta["lfr_window_size"]),
        "lfr_window_shift": int(meta["lfr_window_shift"]),
        "neg_mean": parse("neg_mean"),
        "inv_stddev": parse("inv_stddev"),
    }


def main(out_dir: str, seconds: float, sample_rate: int, model_path: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    wav_path = os.path.join(out_dir, "fbank.wav")
    samples = make_wav(wav_path, seconds, sample_rate)

    fbank = compute_fbank(samples, sample_rate)
    fbank.tofile(os.path.join(out_dir, "fbank_golden.f32"))

    meta = load_metadata(model_path)
    lfr = apply_lfr_and_cmvn(fbank, meta["lfr_window_size"], meta["lfr_window_shift"],
                             meta["neg_mean"], meta["inv_stddev"])
    lfr.tofile(os.path.join(out_dir, "lfr_golden.f32"))

    # 把 CMVN 参数也落盘，测试就不用依赖 F:\ai\sherpa 的模型文件
    meta["neg_mean"].astype(np.float32).tofile(os.path.join(out_dir, "cmvn_neg_mean.f32"))
    meta["inv_stddev"].astype(np.float32).tofile(os.path.join(out_dir, "cmvn_inv_stddev.f32"))

    with open(os.path.join(out_dir, "fbank_shapes.json"), "w", encoding="utf-8") as handle:
        json.dump({
            "sample_rate": sample_rate,
            "samples": int(samples.size),
            "fbank_frames": int(fbank.shape[0]),
            "fbank_dim": int(fbank.shape[1]),
            "lfr_frames": int(lfr.shape[0]),
            "lfr_dim": int(lfr.shape[1]),
            "lfr_window_size": meta["lfr_window_size"],
            "lfr_window_shift": meta["lfr_window_shift"],
        }, handle, indent=2)

    print(f"samples {samples.size}  fbank {fbank.shape}  lfr {lfr.shape}")
    print(f"fbank[0,:6] = {np.round(fbank[0, :6], 4).tolist()}")
    print(f"lfr[0,:6]   = {np.round(lfr[0, :6], 4).tolist()}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(2)
    main(sys.argv[1],
         float(sys.argv[2]) if len(sys.argv) > 2 else 6.0,
         int(sys.argv[3]) if len(sys.argv) > 3 else 16000,
         sys.argv[4] if len(sys.argv) > 4 else r"F:\ai\sherpa\model.int8.onnx")
