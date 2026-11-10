"""端到端参考：huihui.wav → kaldi-native-fbank 特征 → ORT 跑**完整** Paraformer → argmax → 文本。

这是 C# 那条链（fbank → encoder → CIF → decoder → argmax → 文本）的对拍基准。
用法：python tools/scripts/sherpa_e2e_reference.py <model.onnx> <tokens.txt> <wav> <out dir>
"""

import os
import sys
import wave

import numpy as np
import onnxruntime as ort

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fbank_probe import apply_lfr_and_cmvn, compute_fbank, load_metadata  # noqa: E402

BLANK, EOS = 0, 2


def read_wav(path: str) -> tuple[np.ndarray, int]:
    with wave.open(path, "rb") as handle:
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        rate = handle.getframerate()
        raw = handle.readframes(handle.getnframes())

    if width != 2 or channels != 1:
        raise SystemExit(f"expect 16-bit mono, got {width * 8}-bit / {channels} ch")

    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return samples, rate


def decode_text(token_ids, tokens) -> str:
    """Paraformer 的解码规则：跳过 blank、折叠连续重复、遇 EOS 停。"""
    pieces = []
    previous = -1
    for token in token_ids:
        token = int(token)
        if token == BLANK:
            continue
        if token == previous:
            continue
        if token == EOS:
            break
        previous = token
        if 0 <= token < len(tokens):
            pieces.append(tokens[token])

    return "".join(pieces)


def main(model_path: str, tokens_path: str, wav_path: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    samples, rate = read_wav(wav_path)
    if rate != 16000:
        raise SystemExit(f"expect 16 kHz, got {rate}")

    meta = load_metadata(model_path)
    fbank = compute_fbank(samples, rate)
    feats = apply_lfr_and_cmvn(fbank, meta["lfr_window_size"], meta["lfr_window_shift"],
                               meta["neg_mean"], meta["inv_stddev"])
    print(f"wav {samples.size} samples ({samples.size / rate:.2f} s), fbank {fbank.shape}, feats {feats.shape}")

    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    logits, token_num = session.run(None, {
        "speech": feats[None, ...],
        "speech_lengths": np.array([feats.shape[0]], dtype=np.int32),
    })
    token_num = int(np.asarray(token_num).reshape(-1)[0])
    print(f"logits {logits.shape}, token_num {token_num}")

    ids = np.argmax(logits[0][:token_num], axis=-1)

    tokens = []
    with open(tokens_path, encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line:
                continue
            tokens.append(line.rsplit(" ", 1)[0])

    text = decode_text(ids, tokens)
    print(f"token ids ({len(ids)}): {ids.tolist()}")
    print(f"text: {text}")

    feats.astype(np.float32).tofile(os.path.join(out_dir, "e2e_feats.f32"))
    np.array([feats.shape[0]], dtype=np.int32).tofile(os.path.join(out_dir, "e2e_lengths.i32"))
    logits.astype(np.float32).tofile(os.path.join(out_dir, "e2e_logits.f32"))
    with open(os.path.join(out_dir, "e2e_ref_text.txt"), "w", encoding="utf-8") as handle:
        handle.write(text)

    print(f"wrote fixture to {out_dir}")


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print(__doc__)
        raise SystemExit(2)
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
