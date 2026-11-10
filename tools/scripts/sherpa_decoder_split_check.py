"""验证深切后的 decoder.onnx（不含 encoder，2076 节点）是否与全模型 `logits` 一致，
并产出 C# 侧需要的 fixture。

喂给 decoder 的 7 个输入：
  speech_lengths、max_label_len.28、onnx::Slice_6785(memory)、token_num、
  /encoder/after_norm/Add_1_output_0_{quantized,scale,zero_point}

用法：python tools/scripts/sherpa_decoder_split_check.py <model.onnx> <decoder.onnx> <fixture dir>
"""

import json
import os
import sys

import numpy as np
import onnx
import onnxruntime as ort

ENCODER_OUT = [
    "/encoder/after_norm/Add_1_output_0_quantized",
    "/encoder/after_norm/Add_1_output_0_scale",
    "/encoder/after_norm/Add_1_output_0_zero_point",
]


def main(model_path: str, decoder_path: str, fixture_dir: str) -> None:
    with open(os.path.join(fixture_dir, "cif_shapes.json"), encoding="utf-8") as handle:
        shapes = json.load(handle)

    feats = np.fromfile(os.path.join(fixture_dir, "e2e_feats.f32"), dtype=np.float32).reshape(1, -1, 560)
    lengths = np.fromfile(os.path.join(fixture_dir, "e2e_lengths.i32"), dtype=np.int32)

    # 1) 用完整模型拿到 encoder 的量化输出 + frames/fires/token_num
    model = onnx.load(model_path, load_external_data=False)
    infos = {v.name: v for v in list(model.graph.value_info) + list(model.graph.output)}
    wanted = ENCODER_OUT + ["frames", "fires", "token_num"]
    existing = {o.name for o in model.graph.output}
    for name in wanted:
        if name in existing:
            continue
        model.graph.output.append(infos[name] if name in infos else onnx.ValueInfoProto(name=name))

    probe = os.path.join(fixture_dir, "decoder_probe.onnx")
    onnx.save(model, probe)

    session = ort.InferenceSession(probe, providers=["CPUExecutionProvider"])
    results = dict(zip([o.name for o in session.get_outputs()],
                       session.run(None, {"speech": feats, "speech_lengths": lengths})))

    enc_u8 = np.asarray(results["/encoder/after_norm/Add_1_output_0_quantized"], dtype=np.uint8)
    enc_scale = np.array(results["/encoder/after_norm/Add_1_output_0_scale"], dtype=np.float32).reshape(())
    enc_zp = np.array(results["/encoder/after_norm/Add_1_output_0_zero_point"], dtype=np.uint8).reshape(())
    token_num = int(np.asarray(results["token_num"]).reshape(-1)[0])

    frames = np.asarray(results["frames"])
    fires = np.asarray(results["fires"]).reshape(frames.shape[0], -1)
    mask = fires[0] >= 1.0
    memory = frames[0][mask][None, ...].astype(np.float32)

    print(f"enc_u8 {enc_u8.shape} {enc_u8.dtype}, token_num {token_num}, memory {memory.shape}")

    # 2) 深切后的 decoder
    decoder = ort.InferenceSession(decoder_path, providers=["CPUExecutionProvider"])
    feeds = {
        "speech_lengths": lengths,
        "max_label_len.28": np.array(token_num, dtype=np.int64).reshape(()),
        "onnx::Slice_6785": memory,
        "token_num": np.array([token_num], dtype=np.int32),
        "/encoder/after_norm/Add_1_output_0_quantized": enc_u8,
        "/encoder/after_norm/Add_1_output_0_scale": enc_scale,
        "/encoder/after_norm/Add_1_output_0_zero_point": enc_zp,
    }
    logits = decoder.run(None, feeds)[0]

    golden = np.fromfile(os.path.join(fixture_dir, "e2e_logits.f32"), dtype=np.float32)
    diff = np.abs(logits.reshape(-1) - golden)
    print(f"logits {logits.shape} vs full-model golden: max|d| = {diff.max():.3e}, mean|d| = {diff.mean():.3e}")

    # 3) 落 fixture 给 C# 用
    lengths.tofile(os.path.join(fixture_dir, "dec2_speech_lengths.i32"))
    np.array(token_num, dtype=np.int64).tofile(os.path.join(fixture_dir, "dec2_max_len.i64"))
    np.array([token_num], dtype=np.int32).tofile(os.path.join(fixture_dir, "dec2_token_num.i32"))
    memory.tofile(os.path.join(fixture_dir, "dec2_memory.f32"))
    enc_u8.tofile(os.path.join(fixture_dir, "dec2_enc_u8.bin"))
    np.array([enc_scale], dtype=np.float32).tofile(os.path.join(fixture_dir, "dec2_enc_scale.f32"))
    np.array([enc_zp], dtype=np.uint8).tofile(os.path.join(fixture_dir, "dec2_enc_zp.u8"))
    golden.tofile(os.path.join(fixture_dir, "dec2_logits_golden.f32"))
    print(f"wrote dec2_* fixture to {fixture_dir}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(__doc__)
        raise SystemExit(2)
    main(sys.argv[1], sys.argv[2], sys.argv[3])
