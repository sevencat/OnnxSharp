"""SAM3 框提示链路的 ORT 对照（开发期 oracle，不进产品）。

用途：用 onnxruntime 跑同一套 fp32 模型，判断「pred_logits 全 NaN」是引擎实现问题
还是提示编码约定问题。分两步：先用 `vision` 子命令缓存 fpn 特征，再用 `probe`
子命令快速试不同的框格式 / label（不用重复跑 20 多秒的 vision-encoder）。

用法：
    python tools/sam3_ort_probe.py vision <model dir> <image.ppm> <out.npz>
    python tools/sam3_ort_probe.py probe  <model dir> <feats.npz> <format> <label>
"""

import sys

import numpy as np
import onnxruntime as ort


def load_ppm(path: str):
    data = open(path, "rb").read()
    tokens, index = [], 0
    while len(tokens) < 4:
        while data[index:index + 1].isspace():
            index += 1
        start = index
        while not data[index:index + 1].isspace():
            index += 1
        tokens.append(data[start:index])
    index += 1
    width, height = int(tokens[1]), int(tokens[2])
    pixels = np.frombuffer(data[index:index + width * height * 3], dtype=np.uint8)
    return pixels.reshape(height, width, 3)


def preprocess(image: np.ndarray, size: int = 1008):
    height, width = image.shape[:2]
    if width >= height:
        embedded_width, embedded_height = size, max(1, round(height * size / width))
    else:
        embedded_width, embedded_height = max(1, round(width * size / height)), size

    xs = (np.arange(embedded_width) + 0.5) * width / embedded_width - 0.5
    ys = (np.arange(embedded_height) + 0.5) * height / embedded_height - 0.5
    x0 = np.floor(xs).astype(int)
    y0 = np.floor(ys).astype(int)
    wx = (xs - x0).astype(np.float32)[None, :, None]
    wy = (ys - y0).astype(np.float32)[:, None, None]
    x0c = np.clip(x0, 0, width - 1)
    x1c = np.clip(x0 + 1, 0, width - 1)
    y0c = np.clip(y0, 0, height - 1)
    y1c = np.clip(y0 + 1, 0, height - 1)

    top = image[y0c][:, x0c].astype(np.float32) * (1 - wx) + image[y0c][:, x1c].astype(np.float32) * wx
    bottom = image[y1c][:, x0c].astype(np.float32) * (1 - wx) + image[y1c][:, x1c].astype(np.float32) * wx
    resized = top * (1 - wy) + bottom * wy

    canvas = np.full((size, size, 3), 255, np.float32)
    canvas[:embedded_height, :embedded_width] = resized
    return (canvas / 255.0).transpose(2, 0, 1)[None], (embedded_width, embedded_height)


def vision(model_dir: str, image_path: str, out_path: str):
    image = load_ppm(image_path)
    tensor, (embedded_width, embedded_height) = preprocess(image)
    session = ort.InferenceSession(f"{model_dir}/vision-encoder.onnx", providers=["CPUExecutionProvider"])
    outputs = session.run(None, {"images": tensor})
    names = [output.name for output in session.get_outputs()]
    arrays = {name: value for name, value in zip(names, outputs)}
    arrays["ew"] = np.array(embedded_width)
    arrays["eh"] = np.array(embedded_height)
    np.savez(out_path, **arrays)
    for name, value in zip(names, outputs):
        print(f"{name:14} {value.shape} min {value.min():.4f} max {value.max():.4f}")


def probe(model_dir: str, feats_path: str, box_format: str, label: int):
    feats = np.load(feats_path)
    ew, eh = int(feats["ew"]), int(feats["eh"])
    width, height = 1568, 1068
    image_size = 1008

    sx, sy = ew / width, eh / height
    x1, y1, x2, y2 = 0.0 * sx, 0.0 * sy, width * sx, height * sy

    if box_format == "cxcywh":
        box = [(x1 + x2) / 2 / image_size, (y1 + y2) / 2 / image_size, (x2 - x1) / image_size, (y2 - y1) / image_size]
    elif box_format == "xyxy":
        box = [x1 / image_size, y1 / image_size, x2 / image_size, y2 / image_size]
    else:
        box = [x1, y1, x2, y2]

    geometry = ort.InferenceSession(f"{model_dir}/geometry-encoder.onnx", providers=["CPUExecutionProvider"])
    prompt_features, prompt_mask = geometry.run(None, {
        "input_boxes": np.array([[box]], np.float32),
        "input_boxes_labels": np.array([[label]], np.int64),
        "fpn_feat_2": feats["fpn_feat_2"],
        "fpn_pos_2": feats["fpn_pos_2"],
    })

    print(f"format {box_format} label {label}: box {box}")
    print(f"  prompt_features {prompt_features.shape} min {prompt_features.min():.4f} max {prompt_features.max():.4f}")
    print(f"  prompt_mask {prompt_mask.shape} true {int(prompt_mask.sum())}")

    decoder = ort.InferenceSession(f"{model_dir}/decoder.onnx", providers=["CPUExecutionProvider"])
    masks, boxes, logits, presence = decoder.run(None, {
        "fpn_feat_0": feats["fpn_feat_0"],
        "fpn_feat_1": feats["fpn_feat_1"],
        "fpn_feat_2": feats["fpn_feat_2"],
        "fpn_pos_2": feats["fpn_pos_2"],
        "prompt_features": prompt_features,
        "prompt_mask": prompt_mask,
    })

    print(f"  pred_logits  min {logits.min():.4f} max {logits.max():.4f} nan {int(np.isnan(logits).sum())}")
    print(f"  pred_boxes   min {boxes.min():.4f} max {boxes.max():.4f}")
    print(f"  pred_masks   min {masks.min():.4f} max {masks.max():.4f} nan {int(np.isnan(masks).sum())}")
    print(f"  presence     {presence.reshape(-1)[0]:.4f}")


def dump(model_dir: str, feats_path: str, out_dir: str):
    """把 decoder 的 6 个输入按 cxcywh/label=1 算好并落成 .f32/.bin，用于喂 C# 引擎做单模型对照。"""
    import os

    feats = np.load(feats_path)
    ew, eh = int(feats["ew"]), int(feats["eh"])
    width, height = 1568, 1068
    scene = 1008.0
    sx, sy = ew / width, eh / height
    x1, y1, x2, y2 = 0.0, 0.0, width * sx, height * sy
    box = [(x1 + x2) / 2 / scene, (y1 + y2) / 2 / scene, (x2 - x1) / scene, (y2 - y1) / scene]

    geometry = ort.InferenceSession(f"{model_dir}/geometry-encoder.onnx", providers=["CPUExecutionProvider"])
    prompt_features, prompt_mask = geometry.run(None, {
        "input_boxes": np.array([[box]], np.float32),
        "input_boxes_labels": np.array([[1]], np.int64),
        "fpn_feat_2": feats["fpn_feat_2"],
        "fpn_pos_2": feats["fpn_pos_2"],
    })

    os.makedirs(out_dir, exist_ok=True)
    for name in ("fpn_feat_0", "fpn_feat_1", "fpn_feat_2", "fpn_pos_2"):
        feats[name].astype(np.float32).tofile(f"{out_dir}/{name}.f32")
    prompt_features.astype(np.float32).tofile(f"{out_dir}/prompt_features.f32")
    prompt_mask.astype(np.uint8).tofile(f"{out_dir}/prompt_mask.bin")
    print("dumped decoder inputs to", out_dir)


if __name__ == "__main__":
    command = sys.argv[1]
    if command == "vision":
        vision(sys.argv[2], sys.argv[3], sys.argv[4])
    elif command == "dump":
        dump(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        probe(sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]))
