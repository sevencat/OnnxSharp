"""YuNet 对拍夹具：**同一个 640x640 输入**下，OpenCV FaceDetectorYN 的结果 + 原始 ONNX 输出。

用法：python tools/scripts/yunet_probe.py <image> [outdir] [model]

为什么要先把图 resize 成 640x640 再交给 cv2：
cv2.FaceDetectorYN 内部是「按 32 倍数 pad、按原分辨率跑」（不 resize），
而 `F:\ai\models\yunet\yunet.onnx` 是**静态 640x640** 导出 —— 只有把两边喂成同一个输入，
对比才是在比解码公式而不是比输入尺寸。

产出（outdir 默认 .tmp/yunet）：
    yunet_input.f32      预处理后的网络输入（NCHW，BGR，0..255）
    yunet_input.png      resize 后的图（640x640，便于画框肉眼确认）
    yunet_raw_<name>.f32 12 个原始输出
    yunet_faces.txt      cv2 的结果：x,y,w,h,5 个关键点(10 值),score —— 坐标已是 640 空间
"""

import os
import sys

import cv2
import numpy as np
import onnxruntime as ort

image_path = sys.argv[1]
outdir = sys.argv[2] if len(sys.argv) > 2 else ".tmp/yunet"
model = sys.argv[3] if len(sys.argv) > 3 else r"F:\ai\models\yunet\yunet.onnx"
os.makedirs(outdir, exist_ok=True)

session = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
input_meta = session.get_inputs()[0]
target_h = input_meta.shape[2] if isinstance(input_meta.shape[2], int) and input_meta.shape[2] > 0 else 640
target_w = input_meta.shape[3] if isinstance(input_meta.shape[3], int) and input_meta.shape[3] > 0 else 640

image = cv2.imread(image_path, cv2.IMREAD_COLOR)          # BGR
height, width = image.shape[:2]
print(f"image {os.path.basename(image_path)} {width}x{height} -> net {target_w}x{target_h}")

resized = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
blob = resized.astype(np.float32).transpose(2, 0, 1)[None]   # 1x3xHxW, BGR, 0..255
blob.tofile(os.path.join(outdir, "yunet_input.f32"))
resized.tofile(os.path.join(outdir, "yunet_input.bgr"))       # 原始 BGR 字节，供 C# 侧直接当输入
cv2.imwrite(os.path.join(outdir, "yunet_input.png"), resized)

# 模型一起拷进夹具目录，测试就不必依赖 F:\ai（220 KB）
import shutil

shutil.copyfile(model, os.path.join(outdir, "yunet.onnx"))

# ---- OpenCV oracle：喂同一个 640x640 图 ----
detector = cv2.FaceDetectorYN.create(model, "", (target_w, target_h), 0.6, 0.3, 5000, 0)
detector.setInputSize((target_w, target_h))
_, faces = detector.detect(resized)
faces = np.empty((0, 15), np.float32) if faces is None else faces
print(f"cv2.FaceDetectorYN: {len(faces)} faces (640 空间)")
for row in faces:
    print("   " + " ".join(f"{v:9.3f}" for v in row))
np.savetxt(os.path.join(outdir, "yunet_faces.txt"), faces, fmt="%.6f")

# ---- 原始 ONNX 输出（同一个 blob）----
outputs = session.run(None, {input_meta.name: blob})
for meta, value in zip(session.get_outputs(), outputs):
    value.astype(np.float32).tofile(os.path.join(outdir, f"yunet_raw_{meta.name}.f32"))
    print(f"   raw {meta.name:9s} {value.shape}  [{value.min():.3f}, {value.max():.3f}]")
