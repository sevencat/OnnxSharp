"""比对 ab_dump.ps1 落盘的 base / fused 输出，报告每个模型的最坏逐值偏差。"""
import glob
import os
import struct

BASE = os.path.dirname(os.path.abspath(__file__))


def load(path):
    raw = open(path, "rb").read()
    return struct.unpack(f"<{len(raw) // 4}f", raw)


for name in ("rfdetr-small", "rfdetr-medium", "yolo11n", "yolov8s"):
    base_dir = os.path.join(BASE, f"dump-{name}-base")
    fused_dir = os.path.join(BASE, f"dump-{name}-fused")
    worst = 0.0
    count = 0
    for path in sorted(glob.glob(os.path.join(base_dir, "out*.f32"))):
        a = load(path)
        b = load(os.path.join(fused_dir, os.path.basename(path)))
        assert len(a) == len(b), (path, len(a), len(b))
        count += 1
        for x, y in zip(a, b):
            d = abs(x - y)
            if d > worst:
                worst = d
    print(f"{name:<14} outputs={count:<3} max|diff|={worst:.3e}")
