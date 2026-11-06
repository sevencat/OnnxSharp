#!/usr/bin/env python
"""用 onnxruntime 生成 golden 张量（M2 数值验收用）。

用法：
    python tools/scripts/gen_golden.py F:\\ai\\models\\yolo\\yolo11n.onnx F:\\ai\\models\\yolo\\golden\\yolo11n

产出（float32 裸二进制，行主序）：
    <prefix>.input.bin      第一个输入（确定性随机，seed=0）
    <prefix>.output0.bin    第 1 个输出
    <prefix>.output1.bin    第 2 个输出（多输出模型）
"""

import os
import sys

import numpy as np
import onnxruntime as ort


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2

    model_path, prefix = sys.argv[1], sys.argv[2]
    os.makedirs(os.path.dirname(os.path.abspath(prefix)), exist_ok=True)

    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    feeds = {}
    input_infos = session.get_inputs()

    rng = np.random.default_rng(0)
    for index, info in enumerate(input_infos):
        shape = [d if isinstance(d, int) else 1 for d in info.shape]
        data = rng.standard_normal(shape).astype(np.float32)
        feeds[info.name] = data

        path = f"{prefix}.input{index}.bin"
        data.tofile(path)
        print(f"input  {info.name:<16} {shape} -> {path} ({os.path.getsize(path) / 1048576.0:.2f} MB)")

    outputs = session.run(None, feeds)

    for index, (info, value) in enumerate(zip(session.get_outputs(), outputs)):
        array = np.asarray(value, dtype=np.float32)
        path = f"{prefix}.output{index}.bin"
        array.tofile(path)
        print(
            f"output {info.name:<16} {list(array.shape)} min={array.min():.6f} "
            f"max={array.max():.6f} mean={array.mean():.6f} -> {path}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
