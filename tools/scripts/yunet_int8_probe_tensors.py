"""给 `yunet_int8.onnx` 加几个中间张量当输出，用 ORT 录下来，便于定位量化算子的数值问题。

用法：python tools/scripts/yunet_int8_probe_tensors.py [tensor ...]
默认挑开头几层（QuantizeLinear 之后、前三个 QLinearConv、MaxPool 与其 QuantizeLinear）。

产物：
  .tmp/yunet/yunet_int8_probe.onnx     （多了若干输出的模型，两个引擎都能跑）
  .tmp/yunet/probe_<tensor>.f32        （ORT 的参考值，float32）
"""

import os
import sys

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper

MODEL = r"F:\ai\models\yunet\yunet_int8.onnx"
INPUT = os.path.join(".tmp", "yunet", "yunet_input.f32")
OUT_DIR = os.path.join(".tmp", "yunet")
PROBE = os.path.join(OUT_DIR, "yunet_int8_probe.onnx")

DEFAULT = ["input_quantized", "419_quantized", "185_quantized", "422_quantized", "188", "189", "189_quantized", "190_quantized"]


def tensor_type(model, name):
    """从 value_info / 节点输出里找这个张量的 elem_type 与形状。"""
    for collection in (model.graph.value_info, model.graph.output, model.graph.input):
        for info in collection:
            if info.name == name:
                tt = info.type.tensor_type
                dims = [d.dim_value if d.HasField("dim_value") else None for d in tt.shape.dim]
                return tt.elem_type, dims
    return TensorProto.FLOAT, None


def main(names):
    model = onnx.load(MODEL)
    existing = {o.name for o in model.graph.output}

    for name in names:
        if name in existing:
            continue
        elem, dims = tensor_type(model, name)
        shape = None if dims is None else [d if d is not None else "N" for d in dims]
        model.graph.output.append(helper.make_tensor_value_info(name, elem, shape))

    onnx.save(model, PROBE)

    options = ort.SessionOptions()
    options.intra_op_num_threads = 16
    session = ort.InferenceSession(PROBE, options, providers=["CPUExecutionProvider"])
    data = np.fromfile(INPUT, dtype=np.float32).reshape(1, 3, 640, 640)
    outputs = session.run(None, {session.get_inputs()[0].name: data})

    for info, values in zip(session.get_outputs(), outputs):
        array = np.asarray(values)
        flat = array.astype(np.float32).reshape(-1)
        flat.tofile(os.path.join(OUT_DIR, f"probe_{info.name}.f32"))
        print(f"  {info.name:16} {str(array.shape):16} {array.dtype}  "
              f"min {flat.min():.5f}  max {flat.max():.5f}  mean {flat.mean():.5f}")

    print(f"wrote {PROBE}")


if __name__ == "__main__":
    main(sys.argv[1:] or DEFAULT)
