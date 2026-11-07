import sys
import numpy as np
import cv2
import onnxruntime as ort

MODEL = sys.argv[1]
INBIN = sys.argv[2]
SIZE = int(sys.argv[3])

x = np.fromfile(INBIN, dtype=np.float32).reshape(1, 3, SIZE, SIZE)

net = cv2.dnn.readNetFromONNX(MODEL)
net.setInput(x.copy())
outs_cv = net.forward()

so = ort.SessionOptions()
so.intra_op_num_threads = 16
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
s = ort.InferenceSession(MODEL, so, providers=["CPUExecutionProvider"])
outs_ort = s.run(None, {s.get_inputs()[0].name: x})

if isinstance(outs_cv, np.ndarray):
    outs_cv = [outs_cv]
if isinstance(outs_cv, (list, tuple)) and len(outs_cv) and isinstance(outs_cv[0], (list, tuple)):
    outs_cv = list(outs_cv[0])

print("cv outputs:", [o.shape for o in outs_cv], " ort outputs:", [o.shape for o in outs_ort])

for i, oc in enumerate(outs_cv):
    best = None
    for j, oo in enumerate(outs_ort):
        if tuple(oc.shape) != tuple(oo.shape):
            continue
        d = np.abs(oc.astype(np.float64) - oo.astype(np.float64))
        best = (j, float(d.max()), float(d.mean()))
        break
    if best is None:
        print("out[%d] %s: no matching ort output" % (i, oc.shape))
    else:
        print("out[%d] -> ort[%d] %s max|diff|=%.6g mean|diff|=%.6g" % (i, best[0], oc.shape, best[1], best[2]))
