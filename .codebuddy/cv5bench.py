import sys, time, statistics
import numpy as np
import cv2

MODEL = sys.argv[1]
INBIN = sys.argv[2]
SIZE = int(sys.argv[3])
ITERS = int(sys.argv[4]) if len(sys.argv) > 4 else 30
WARM = int(sys.argv[5]) if len(sys.argv) > 5 else 10
THREADS = int(sys.argv[6]) if len(sys.argv) > 6 else 16

cv2.setNumThreads(THREADS)

net = cv2.dnn.readNetFromONNX(MODEL)
try:
    net.setPreferableEngine(cv2.dnn.ENGINE_NEW)
    engine = "ENGINE_NEW"
except Exception as e:
    engine = "default(%s)" % e

x = np.fromfile(INBIN, dtype=np.float32).reshape(1, 3, SIZE, SIZE)
blob = x.copy()

for _ in range(WARM):
    net.setInput(blob)
    net.forward()

ts = []
for _ in range(ITERS):
    t = time.perf_counter()
    net.setInput(blob)
    net.forward()
    ts.append((time.perf_counter() - t) * 1000.0)

print("engine=%s threads=%d min=%.1f median=%.1f mean=%.1f" % (
    engine, THREADS, min(ts), statistics.median(ts), statistics.mean(ts)))
