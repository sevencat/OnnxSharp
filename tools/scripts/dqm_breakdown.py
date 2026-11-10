"""把 `ONNXSHARP_DQM_BREAKDOWN=1` 打出来的逐算子分段时间按形状汇总。

用法：
    set ONNXSHARP_DQM_BREAKDOWN=1
    OnnxSharp.Cli profile .tmp/sherpa/encoder.onnx ... 2> .tmp/dqm_break.txt
    python tools/scripts/dqm_breakdown.py .tmp/dqm_break.txt

每行形如（stderr，PowerShell 重定向后可能带前缀，所以按 token 解析）：
    [dqm] m=96 k=320 n=1280  minmax 0.004 quant 0.012 gemm 0.107 epi 0.031 ms

统计取**每个形状的最小值**（机器有后台负载时均值会被重尾放大）。
"""

import collections
import sys


def parse(path):
    rows = collections.defaultdict(list)

    # PowerShell 5.1 的 `2> file` 写的是 UTF-16LE（带 BOM），管道里是普通文本 —— 两种都认
    with open(path, "rb") as handle:
        raw = handle.read()

    if raw[:2] == b"\xff\xfe":
        text = raw.decode("utf-16-le", errors="ignore")
    elif raw[:3] == b"\xef\xbb\xbf":
        text = raw[3:].decode("utf-8", errors="ignore")
    else:
        text = raw.decode("utf-8", errors="ignore")

    for line in text.splitlines():
        index = line.find("[dqm]")
        if index < 0:
            continue

        tokens = line[index:].split()
        try:
            m = int(tokens[1].split("=")[1])
            k = int(tokens[2].split("=")[1])
            n = int(tokens[3].split("=")[1])
            minmax = float(tokens[5])
            quant = float(tokens[7])
            gemm = float(tokens[9])
            epi = float(tokens[11])
        except (IndexError, ValueError):
            continue

        rows[(m, k, n)].append((minmax, quant, gemm, epi))

    return rows


def main(path):
    rows = parse(path)
    if not rows:
        print("no [dqm] lines found")
        return

    total = [0.0, 0.0, 0.0, 0.0]
    print("    m     k     n  ops   minmax   quant    gemm     epi   (ms, min per shape)")
    for (m, k, n), samples in sorted(rows.items(), key=lambda item: -len(item[1])):
        best = [min(sample[i] for sample in samples) for i in range(4)]
        print(f"{m:5} {k:5} {n:5} {len(samples):4}   "
              f"{best[0]:6.3f}  {best[1]:6.3f}  {best[2]:6.3f}  {best[3]:6.3f}")
        for i in range(4):
            total[i] += best[i] * len(samples)

    print(f"TOTAL over {sum(len(v) for v in rows.values())} ops: "
          f"minmax {total[0]:.1f}  quant {total[1]:.1f}  gemm {total[2]:.1f}  epi {total[3]:.1f} ms")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".tmp/dqm_break.txt")
