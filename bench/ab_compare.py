"""A/B 逐值比对：同一模型跑两次前向、把输出转储成 F32，逐元素比对。

`bench/results/` 已被 `.gitignore` 忽略，所以这个脚本放在 `bench/` 下（会进版本库），
它是 `planv2.md §1.2` 里「任何改动都要 max|diff| = 0」的执行工具。

用法：
    # 自检：同一份二进制跑两次，结果必须逐位一致（max|diff| = 0）
    python bench/ab_compare.py --names rfdetr-small

    # 改代码后：比较两次转储目录（改动前 dump 到 base/，改动后 dump 到 new/）
    python bench/ab_compare.py --a bench/ab/base --b bench/ab/new

    # 保留转储现场（默认成功后删除临时目录）
    python bench/ab_compare.py --names yolo11n --keep

退出码：0 = 全部逐位一致；1 = 存在差异；2 = 用法/运行错误。
"""

import argparse
import array
import filecmp
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
INPUTS = os.path.join(HERE, "inputs")
OURS_CLI = os.path.join(ROOT, "tools", "OnnxSharp.Cli", "bin", "Release", "net10.0", "OnnxSharp.Cli.exe")


def load_models(models_path):
    with open(models_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def ensure_input(model_path):
    """与 run_all.py 同口径：按静态输入形状生成全 0.5 的 FP32 输入（缓存复用）。"""
    import numpy as np
    import onnx

    proto = onnx.load(model_path, load_external_data=False)
    dims = [d.dim_value if d.dim_value > 0 else 1 for d in proto.graph.input[0].type.tensor_type.shape.dim]
    while len(dims) < 4:
        dims.append(1)
    n, c, h, w = dims[:4]
    os.makedirs(INPUTS, exist_ok=True)
    path = os.path.join(INPUTS, f"i{n}x{c}x{h}x{w}.f32")
    if not os.path.exists(path):
        np.full(n * c * h * w, 0.5, dtype=np.float32).tofile(path)
    return path


def dump(model_path, out_dir):
    """跑一次前向，输出写成 out0.f32 / out1.f32 …"""
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)
    inp = ensure_input(model_path)

    proc = subprocess.run(
        [OURS_CLI, "run", model_path, "--input", inp, "--dump", out_dir],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout).strip().splitlines()
        raise RuntimeError(f"run 失败（exit {proc.returncode}）：{tail[-1] if tail else 'no output'}")

    files = sorted(f for f in os.listdir(out_dir) if f.startswith("out") and f.endswith(".f32"))
    if not files:
        raise RuntimeError(f"{out_dir} 里没有 out*.f32，dump 没生效？")
    return files


def read_f32(path):
    buf = array.array("f")
    with open(path, "rb") as fh:
        buf.fromfile(fh, os.path.getsize(path) // 4)
    return buf


def compare_pair(a_path, b_path):
    """返回 (max_abs, max_rel, first_diff_index, nan_mismatch, bit_identical)。"""
    if os.path.getsize(a_path) != os.path.getsize(b_path):
        return None
    if filecmp.cmp(a_path, b_path, shallow=False):
        return 0.0, 0.0, -1, 0, True

    a, b = read_f32(a_path), read_f32(b_path)
    max_abs = max_rel = 0.0
    first = -1
    nan_mismatch = 0
    for i, (x, y) in enumerate(zip(a, b)):
        if x != x or y != y:  # 差值会是 NaN，单独计数
            if (x != x) != (y != y):
                nan_mismatch += 1
            continue
        d = abs(x - y)
        if d:
            if first < 0:
                first = i
            if d > max_abs:
                max_abs = d
            denom = max(abs(x), abs(y))
            if denom and d / denom > max_rel:
                max_rel = d / denom
    return max_abs, max_rel, first, nan_mismatch, False


def report(name, a_dir, b_dir):
    a_files = sorted(f for f in os.listdir(a_dir) if f.endswith(".f32"))
    b_files = sorted(f for f in os.listdir(b_dir) if f.endswith(".f32"))
    if a_files != b_files:
        print(f"[FAIL] {name}: 输出文件集合不同 A={a_files} B={b_files}")
        return False

    ok = True
    for fname in a_files:
        a_path = os.path.join(a_dir, fname)
        b_path = os.path.join(b_dir, fname)
        res = compare_pair(a_path, b_path)
        if res is None:
            print(f"[FAIL] {name}/{fname}: 字节数不同 {os.path.getsize(a_path)} vs {os.path.getsize(b_path)}")
            ok = False
            continue
        max_abs, max_rel, first, nan_mismatch, identical = res
        if identical:
            print(f"[ OK ] {name}/{fname}: 逐位一致")
            continue
        problems = []
        if max_abs > 0:
            problems.append(f"max|diff|={max_abs:.6g} max_rel={max_rel:.3g} 首个差异@{first}")
        if nan_mismatch:
            problems.append(f"NaN 错配={nan_mismatch}")
        if problems:
            ok = False
            print(f"[FAIL] {name}/{fname}: " + "，".join(problems))
        else:
            print(f"[ OK ] {name}/{fname}: 非逐位一致但所有元素相等（仅 ±0/-0 或 NaN 位型差异）")
    return ok


def main():
    ap = argparse.ArgumentParser(description="OnnxSharp A/B 逐值比对（max|diff| 必须为 0）")
    ap.add_argument("--names", default="", help="逗号分隔的模型名（默认全部）")
    ap.add_argument("--models", default=os.path.join(HERE, "models.json"))
    ap.add_argument("--a", default="", help="改动前的 dump 目录；不传则现场跑 baseline")
    ap.add_argument("--b", default="", help="改动后的 dump 目录；不传则现场跑第二次")
    ap.add_argument("--keep", action="store_true", help="保留现场转储（默认成功后删除）")
    args = ap.parse_args()

    if (args.a and not args.b) or (args.b and not args.a):
        ap.error("--a 与 --b 必须同时给出")
    if not os.path.exists(OURS_CLI):
        print(f"找不到 {OURS_CLI}，先 dotnet build -c Release")
        return 2

    models = load_models(args.models)
    by_name = {m["name"]: m for m in models}
    names = [n.strip() for n in args.names.split(",") if n.strip()] or [m["name"] for m in models]
    unknown = [n for n in names if n not in by_name]
    if unknown:
        ap.error(f"未知模型: {unknown}；可选: {sorted(by_name)}")

    all_ok = True
    tmp_root = None
    try:
        if args.a and args.b:
            pairs = [(n, os.path.join(args.a, n), os.path.join(args.b, n)) for n in names]
        else:
            if args.keep:
                base = os.path.abspath(os.path.join(HERE, "ab"))
                os.makedirs(base, exist_ok=True)
                tmp_root = tempfile.mkdtemp(prefix="run_", dir=base)
            else:
                tmp_root = tempfile.mkdtemp(prefix="ab_")
            pairs = []
            for name in names:
                model_path = by_name[name]["path"]
                if not os.path.exists(model_path):
                    print(f"[SKIP] {name}: 模型不存在 {model_path}")
                    continue
                a_dir = os.path.join(tmp_root, "A", name)
                b_dir = os.path.join(tmp_root, "B", name)
                print(f"[dump] {name} A ...", flush=True)
                dump(model_path, a_dir)
                print(f"[dump] {name} B ...", flush=True)
                dump(model_path, b_dir)
                pairs.append((name, a_dir, b_dir))
            if tmp_root:
                print(f"[tmp ] {tmp_root}")

        for name, a_dir, b_dir in pairs:
            if not os.path.isdir(a_dir) or not os.path.isdir(b_dir):
                print(f"[SKIP] {name}: 目录缺失 A={a_dir} B={b_dir}")
                continue
            all_ok &= report(name, a_dir, b_dir)
    except Exception as exc:  # noqa: BLE001
        print(f"[ERR ] {type(exc).__name__}: {exc}")
        return 2
    finally:
        if tmp_root and os.path.isdir(tmp_root) and not args.keep:
            shutil.rmtree(tmp_root, ignore_errors=True)

    print()
    print("结果：" + ("全部逐位一致" if all_ok else "存在差异，改动不可接受"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
