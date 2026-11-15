"""ORT 侧诊断：导出 ORT 优化后的图、对比算子直方图、可选逐算子 profile。

目的：解释 ONNXRuntime 在 RF-DETR 上比自研引擎快的「结构性原因」——
      哪些算子被融合掉、哪些权重变成预打包的 Gemm、耗时又集中在哪。

用法：
    python bench/ort_inspect.py --names rfdetr-small
    python bench/ort_inspect.py --names rfdetr-small --profile --iters 10

产物：bench/results/<name>.ort-opt.onnx（优化后的图，可用 onnx 工具继续看）
"""

import argparse
import collections
import gc
import json
import os
import re
import statistics
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

# 关注这些算子：融合是否命中，直接决定「逐算子开销 + 中间张量往返」有多少
WATCH = [
    "Gelu", "QuickGelu", "BiasGelu", "FastGelu", "Erf",
    "LayerNormalization", "SkipLayerNormalization", "SimplifiedLayerNormalization",
    "Attention", "MultiHeadAttention", "GroupQueryAttention",
    "Gemm", "MatMul", "FusedMatMul", "MatMulNBits",
    "Transpose", "Concat", "Reshape", "Softmax", "Add", "Mul", "Div", "Sub",
]


def load_models(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def histogram(proto):
    counter = collections.Counter()
    for node in proto.graph.node:
        counter[node.op_type] += 1
    return counter


def initializer_bytes(proto):
    total = 0
    for init in proto.graph.initializer:
        if init.raw_data:
            total += len(init.raw_data)
        else:
            n = 1
            for d in init.dims:
                n *= int(d)
            total += n * 4
    return total


def print_hist(title, counter):
    print(f"\n== {title} ==")
    print(f"   nodes: {sum(counter.values())}")
    width = max([len(k) for k in counter] + [8])
    for op, n in counter.most_common():
        mark = "  *" if op in WATCH else ""
        print(f"   {op:<{width}} {n:>5}{mark}")


def print_diff(before, after):
    added = {k: v for k, v in after.items() if k not in before}
    removed = {k: v for k, v in before.items() if k not in after}
    changed = {
        k: (before[k], after[k])
        for k in before
        if k in after and before[k] != after[k]
    }

    print("\n== ORT 图变换 diff（* 为关注算子）==")
    if added:
        print("   新增:", ", ".join(f"{k}x{v}" for k, v in sorted(added.items())))
    if removed:
        print("   消失:", ", ".join(f"{k}x{v}" for k, v in sorted(removed.items())))
    if changed:
        print(
            "   数量变化:",
            ", ".join(f"{k} {a}->{b}" for k, (a, b) in sorted(changed.items())),
        )
    if not (added or removed or changed):
        print("   （无变化）")


def make_feeds(sess):
    feeds = {}
    for inp in sess.get_inputs():
        shape = [d if isinstance(d, int) and d > 0 else 640 for d in inp.shape]
        feeds[inp.name] = np.full(shape, 0.5, dtype=np.float32)
    return feeds


def stats_ms(ms):
    ms = sorted(ms)
    n = len(ms)
    return {
        "iters": n,
        "min_ms": round(ms[0], 2),
        "median_ms": round(ms[int((n - 1) * 0.5)], 2),
        "mean_ms": round(statistics.fmean(ms), 2),
    }


def build_session(model_path, threads, opt_level="all", opt_out=None, profiling=False):
    import onnxruntime as ort

    levels = {
        "disable": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
        "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
        "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
        "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
    }

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = levels[opt_level]
    if opt_out:
        opts.optimized_model_filepath = opt_out
    if profiling:
        opts.enable_profiling = True
    return ort.InferenceSession(model_path, opts, providers=["CPUExecutionProvider"])


def time_session(sess, iters, warmup):
    feeds = make_feeds(sess)
    outs = [o.name for o in sess.get_outputs()]
    for _ in range(warmup):
        sess.run(outs, feeds)

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        sess.run(outs, feeds)
        times.append((time.perf_counter() - t0) * 1000.0)
    return stats_ms(times), times


def profile_ops(trace_path, name_to_type, iters, top=15):
    with open(trace_path, "r", encoding="utf-8") as f:
        events = json.load(f)

    by_type = collections.Counter()
    by_name = collections.Counter()

    def resolve(op_name):
        # trace 里的 op_name 有时就是节点名，有时直接是 OpType（不同版本/导出方式不同），
        # 所以三级回退：精确名字 -> 去掉 _<数字> 后缀 -> 原样当作类型
        if op_name in name_to_type:
            return name_to_type[op_name]
        stripped = re.sub(r"_\d+$", "", op_name)
        if stripped in name_to_type:
            return name_to_type[stripped]
        return stripped

    for e in events:
        if e.get("cat") != "Node" or "dur" not in e:
            continue
        op_name = e.get("args", {}).get("op_name", "")
        op_type = resolve(op_name)
        dur_ms = e["dur"] / 1000.0
        by_type[op_type] += dur_ms
        by_name[f"{op_name} [{op_type}]"] += dur_ms

    total = 0.0
    print(f"\n== ORT 逐算子耗时（profiling 会放大绝对值，只看占比；{iters} 次均值）==")
    print(f"   {'op_type':<32}{'ms/run':>9}{'share':>8}")
    # 先算总时长再打印占比
    total = sum(by_type.values())
    for op, dur in by_type.most_common(top):
        share = 100.0 * dur / total if total else 0.0
        print(f"   {op:<32}{dur / iters:>9.3f}{share:>7.1f}%")
    print(f"   {'(合计)':<32}{total / iters:>9.3f}{100.0:>7.1f}%")

    print("\n   -- 单个节点 top --")
    for name, dur in by_name.most_common(top):
        print(f"   {name:<56}{dur / iters:>9.3f}")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=os.path.join(HERE, "models.json"))
    ap.add_argument("--names", default="")
    # 默认 4 线程：16 线程满载会撞温度墙（% Processor Performance 掉到 21%），
    # 耗时会随机涨 5-7 倍。需要 16 线程口径时显式传 --threads 16。
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--profile", action="store_true", help="额外做一次逐算子 profile")
    ap.add_argument("--no-bench", action="store_true", help="只做图分析，不计时")
    ap.add_argument(
        "--compare-levels",
        action="store_true",
        help="依次跑 disable/basic/extended/all 四档优化，量化「图优化值多少」",
    )
    ap.add_argument("--rounds", type=int, default=2, help="优化等级对比的轮数（取各轮最好 min）")
    ap.add_argument(
        "--opt-level",
        default="all",
        choices=("disable", "basic", "extended", "all"),
        help="计时用 session 的图优化等级",
    )
    ap.add_argument(
        "--no-dump",
        action="store_true",
        help="跳过导图/直方图（同进程少建一个 session、少读两份 110MB 图，计时更干净）",
    )
    args = ap.parse_args()

    import onnx

    models = load_models(args.models)
    wanted = [s.strip() for s in args.names.split(",") if s.strip()]
    if wanted:
        models = [m for m in models if m["name"] in wanted]

    os.makedirs(RESULTS, exist_ok=True)

    for m in models:
        path = m["path"]
        if not os.path.exists(path):
            print(f"\n### {m['name']}: model not found ({path})")
            continue

        print(f"\n{'#' * 70}\n### {m['name']}  ({path})\n{'#' * 70}")

        if not args.no_dump:
            orig = onnx.load(path, load_external_data=False)
            hist_before = histogram(orig)
            print_hist("原始图（ONNX 导出结果）", hist_before)
            print(f"   initializer: {initializer_bytes(orig) / 1024 / 1024:.2f} MB")

            opt_path = os.path.join(RESULTS, f"{m['name']}.ort-opt.onnx")
            # 导图专用 session，导完立刻销毁：带 optimized_model_filepath 的 session
            # 实测推理明显更慢（同参数下 300-700ms vs 148ms），不能拿它计时
            dump_sess = build_session(path, args.threads, opt_out=opt_path)
            del dump_sess
            gc.collect()

            opt = onnx.load(opt_path, load_external_data=False)
            hist_after = histogram(opt)
            print_hist("ORT 优化后的图（ORT_ENABLE_ALL）", hist_after)
            print(f"   initializer: {initializer_bytes(opt) / 1024 / 1024:.2f} MB")
            print_diff(hist_before, hist_after)
            del orig, opt
            gc.collect()

        sess = build_session(path, args.threads, opt_level=args.opt_level)

        if not args.no_bench:
            feeds = make_feeds(sess)
            outs = [o.name for o in sess.get_outputs()]
            for _ in range(args.warmup):
                sess.run(outs, feeds)

            times = []
            for _ in range(args.iters):
                t0 = time.perf_counter()
                sess.run(outs, feeds)
                times.append((time.perf_counter() - t0) * 1000.0)

            print("\n== ORT CPU 计时（ms，同 bench/run_python_bench.py 口径）==")
            print("   " + json.dumps(stats_ms(times), ensure_ascii=False))

        if args.compare_levels:
            # 逐档独立 session、测完立即销毁：多个 session 各自持有一个 16 线程的池，
            # 空闲池的自旋等待会抢核，交错采样反而比背景负载更污染数据。
            # 先把上一段用的 session 释放掉，否则它那份 packed 权重会和被测 session 抢页/带宽
            del sess
            gc.collect()

            print("\n== 优化等级对比（逐档独立 session，取各轮最好 min） ==")
            levels = ("disable", "basic", "extended", "all")
            best = {}
            for _ in range(args.rounds):
                for lv in levels:
                    s = build_session(path, args.threads, opt_level=lv)
                    st, _ = time_session(s, args.iters, args.warmup)
                    prev = best.get(lv)
                    if prev is None or st["min_ms"] < prev["min_ms"]:
                        best[lv] = st
                    del s
                    gc.collect()

            baseline = best["disable"]["min_ms"]
            for lv in levels:
                st = best[lv]
                speedup = baseline / st["min_ms"] if st["min_ms"] else 0.0
                print(
                    f"   {lv:<9} min {st['min_ms']:>8.2f}  median {st['median_ms']:>8.2f}"
                    f"  mean {st['mean_ms']:>8.2f}  x{speedup:>5.2f}"
                )

        if args.profile:
            # 单独建一个开着 profiling 的会话，只跑 1 次：trace 有逐节点记事件的开销，
            # 混进计时会失真；单次 run 的占比足以看结构。
            ptrace_model = os.path.join(RESULTS, f"{m['name']}.ort-prof.onnx")
            psess = build_session(path, args.threads, opt_out=ptrace_model, profiling=True)
            pfeeds = make_feeds(psess)
            pouts = [o.name for o in psess.get_outputs()]
            psess.run(pouts, pfeeds)
            trace = psess.end_profiling()
            prof_model = onnx.load(ptrace_model, load_external_data=False)
            name_to_type = {n.name: n.op_type for n in prof_model.graph.node}
            profile_ops(trace, name_to_type, 1)
            os.remove(trace)
            os.remove(ptrace_model)
            del psess


if __name__ == "__main__":
    sys.exit(main())
