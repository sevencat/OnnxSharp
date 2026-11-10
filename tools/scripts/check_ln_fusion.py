"""按 C# 的 FuseLayerNorm 条件逐条检查，打印拒绝原因分布。"""

import collections
import sys

import onnx

m = onnx.load(sys.argv[1] if len(sys.argv) > 1 else "encoder.onnx", load_external_data=False)
g = m.graph
prod = {o: n for n in g.node for o in n.output}
inits = {i.name for i in g.initializer}
cons = collections.defaultdict(list)
uses = collections.Counter()
for n in g.node:
    for v in n.input:
        cons[v].append(n)
        uses[v] += 1

reasons = collections.Counter()
one = lambda name, t: prod.get(name) if name in prod and prod[name].op_type == t else None

for tail in g.node:
    if tail.op_type != "Add" or len(tail.input) != 2 or len(tail.output) != 1:
        continue

    beta = next((v for v in tail.input if v in inits), None)
    if beta is None:
        reasons["no const beta"] += 1
        continue
    mul_arg = next(v for v in tail.input if v != beta)
    mul = one(mul_arg, "Mul")
    if mul is None:
        reasons["beta side not Mul"] += 1
        continue

    gamma = next((v for v in mul.input if v in inits), None)
    if gamma is None:
        reasons["no const gamma"] += 1
        continue
    div_arg = next(v for v in mul.input if v != gamma)
    div = one(div_arg, "Div")
    if div is None:
        reasons["gamma side not Div"] += 1
        continue

    sub_arg, sqrt_arg = div.input[0], div.input[1]
    sqrt = one(sqrt_arg, "Sqrt")
    if sqrt is None:
        reasons["div input1 not Sqrt"] += 1
        continue
    sub = one(sub_arg, "Sub")
    if sub is None:
        reasons["div input0 not Sub"] += 1
        continue

    eps_add = prod.get(sqrt.input[0])
    if eps_add is None or eps_add.op_type != "Add":
        reasons["sqrt input not Add"] += 1
        continue
    eps = next((v for v in eps_add.input if v in inits), None)
    if eps is None:
        reasons["no const eps"] += 1
        continue
    var = next(v for v in eps_add.input if v != eps)
    vmean = prod.get(var)
    if vmean is None or vmean.op_type != "ReduceMean":
        reasons["eps side not ReduceMean"] += 1
        continue

    pow_node = prod.get(vmean.input[0])
    if pow_node is None or pow_node.op_type not in ("Pow", "Mul"):
        reasons["var not Pow/Mul"] += 1
        continue
    if pow_node.op_type == "Pow" and pow_node.input[0] != sub_arg:
        reasons["Pow not of Sub"] += 1
        continue
    if pow_node.op_type == "Mul" and not (pow_node.input[0] == sub_arg == pow_node.input[1]):
        reasons["Mul not x*x"] += 1
        continue

    x = sub.input[0]
    xmean = prod.get(sub.input[1])
    if xmean is None or xmean.op_type != "ReduceMean" or xmean.input[0] != x:
        reasons["sub not x - mean(x)"] += 1
        continue

    if uses[mul_arg] != 1 or uses[div_arg] != 1 or uses[sqrt_arg] != 1 or uses[eps] != 1 \
            or uses[var] != 1 or uses[vmean.input[0]] != 1 or uses[xmean.output[0]] != 1:
        reasons["multi consumer"] += 1
        continue

    reasons["would fuse"] += 1

print("Add total:", sum(1 for n in g.node if n.op_type == "Add"))
for reason, count in reasons.most_common():
    print(f"  {reason:32s} {count}")
