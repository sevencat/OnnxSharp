"""按 C# 融合器的匹配条件逐条检查 ONNX 图，打印每条拒绝原因（定位为什么没融合）。"""

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

for mm in g.node:
    if mm.op_type != "MatMulInteger" or len(mm.input) < 2 or len(mm.output) != 1:
        continue

    a = prod.get(mm.input[0])
    if a is None or a.op_type != "DynamicQuantizeLinear":
        reasons["A not DQL"] += 1
        continue
    if len(a.output) < 3:
        reasons["dql outputs<3"] += 1
        continue
    if uses[a.output[0]] != 1:
        reasons[f"y uses={uses[a.output[0]]}"] += 1
        continue
    if uses[a.output[1]] != 1:
        reasons[f"scale uses={uses[a.output[1]]}"] += 1
        continue
    if uses[a.output[2]] != 1:
        reasons[f"zp uses={uses[a.output[2]]}"] += 1
        continue
    if len(mm.input) > 2 and mm.input[2] and mm.input[2] != a.output[2]:
        reasons["a_zp mismatch"] += 1
        continue
    if len(mm.input) > 3 and mm.input[3] and mm.input[3] not in inits:
        reasons["w_zp not const"] += 1
        continue

    scale_consumers = cons.get(a.output[1], [])
    if len(scale_consumers) != 1 or scale_consumers[0].op_type != "Mul":
        reasons["scale consumer not Mul"] += 1
        continue
    scale_mul = scale_consumers[0]

    if uses[mm.output[0]] != 1:
        reasons[f"acc uses={uses[mm.output[0]]}"] += 1
        continue
    cast = cons.get(mm.output[0], [None])[0]
    if cast is None or cast.op_type != "Cast":
        reasons["acc consumer not Cast"] += 1
        continue

    if uses[cast.output[0]] != 1:
        reasons[f"cast uses={uses[cast.output[0]]}"] += 1
        continue
    mul = cons.get(cast.output[0], [None])[0]
    if mul is None or mul.op_type != "Mul":
        reasons[f"cast consumer not Mul ({mul.op_type if mul else '?'})"] += 1
        continue
    if scale_mul.output[0] not in mul.input:
        reasons["mul not using combined scale"] += 1
        continue

    reasons["would fuse"] += 1

print("MatMulInteger total:", sum(1 for n in g.node if n.op_type == "MatMulInteger"))
for reason, count in reasons.most_common():
    print(f"  {reason:40s} {count}")
