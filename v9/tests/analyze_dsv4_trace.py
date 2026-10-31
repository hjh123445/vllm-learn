"""分析 /tmp/ours_dsv4_trace.json（v6.3 DSV4 8 卡 decode 8 步的 torch profiler trace）。"""
import json, collections, sys

tr = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "/tmp/ours_dsv4_trace.json"))
ev = [e for e in tr["traceEvents"] if e.get("ph") == "X" and e.get("cat") == "cpu_op"]
ev.sort(key=lambda e: e["ts"])
t0, t1 = ev[0]["ts"], max(e["ts"] + e["dur"] for e in ev)
print(f"窗口 {(t1-t0)/1e6:.2f}s / 8 步 = {(t1-t0)/8e3:.0f}ms/步, {len(ev)} ops")
c = collections.Counter(e["name"] for e in ev)
print("每步 op 数 top:")
for n, k in c.most_common(10):
    print(f"  {k//8:6d}/步 {n}")
to_ops = [e for e in ev if e["name"] in ("aten::to", "aten::_to_copy")]
print(f"跨卡搬运 aten::to*: {len(to_ops)} 次, {sum(e['dur'] for e in to_ops)/1e3:.0f}ms 总")
