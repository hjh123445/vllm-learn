# benchmark/serve_bench.py
"""通用流式压测：对任意 OpenAI 兼容的 /v1/chat/completions 端点打流式负载。

通用性：不关心对面是谁——我们的 v5 server、vLLM serve、任何兼容端点，
同一个客户端、同一套计时口径。

每个请求测什么（全部从 SSE 流的时间戳得出）：
  TTFT  请求发出 → 首个内容 chunk 到达
  TPOT  (末 chunk − 首 chunk) / (chunk 数 − 1)
  E2E   请求发出 → [DONE]
负载从什么来：与 harness.py 同一套固定问题集（保证可复现、可与其它测试对比）。

用法:
  python serve_bench.py --url http://127.0.0.1:8000 --num-requests 64 --arrive-interval 0.05
输出: 终端表格 + results/serve_bench_<tag>.json
"""
import argparse
import json
import os
import sys
import threading
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from harness import QUESTIONS  # noqa: E402  同一套负载问题集


def one_request(url: str, prompt: str, max_tokens: int, temperature: float, out: dict, idx: int):
    """发一条流式请求，把 (ttft, tpot, e2e, ntok) 记进 out[idx]。"""
    t0 = time.perf_counter()
    body = json.dumps({"messages": [{"role": "user", "content": prompt}],
                       "stream": True, "max_tokens": max_tokens,
                       "temperature": temperature}).encode()
    req = urllib.request.Request(url + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    ttft, ntok, last_t = None, 0, t0
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            for raw in r:
                line = raw.strip()
                if not line.startswith(b"data: "):
                    continue
                p = line[6:]
                if p == b"[DONE]":
                    break
                d = json.loads(p)
                if d["choices"][0]["delta"].get("content"):
                    now = time.perf_counter()
                    if ttft is None:
                        ttft = now - t0
                    last_t = now
                    ntok += 1
        out[idx] = {"ttft": ttft, "e2e": time.perf_counter() - t0,
                    "tpot": (last_t - t0 - ttft) / max(ntok - 1, 1) if ntok > 1 and ttft else 0.0,
                    "ntok": ntok}
    except Exception as e:
        out[idx] = {"error": str(e)}


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(int(len(xs) * p), len(xs) - 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True, help="服务地址，如 http://127.0.0.1:8000")
    ap.add_argument("--tag", default=None, help="结果文件名标签（默认取 url 端口）")
    ap.add_argument("--num-requests", type=int, default=64)
    ap.add_argument("--arrive-interval", type=float, default=0.05,
                    help="请求到达间隔（秒）。0 = 全部同时炸过去（最大压力）")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()

    out = {}
    threads = []
    t_start = time.perf_counter()
    for i in range(args.num_requests):
        th = threading.Thread(target=one_request,
                               args=(args.url, QUESTIONS[i % len(QUESTIONS)],
                                     args.max_tokens, args.temperature, out, i))
        th.start()
        threads.append(th)
        if args.arrive_interval > 0 and i < args.num_requests - 1:
            time.sleep(args.arrive_interval)
    for th in threads:
        th.join()
    wall = time.perf_counter() - t_start

    errors = [v for v in out.values() if "error" in v]
    ok = [v for v in out.values() if "error" not in v]
    ttft = [v["ttft"] * 1e3 for v in ok]
    tpot = [v["tpot"] * 1e3 for v in ok]
    e2e = [v["e2e"] * 1e3 for v in ok]
    total_tok = sum(v["ntok"] for v in ok)

    r = {
        "url": args.url, "tag": args.tag or args.url.split(":")[-1],
        "error_samples": [v["error"] for v in errors[:3]],
        "num_requests": args.num_requests,
        "arrive_interval_s": args.arrive_interval,
        "max_tokens": args.max_tokens, "temperature": args.temperature,
        "wall_s": round(wall, 2),
        "errors": len(errors),
        "ttft_avg_ms": round(sum(ttft) / len(ttft), 1) if ttft else None,
        "ttft_p50_ms": round(pct(ttft, 0.5), 1) if ttft else None,
        "ttft_p95_ms": round(pct(ttft, 0.95), 1) if ttft else None,
        "ttft_p99_ms": round(pct(ttft, 0.99), 1) if ttft else None,
        "tpot_avg_ms": round(sum(tpot) / len(tpot), 1) if tpot else None,
        "e2e_p50_ms": round(pct(e2e, 0.5), 1) if e2e else None,
        "e2e_max_ms": round(max(e2e), 1) if e2e else None,
        "total_out_tokens": total_tok,
        "out_tok_per_s": round(total_tok / wall, 1),
    }
    print(json.dumps(r, ensure_ascii=False, indent=1))
    out_path = os.path.join(HERE, "results", f"serve_bench_{r['tag']}_{args.num_requests}req.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(r, f, ensure_ascii=False, indent=1)
    print(f"saved to {out_path}")


if __name__ == "__main__":
    main()
