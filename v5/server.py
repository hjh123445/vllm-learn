# v5/server.py
"""v5 在线推理服务：stdlib HTTP + OpenAI 兼容接口（零额外依赖）。

  GET  /health
  POST /v1/chat/completions   {"messages": [...], "stream": true/false,
                                "max_tokens": 128, "temperature": 0.7}

流式（stream=true）按 OpenAI 的 SSE 格式逐 token 推送：
  data: {"choices": [{"delta": {"content": "..."}}]}
  ...
  data: [DONE]

用法: python server.py --model ../Qwen3-0.6B --port 8000 --gpu-mem-util 0.3
"""
import argparse
import json
import os
import sys
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine.async_engine import AsyncEngine
from samplingParams import SamplingParams


class Handler(BaseHTTPRequestHandler):
    engine: AsyncEngine = None
    tokenizer = None
    protocol_version = "HTTP/1.1"      # keep-alive

    def log_message(self, fmt, *args):  # 静默访问日志
        pass

    # ---------------- helpers ----------------

    def _json(self, code, obj):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # ---------------- GET /health ----------------

    def do_GET(self):
        if self.path == "/health":
            self._json(200, {"status": "ok", "running": not self.engine.engine.is_finished()})
        else:
            self._json(404, {"error": "not found"})

    # ---------------- POST /v1/chat/completions ----------------

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self._json(404, {"error": "not found"})
            return
        n = int(self.headers.get("Content-Length", 0))
        try:
            req = json.loads(self.rfile.read(n))
        except Exception as e:
            self._json(400, {"error": f"bad json: {e}"})
            return
        messages = req.get("messages")
        if not messages:
            self._json(400, {"error": "messages required"})
            return
        stream = bool(req.get("stream", False))
        sp = SamplingParams(
            temperature=float(req.get("temperature", 0.7)),
            max_tokens=int(req.get("max_tokens", 128)),
        )
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False,
                                                    add_generation_prompt=True)
        rid = "chatcmpl-" + uuid.uuid4().hex[:12]
        t0 = time.perf_counter()
        _, st = self.engine.add(prompt, sp)

        if stream:
            # SSE + chunked 传输
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()

            def send(obj):
                payload = (b"data: " + json.dumps(obj, ensure_ascii=False).encode() + b"\n\n")
                self.wfile.write(f"{len(payload):x}\r\n".encode() + payload + b"\r\n")
                self.wfile.flush()

            send({"id": rid, "object": "chat.completion.chunk",
                  "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]})
            ids, prev_text, first_t = [], "", None
            while True:
                t = st.q.get()
                if t is None:
                    break
                if first_t is None:
                    first_t = time.perf_counter() - t0      # 首 token 到达
                ids.append(t)
                # 增量文本 = 全量 decode 的差量（BPE 字节级切词，单 token decode 不安全）
                full = self.tokenizer.decode(ids, skip_special_tokens=True)
                delta = full[len(prev_text):]
                prev_text = full
                if delta:
                    send({"id": rid, "object": "chat.completion.chunk",
                          "choices": [{"index": 0, "delta": {"content": delta}, "finish_reason": None}]})
            send({"id": rid, "object": "chat.completion.chunk",
                  "choices": [{"index": 0, "delta": {},
                               "finish_reason": st.finish_reason or "stop"}]})
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
            print(f"[server] {rid} stream done: {len(ids)} tok, "
                  f"ttft {first_t*1e3:.0f}ms, total {time.perf_counter()-t0:.2f}s", flush=True)
        else:
            toks = []
            while True:
                t = st.q.get()
                if t is None:
                    break
                toks.append(t)
            text = self.tokenizer.decode(toks, skip_special_tokens=True)
            self._json(200, {"id": rid, "object": "chat.completion",
                             "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                                           "finish_reason": st.finish_reason or "stop"}],
                             "usage": {"completion_tokens": len(toks)}})
            print(f"[server] {rid} done: {len(toks)} tok, {time.perf_counter()-t0:.2f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "Qwen3-0.6B"))
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--max-num-seqs", type=int, default=8)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-mem-util", type=float, default=0.3)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    engine = AsyncEngine(args.model, max_num_seqs=args.max_num_seqs,
                         max_model_len=args.max_model_len,
                         gpu_memory_utilization=args.gpu_mem_util, enforce_eager=True)
    engine.start()

    Handler.engine = engine
    Handler.tokenizer = tok
    ThreadingHTTPServer.request_queue_size = 256   # 64+ 并发同时建连不打爆 accept 队列（默认 5）
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[server] http://{args.host}:{args.port}/v1/chat/completions  (model={os.path.basename(args.model)})")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        engine.stop()


if __name__ == "__main__":
    main()
