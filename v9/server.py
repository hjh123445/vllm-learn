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
import signal
import sys
import threading
import time
import uuid
import math
import queue
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine.async_engine import AsyncEngine, EngineOverloaded
from samplingParams import SamplingParams


class Handler(BaseHTTPRequestHandler):
    engine: AsyncEngine = None
    tokenizer = None
    protocol_version = "HTTP/1.1"      # keep-alive
    request_timeout = 300.0
    max_body_bytes = 1024 * 1024
    max_output_tokens = 4096
    metrics_path = None
    metrics_lock = threading.Lock()

    def setup(self):
        super().setup()
        self.connection.settimeout(120)

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
            stats = self.engine.stats()
            self._json(200 if stats['accepting'] else 503, dict(status="ok" if stats['accepting'] else "unavailable", **stats))
        else:
            self._json(404, {"error": "not found"})

    # ---------------- POST /v1/chat/completions ----------------

    def do_POST(self):
        started = time.perf_counter()
        if self.path != "/v1/chat/completions":
            self._json(404, {"error": "not found"})
            return
        rid, st, request_id = "chatcmpl-" + uuid.uuid4().hex[:12], None, None
        stream_started = False
        record = dict(id=rid, start=started, status='error')
        try:
            if self.headers.get('Transfer-Encoding'):
                raise ValueError('chunked request bodies are unsupported')
            n = int(self.headers.get("Content-Length", 0))
            if not 0 < n <= self.max_body_bytes:
                raise ValueError('request body missing or exceeds size limit')
            req = json.loads(self.rfile.read(n))
            if not isinstance(req, dict):
                raise ValueError('body must be a JSON object')
            messages = req.get('messages')
            if not isinstance(messages, list) or not messages:
                raise ValueError('messages required')
            if any(not isinstance(m, dict) or not isinstance(m.get('content'), str)
                   or m.get('role') not in ('system','user','assistant') for m in messages):
                raise ValueError('this endpoint currently supports text messages only')
            stream = req.get('stream', False)
            limit = req.get('max_tokens', 128)
            temperature = float(req.get('temperature', .7))
            if type(stream) is not bool or type(limit) is not int or not 1 <= limit <= self.max_output_tokens:
                raise ValueError('invalid stream or max_tokens')
            if not math.isfinite(temperature) or temperature < 0:
                raise ValueError('invalid temperature')
            if req.get('n', 1) != 1 or req.get('stop') or 'top_p' in req:
                raise ValueError('n>1, stop strings and top_p are not implemented')
            record['parsed'] = time.perf_counter()
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            record['templated'] = time.perf_counter()
            prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
            record['tokenized'] = time.perf_counter()
            record['prompt_tokens'] = len(prompt_ids)
            if self.metrics_path: record['input_ids'] = prompt_ids
            request_id, st = self.engine.add(prompt_ids, SamplingParams(temperature=temperature,max_tokens=limit))
            record['submitted'] = time.perf_counter()

            def send(obj):
                data = obj if isinstance(obj,str) else json.dumps(obj,ensure_ascii=False)
                payload = ('data: ' + data + '\n\n').encode()
                self.wfile.write(f"{len(payload):x}\r\n".encode() + payload + b"\r\n")
                self.wfile.flush()
            def chunk(delta, reason=None):
                return dict(id=rid,object='chat.completion.chunk',choices=[dict(index=0,delta=delta,finish_reason=reason)])
            if stream:
                self.send_response(200)
                self.send_header('Content-Type','text/event-stream')
                self.send_header('Cache-Control','no-cache')
                self.send_header('Transfer-Encoding','chunked')
                self.end_headers()
                stream_started = True
                send(chunk(dict(role='assistant')))
            ids, emitted = [], ''
            deadline = started + self.request_timeout
            while True:
                remaining = deadline-time.perf_counter()
                if remaining <= 0:
                    raise TimeoutError('request deadline exceeded')
                try:
                    token = st.q.get(timeout=min(1., remaining))
                except queue.Empty:
                    continue
                if token is None:
                    if st.error:
                        raise st.error
                    break
                record.setdefault('first_token_consumed',time.perf_counter())
                ids.append(token)
                if stream:
                    full = self.tokenizer.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False).rstrip('\ufffd')
                    if not full.startswith(emitted):
                        raise RuntimeError('tokenizer revised an already emitted text prefix')
                    delta=full[len(emitted):]
                    if delta:
                        send(chunk(dict(content=delta)))
                        record.setdefault('first_content_flushed',time.perf_counter())
                        emitted=full
            text=self.tokenizer.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)
            if stream:
                if not text.startswith(emitted):
                    raise RuntimeError('unstable terminal text prefix')
                if len(text)>len(emitted):send(chunk(dict(content=text[len(emitted):])))
                send(chunk({},st.finish_reason))
                send('[DONE]')
                self.wfile.write(b'0\r\n\r\n');self.wfile.flush()
            else:
                self._json(200,dict(id=rid,object='chat.completion',choices=[dict(index=0,
                    message=dict(role='assistant',content=text),finish_reason=st.finish_reason)],
                    usage=dict(prompt_tokens=len(prompt_ids),completion_tokens=len(ids),total_tokens=len(prompt_ids)+len(ids))))
            record.update(status='ok',completion_tokens=len(ids))
            if self.metrics_path: record['output_ids'] = ids
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            record['status']='disconnected'
        except Exception as exc:
            record['error']=str(exc)
            code=429 if isinstance(exc,EngineOverloaded) else (400 if isinstance(exc,(ValueError,TypeError)) else (504 if isinstance(exc,TimeoutError) else 500))
            record['http_status']=code
            try:
                if stream_started:
                    send(dict(error=dict(message=str(exc),code=code)))
                    send('[DONE]')
                    self.wfile.write(b'0\r\n\r\n');self.wfile.flush()
                else:
                    self.close_connection=True
                    self._json(code,dict(error=dict(message=str(exc),code=code)))
            except OSError: pass
        finally:
            if request_id is not None:self.engine.abort(request_id)
            record['finished']=time.perf_counter()
            if st is not None:record['engine_times']=dict(st.times)
            if self.metrics_path:
                with self.metrics_lock:
                    with open(self.metrics_path,'a',encoding='utf-8') as f:
                        f.write(json.dumps(record,ensure_ascii=False)+'\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "Qwen3-0.6B"))
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--max-num-seqs", type=int, default=8)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-mem-util", type=float, default=0.3)
    ap.add_argument('--max-num-batched-tokens',type=int,default=2048)
    ap.add_argument('--max-kvcache-blocks',type=int,default=None)
    ap.add_argument('--disable-prefix-caching',action='store_true')
    ap.add_argument('--batch-execution-mode', choices=('batched', 'per_request'),
                    default='batched',
                    help='per_request preserves single-request numerical shapes at a throughput cost')
    ap.add_argument('--max-pending-requests',type=int,default=1024)
    ap.add_argument('--stream-buffer-tokens',type=int,default=256)
    ap.add_argument('--request-timeout',type=float,default=300.)
    ap.add_argument('--metrics-path',default=None)
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--expert-parallel-size", type=int, default=1)
    ap.add_argument("--pipeline-parallel-size", type=int, default=1)
    ap.add_argument("--pipeline-microbatches", type=int, default=1,
                    help="legacy fixed-count schedule is unsupported; use token budget")
    ap.add_argument("--zero-bubble", action="store_true",
                    help="reserved; training DualPipe is not implemented by this inference runtime")
    ap.add_argument("--pipeline-microbatch-tokens", type=int, default=2048)
    ap.add_argument("--gpus", default=None,
                    help="comma-separated physical GPU ids, e.g. 0,1,2,3")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    template_path = os.path.join(args.model, 'chat_template.json')
    if os.path.isfile(template_path):
        with open(template_path,encoding='utf-8') as f:
            tok.chat_template = json.load(f)['chat_template']
    gpus = ([int(x) for x in args.gpus.split(",") if x.strip()]
            if args.gpus else None)
    engine = AsyncEngine(
        args.model, max_num_seqs=args.max_num_seqs,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem_util, enforce_eager=True,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_kvcache_blocks=args.max_kvcache_blocks,enable_prefix_caching=not args.disable_prefix_caching,
        batch_execution_mode=args.batch_execution_mode,
        max_pending_requests=args.max_pending_requests,stream_buffer_tokens=args.stream_buffer_tokens,
        tensor_parallel_size=args.tensor_parallel_size,
        expert_parallel_size=args.expert_parallel_size,
        pipeline_parallel_size=args.pipeline_parallel_size,
        pipeline_microbatches=args.pipeline_microbatches,
        pipeline_schedule=("zero_bubble" if args.zero_bubble else "forward"),
        pipeline_microbatch_tokens=args.pipeline_microbatch_tokens,
        gpus=gpus)
    engine.start()

    Handler.engine = engine
    Handler.tokenizer = tok
    Handler.request_timeout=args.request_timeout
    Handler.metrics_path=args.metrics_path
    ThreadingHTTPServer.request_queue_size = 256   # 64+ 并发同时建连不打爆 accept 队列（默认 5）
    srv = ThreadingHTTPServer((args.host, args.port), Handler)

    # 优雅退出：SIGTERM/SIGINT 都走清理路径（不注册的话 SIGTERM 直接击杀进程，
    # finally 不执行、在途连接被 RST）。srv.shutdown 必须从别的线程调
    # （signal handler 跑在 serve_forever 同一个主线程里，直接调会死锁）。
    def graceful_shutdown(signum, frame):
        print(f"[server] 收到 {signal.Signals(signum).name}，正在关闭……", flush=True)
        engine.stop()
        threading.Thread(target=srv.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, graceful_shutdown)
    signal.signal(signal.SIGINT, graceful_shutdown)

    print(f"[server] http://{args.host}:{args.port}/v1/chat/completions  (model={os.path.basename(args.model)})")
    try:
        srv.serve_forever()
    finally:
        engine.stop()                      # 幂等：stop 走两遍安全
        srv.server_close()
        print("[server] 已退出", flush=True)


if __name__ == "__main__":
    main()
