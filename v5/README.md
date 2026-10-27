# v5：在线推理

v4 的引擎 + **异步化 + HTTP 流式服务**。请求随到随入：客户端提交后立刻拿到流式队列，
引擎后台线程持续调度生成，每步把新 token 推进各请求的队列——这就是"在线"
区别于批处理的本质行为。

## 快速开始

```bash
cd v5 && python server.py --port 8000 --gpu-mem-util 0.3 --max-num-seqs 32
# 另一个终端：
curl http://127.0.0.1:8000/v1/chat/completions -d '{
  "messages": [{"role": "user", "content": "你好"}], "stream": true, "max_tokens": 64}'   # SSE 流式
python ../benchmark/serve_bench.py --url http://127.0.0.1:8000 --num-requests 64 --arrive-interval 0.02   # 压测
```

## 代码结构

```
server.py               stdlib HTTP 服务（零额外依赖）：OpenAI 兼容接口 + SSE + 优雅退出
engine/async_engine.py  异步引擎：后台循环线程 + 每请求流式队列
engine/ 其余             = v4（nano 调度栈 + 手写 triton kernel 引擎，批量接口不变）
```

## 主要技术原理与实现

### 线程模型（单卡单进程内）

```
HTTP 处理线程（多个）                     引擎循环线程（唯一写者，daemon）
─────────────────────                    ─────────────────────
tokenize → async_engine.add()            被 add() 的 wake 事件唤醒
        ↓ 立刻返回                        schedule → run → postprocess
拿到 queue.Queue                          → 把每序列的新 token 推进它的 queue
从 queue 逐个取 token → SSE 写出          空转时 wait 让出 CPU
```

**线程安全的依据**（不加锁）：scheduler 的 `waiting` 是 deque，`append`/`popleft`
在 CPython 里原子；block_manager 和 Sequence 状态**只有循环线程写**；
HTTP 线程只通过 `add()` 往 waiting 投放数据——**单写多读无竞态**。

### 增量文本（`server.py`）

BPE 字节级切词，**单 token decode 不安全**（可能切出半个字），所以每个新 token 到达后
**全量 decode 取差量**作为 delta 推给客户端。

### SSE 流式

`Transfer-Encoding: chunked` 手写分块（stdlib http.server 不自动处理）：
每个 chunk 是 `data: {...}\n\n`，按 http chunked 协议加长度前缀；结束发 `data: [DONE]` + `0\r\n\r\n`。

### 优雅退出

SIGTERM/SIGINT 都注册了 handler → `engine.stop()`（join 引擎线程）→ `srv.shutdown()`
（**必须从别的线程调**——signal handler 跑在 serve_forever 同一个主线程里）→ `server_close()`。
实测：`kill -TERM <真实pid>` → 日志三步 → 显存全释放。
已知边界：在途请求不 drain（连接直接断），vLLM 会 drain。

## 与 v4 的差异

引擎 = v4 原样；v5 只是异步化和协议层。批量接口（`example.py`）不变。

## 实测（serve_bench.py，同客户端同口径）

- 4 路并发流式：TTFT 16-23ms，全部 1.15s 完成
- 运行中 1s 后到达的新请求：TTFT 24ms 立即准入（有空闲 slot 即插）
- 128 请求 × 128 tok（20ms 到达流）：TPOT 38.8ms / 1800 tok/s / TTFT 20.6ms
  （vs vLLM 0.31 graph：2.8ms / 5551 tok/s——差距全部来自 CUDA graph，见 PERFORMANCE_NOTES §9）

## 已知限制

- 在途请求不 drain；没有请求取消（客户端断连请求会生成完才被驱逐）
- GIL 下 HTTP 线程的 tokenize 会偶发抢走 GIL 抬高 TPOT（decode 本来就是 CPU 发射瓶颈）
- stdlib http.server 的 accept 队列已调 256，但**瞬时 SYN 风暴**（64+ 同时建连）仍可能 reset——
  稳态满压（20ms 到达流）0 错误；生产应换 asyncio/独立引擎进程（vLLM AsyncLLMEngine 的做法）
