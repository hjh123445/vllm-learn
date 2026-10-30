# v9/engine/mq.py
"""v9.4 消息通道：EngineCore <-> WorkerProc（vLLM v1 形态）。

对标 vLLM v1 的 shm_broadcast.MessageQueue（1 写者 N 读者 + 每 worker 一条
response 通道）。本版用 mp.Pipe 实现同语义（pickle 的结构化消息，每条
自带边界）。

为什么不用 mp.Queue：节点 56 实测 mp.Queue + spawn 组合子进程 SIGSEGV
（exitcode -11、无 traceback；Pipe 正常）——环境级问题（Queue 的 feeder
线程在 spawn bootstrap 时段错误），Pipe 规避。vLLM 的零拷贝 shm 广播
（tensor 直接共享内存）是后续优化。

与 v6.4 固定张量 NCCL 广播的本质区别（VLLM_ARCH.md 0）：
- NCCL 方案要求所有 rank 形状逐字节一致（变长=错位死锁，哨兵=歧义）；
- 消息通道传 Python 对象，每条消息独立、自带边界——EngineCore 主进程
  存在后这是正确形态（vLLM 同款）。
"""

import multiprocessing as mp

# 关停哨兵（显式对象身份，与任何合法消息不等）
SHUTDOWN = {"type": "shutdown"}


def make_channels(world_size: int):
    """每 worker 一对全双工 Pipe。

    返回 (bc, rc):
      bc[r] = (parent_conn, child_conn)  # EngineCore -> worker r（调度下发）
      rc[r] = (parent_conn, child_conn)  # worker r -> EngineCore（ready/output）
    spawn 时 child 端随 args 传给 worker 进程（Pipe 可 pickle）。"""
    bc = [mp.Pipe() for _ in range(world_size)]
    rc = [mp.Pipe() for _ in range(world_size)]
    return bc, rc


def broadcast(msg, bc_parents):
    """EngineCore 侧：同一消息发给每个 worker。"""
    for conn in bc_parents:
        conn.send(msg)
