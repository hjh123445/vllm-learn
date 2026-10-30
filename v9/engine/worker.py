# v9/engine/worker.py
"""v9.4 WorkerProc：每 GPU 一个子进程，内含 ModelRunner（vLLM v1 形态）。

对标 vLLM: v1/executor/multiproc_executor.py 的 WorkerProc.worker_main +
v1/worker/worker_base.py 的 WorkerWrapperBase。

进程拓扑（VLLM_ARCH.md §2）:
    EngineCore 进程（scheduler + MQ 写端）
      └─ spawn N 个本进程（每 GPU 一个）
           主循环: MQ.recv(scheduler_output) → model_runner.run → response MQ.send

与 v6.4 的"N 个全等进程 + NCCL broadcast"的区别：worker 进程里【没有
scheduler】，每步的 forward 输入完全来自广播的消息（用完即弃，不镜像
跨步状态——v6.2 字段漏镜像的坑在结构上消失）。
"""

import os


def worker_main(rank, world_size, tp_size, ep_size, model_path,
                broadcast_conn, response_conn, engine_kwargs):
    """WorkerProc 入口（multiprocessing.Process 的 target）。

    rank:      全局 worker 序号（0..world_size-1）
    tp/ep:     本 EngineCore 内的并行坐标（ParallelCoords 的 (tp,ep) 笛卡尔积）
    broadcast_q/response_q: 引擎侧传来的 Queue（mp.Queue 可 pickle 跨进程）
    """
    # 设备由 launcher 通过 CUDA_VISIBLE_DEVICES 控制（每 worker 只见一张卡,
    # spawn 继承父进程 env → EngineCore 在 spawn 前为每个 worker 设置子进程 env）
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = engine_kwargs.get("master_addr", "127.0.0.1")
    os.environ["MASTER_PORT"] = str(engine_kwargs.get("master_port", 29600))

    import torch
    torch.cuda.set_device(0)   # CUDA_VISIBLE_DEVICES 已隔离 → 逻辑 0 即本卡
    import torch.distributed as dist

    # v8.2 教训三件套：device_id 显式绑定 + 组预热 + 对称收发
    if world_size > 1:
        dist.init_process_group(
            "nccl", device_id=torch.device(f"cuda:{torch.cuda.current_device()}"))

    # ---- 并行坐标（构造模型之前注入——v9.3 组件化的约定）----
    # 坐标: rank = ep_rank * tp_size + tp_rank（tp 连续段同属一个 ep 组）
    tp_rank, ep_rank = rank % tp_size, rank // tp_size
    from layers.parallel import configure_parallel, STATE
    tp_group = moe_group = None
    if world_size > 1:
        # tp 组 =【同 ep_rank】的所有 rank（rank // tp_size == ep_rank）——
        # v9.4a 第一版写成 r % tp_size == tp_rank → TP2 时是单成员组，
        # allreduce 变 no-op，部分和流进 norm（e2e 从 tok0 全错的根因）
        tp_group = dist.new_group(ranks=[r for r in range(world_size)
                                         if r // tp_size == ep_rank])
        if tp_size * ep_size > 1:
            moe_group = dist.new_group(ranks=list(range(world_size)))
        # 组预热（惰性 wire-up 竞态——v8.2 的 25 分钟死锁）
        for g in (tp_group, moe_group):
            if g is not None:
                t = torch.zeros(1, device="cuda")
                dist.all_reduce(t, group=g)
        torch.cuda.synchronize()
    configure_parallel(tp_size=tp_size, tp_rank=tp_rank,
                       ep_size=ep_size, ep_rank=ep_rank,
                       tp_group=tp_group, moe_group=moe_group)

    # ---- DCP 注入（v8.2 的 set_dcp，worker 内生效）----
    dcp_size = engine_kwargs.get("dcp_size", 1)
    if dcp_size > 1:
        from layers.dcp import set_dcp
        # cp 维坐标 = rank // tp_size（与 v8.2 的 tp*cp+cp 编码对齐）
        cp_rank = rank % dcp_size if dcp_size == tp_size else rank // tp_size
        cp_group = dist.new_group(ranks=[r for r in range(world_size)
                                         if (r // tp_size) == (rank // tp_size)])
        t = torch.zeros(1, device="cuda"); dist.all_reduce(t, group=cp_group)
        set_dcp(active=True, size=dcp_size, rank=cp_rank, group=cp_group)

    # ---- 模型（v9.3 组件: 构造即本 rank 切片）----
    from config import Config
    from engine.model_runner import ModelRunner
    config = Config(model_path, **engine_kwargs.get("config_kwargs", {}))
    runner = ModelRunner(config, dtype=None)

    from engine.mq import SHUTDOWN
    # ready 带块数（EngineCore 的 BlockManager 用它建——worker 与 master 的
    # 池子视图一致性 = 相同的 warmup+预算公式，构造序差异不再产生 0 块池）
    ready_sig = {"type": "ready", "rank": rank,
                 "num_kvcache_blocks": config.num_kvcache_blocks}
    response_conn.send(ready_sig)

    while True:
        msg = broadcast_conn.recv()
        if msg is SHUTDOWN or msg.get("type") == "shutdown":
            break
        # scheduler_output → forward 输入（用完即弃的快照）
        from engine.core import rebuild_specs
        specs = rebuild_specs(msg)
        is_prefill = msg["is_prefill"]
        token_ids = []
        if specs:
            token_ids = runner.run(specs, is_prefill)
        response_conn.send({"type": "output", "tokens": token_ids,
                            "n": len(specs)})

    if world_size > 1:
        dist.destroy_process_group()
