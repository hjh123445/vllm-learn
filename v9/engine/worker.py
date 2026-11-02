# v9/engine/worker.py
"""v9.4 WorkerProc：每 GPU 一个子进程，内含 ModelRunner（vLLM v1 形态）。

对标 vLLM: v1/executor/multiproc_executor.py 的 WorkerProc.worker_main +
v1/worker/worker_base.py 的 WorkerWrapperBase。

进程拓扑（VLLM_ARCH.md §2）:
    EngineCore 进程（scheduler + MQ 写端）
      └─ spawn N 个本进程（每 GPU 一个）
           主循环: MQ.recv(scheduler_output) → model_runner.run → response MQ.send

PP stage 内的 activation 由 PipelineTransport 独立管理：两个 lane 预投递
irecv，worker 仍按 scheduler 消息顺序执行模型 forward。

与 v6.4 的"N 个全等进程 + NCCL broadcast"的区别：worker 进程里【没有
scheduler】，每步的 forward 输入完全来自广播的消息（用完即弃，不镜像
跨步状态——v6.2 字段漏镜像的坑在结构上消失）。
"""

import os


def worker_main(rank, world_size, tp_size, ep_size, model_path,
                broadcast_conn, response_conn, engine_kwargs, gpu_id=None):
    """WorkerProc 入口（multiprocessing.Process 的 target）。

    rank:      全局 worker 序号（0..world_size-1）
    tp/ep:     本 EngineCore 内的并行坐标（ParallelCoords 的 (tp,ep) 笛卡尔积）
    broadcast_q/response_q: 引擎侧传来的 Queue（mp.Queue 可 pickle 跨进程）
    """
    debug = os.getenv("VLLM_LEARN_DEBUG", "0") == "1"
    # 设备由 launcher 通过 CUDA_VISIBLE_DEVICES 控制（每 worker 只见一张卡,
    # spawn 继承父进程 env → EngineCore 在 spawn 前为每个 worker 设置子进程 env）
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = engine_kwargs.get("master_addr", "127.0.0.1")
    os.environ["MASTER_PORT"] = str(engine_kwargs.get("master_port", 29600))

    import torch
    # Normally the wrapper restricts CUDA_VISIBLE_DEVICES to one card.  The
    # explicit physical id is a fallback for spawn implementations that import
    # the parent environment before the wrapper runs; this prevents two ranks
    # silently binding the same default device.
    requested = int(gpu_id) if gpu_id is not None else 0
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if "," not in visible and visible.strip() and requested >= torch.cuda.device_count():
        requested = 0
    torch.cuda.set_device(requested if requested < torch.cuda.device_count() else 0)
    print(f"[worker-device] rank={rank} gpu_id={gpu_id} visible={visible!r} "
          f"count={torch.cuda.device_count()} current={torch.cuda.current_device()}",
          flush=True)
    import torch.distributed as dist

    # v8.2 教训三件套：device_id 显式绑定 + 组预热 + 对称收发
    if world_size > 1:
        dist.init_process_group(
            "nccl", device_id=torch.device(f"cuda:{torch.cuda.current_device()}"))

    # ---- 四维坐标（row-major 解码，对齐 ParallelCoords DIMS=(tp,cp,ep,pp)）----
    # rank = pp*(tp*cp*ep) + ep*(tp*cp) + tp*cp + cp
    cp_size = engine_kwargs.get("dcp_size", 1)
    pp_size = engine_kwargs.get("pp_size", 1)
    inner = tp_size * cp_size * ep_size
    pp_rank = rank // inner
    r = rank % inner
    ep_rank = r // (tp_size * cp_size)
    r = r % (tp_size * cp_size)
    tp_rank = r // cp_size
    cp_rank = r % cp_size
    from layers.parallel import configure_parallel, STATE
    tp_group = ep_group = moe_group = cp_group = None
    if world_size > 1:
        # tp 组 = 同 (pp, ep, cp) 沿 tp 变化——v9.4a 曾写成 r%tp_size==tp_rank
        # → 单成员组 → allreduce 变 no-op（e2e 全错的根因，勿回退）
        if tp_size > 1:
            tp_group = dist.new_group(ranks=[
                pp_rank * inner + ep_rank * tp_size * cp_size + t * cp_size + cp_rank
                for t in range(tp_size)])
        if ep_size > 1:
            ep_group = dist.new_group(ranks=[
                pp_rank * inner + e * tp_size * cp_size + tp_rank * cp_size + cp_rank
                for e in range(ep_size)])
        # moe 组 = 同 (pp, cp) 的 tp×ep 张成（MoE 部分和聚合域）
        if tp_size * ep_size > 1:
            moe_group = dist.new_group(ranks=[
                pp_rank * inner + e * tp_size * cp_size + t * cp_size + cp_rank
                for e in range(ep_size) for t in range(tp_size)])
        # cp 组 = 同 (pp, ep, tp) 沿 cp 变化（DCP partial/LSE 合并域）
        if cp_size > 1:
            cp_group = dist.new_group(ranks=[
                pp_rank * inner + ep_rank * tp_size * cp_size + tp_rank * cp_size + c
                for c in range(cp_size)])
        # 组预热（惰性 wire-up 竞态——v8.2 的 25 分钟死锁）
        for g in (tp_group, ep_group, moe_group, cp_group):
            if g is not None:
                t = torch.zeros(1, device="cuda")
                dist.all_reduce(t, group=g)
        torch.cuda.synchronize()
    configure_parallel(tp_size=tp_size, tp_rank=tp_rank,
                       ep_size=ep_size, ep_rank=ep_rank,
                       tp_group=tp_group, ep_group=ep_group, moe_group=moe_group,
                       cp_size=cp_size, cp_rank=cp_rank,
                       pp_size=pp_size, pp_rank=pp_rank)

    # Adjacent PP stages use two pre-posted activation lanes.  This object is
    # created before the first scheduler message, so stage>0 is already
    # listening when stage0 issues its first asynchronous send.
    if pp_size > 1:
        from engine.pipeline_transport import PipelineTransport
        STATE.pipeline_transport = PipelineTransport(
            pp_size=pp_size, pp_rank=pp_rank, inner_size=inner,
            lanes=int(engine_kwargs.get("pipeline_lanes", 2)),
            credit=int(engine_kwargs.get("pipeline_credit", 4)),
            device="cuda")

    # ---- DCP 注入（v8.2 的 set_dcp，worker 内生效）----
    if cp_size > 1:
        from layers.dcp import set_dcp
        set_dcp(active=True, size=cp_size, rank=cp_rank, group=cp_group)

    # ---- 模型（v9.3 组件: 构造即本 rank 切片）----
    from config import Config
    from engine.model_runner import ModelRunner
    config = Config(model_path, **engine_kwargs.get("config_kwargs", {}))
    runner = ModelRunner(config, dtype=None)

    from engine.mq import SHUTDOWN
    # PP response 路由：采样在 last stage——只有 last stage 的 inner-rank0
    # 发 response（EngineCore 从它读），其余 worker 静默
    is_resp_rank = (pp_rank == pp_size - 1 and rank % inner == 0) \
        if pp_size > 1 else True
    # PP 时 ready 只由 stage0 发（core 的 ready 循环只收 stage0 的 rank0——
    # last stage 若也发，其 ready 会残留在 response 管道里，step 的第一次
    # recv 收到的是它而不是 output → KeyError 'tokens'）
    is_ready_rank = (pp_rank == 0 and rank % inner == 0) \
        if pp_size > 1 else True
    # ready 带块数（EngineCore 的 BlockManager 用它建——worker 与 master 的
    # 池子视图一致性 = 相同的 warmup+预算公式，构造序差异不再产生 0 块池）
    ready_sig = {"type": "ready", "rank": rank,
                 "num_kvcache_blocks": config.num_kvcache_blocks}
    if is_ready_rank:
        response_conn.send(ready_sig)
        if debug:
            print(f"[worker-ready] rank={rank} blocks={config.num_kvcache_blocks}", flush=True)

    while True:
        msg = broadcast_conn.recv()
        if msg is SHUTDOWN or msg.get("type") == "shutdown":
            break
        if debug:
            print(f"[worker-recv] rank={rank} mb={msg.get('microbatch_id')} n={len(msg.get('scheduled', []))}", flush=True)
        # scheduler_output → forward 输入（用完即弃的快照）
        from engine.core import rebuild_specs
        specs = rebuild_specs(msg)
        is_prefill = msg["is_prefill"]
        token_ids = []
        if specs:
            if debug:
                print(f"[worker-run] rank={rank} mb={msg.get('microbatch_id')}", flush=True)
            token_ids = runner.run(specs, is_prefill,
                                   pipeline_lane=msg.get("pipeline_lane", 0),
                                   microbatch_id=msg.get("microbatch_id", 0))
            if debug:
                print(f"[worker-done] rank={rank} mb={msg.get('microbatch_id')}", flush=True)
        if is_resp_rank:
            response_conn.send({"type": "output", "tokens": token_ids,
                                "seq_ids": [s.seq_id for s in specs],
                                "microbatch_id": msg.get("microbatch_id", 0),
                                "n": len(specs)})

    if pp_size > 1 and STATE.pipeline_transport is not None:
        STATE.pipeline_transport.drain()
    if world_size > 1:
        dist.destroy_process_group()
