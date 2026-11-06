# v9/engine/worker.py
"""v9.4 WorkerProc：每 GPU 一个子进程，内含 ModelRunner（vLLM v1 形态）。

对标 vLLM: v1/executor/multiproc_executor.py 的 WorkerProc.worker_main +
v1/worker/worker_base.py 的 WorkerWrapperBase。

进程拓扑:
    EngineCore 进程（scheduler + MQ 写端）
      └─ spawn N 个本进程（每 GPU 一个）
           主循环: MQ.recv(scheduler_output) → model_runner.run → response MQ.send

PP activation 由 PipelineTransport 管理；receive 在处理消息时懒投递，
worker 按消息顺序执行 forward；PP形状来自同一份调度消息。

与 v6.4 的"N 个全等进程 + NCCL broadcast"的区别：worker 进程里【没有
scheduler】，每步的 forward 输入完全来自广播的消息（用完即弃，不镜像
跨步状态——v6.2 字段漏镜像的坑在结构上消失）。
"""

import os


def worker_main(local_rank, model_path, broadcast_conn, response_conn,
                engine_kwargs, gpu_id):
    """WorkerProc 入口（multiprocessing.Process 的 target）。

    local_rank: 本EngineCore内的worker序号。
    所有并行坐标从唯一的ParallelTopology推导，不重复传world/EP大小。
    """
    topology = engine_kwargs["topology"]
    local_world_size = topology.local_world_size
    rank = engine_kwargs["dp_rank"] * local_world_size + local_rank
    dp_rank, pp_rank, tp_rank, cp_rank = topology.coords(rank)
    tp_size, pp_size, cp_size, dp_size = (
        topology.tp_size, topology.pp_size, topology.cp_size, topology.dp_size)
    world_size = topology.world_size
    enable_ep = topology.enable_expert_parallel
    debug = os.getenv("VLLM_LEARN_DEBUG", "0") == "1"
    # GPU ordinal is local to the inherited CUDA_VISIBLE_DEVICES list.
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = engine_kwargs.get("master_addr", "127.0.0.1")
    os.environ["MASTER_PORT"] = str(engine_kwargs.get("master_port", 29600))
    cpu_threads = max(1, int(os.getenv("VLLM_LEARN_CPU_THREADS", "4")))
    os.environ.setdefault("OMP_NUM_THREADS", str(cpu_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(cpu_threads))

    import torch
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", cpu_threads)))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    requested = int(gpu_id)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not 0 <= requested < torch.cuda.device_count():
        raise ValueError(f"rank {rank}: GPU ordinal {requested} outside visible devices {visible!r}")
    torch.cuda.set_device(requested)
    if debug:
        print(f"[worker-device] rank={rank} gpu_id={gpu_id} visible={visible!r}", flush=True)
    import torch.distributed as dist

    # v8.2 教训三件套：device_id 显式绑定 + 组预热 + 对称收发
    if world_size > 1:
        if debug:
            print(f"[worker-dist-before] rank={rank}", flush=True)
        dist.init_process_group(
            "nccl", device_id=torch.device(f"cuda:{torch.cuda.current_device()}"))
        if debug:
            print(f"[worker-dist-after] rank={rank}", flush=True)

    # Global rank = ((dp * pp_size + pp) * tp_size + tp) * cp_size + cp.
    # EP overlays DP x TP, independent of PP/CP.
    inner = tp_size * cp_size
    from layers.parallel import configure_parallel, STATE
    from engine.process_groups import create_groups
    groups = create_groups(rank, tp_size, cp_size, dp_size, pp_size, enable_ep)
    pp_recv_group, pp_send_group = None, None
    if pp_size > 1 and os.getenv("VLLM_LEARN_PP_P2P_GROUPS", "dedicated") == "dedicated":
        from engine.process_groups import create_pipeline_edge_groups
        pp_recv_group, pp_send_group = create_pipeline_edge_groups(rank, inner, pp_size, dp_size)
        groups.update(pp_recv=pp_recv_group, pp_send=pp_send_group)
    tp_group, ep_group, cp_group = (groups[k] for k in ("tp", "ep", "cp"))
    for group in groups.values():
        if group is not None:
            dist.all_reduce(torch.zeros(1, device="cuda"), group=group)
    torch.cuda.synchronize()
    configure_parallel(tp_size=tp_size, tp_rank=tp_rank,
                       tp_group=tp_group, ep_group=ep_group,
                       cp_size=cp_size, cp_rank=cp_rank,
                       pp_size=pp_size, pp_rank=pp_rank,
                       dp_size=dp_size, dp_rank=dp_rank,
                       enable_expert_parallel=enable_ep)

    # The transport owns bounded send buffers, not virtual engines. Queue
    # depth/backpressure lives in EngineCore + MultiprocessExecutor.
    if pp_size > 1:
        from engine.pipeline_transport import PipelineTransport
        transport_cls = engine_kwargs.get("transport_cls") or PipelineTransport
        STATE.pipeline_transport = transport_cls(
            pp_size=pp_size, pp_rank=pp_rank, inner_size=inner,
            credit=int(engine_kwargs.get("pipeline_credit", 4)),
            async_enabled=engine_kwargs.get("pipeline_mode", "optimized") != "baseline",
            device="cuda", recv_group=pp_recv_group, send_group=pp_send_group,
            rank_offset=dp_rank * local_world_size)

    # ---- DCP 注入（v8.2 的 set_dcp，worker 内生效）----
    if cp_size > 1:
        from layers.dcp import set_dcp
        set_dcp(active=True, size=cp_size, rank=cp_rank, group=cp_group)

    # ---- 模型（v9.3 组件: 构造即本 rank 切片）----
    from config import Config
    from engine.model_runner import ModelRunner
    if debug:
        print(f"[worker-config-before] rank={rank} pp={pp_rank} tp={tp_rank} ep={STATE.ep_rank}", flush=True)
    config = Config(model_path, **engine_kwargs.get("config_kwargs", {}))
    if debug:
        print(f"[worker-config-after] rank={rank}", flush=True)
    runner_cls = engine_kwargs.get("runner_cls") or ModelRunner
    runner = runner_cls(config, dtype=None)
    if debug:
        print(f"[worker-runner-after] rank={rank} blocks={config.num_kvcache_blocks}", flush=True)

    from engine.mq import SHUTDOWN
    # PP response 路由：采样在 last stage——只有 last stage 的 inner-rank0
    # 发 response（EngineCore 从它读），其余 worker 静默
    # A TP/EP worker group produces one logical result.  EngineCore reads
    # that result from rank 0 only; letting every rank send into its response
    # slot leaves the other slots without an ACK and blocks the next step.
    is_resp_rank = (pp_rank == pp_size - 1 and tp_rank == 0 and cp_rank == 0)
    # 每个 PP stage 都必须报告 ready。EngineCore 会等齐所有 stage 后再
    # 调度请求；否则 stage0 可能已经进入真实 forward，而 stage1 仍在
    # warmup，P2P mailbox 的消息顺序就会错位。
    # One readiness signal per EP replica (the first TP/CP rank in each
    # replica).  Waiting only for global rank 0 makes TP/EP/DP supervisors
    # hang forever when the other logical replicas never publish readiness.
    is_ready_rank = (tp_rank == 0 and cp_rank == 0)
    # ready 带块数（EngineCore 的 BlockManager 用它建——worker 与 master 的
    # 池子视图一致性 = 相同的 warmup+预算公式，构造序差异不再产生 0 块池）
    ready_sig = {"type": "ready", "rank": rank,
                 "num_kvcache_blocks": config.num_kvcache_blocks}
    if is_ready_rank:
        response_conn.send(ready_sig)
        if debug:
            print(f"[worker-ready] rank={rank} blocks={config.num_kvcache_blocks}", flush=True)

    response_pending = False
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
        if specs or (enable_ep and dp_size > 1):
            if debug:
                print(f"[worker-run] rank={rank} mb={msg.get('microbatch_id')}", flush=True)
            token_ids = runner.run(specs, is_prefill,
                                   microbatch_id=msg.get("microbatch_id", 0))
        if is_resp_rank:
            if response_pending:
                response_conn.wait_ack()
            response_conn.send({"type": "output", "tokens": token_ids,
                                "seq_ids": [s.seq_id for s in specs],
                                "microbatch_id": msg.get("microbatch_id", 0),
                                "n": len(specs)}, wait_ack=False)
            response_pending = True

    if response_pending:
        response_conn.wait_ack()
    runner.shutdown()
    if pp_size > 1 and STATE.pipeline_transport is not None:
        STATE.pipeline_transport.shutdown()
    if world_size > 1:
        torch.cuda.synchronize()  # Idle DP replies can precede device collective completion.
        dist.destroy_process_group()
    # Release the child-side pipes and shared-memory handles.  The parent
    # owns unlinking; closing here keeps multiprocessing.resource_tracker from
    # reporting leaked slots after every matrix configuration.
    broadcast_conn.close()
    response_conn.close()
