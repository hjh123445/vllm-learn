# v6/layers/moe.py
"""MoE 路由、本地专家权重与变长 EP dispatch/combine。

设计（对齐我们"平铺批 + context"的架构）：
  - 输入是平铺的 [total_q, hidden]（与 attention 同一约定）；
  - router：softmax(fp32) → topk → normalize（HF Qwen3MoeTopKRouter 逐行同数学）；
  - experts：按 expert_idx 排序 token，Python 循环调用各非空专家的两个 GEMM；
    尚不是单次 fused grouped GEMM。w13 为 [E_local, 2*I_local, H]，
    w2 为 [E_local, H, I_local]。

EP 使用 DP×TP 派生 ep_group 做变长 dispatch/combine；模型层只负责路由和本地
grouped expert，通信协议仍由组件封装，避免 model_runner 了解专家布局。"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


def _reduce_routes(routes, expert_ids):
    """Reduce returned top-k routes; the HF order is a diagnostic oracle."""
    if os.getenv('VLLM_LEARN_MOE_REDUCTION', 'fp32') == 'hf_reference':
        order = expert_ids.argsort(dim=-1)
        routes = routes.gather(1, order[..., None].expand_as(routes))
        output = torch.zeros_like(routes[:, 0])
        for slot in range(routes.shape[1]):
            output = output + routes[:, slot]
        return output
    return routes.float().sum(dim=1).to(routes.dtype)


class MoERouter(nn.Module):
    """模型 dtype 的 logits、FP32 softmax 与 top-k；支持显式 FP32 消融。"""

    def __init__(self, hidden_size: int, num_experts: int, top_k: int,
                 norm_topk_prob: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_experts, hidden_size))
        self.top_k = top_k
        self.norm_topk_prob = norm_topk_prob
        self.num_experts = num_experts
        # HF computes logits in model dtype before FP32 softmax. An explicit
        # FP32 logits policy remains available for numerical ablations.
        self.logits_dtype = os.getenv("VLLM_LEARN_ROUTER_LOGITS_DTYPE", "model")
        if self.logits_dtype not in ("float32", "model"):
            raise ValueError("router logits dtype must be float32 or model")

    def weight_loader(self, param, loaded, shard_id=None):
        """HF 名 mlp.gate.weight（loader 已映射到 router.weight）。直装。"""
        assert param.shape == loaded.shape
        param.data.copy_(loaded)

    def forward(self, x: torch.Tensor):
        """x: [T, H] → (topk_idx [T,k], topk_w [T,k])。"""
        logits = (F.linear(x, self.weight) if self.logits_dtype == "model"
                  else F.linear(x.float(), self.weight.float()))
        probs = logits.softmax(dim=-1, dtype=torch.float32)
        w, idx = probs.topk(self.top_k, dim=-1)
        if self.norm_topk_prob:
            w = w / w.sum(dim=-1, keepdim=True)
        return idx, w.to(x.dtype)


class GroupedExperts(nn.Module):
    """128 个专家的分组计算。权重三块：
      w13: [E, 2*I, H]  gate|up 融合（一个 gemm 出两半）
      w2:  [E, H, I]    down
    forward 按 expert 分组排序后 gather → 两个 grouped gemm → scatter 加权回去。"""

    def __init__(self, num_experts: int, hidden_size: int, intermediate: int):
        super().__init__()
        from layers.parallel import STATE
        self.hidden = hidden_size
        # ---- EP: 本地专家区间 ----
        per = num_experts // STATE.ep_size
        assert per * STATE.ep_size == num_experts, \
            f"E={num_experts} 切不动 EP{STATE.ep_size}"
        self.e_start = STATE.ep_rank * per
        self.num_experts = per                    # 本地专家数（E_local）
        # EP replaces expert tensor sharding: each owner holds WHOLE experts.
        expert_tp = 1 if STATE.enable_expert_parallel else STATE.tp_size
        assert intermediate % expert_tp == 0, f"I={intermediate} cannot shard over TP{expert_tp}"
        self.inter = intermediate // expert_tp
        self.i_start = 0 if STATE.enable_expert_parallel else STATE.tp_rank * self.inter
        self.quantized = False
        # Opt-in native grouped GEMM: two grouped API calls for local experts,
        # without copying weights per route or reading expert counts on CPU.
        # The backend determines the actual number of CUDA kernel launches.
        # Keep the validated reference loop as the default.  ``auto`` is an
        # explicit performance mode selected by the benchmark/serving policy.
        self.backend = os.getenv('VLLM_LEARN_EXPERT_BACKEND', 'loop')
        if self.backend not in ('auto', 'loop', 'torch_grouped'):
            raise ValueError('expert backend must be auto, loop, or torch_grouped')
        self.w13 = nn.Parameter(torch.empty(self.num_experts, 2 * self.inter, hidden_size))
        self.w2 = nn.Parameter(torch.empty(self.num_experts, hidden_size, self.inter))

    def quantize_int4(self, group: int = 128):
        """在线量化：装载完 bf16 权重后调用，就地转 int4 打包 buffer。
        权重显存 -75%（专家占模型 ~80%，是量化收益主体）。"""
        from layers.quantized import rtn_quantize
        E, _, H = self.w13.shape
        I = self.w2.shape[-1]
        p13, s13 = rtn_quantize(self.w13.data.float().reshape(E * 2 * I, H), group)
        p2, s2 = rtn_quantize(self.w2.data.float().reshape(E * H, I), group)
        # 平铺 2D 存储避免 reshape 维度混乱（packed 维 = 最后一维 K/2）
        self.register_buffer("w13_packed", p13)            # [E*2I, H/2]
        self.register_buffer("w13_scales", s13)            # [E*2I, H/g]
        self.register_buffer("w2_packed", p2)              # [E*H, I/2]
        self.register_buffer("w2_scales", s2)              # [E*H, I/g]
        self.group = group
        del self.w13, self.w2                      # 释放 bf16 权重
        self.quantized = True

    def weight_loader(self, param, loaded, shard_id=None):
        """weight_loader 协议（与 BaseLinear/RmsNorm 同约定）：
        loader 做名字映射后调 param 所属模块的 weight_loader，shard_id 携带
        融合信息——(expert_idx, half) 装进 w13 的对应半区，(expert_idx, None) 装 w2。

        并行处理（v9.3）：loaded 是【全量单专家】张量，组件内部切好再装——
          非本地专家（EP 区间外）→ 跳过
          w13: gate/up 半区里各取 inter 的本 rank 行段；w2: 列段
        TP 切缝与 gate|up 边界对齐（I % tp==0 保证不跨缝）；
        loader 完全不感知并行。"""
        i, half = shard_id
        if not (self.e_start <= i < self.e_start + self.num_experts):
            return                                     # EP: 非本地专家
        li = i - self.e_start                          # 本地专家下标
        s, e = self.i_start, self.i_start + self.inter
        if param is self.w13:
            # loaded 是 gate 或 up 的【半张量】[I_full, H]（loader 的 _ckpt_name
            # 已按 gate/up 拆 shard）——直接取本 rank 的 inter 行段
            param.data[li, half * self.inter:(half + 1) * self.inter, :] = loaded[s:e, :]
        else:
            param.data[li] = loaded[:, s:e]


    def forward(self, x: torch.Tensor, topk_idx: torch.Tensor, topk_w: torch.Tensor) -> torch.Tensor:
        """x: [T, H]; topk_idx: [T, k]; topk_w: [T, k] → [T, H]。
        按专家分组：每专家一次 [n_e, H] gemm（激活只含该专家的 token，
        权重不复制——朴素 einsum 会把专家权重按 token 复制 T*k 份直接爆显存）。
        counts 的 CPU 同步每层一次（v6.1 优化：预分配分段表）。"""
        T, H = x.shape
        k = topk_idx.shape[1]
        E = self.num_experts

        # (T, k) → 按 expert 分组的 (token, weight) 列表
        flat_expert = topk_idx.reshape(-1)
        flat_token = torch.arange(T, device=x.device).repeat_interleave(k)
        flat_w = topk_w.reshape(-1)

        route_out = torch.empty((T * k, H), device=x.device, dtype=x.dtype)
        counts = torch.bincount(flat_expert, minlength=E)
        offsets = counts.cumsum(0)                     # 每个专家在排序后序列的终点
        # 稳定排序：同一专家的 token 连续排布
        order = torch.argsort(flat_expert, stable=True)
        sorted_token = flat_token[order]
        sorted_w = flat_w[order]

        xs = x[sorted_token]
        out = self.compute_sorted(xs, offsets)
        route_out[order] = out * sorted_w[:, None].to(out.dtype)
        return _reduce_routes(route_out.view(T, k, H), topk_idx)

    def compute_sorted(self, xs, offsets, offsets_host=None):
        """Compute expert-contiguous routes using local weights only.

        ``offsets[e]`` is the exclusive end of expert e, including empty
        experts. Native grouped GEMM keeps this prefix sum on the GPU.
        Explicit selection fails on unsupported platforms instead of silently
        falling back to the Python loop and mislabelling measurements.
        ``offsets_host``: caller-supplied host endpoints — skips the
        ``offsets.tolist()`` D2H sync in the loop backend.
        """
        if self.quantized:
            from layers.quantized import dequantize
            g = self.group
            # 平铺 [E*2I, H/2] → 按需 dequant（本版全量；v9.2 kernel 化后按 tile）
            w13 = dequantize(self.w13_packed, self.w13_scales, g).reshape(self.num_experts, -1, self.hidden).to(xs.dtype)
            w2 = dequantize(self.w2_packed, self.w2_scales, g).reshape(self.num_experts, self.hidden, self.inter).to(xs.dtype)
        else:
            w13, w2 = self.w13, self.w2
        if xs.shape[0] == 0:
            return xs.new_empty((0, self.hidden))
        backend = self.resolve_backend(xs)
        return self._compute_sorted_backend(xs, offsets, w13, w2, backend,
                                            offsets_host)

    def resolve_backend(self, xs):
        """Select the grouped kernel for supported CUDA BF16 inputs.

        Explicit ``torch_grouped`` stays strict for reproducible ablations;
        ``auto`` falls back to the readable loop on unsupported platforms.
        """
        if self.backend != 'auto':
            return self.backend
        grouped_mm = getattr(F, 'grouped_mm', None) or getattr(torch, '_grouped_mm', None)
        if xs.is_cuda and xs.dtype == torch.bfloat16 and grouped_mm is not None:
            return 'torch_grouped'
        return 'loop'

    def _compute_sorted_backend(self, xs, offsets, w13, w2, backend,
                                offsets_host=None):
        if backend == 'torch_grouped':
            grouped_mm = getattr(F, 'grouped_mm', None) or getattr(torch, '_grouped_mm', None)
            if grouped_mm is None or not xs.is_cuda or xs.dtype != torch.bfloat16:
                raise RuntimeError('torch_grouped requires CUDA BF16 and a compatible native grouped_mm')
            offs = offsets.to(dtype=torch.int32)
            h13 = grouped_mm(xs, w13.transpose(-2, -1), offs=offs)
            gate, up = h13.chunk(2, dim=-1)
            act = F.silu(gate) * up
            return grouped_mm(act, w2.transpose(-2, -1), offs=offs)
        output = xs.new_empty((xs.shape[0], self.hidden))
        start = 0
        ends = offsets_host if offsets_host is not None else offsets.tolist()
        for e, end in enumerate(ends):
            if end == start:                           # 本批没 token 命中该专家
                continue
            h13 = xs[start:end] @ w13[e].t()           # [n_e, 2I]
            gate, up = h13.chunk(2, dim=-1)
            act = F.silu(gate) * up                    # [n_e, I]
            out = act @ w2[e].t()                      # [n_e, H]
            output[start:end] = out
            start = end
        return output


class MoEMLP(nn.Module):
    """一层完整 MoE：router + experts。无共享专家（Qwen3-Omni 配置为 0）。

    EP开启时，Attention TP输出在副本内复制；先按TP分配token ownership，
    再跨DP×TP的专家池分发。owner持完整专家，combine后仅在原TP组恢复。
    EP关闭时保留专家intermediate TP与partial all-reduce。
    """

    def __init__(self, config):
        super().__init__()
        # Routing probabilities are already known by the source. Keeping the
        # BF16 multiply there removes one metadata collective per MoE layer.
        self.weight_at_source = os.getenv('VLLM_LEARN_EP_WEIGHT_AT_SOURCE', '0') == '1'
        # BF16 GEMM algorithms depend on M. Routing an ownership slice can
        # round logits differently from the identical replicated attention
        # batch, changing top-k weights and occasionally expert selection.
        # Default to the same router shape as EP1; local routing is opt-in.
        self.local_router = os.getenv('VLLM_LEARN_EP_LOCAL_ROUTER', '0') == '1'
        # Experimental ownership-aware fast path (see _ep_forward_fast):
        # counts provider + packed dispatch payload + source-side weighting
        # -> counts allgather + dispatch + combine (3 collectives/layer,
        # down from 5 in _ep_forward_alltoallv) and ONE D2H per layer.
        self.fastmeta = os.getenv('VLLM_LEARN_EP_FASTMETA', '0') == '1'
        # Optional second-communicator pipeline: parity-split routes,
        # dispatch(B) overlaps experts(A), combine(A) overlaps experts(B).
        self.overlap = (self.fastmeta
                        and os.getenv('VLLM_LEARN_EP_OVERLAP', '0') == '1')
        # Counts provider. 'exchange' (DP x TP coupled form, upstream
        # ownership topology): local bincount over OWNED routes + one tiny
        # packed allgather + one D2H — valid for ANY per-rank input batch,
        # including empty DP replicas. 'analytic': only valid when dp==1 and
        # the router ran on the full replicated batch (default), because the
        # ownership split is then derivable locally; anything else falls back
        # to exchange. Equivalence on identical inputs is token-level
        # checkable.
        self.meta_mode = os.getenv('VLLM_LEARN_EP_META', 'exchange')
        self._ov_streams = None
        self._rowmap_cache = {}
        self.router = MoERouter(config.hidden_size, config.num_experts,
                                config.num_experts_per_tok,
                                getattr(config, "norm_topk_prob", True))
        self.experts = GroupedExperts(config.num_experts, config.hidden_size,
                                      config.moe_intermediate_size)
        # v9.3: 并行状态在 GroupedExperts 内部（构造时从 ParallelState 读）,
        # MoEMLP 只负责 router + 本地计算 + 部分和聚合

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from layers.parallel import STATE, tp_allreduce
        if STATE.enable_expert_parallel and STATE.ep_size > 1:
            return self._forward_tp_replicated_input(x)
        idx, w = self.router(x)
        return tp_allreduce(self.experts(x, idx, w))

    def _forward_tp_replicated_input(self, x):
        """TP token ownership -> DP x TP all-to-allv -> TP output restore.

        Only TP peers have identical inputs; DP batches may differ or be empty.
        Every route is evaluated once by a whole-expert owner. No expert TP
        partial sum exists in this mode. Restore the original TP-replicated
        attention input layout WITHOUT mixing different DP requests.
        """
        from layers.parallel import STATE
        import torch.distributed as dist
        tokens, hidden = x.shape
        # Attention output is replicated only WITHIN this DP replica's TP
        # group. Other DP replicas have independent token counts and contents.
        rows = (tokens + STATE.tp_size - 1) // STATE.tp_size
        start = STATE.tp_rank * rows
        local_x = x[start:min(start + rows, tokens)]
        if self.local_router:
            ids, weights = self.router(local_x)
            full = None
        else:
            all_ids, all_weights = self.router(x)
            ids, weights = all_ids[start:start + rows], all_weights[start:start + rows]
            full = all_ids
        if self.fastmeta and x.dtype == torch.bfloat16:
            local_out = self._ep_forward_fast(local_x, ids, weights,
                                              full=full, start=start,
                                              rows=rows)
        else:
            local_out = self._ep_forward_alltoallv(local_x, ids, weights)
        if STATE.tp_size == 1 or tokens == 0:
            return local_out
        # NCCL all_gather_into_tensor requires equal sizes. This pads only the
        # TP ownership boundary (at most tp_size-1 rows in this replica),
        # never expert activations to a hottest-expert capacity.
        padded = torch.zeros((rows, hidden), device=x.device, dtype=x.dtype)
        padded[:local_out.shape[0]].copy_(local_out)
        output = torch.empty((rows * STATE.tp_size, hidden), device=x.device, dtype=x.dtype)
        dist.all_gather_into_tensor(output, padded, group=STATE.tp_group)
        return output[:tokens]

    def _experts_local(self, xs, eids, ws=None):
        """本地专家子集的分组计算（eids 是本地编号）。"""
        E = self.experts.num_experts
        final = torch.empty_like(xs)
        counts = torch.bincount(eids, minlength=E)
        offsets = counts.cumsum(0)
        order = torch.argsort(eids, stable=True)
        xs_s = xs[order]
        out = self.experts.compute_sorted(xs_s, offsets)
        final[order] = out if ws is None else out * ws[order, None].to(out.dtype)
        return final

    def _ep_forward_alltoallv(self, x, idx, w):
        """Variable-length EP dispatch and combine using NCCL all-to-allv.

        The flattened route entries are grouped by destination EP rank.  The
        destination computes only its local experts and returns outputs in the
        same per-peer order, allowing the sender to scatter-add directly into
        its original token rows.  ``send_counts``/``recv_counts`` are exchanged
        first, so hot experts create real larger messages instead of padding.
        """
        from layers.parallel import STATE
        if STATE.ep_group is None:
            raise RuntimeError("EP all-to-allv requires an ep_group")
        import torch.distributed as dist
        T, H = x.shape
        k = idx.shape[1]
        experts_per_rank = self.experts.num_experts
        flat_e = idx.reshape(-1).to(torch.int64)
        flat_w = w.reshape(-1)
        flat_t = torch.arange(T, device=x.device, dtype=torch.int64).repeat_interleave(k)
        dest = torch.div(flat_e, experts_per_rank, rounding_mode="floor")
        order = torch.argsort(dest, stable=True)
        send_e = flat_e[order].contiguous()
        send_w = flat_w[order].contiguous()
        send_t = flat_t[order].contiguous()
        send_x = x[send_t].contiguous()
        world = STATE.ep_size
        send_counts = torch.bincount(dest, minlength=world).to(torch.int64)
        recv_counts = torch.empty_like(send_counts)
        dist.all_to_all_single(recv_counts, send_counts, group=STATE.ep_group)
        ss = [int(v) for v in send_counts.tolist()]
        rs = [int(v) for v in recv_counts.tolist()]

        def a2a(inp, out_shape, splits_in, splits_out):
            out = torch.empty(out_shape, dtype=inp.dtype, device=inp.device)
            dist.all_to_all_single(out, inp, output_split_sizes=splits_out,
                                   input_split_sizes=splits_in, group=STATE.ep_group)
            return out

        recv_x = a2a(send_x, (sum(rs), H), ss, rs)
        recv_e = a2a(send_e, (sum(rs),), ss, rs)
        recv_w = None if self.weight_at_source else a2a(send_w, (sum(rs),), ss, rs)
        # Convert global expert ids to this rank's local interval.
        local_e = recv_e - STATE.ep_rank * experts_per_rank
        recv_out = self._experts_local(recv_x, local_e, recv_w)
        returned = a2a(recv_out, (sum(ss), H), rs, ss)
        # Each flattened route has exactly one output. Restore route order,
        # then sum top-k in FP32 instead of BF16 atomic index_add_. Atomic
        # accumulation order varies across launches and can change greedy
        # tokens when scores are close (and can diverge across EP replicas).
        route_out = torch.empty_like(returned)
        route_out[order] = (returned * send_w[:, None].to(returned.dtype)
                            if self.weight_at_source else returned)
        return _reduce_routes(route_out.view(T, k, H), idx)

    # ============ experimental fast path (VLLM_LEARN_EP_FASTMETA) ============

    def _meta_host(self, *tensors):
        """一次 D2H 同步取回全部元数据(list[int] 拼接)。"""
        return torch.cat(tensors).tolist()

    def _local_experts_host(self, recv, rows, me, per, ends_host, H):
        """接收缓冲 → 专家计算（带 host 偏移，无 .tolist() 同步）。"""
        rx = recv[:, :H]
        le = recv[:, H].contiguous().view(torch.int16).to(torch.int64) \
            - me * per
        rorder = torch.argsort(le, stable=True)
        xs_s = rx[rorder]
        offs_gpu = torch.tensor(ends_host, dtype=torch.int32,
                                device=rx.device)
        out_s = self.experts.compute_sorted(xs_s, offs_gpu,
                                            offsets_host=ends_host)
        final = torch.empty((rows, H), dtype=rx.dtype, device=rx.device)
        final[rorder] = out_s
        return final

    def _pack(self, x, rows, k, flat_e, H):
        """把 (token 行 | int16 专家号) 打包成单个 bf16 载荷 [n, H+1]。"""
        return torch.cat([x[rows // k],
                          flat_e[rows].to(torch.int16).unsqueeze(1)
                                      .view(torch.bfloat16)], dim=1)

    def _counts_host(self, gcE, gcD, world, me, per, E):
        """exchange provider（DP×TP 耦合形态）：本地 bincount（仅本 rank
        owned 路由）+ 一次打包 allgather(E+world 个 int32) + **每层一次**
        D2H → (send_splits, recv_splits, 本地专家终点)。对任意 per-rank
        输入批次成立，包括空 DP 副本；与 analytic 在同输入下逐 token 等价。
        """
        from layers.parallel import STATE
        import torch.distributed as dist
        W = E + world
        cnt = torch.cat([gcE, gcD]).to(torch.int32)
        gathered = torch.empty(world * W, dtype=torch.int32,
                               device=gcE.device)
        dist.all_gather_into_tensor(gathered, cnt, group=STATE.ep_group)
        meta = gathered.tolist()                      # <<< 每层唯一一次 D2H
        ssl = meta[me * W + E: me * W + E + world]
        rs = [meta[j * W + E + me] for j in range(world)]
        ends, acc = [], 0
        for e in range(me * per, (me + 1) * per):
            acc += sum(meta[j * W + e] for j in range(world))
            ends.append(acc)
        return ssl, rs, ends

    def _counts_host_analytic(self, full_ids, rows, world, me, per, E):
        """analytic provider（ownership 拓扑）：仅当 DP=1 且 router 作用于
        全量复制 batch 时成立——ownership 切分规则确定（owner = row//rows），
        全组 send/专家计数可由 [owner,dest]/[owner,expert] 两个 bincount
        本地推出，一次 D2H、零额外交集通信。其他情形一律 fall back 到
        exchange（调用侧保证）。"""
        T_full, k = full_ids.shape
        rows = max(rows, 1)
        flat_e = full_ids.reshape(-1).to(torch.int64)
        dest = torch.div(flat_e, per, rounding_mode="floor")
        owner = torch.arange(T_full, device=full_ids.device,
                             dtype=torch.int64).repeat_interleave(k) // rows
        sd = torch.bincount(owner * world + dest, minlength=world * world)
        oe = torch.bincount(owner * E + flat_e, minlength=world * E)
        meta = self._meta_host(sd, oe)                # <<< 每层唯一一次 D2H
        ssl = [meta[me * world + d] for d in range(world)]
        rs = [meta[j * world + me] for j in range(world)]
        base = world * world
        ends, acc = [], 0
        for e in range(me * per, (me + 1) * per):
            acc += sum(meta[base + j * E + e] for j in range(world))
            ends.append(acc)
        return ssl, rs, ends

    def _ep_forward_fast(self, x, idx, w, full=None, start=0, rows=None):
        """ownership 拓扑下的实验性 EP 快路径（VLLM_LEARN_EP_FASTMETA）。

        相对默认 _ep_forward_alltoallv：
          - 计数链：counts 专用 all_to_all + 2 次 tolist + 专家循环
            offsets tolist → 一次打包（allgather 或解析）+ **一次** D2H；
          - 载荷打包 (x|int16 e) + 源侧加权：每层集合通信 5 次 → 3 次
            （counts-allgather + dispatch + combine）；
          - 专家循环的 host 偏移由 provider 直接给出（无额外 D2H）。
        数学与默认路径一致（GEMM 形状不变、top-k FP32 归并相同）。
        full：DP=1 且 router 作用于全量复制 batch 时的完整路由表 [T,k]，
        供 analytic provider 使用；空本地批次（T=0）照常参与全部集合通信
        ——空闲 DP 副本必须接收并执行其他副本发来的 route。
        """
        from layers.parallel import STATE
        import torch.distributed as dist
        T, H = x.shape
        k = idx.shape[1]
        world, me = STATE.ep_size, STATE.ep_rank
        per = self.experts.num_experts
        E = per * world
        flat_e = idx.reshape(-1).to(torch.int64)
        flat_w = w.reshape(-1)
        dest = torch.div(flat_e, per, rounding_mode="floor")
        if self.overlap and STATE.ep_group2 is not None and k % 2 == 0:
            return self._ep_forward_fast_overlap(x, idx, w, flat_e, flat_w,
                                                 dest)
        with torch.autograd.profiler.record_function("moe.dispatch.meta"):
            if (self.meta_mode == 'analytic' and full is not None
                    and STATE.dp_size == 1):
                ssl, rs, ends_host = self._counts_host_analytic(
                    full, rows, world, me, per, E)
            else:
                gcE = torch.bincount(flat_e, minlength=E)
                gcD = torch.bincount(dest, minlength=world)
                ssl, rs, ends_host = self._counts_host(gcE, gcD, world, me,
                                                       per, E)
        order = torch.argsort(dest, stable=True)
        packed = self._pack(x, order, k, flat_e, H)
        with torch.autograd.profiler.record_function("moe.dispatch"):
            recv = torch.empty((sum(rs), H + 1), dtype=x.dtype,
                               device=x.device)
            dist.all_to_all_single(recv, packed, output_split_sizes=rs,
                                   input_split_sizes=ssl, group=STATE.ep_group)
        with torch.autograd.profiler.record_function("moe.expert"):
            final = self._local_experts_host(recv, recv.shape[0], me, per,
                                             ends_host, H)
        with torch.autograd.profiler.record_function("moe.combine"):
            returned = torch.empty((order.numel(), H), dtype=x.dtype,
                                   device=x.device)
            dist.all_to_all_single(returned, final, output_split_sizes=ssl,
                                   input_split_sizes=rs,
                                   group=STATE.ep_group)
        full = torch.empty_like(returned)
        full[order] = returned * flat_w[order].unsqueeze(1).to(returned.dtype)
        return _reduce_routes(full.view(T, k, H), idx)

    def _ep_forward_fast_overlap(self, x, idx, w, flat_e, flat_w, dest):
        """双通信域流水的 fastmeta 变体。

        路由按 top-k 槽位奇偶（k=8，偶槽/奇槽各 4 个、天然均衡）拆成两路，
        分别走 STATE.ep_group / ep_group2。事件编排：
            dispatch A ‖ dispatch B（两条 NCCL comm 并行）
            experts  A ‖ dispatch B
            combine  A ‖ experts  B
            combine  B ‖ （本层尾部）
        每个专家 GEMM 的 M 维变为原来一半——数值需 token 级对拍确认。
        ownership 拓扑下计数一律走 exchange provider（本地 bincount + 一次
        打包 allgather + 一次 D2H），对任意输入分布与空副本成立。
        """
        from layers.parallel import STATE
        import torch.distributed as dist
        T, H = x.shape
        k = idx.shape[1]
        world, me = STATE.ep_size, STATE.ep_rank
        per = self.experts.num_experts
        E = per * world
        k2 = k // 2
        dmat = dest.view(T, k)
        emat = flat_e.view(T, k)
        dA = dmat[:, 0::2].reshape(-1)
        dB = dmat[:, 1::2].reshape(-1)
        eA = emat[:, 0::2].reshape(-1)
        eB = emat[:, 1::2].reshape(-1)
        with torch.autograd.profiler.record_function("moe.dispatch.meta"):
            bcA = torch.bincount(dA, minlength=world)
            bcB = torch.bincount(dB, minlength=world)
            bcEA = torch.bincount(eA, minlength=E)
            bcEB = torch.bincount(eB, minlength=E)
            # exchange：每 rank 打包 [dA(world), dB(world), eA(E), eB(E)]
            # 一次 allgather + 一次 D2H——对任意 per-rank 批次成立（含空副本）。
            cnt = torch.cat([bcA, bcB, bcEA, bcEB]).to(torch.int32)
            W2 = cnt.numel()
            gathered = torch.empty(world * W2, dtype=torch.int32,
                                   device=x.device)
            dist.all_gather_into_tensor(gathered, cnt,
                                        group=STATE.ep_group)
            g = gathered.tolist()                     # <<< 每层唯一一次 D2H
            row = me * W2
            ssA = g[row: row + world]
            ssB = g[row + world: row + 2 * world]
            rsA = [g[j * W2 + me] for j in range(world)]
            rsB = [g[j * W2 + world + me] for j in range(world)]
            endsA, acc = [], 0
            for e in range(me * per, (me + 1) * per):
                acc += sum(g[j * W2 + 2 * world + e] for j in range(world))
                endsA.append(acc)
            endsB, acc = [], 0
            for e in range(me * per, (me + 1) * per):
                acc += sum(g[j * W2 + 2 * world + E + e]
                           for j in range(world))
                endsB.append(acc)
        # A 子域位置 p ↔ flat route id: token p//k2, slot p%k2 → t*k + 2j
        key = (T, k)
        maps = self._rowmap_cache.get(key)
        if maps is None:
            p = torch.arange(T * k2, device=x.device)
            rmA = (p // k2) * k + 2 * (p % k2)
            maps = (rmA, rmA + 1)
            self._rowmap_cache[key] = maps
        rmA, rmB = maps
        orderA = torch.argsort(dA, stable=True)
        orderB = torch.argsort(dB, stable=True)
        rowsA = rmA[orderA]
        rowsB = rmB[orderB]
        packedA = self._pack(x, rowsA, k, flat_e, H)
        packedB = self._pack(x, rowsB, k, flat_e, H)
        if self._ov_streams is None:
            self._ov_streams = (torch.cuda.Stream(), torch.cuda.Stream())
        sA, sB = self._ov_streams
        main = torch.cuda.current_stream()
        ev_pack = torch.cuda.Event()
        ev_pack.record(main)
        recvA = torch.empty((sum(rsA), H + 1), dtype=x.dtype, device=x.device)
        recvB = torch.empty((sum(rsB), H + 1), dtype=x.dtype, device=x.device)
        with torch.autograd.profiler.record_function("moe.dispatch"):
            with torch.cuda.stream(sA):
                sA.wait_event(ev_pack)
                dist.all_to_all_single(recvA, packedA, output_split_sizes=rsA,
                                       input_split_sizes=ssA,
                                       group=STATE.ep_group)
                ev_dA = torch.cuda.Event()
                ev_dA.record(sA)
            with torch.cuda.stream(sB):
                sB.wait_event(ev_pack)
                dist.all_to_all_single(recvB, packedB, output_split_sizes=rsB,
                                       input_split_sizes=ssB,
                                       group=STATE.ep_group2)
                ev_dB = torch.cuda.Event()
                ev_dB.record(sB)
        with torch.autograd.profiler.record_function("moe.expert"):
            main.wait_event(ev_dA)
            finalA = self._local_experts_host(recvA, recvA.shape[0], me, per,
                                              endsA, H)
        ev_eA = torch.cuda.Event()
        ev_eA.record(main)
        combA = torch.empty((rowsA.numel(), H), dtype=x.dtype, device=x.device)
        with torch.autograd.profiler.record_function("moe.combine"):
            with torch.cuda.stream(sA):
                sA.wait_event(ev_eA)
                dist.all_to_all_single(combA, finalA, output_split_sizes=ssA,
                                       input_split_sizes=rsA,
                                       group=STATE.ep_group)
                ev_cA = torch.cuda.Event()
                ev_cA.record(sA)
        with torch.autograd.profiler.record_function("moe.expert"):
            main.wait_event(ev_dB)
            finalB = self._local_experts_host(recvB, recvB.shape[0], me, per,
                                              endsB, H)
        ev_eB = torch.cuda.Event()
        ev_eB.record(main)
        combB = torch.empty((rowsB.numel(), H), dtype=x.dtype, device=x.device)
        with torch.autograd.profiler.record_function("moe.combine"):
            with torch.cuda.stream(sB):
                sB.wait_event(ev_eB)
                dist.all_to_all_single(combB, finalB, output_split_sizes=ssB,
                                       input_split_sizes=rsB,
                                       group=STATE.ep_group2)
                ev_cB = torch.cuda.Event()
                ev_cB.record(sB)
        main.wait_event(ev_cA)
        main.wait_event(ev_cB)
        full = torch.empty((T * k, H), dtype=x.dtype, device=x.device)
        full[rowsA] = combA * flat_w[rowsA].unsqueeze(1).to(combA.dtype)
        full[rowsB] = combB * flat_w[rowsB].unsqueeze(1).to(combB.dtype)
        return _reduce_routes(full.view(T, k, H), idx)
