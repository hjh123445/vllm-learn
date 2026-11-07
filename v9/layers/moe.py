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

from layers.kernels import silu_and_mul
from layers.batch_invariant import enabled as batch_invariant, linear as forward_linear, all_reduce as fixed_all_reduce, expert_sum


def _bucket_order(keys, num_buckets, counts=None):
    """Stable grouping order via counting scatter; equals argsort(stable=True).

    When the key domain is small (EP ranks or local experts), counting the
    in-bucket rank avoids the multi-pass int64 radix sort: for short decode
    batches sorting was several times the cost of the gather it orders.
    ``counts`` may be reused when the caller already histogrammed ``keys``.
    """
    n = keys.numel()
    if counts is None:
        counts = torch.bincount(keys, minlength=num_buckets)
    starts = counts.cumsum(0) - counts
    rank_in_key = (keys.unsqueeze(1) == torch.arange(num_buckets, device=keys.device,
                                                     dtype=keys.dtype)).to(torch.int32).cumsum(0)
    rank_of = rank_in_key.gather(1, keys.unsqueeze(1)).squeeze(1).to(torch.int64) - 1
    positions = starts[keys] + rank_of
    order = torch.empty(n, device=keys.device, dtype=torch.int64)
    order.scatter_(0, positions, torch.arange(n, device=keys.device))
    return order


def _moe_ordering():
    """'argsort' (default) keeps the single fused radix kernel; 'counting'
    is an ablation. Measured on 8xA800 EP8: counting loses 1.5-3.8% tok/s
    — this engine is CPU-launch-bound, so 7 small kernels cost more than
    one multi-pass sort. radixSort itself was only ~2% of busy time."""
    return os.getenv('VLLM_LEARN_MOE_ORDER', 'argsort')


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


def _local_route_weights(idx, weights, first_expert, local_experts):
    """Fixed-size scatter: no boolean selection/nonzero or host offsets.

    Invalid routes scatter zero into a clamped local slot. Top-k expert IDs
    are unique per token, so every valid destination has one nonzero value.
    Returns [E_local, T]; shape depends only on the decode batch bucket.
    """
    local = idx - first_expert
    valid = (local >= 0) & (local < local_experts)
    out = torch.zeros((idx.shape[0], local_experts), device=idx.device, dtype=torch.float32)
    out.scatter_add_(1, local.clamp(0, local_experts - 1), weights.float() * valid)
    return out.t().contiguous()


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
        logits = (forward_linear(x, self.weight) if self.logits_dtype == "model"
                  else forward_linear(x.float(), self.weight.float()))
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
        # Numerical oracle: keep expert GEMMs, route weighting and the expert
        # TP reduction in FP32. BF16 complete/partitioned GEMMs otherwise have
        # different rounding boundaries. This is opt-in, not a speed policy.
        self.precision = os.getenv('VLLM_LEARN_EXPERT_PRECISION', 'model')
        if self.precision not in ('model', 'float32'):
            raise ValueError('expert precision must be model or float32')
        if self.precision == 'float32' and self.backend == 'torch_grouped':
            raise ValueError('float32 expert oracle requires loop or auto backend')
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

        counts = torch.bincount(flat_expert, minlength=E)
        offsets = counts.cumsum(0)                     # 每个专家在排序后序列的终点
        # 同一专家的 token 连续排布（counting 与 argsort(stable) 逐位一致）
        order = (_bucket_order(flat_expert, E, counts) if _moe_ordering() == 'counting'
                 else torch.argsort(flat_expert, stable=True))
        sorted_token = flat_token[order]
        sorted_w = flat_w[order]

        xs = x[sorted_token]
        out = self.compute_sorted(xs, offsets)
        route_out = torch.empty((T * k, H), device=x.device, dtype=out.dtype)
        route_out[order] = out * sorted_w[:, None].to(out.dtype)
        return _reduce_routes(route_out.view(T, k, H), topk_idx)

    def compute_sorted(self, xs, offsets):
        """Compute expert-contiguous routes using local weights only.

        ``offsets[e]`` is the exclusive end of expert e, including empty
        experts. Native grouped GEMM keeps this prefix sum on the GPU.
        Explicit selection fails on unsupported platforms instead of silently
        falling back to the Python loop and mislabelling measurements.
        """
        if self.quantized:
            from layers.quantized import dequantize
            g = self.group
            # 平铺 [E*2I, H/2] → 按需 dequant（本版全量；v9.2 kernel 化后按 tile）
            w13 = dequantize(self.w13_packed, self.w13_scales, g).reshape(self.num_experts, -1, self.hidden).to(xs.dtype)
            w2 = dequantize(self.w2_packed, self.w2_scales, g).reshape(self.num_experts, self.hidden, self.inter).to(xs.dtype)
        else:
            w13, w2 = self.w13, self.w2
        if self.precision == 'float32':
            xs, w13, w2 = xs.float(), w13.float(), w2.float()
        if xs.shape[0] == 0:
            return xs.new_empty((0, self.hidden))
        backend = self.resolve_backend(xs)
        return self._compute_sorted_backend(xs, offsets, w13, w2, backend)

    def resolve_backend(self, xs):
        """Select the grouped kernel for supported CUDA BF16 inputs.

        Explicit ``torch_grouped`` stays strict for reproducible ablations;
        ``auto`` falls back to the readable loop on unsupported platforms.
        """
        if batch_invariant():
            return 'loop'
        if self.backend != 'auto':
            return self.backend
        grouped_mm = getattr(F, 'grouped_mm', None) or getattr(torch, '_grouped_mm', None)
        if xs.is_cuda and xs.dtype == torch.bfloat16 and grouped_mm is not None:
            return 'torch_grouped'
        return 'loop'

    def _compute_sorted_backend(self, xs, offsets, w13, w2, backend):
        if backend == 'torch_grouped':
            grouped_mm = getattr(F, 'grouped_mm', None) or getattr(torch, '_grouped_mm', None)
            if grouped_mm is None or not xs.is_cuda or xs.dtype != torch.bfloat16:
                raise RuntimeError('torch_grouped requires CUDA BF16 and a compatible native grouped_mm')
            offs = offsets.to(dtype=torch.int32)
            h13 = grouped_mm(xs, w13.transpose(-2, -1), offs=offs)
            act = silu_and_mul(h13)
            return grouped_mm(act, w2.transpose(-2, -1), offs=offs)
        output = xs.new_empty((xs.shape[0], self.hidden))
        start = 0
        for e, end in enumerate(offsets.tolist()):
            if end == start:                           # 本批没 token 命中该专家
                continue
            h13 = forward_linear(xs[start:end], w13[e])           # [n_e, 2I]
            act = silu_and_mul(h13)                    # [n_e, I]
            out = forward_linear(act, w2[e])                      # [n_e, H]
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
        # Packed dispatch: one (x|e|w) payload all-to-all replaces three,
        # counts exchange overlaps the packing on a side stream, and both
        # split vectors reach the host in one D2H read. Opt-in experiment.
        self.ep_packed = os.getenv('VLLM_LEARN_EP_PACKED', '0') == '1'
        # Dense-EP decode:定形 decode 路径(CUDA graph 用)。TP 副本内 x 逐位
        # 相同 → router 输出逐位相同 → 免 dispatch;本地专家全 token dense
        # grouped GEMM + 权重矩阵聚合 + 一次 allreduce,全部静态形状。
        self.ep_dense_decode = os.getenv('VLLM_LEARN_EP_DENSE_DECODE', '0') == '1'
        self.dense_backend = os.getenv('VLLM_LEARN_DENSE_EXPERT_BACKEND', 'bmm')
        if self.dense_backend not in ('bmm', 'torch_grouped'):
            raise ValueError('dense expert backend must be bmm or torch_grouped')
        self._ep_comm_stream = None
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
            if self.ep_dense_decode and STATE.dp_size == 1:
                from utils.context import get_context as _gc
                if not _gc().is_prefill:
                    return self._dense_ep_decode(x)
            return self._forward_tp_replicated_input(x)
        idx, w = self.router(x)
        # Cast only AFTER summing expert-TP partials in oracle mode.
        return tp_allreduce(self.experts(x, idx, w)).to(x.dtype)

    def _dense_ep_decode(self, x: torch.Tensor) -> torch.Tensor:
        """定形 EP decode(大部分配在 CUDA graph 内,也可 eager 消融)。

        TP 副本内 x 逐位相同 → 各 rank 的 router 输出逐位相同 → 无需
        dispatch/counts 交换:本地 E_per 个专家对全部 T 个 token 做 dense
        BMM(或显式选择 grouped GEMM),路由权重用 [E_per, T] 矩阵
        散射聚合,最后 EP 组一次 FP32 allreduce。要求非量化专家、DP=1。
        每个 token 计算全部专家,总 FLOPs 约为稀疏 top-k 的 E/k 倍;
        是否合算取决于负载,不能把它当成通用最优。数学上与 dispatch
        等价不代表 BF16 逐位一致;这里只保证已测的同形状 graph/eager。"""
        from layers.parallel import STATE
        import torch.distributed as dist
        exp = self.experts
        assert not exp.quantized and exp.precision != 'float32'
        T, H = x.shape
        E_per = exp.num_experts                  # EP 构造已切成本地专家数
        r0 = STATE.ep_rank * E_per
        idx, w = self.router(x)                  # [T,k];TP 组内逐位一致
        wm = _local_route_weights(idx, w, r0, E_per)
        if self.dense_backend == 'bmm':
            # Static expert batches have no CPU offset metadata. Native
            # grouped_mm on A800 may copy offsets internally during capture.
            xs = x.unsqueeze(0).expand(E_per, T, H)
            h13 = forward_linear(xs, exp.w13)
            act = silu_and_mul(h13.reshape(E_per * T, -1)).view(E_per, T, exp.inter)
            out = forward_linear(act, exp.w2)
        else:
            xs = x.repeat(E_per, 1)
            offs = torch.arange(1, E_per + 1, device=x.device, dtype=torch.int32) * T
            gmm = getattr(F, 'grouped_mm', None) or getattr(torch, '_grouped_mm', None)
            if gmm is None:
                raise RuntimeError('dense grouped decode requires native grouped_mm')
            h13 = gmm(xs, exp.w13.transpose(-2, -1), offs=offs)
            act = silu_and_mul(h13)
            out = gmm(act, exp.w2.transpose(-2, -1), offs=offs)
        # Match the sparse path's BF16 route multiply, then accumulate in
        # FP32 locally AND across ranks. Reduction order still differs.
        weighted = out.view(E_per, T, H) * wm.to(out.dtype).unsqueeze(-1)
        y = expert_sum(weighted) if batch_invariant() else weighted.float().sum(0)
        if batch_invariant():
            y = fixed_all_reduce(y, STATE.ep_group)
        else:
            dist.all_reduce(y, group=STATE.ep_group)
        return y.to(x.dtype)

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
        else:
            all_ids, all_weights = self.router(x)
            ids, weights = all_ids[start:start + rows], all_weights[start:start + rows]
        local_out = (self._ep_forward_alltoallv_packed(local_x, ids, weights)
                     if self.ep_packed else self._ep_forward_alltoallv(local_x, ids, weights))
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
        counts = torch.bincount(eids, minlength=E)
        offsets = counts.cumsum(0)
        order = (_bucket_order(eids, E, counts) if _moe_ordering() == 'counting'
                 else torch.argsort(eids, stable=True))
        xs_s = xs[order]
        out = self.experts.compute_sorted(xs_s, offsets)
        final = torch.empty_like(out)
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
        world = STATE.ep_size
        send_counts = torch.bincount(dest, minlength=world).to(torch.int64)
        order = (_bucket_order(dest, world, send_counts) if _moe_ordering() == 'counting'
                 else torch.argsort(dest, stable=True))
        send_e = flat_e[order].contiguous()
        send_w = flat_w[order].contiguous()
        send_t = flat_t[order].contiguous()
        send_x = x[send_t].contiguous()
        recv_counts = torch.empty_like(send_counts)
        dist.all_to_all_single(recv_counts, send_counts, group=STATE.ep_group)
        # PyTorch variable splits require host lists, not CUDA offsets.
        ss = send_counts.tolist()
        rs = recv_counts.tolist()

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
        return _reduce_routes(route_out.view(T, k, H), idx).to(x.dtype)

    def _comm_stream(self):
        if self._ep_comm_stream is None:
            self._ep_comm_stream = torch.cuda.Stream()
        return self._ep_comm_stream

    def _ep_forward_alltoallv_packed(self, x, idx, w):
        """Three-collective EP dispatch/combine with overlapped counts exchange.

        One uint8 payload row per route (x bf16 | e int32 | w bf16) replaces
        the three payload all-to-alls, so a MoE layer issues counts + payload
        + combine instead of five transfers.  The counts exchange runs on a
        side stream while the payload is gathered, and both split vectors
        reach the host in a single D2H read (the unpacked path syncs twice).
        Route output scaling stays at the source, which is the exact same
        multiply order as ``weight_at_source="1"`` on the unpacked path.
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
        world = STATE.ep_size
        send_counts = torch.bincount(dest, minlength=world).to(torch.int64)
        order = (_bucket_order(dest, world, send_counts) if _moe_ordering() == 'counting'
                 else torch.argsort(dest, stable=True))
        n = flat_e.numel()

        recv_counts = torch.empty_like(send_counts)
        current = torch.cuda.current_stream() if x.is_cuda else None
        if current is not None:
            comm = self._comm_stream()
            comm.wait_stream(current)
            with torch.cuda.stream(comm):
                dist.all_to_all_single(recv_counts, send_counts, group=STATE.ep_group)

        send_t = flat_t[order]
        send_e = flat_e[order]
        send_w = flat_w[order]
        send_x = x.index_select(0, send_t)
        # Row: e int32 (4B) | x (esz*H) | w (esz); pad the row stride to a
        # multiple of 4B so the strided int32 view on the receiver is legal.
        esz = x.element_size()
        row_bytes = (4 + esz * H + esz + 3) // 4 * 4
        payload = torch.empty((n, row_bytes), device=x.device, dtype=torch.uint8)
        payload[:, :4].copy_(send_e.to(torch.int32).view(torch.uint8).view(n, 4))
        payload[:, 4:4 + esz * H].copy_(send_x.view(torch.uint8).view(n, esz * H))
        payload[:, 4 + esz * H:4 + esz * H + esz].copy_(send_w.view(torch.uint8).view(n, esz))

        if current is None:
            dist.all_to_all_single(recv_counts, send_counts, group=STATE.ep_group)
        else:
            current.wait_stream(comm)
        # One host sync for both split vectors (the reference path syncs twice).
        both = torch.cat((send_counts, recv_counts)).tolist()
        ss, rs = both[:world], both[world:]

        in_bytes = [s * row_bytes for s in ss]
        out_bytes = [r * row_bytes for r in rs]
        m = sum(rs)
        recv_flat = torch.empty((m * row_bytes,), device=x.device, dtype=torch.uint8)
        dist.all_to_all_single(recv_flat, payload.view(-1), output_split_sizes=out_bytes,
                               input_split_sizes=in_bytes, group=STATE.ep_group)
        packed_recv = recv_flat.view(m, row_bytes)
        recv_x = packed_recv[:, 4:4 + esz * H].view(x.dtype)
        recv_e = packed_recv[:, :4].view(torch.int32).squeeze(1).to(torch.int64)

        local_e = recv_e - STATE.ep_rank * experts_per_rank
        recv_out = self._experts_local(recv_x, local_e, None)
        returned_flat = torch.empty((sum(ss) * H,), device=x.device, dtype=recv_out.dtype)
        dist.all_to_all_single(returned_flat, recv_out.reshape(-1),
                               output_split_sizes=[s * H for s in ss],
                               input_split_sizes=[r * H for r in rs], group=STATE.ep_group)
        returned = returned_flat.view(sum(ss), H)
        route_out = torch.empty_like(returned)
        route_out[order] = returned * send_w[:, None].to(returned.dtype)
        return _reduce_routes(route_out.view(T, k, H), idx).to(x.dtype)
