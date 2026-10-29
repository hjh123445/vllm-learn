# v6/layers/moe.py
"""v6 MoE 层：router + grouped expert 计算。

设计（对齐我们"平铺批 + context"的架构）：
  - 输入是平铺的 [total_q, hidden]（与 attention 同一约定）；
  - router：softmax(fp32) → topk → normalize（HF Qwen3MoeTopKRouter 逐行同数学）；
  - experts：**分组 gemm**——按 expert_idx 排序 token，同一专家的 token 连续排布后
    一次 batched gemm（不是 HF 的逐专家循环，也不是 dense 大矩阵乘全部专家）。
    权重布局 [num_experts, 3*I, H]（gate|up 融合在第一个 gemm，down 是第二个）。

EP 扩展点（v6.2）：本文件只做"本地专家子集"的计算，dispatch/combine 的
all-to-all 在 model_runner 层做（experts 的输入输出张量按 expert 分片搬运）。"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


class MoERouter(nn.Module):
    """top-k 路由。fp32 softmax 保证与 HF 逐位一致（norm_topk_prob=True）。"""

    def __init__(self, hidden_size: int, num_experts: int, top_k: int,
                 norm_topk_prob: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_experts, hidden_size))
        self.top_k = top_k
        self.norm_topk_prob = norm_topk_prob
        self.num_experts = num_experts

    def weight_loader(self, param, loaded, shard_id=None):
        """HF 名 mlp.gate.weight（loader 已映射到 router.weight）。直装。"""
        assert param.shape == loaded.shape
        param.data.copy_(loaded)

    def forward(self, x: torch.Tensor):
        """x: [T, H] → (topk_idx [T,k], topk_w [T,k])。"""
        logits = F.linear(x.float(), self.weight.float())
        probs = logits.softmax(dim=-1)
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
        self.num_experts = num_experts
        self.hidden = hidden_size
        self.inter = intermediate
        # 权重以 fp 存储由 loader 决定；结构上保持完整专家维度（EP 时 loader 只装本地切片）
        self.w13 = nn.Parameter(torch.empty(num_experts, 2 * intermediate, hidden_size))
        self.w2 = nn.Parameter(torch.empty(num_experts, hidden_size, intermediate))

    def weight_loader(self, param, loaded, shard_id):
        """weight_loader 协议（与 BaseLinear/RmsNorm 同约定）：
        loader 做名字映射后调 param 所属模块的 weight_loader，shard_id 携带
        融合信息——(expert_idx, half) 装进 w13 的对应半区，(expert_idx, None) 装 w2。
        装载逻辑在组件内（loader 不感知 w13 的内部布局）。"""
        i, half = shard_id
        if param is self.w13:
            I = loaded.shape[0]
            param.data[i, half * I:(half + 1) * I, :] = loaded
        else:
            param.data[i] = loaded

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

        final = torch.zeros_like(x)
        counts = torch.bincount(flat_expert, minlength=E)
        offsets = counts.cumsum(0)                     # 每个专家在排序后序列的终点
        # 稳定排序：同一专家的 token 连续排布
        order = torch.argsort(flat_expert, stable=True)
        sorted_token = flat_token[order]
        sorted_w = flat_w[order]

        off_list = offsets.tolist()                    # [E]（一次 CPU 同步）
        start = 0
        for e, end in enumerate(off_list):
            if end == start:                           # 本批没 token 命中该专家
                continue
            toks = sorted_token[start:end]             # 该专家的 token 索引
            xs = x[toks]                               # [n_e, H]
            h13 = xs @ self.w13[e].t()                 # [n_e, 2I]
            gate, up = h13.chunk(2, dim=-1)
            act = F.silu(gate) * up                    # [n_e, I]
            out = act @ self.w2[e].t()                 # [n_e, H]
            final.index_add_(0, toks, out * sorted_w[start:end, None].to(out.dtype))
            start = end
        return final


class MoEMLP(nn.Module):
    """一层完整 MoE：router + experts。无共享专家（Qwen3-Omni 配置为 0）。

    EP 模式（v6.2）：expert_parallel=True 时——
      1. 本地 experts 只持有自己一半专家的权重（loader/shard 切）；
      2. router 全卡跑（全专家 logits，冗余）；
      3. dispatch: 把 (token, expert, weight) 中 expert 属于对端的条目 all-to-all 发过去；
      4. 本地算完 partial 和，combine: all-to-all 收回对端算的结果，就地累加。
    通信量 = 2 × tokens × h × 2B（每层），与 TP 的 b×s×h×N 倍形成对照。"""

    def __init__(self, config):
        super().__init__()
        self.router = MoERouter(config.hidden_size, config.num_experts,
                                config.num_experts_per_tok,
                                getattr(config, "norm_topk_prob", True))
        self.experts = GroupedExperts(config.num_experts, config.hidden_size,
                                      config.moe_intermediate_size)
        # EP 状态（由 shard_model_for_ep 设置）
        self.expert_parallel = False
        self.ep_rank = 0
        self.ep_size = 1
        self.local_expert_start = 0        # 本地专家区间 [start, start + local_E)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        import torch.distributed as dist
        idx, w = self.router(x)
        if not self.expert_parallel or not dist.is_initialized() or dist.get_world_size() == 1:
            return self.experts(x, idx, w)
        return self._ep_forward_clean(x, idx, w)

    def _experts_local(self, xs, eids, ws):
        """本地专家子集的分组计算（eids 是本地编号）。"""
        E = self.local_E
        final = torch.zeros_like(xs)
        counts = torch.bincount(eids, minlength=E)
        offsets = counts.cumsum(0)
        order = torch.argsort(eids, stable=True)
        xs_s, ws_s = xs[order], ws[order]
        start = 0
        for e, end in enumerate(offsets.tolist()):
            if end == start:
                continue
            toks = order[start:end]
            h13 = xs_s[start:end] @ self.experts.w13[e].t()
            gate, up = h13.chunk(2, dim=-1)
            act = torch.nn.functional.silu(gate) * up
            out = act @ self.experts.w2[e].t()
            final[toks] = out * ws_s[start:end, None].to(out.dtype)
            start = end
        return final

    def _ep_forward_clean(self, x, idx, w):
        """EP 的清晰实现：dispatch 用 all_gather（v6.4 换 all_to_all 省带宽）。
        每卡算自己专家的部分和 → all_reduce 聚合。"""
        import torch.distributed as dist
        T, H = x.shape
        k = idx.shape[1]
        flat_expert = idx.reshape(-1)                      # [T*k] 全局 expert 号
        flat_w = w.reshape(-1)
        # 每张卡独立算自己本地专家的部分和（非本地条目跳过）——零 dispatch 通信！
        # 这是 "DP-style EP"：输入 x 全卡冗余（TP allreduce 已保证），
        # 每卡只算本地专家, 输出 all_reduce。
        out = torch.zeros(T, H, device=x.device, dtype=x.dtype)
        local_mask = (flat_expert >= self.local_expert_start) & \
                     (flat_expert < self.local_expert_start + self.local_E)
        if local_mask.any():
            eids_local = flat_expert[local_mask] - self.local_expert_start
            # 每个条目的全局行号 → token 号
            rows = torch.nonzero(local_mask).squeeze(1)
            toks = rows // k
            ws = flat_w[local_mask]
            xs = x[toks]
            outs = self._experts_local(xs, eids_local, ws)
            out.index_add_(0, toks, outs)
        dist.all_reduce(out)
        return out
