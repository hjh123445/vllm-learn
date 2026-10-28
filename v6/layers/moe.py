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
        self.weight.weight_loader = self._load
        self.num_experts = num_experts

    def _load(self, param, loaded):
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
        self.w13.weight_loader = self._load
        self.w2.weight_loader = self._load

    def _load(self, param, loaded):
        # loader 负责：gate_proj/up_proj 两个 [I,H] → 拼进 w13 的两半
        param.data.copy_(loaded)

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
    """一层完整 MoE：router + experts。无共享专家（Qwen3-Omni 配置为 0）。"""

    def __init__(self, config):
        super().__init__()
        self.router = MoERouter(config.hidden_size, config.num_experts,
                                config.num_experts_per_tok,
                                getattr(config, "norm_topk_prob", True))
        self.experts = GroupedExperts(config.num_experts, config.hidden_size,
                                      config.moe_intermediate_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        idx, w = self.router(x)
        return self.experts(x, idx, w)
