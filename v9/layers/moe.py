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
        from layers.parallel import STATE
        self.hidden = hidden_size
        # ---- EP: 本地专家区间 ----
        per = num_experts // STATE.ep_size
        assert per * STATE.ep_size == num_experts, \
            f"E={num_experts} 切不动 EP{STATE.ep_size}"
        self.e_start = STATE.ep_rank * per
        self.num_experts = per                    # 本地专家数（E_local）
        # ---- TP: inter 切片（w13 的行 / w2 的列）----
        assert intermediate % STATE.tp_size == 0, \
            f"I={intermediate} 切不动 TP{STATE.tp_size}"
        self.inter = intermediate // STATE.tp_size       # I_local
        self.i_start = STATE.tp_rank * self.inter
        self.quantized = False
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

        final = torch.zeros_like(x)
        counts = torch.bincount(flat_expert, minlength=E)
        offsets = counts.cumsum(0)                     # 每个专家在排序后序列的终点
        # 稳定排序：同一专家的 token 连续排布
        order = torch.argsort(flat_expert, stable=True)
        sorted_token = flat_token[order]
        sorted_w = flat_w[order]

        off_list = offsets.tolist()                    # [E]（一次 CPU 同步）
        if self.quantized:
            from layers.quantized import dequantize
            g = self.group
            # 平铺 [E*2I, H/2] → 按需 dequant（本版全量；v9.2 kernel 化后按 tile）
            w13 = dequantize(self.w13_packed, self.w13_scales, g)                 .reshape(self.num_experts, -1, self.hidden).to(x.dtype)
            w2 = dequantize(self.w2_packed, self.w2_scales, g)                 .reshape(self.num_experts, self.hidden, self.inter).to(x.dtype)
        else:
            w13, w2 = self.w13, self.w2
        start = 0
        for e, end in enumerate(off_list):
            if end == start:                           # 本批没 token 命中该专家
                continue
            toks = sorted_token[start:end]             # 该专家的 token 索引
            xs = x[toks]                               # [n_e, H]
            h13 = xs @ w13[e].t()                      # [n_e, 2I]
            gate, up = h13.chunk(2, dim=-1)
            act = F.silu(gate) * up                    # [n_e, I]
            out = act @ w2[e].t()                      # [n_e, H]
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
        # v9.3: 并行状态在 GroupedExperts 内部（构造时从 ParallelState 读）,
        # MoEMLP 只负责 router + 本地计算 + 部分和聚合

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from layers.parallel import STATE, moe_allreduce
        idx, w = self.router(x)
        if not (STATE.tp_enabled or STATE.ep_size > 1):
            return self.experts(x, idx, w)
        out = self._ep_forward_local(x, idx, w)
        return moe_allreduce(out)          # tp×ep 张成组上的部分和聚合

    def _experts_local(self, xs, eids, ws):
        """本地专家子集的分组计算（eids 是本地编号）。"""
        E = self.experts.num_experts
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

    def _ep_forward_local(self, x, idx, w):
        """本地专家部分和（DP-style EP，v6.2 设计）：
        输入 x 组内冗余（上游 RowParallel 的 allreduce 已保证各 tp rank 一致），
        每卡只算本地专家区间内的 (token, expert) 条目，非本地条目贡献 0，
        由调用方 moe_allreduce 聚合（tp 切 inter 的部分和 + ep 切专家的部分和
        在同一个组上一次 allreduce 完成）。
        v6.4 起可换 all_to_all 省带宽（dispatch/combine），本版保留冗余计算换零
        dispatch 通信的简单形态。"""
        T, H = x.shape
        k = idx.shape[1]
        flat_expert = idx.reshape(-1)                      # [T*k] 全局 expert 号
        flat_w = w.reshape(-1)
        ex = self.experts
        out = torch.zeros(T, H, device=x.device, dtype=x.dtype)
        local_mask = (flat_expert >= ex.e_start) & \
                     (flat_expert < ex.e_start + ex.num_experts)
        if local_mask.any():
            eids_local = flat_expert[local_mask] - ex.e_start
            rows = torch.nonzero(local_mask).squeeze(1)
            toks = rows // k
            ws = flat_w[local_mask]
            xs = x[toks]
            outs = self._experts_local(xs, eids_local, ws)
            out.index_add_(0, toks, outs)
        return out
