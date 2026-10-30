# v5/layers/attention.py
"""v4 attention：cache 写入 + 范数 + rope 融成一个 kernel（kernels.py），
直接注意力换成手写 triton FlashAttention（flash_attn.py）。

v8.2 新增 DCP 分支：KV cache 按块染色分片到 cp 维各 rank——
  1. store 门控：slot_mapping 重映射（全局块→本地槽），非本 rank 颜色置 -1，
     融合 kernel 的 store 带 mask（负 slot 不写）；
  2. partial attention：只对本 rank 颜色的行算（未归一化 part + m + l）；
  3. all_gather partial/m/l → 本地 LSE 合并（数学 = v8.0）。
TP 与 DCP 正交：TP 切头（cp 组内各 rank 头相同），DCP 切位置（cp 组内
各 rank 行不同）——cp 组恰好是 LSE 合并需要的通信域（coords.group("cp")）。
"""
import torch
import torch.distributed as dist
from torch import nn

from utils.context import get_context
from layers.kernels import fused_qk_norm_rope_store
from layers.flash_attn import decode_attn, prefill_attn
from layers.dcp import get_dcp


class Attention(nn.Module):
    def __init__(self, num_heads, num_kv_heads, head_dim, scale):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.scale = scale
        self.k_cache = self.v_cache = torch.tensor([])   # ModelRunner 分配后指过来

    def forward(self, q, k, v, positions, q_norm_w, k_norm_w, cos_sin_cache, eps):
        """q/k/v: [total_q, heads, head_dim]（投影输出，未 norm 未 rope）。
        这里完成 norm + rope + 写 cache + 注意力，全部 triton。"""
        ctx = get_context()
        dcp = get_dcp()
        slot_mapping = self._dcp_slots(ctx) if dcp.active else ctx.slot_mapping
        q = fused_qk_norm_rope_store(q, k, v, q_norm_w, k_norm_w, cos_sin_cache,
                                     positions, slot_mapping,
                                     self.k_cache, self.v_cache, eps)
        if dcp.active:
            return self._dcp_attention(q)
        if ctx.is_prefill:
            return prefill_attn(q, self.k_cache, self.v_cache, ctx.block_tables,
                                 ctx.cu_seqlens_q, ctx.cu_seqlens_k,
                                 ctx.max_seqlen_q, self.num_kv_heads, self.scale)
        return decode_attn(q, self.k_cache, self.v_cache, ctx.block_tables,
                           ctx.context_lens, self.num_kv_heads, self.scale)

    # ================= v8.2 DCP 路径 =================

    def _dcp_slots(self, ctx) -> torch.Tensor:
        """全局槽位（g*bs+off）→ 本地槽位（local(g)*bs+off）；非本 rank 颜色 → -1。
        纯 python 循环逐 token 映射（每步几百个，开销可忽略）。"""
        dcp = get_dcp()
        bs = self.k_cache.shape[1]
        out = []
        for s in ctx.slot_mapping.tolist():
            g, o = divmod(s, bs)
            out.append(dcp.local_slot(g) * bs + o if dcp.owned(g) else -1)
        return torch.tensor(out, dtype=torch.int32, device=ctx.slot_mapping.device)

    def _dcp_attention(self, q) -> torch.Tensor:
        """partial（本 rank 颜色的行）+ all_gather + LSE 合并。
        q: [T, H, D]（已 rope，H 已是 TP 切分后的本 rank 头数）→ [T, H*D]。"""
        dcp = get_dcp()
        ctx = get_context()
        T, H, D = q.shape
        HK = self.num_kv_heads
        bs = self.k_cache.shape[1]
        dev = q.device

        # ---- 1. 本 rank 持有的行：[(seq, 全局位置, 本地扁平位置)] ----
        bt = ctx.block_tables.tolist()
        lens = ctx.cu_seqlens_k.tolist()[1:] if ctx.is_prefill else ctx.context_lens.tolist()
        row_seq, row_pos, row_loc = [], [], []
        for i, btrow in enumerate(bt):
            L = lens[i]
            for j in range((L + bs - 1) // bs):
                g = btrow[j]
                if dcp.owned(g):
                    s = dcp.local_slot(g)
                    for pos in range(j * bs, min((j + 1) * bs, L)):
                        row_seq.append(i)
                        row_pos.append(pos)
                        row_loc.append(s * bs + pos % bs)
        L_loc = len(row_loc)

        # ---- 2. q 的 (seq, 位置)：prefill 从 cu_seqlens 推，decode 是末位 ----
        if ctx.is_prefill:
            cq, ck = ctx.cu_seqlens_q.tolist(), ctx.cu_seqlens_k.tolist()
            qseq, qpos = [], []
            for i in range(len(cq) - 1):
                q_len, k_len = cq[i + 1] - cq[i], ck[i + 1] - ck[i]
                qseq.extend([i] * q_len)
                qpos.extend(range(k_len - q_len, k_len))
        else:
            qseq = list(range(T))
            qpos = [l - 1 for l in lens]
        qseq = torch.tensor(qseq, device=dev)
        qpos = torch.tensor(qpos, device=dev)

        # ---- 3. partial attention（未归一化——空 rank 时 l=0 天然零贡献）----
        if L_loc:
            idx = torch.tensor(row_loc, dtype=torch.int64, device=dev)
            k_loc = self.k_cache.reshape(-1, HK, D)[idx]      # [L_loc, HK, D]
            v_loc = self.v_cache.reshape(-1, HK, D)[idx]
            rep = H // HK
            k_loc = k_loc.repeat_interleave(rep, dim=1)       # GQA → [L_loc, H, D]
            v_loc = v_loc.repeat_interleave(rep, dim=1)
            sc = torch.einsum("qhd,lhd->hql", q.float(), k_loc.float()) * self.scale
            vis = (torch.tensor(row_seq, device=dev)[None, :] == qseq[:, None]) & \
                  (torch.tensor(row_pos, device=dev)[None, :] <= qpos[:, None])
            sc = sc.masked_fill(~vis[None], float("-inf"))    # [H, T, L_loc]
            m = sc.max(-1).values                             # [H, T]
            m_safe = torch.where(m.isneginf(), torch.zeros_like(m), m)  # 防 NaN
            p = torch.exp(sc - m_safe[..., None])
            l = p.sum(-1)                                     # [H, T]
            part = torch.einsum("hql,lhd->qhd", p, v_loc.float())       # [T, H, D]
        else:
            part = torch.zeros(T, H, D, device=dev)
            m = torch.full((H, T), float("-inf"), device=dev)
            l = torch.zeros(H, T, device=dev)

        # ---- 4. all_gather + LSE 合并（v8.0 数学，w_i ∝ exp(m_i-m*)·l_i）----
        # 3 个张量 pack 成 1 次 all_gather：通信次数 -2/3，且杜绝"部分 gather
        # 已发、部分未发"的中间态（多 rank python 快慢不一时最安全的模式）。
        N = dcp.size if (dcp.group is not None) else 1
        mT, lT = m.t().contiguous(), l.t().contiguous()          # [T, H]
        pad = (-(T * H * D + 2 * T * H)) % N                      # 可整除 N（nccl ag 均分）
        flat = torch.cat([part.reshape(-1).float(), mT.reshape(-1).float(),
                          lT.reshape(-1).float(),
                          torch.zeros(pad, device=dev)])
        bufs = [torch.empty_like(flat) for _ in range(N)]
        if dcp.group is not None:
            dist.all_gather(bufs, flat, group=dcp.group)
        stack = torch.stack(bufs)                                 # [N, S]
        off1, off2 = T * H * D, T * H * D + T * H
        parts = stack[:, :off1].reshape(N, T, H, D)
        ms = stack[:, off1:off2].reshape(N, T, H)
        ls = stack[:, off2:off2 + T * H].reshape(N, T, H)
        m_star = ms.max(0).values                                 # [T, H]
        e = torch.exp(ms - m_star[None])                          # [N, T, H]（空 rank e=0）
        # 约定 (B)：part 未归一化（含 Σexp(s-m_i)·v），合并 = Σ e·part / Σ e·l
        out = torch.einsum("nqhd,nqh->qhd", parts, e) / (e * ls).sum(0)[..., None]
        return out.to(q.dtype).reshape(T, H * D)
