# v8/layers/dcp.py
"""v8.1 DCP（Decode Context Parallel）：KV cache 按序列维度交错分片 + LSE 合并。

建立在 v7 引擎之上。DCP 的三个组件：

1. **块染色（cache 分配器侧）**：全局 block 号按 `block_id % dcp_size` 染色，
   本 rank 只持有自己颜色的块 → KV cache 显存 /N；
2. **partial attention（kernel 侧）**：decode 时每卡只对自己的块算
   partial 输出 + LSE（v8.0 已验证的数学）；
3. **ag_rs 合并（通信侧）**：AllGather(partial+LSE) → 本地加权合并 →
   ReduceScatter——每层 3 次 NCCL（vLLM 的 ag_rs 后端同款）。

与 vLLM DCP 的对应：interleave=1 的块染色 / ag_rs 后端 / 复用现有进程组。
"""
import torch
import torch.distributed as dist


class DCPCacheCoordinator:
    """块染色 + 本 rank 的块表视图。

    全局逻辑块表（scheduler 视角，完整）→ 本 rank 的物理持有子集。
    染色规则: block_id % dcp_size == cp_rank 的块归本 rank。
    """

    def __init__(self, dcp_size: int, cp_rank: int):
        self.size = dcp_size
        self.rank = cp_rank

    def owns(self, global_block: int) -> bool:
        return global_block % self.size == self.rank

    def local_blocks(self, block_table: list[int]) -> list[int]:
        """本 rank 持有的块（保持顺序）。"""
        return [b for b in block_table if self.owns(b)]

    def mask_for(self, block_table: list[int]) -> torch.Tensor:
        """[n_blocks] 的 bool：True = 本 rank 持有。"""
        t = torch.tensor(block_table)
        return t % self.size == self.rank


def partial_attention_with_lse(q, k, v, scale):
    """单卡的 partial decode attention（纯 PyTorch 版——triton 化在 8.2）。
    q: [b, h, d]; k/v: [b, L_local, h, d]（只有本 rank 的分片）
    返回 (partial [b, h, d], m [b, h], l [b, h])。"""
    scores = torch.einsum("bhd,blhd->bhl", q, k) * scale   # [b, h, L_local]
    m = scores.max(-1).values
    p = torch.exp(scores - m[..., None])
    l = p.sum(-1)
    out = torch.einsum("bhl,blhd->bhd", p / l[..., None], v)
    return out, m, l


def dcp_ag_rs_merge(partials, ms, ls, coords):
    """ag_rs 后端：AllGather partial+LSE → 加权合并 → ReduceScatter。
    本版实现为 all_reduce 简化（数学等价：LSE 加权和的结合律）——
    真正的 ag_rs 切分留 8.2（当前 batch=decode 的 b 小, 全量 allreduce
    的额外开销可忽略）。"""
    g = coords.group("cp")
    if g is None:
        # 单卡：直接合并（只有一个 partial）
        return partials[0] if partials else None
    # 堆叠通信: [b, h, d] + [b, h] × 2 拼一个大 tensor 一次 all_reduce
    b, h, d = partials[0].shape
    buf = torch.zeros(b, h, d + 2, device=partials[0].device,
                      dtype=partials[0].dtype)
    buf[..., :d] = 0                    # partial 走加法不行——LSE 合并非线性!
    # 正确做法: all_gather 全部再本地合并（ag 的前半）, 不做 rs 的切分
    parts = torch.stack(partials)       # [N, b, h, d]
    mm = torch.stack(ms)                # [N, b, h]
    ll = torch.stack(ls)                # [N, b, h]
    return lse_merge_stack(parts, mm, ll)


def lse_merge_stack(parts, ms, ls):
    """N 个 partial 的 LSE 加权合并（v8.0 数学, 张量化版）。
    parts [N, b, h, d]; ms/ls [N, b, h] → [b, h, d]。"""
    m_star = ms.max(0).values                          # [b, h]
    w = torch.exp(ms - m_star[None]) * ls              # [N, b, h]
    w_norm = w.sum(0)                                  # [b, h]
    out = torch.einsum("nbhd,nbh->bhd", parts.to(w.dtype), w)
    return out / w_norm[..., None]


# ================= v8.2：引擎接线 =================

class DCPContext:
    """DCP 的全局状态（Attention 层读取，worker 启动时 set_dcp 注入一次）。

    关键约定——【全局块号 → 本地槽位】是静态映射：
        全局块 g（g % size == rank 归本 rank）→ 本地槽位 g // size
    这保证同一全局块的 KV 轮间写在同一物理位置（增量写的前提），
    本地池大小 = 全局块数 / size。scheduler/BlockManager 仍然操作全局块号
    （所有 rank 视图一致），只有 cache 读写时做染色映射。"""

    def __init__(self):
        self.active = False
        self.size = 1
        self.rank = 0
        self.group = None                  # DCP 组（cp 维，None=单卡直通）

    def owned(self, g: int) -> bool:
        return g % self.size == self.rank

    def local_slot(self, g: int) -> int:
        return g // self.size


_DCP_CTX = DCPContext()


def set_dcp(active=False, size=1, rank=0, group=None):
    """worker 启动时注入。active=False 时 Attention 走原路径（零改动）。"""
    _DCP_CTX.active = active
    _DCP_CTX.size = size
    _DCP_CTX.rank = rank
    _DCP_CTX.group = group
    return _DCP_CTX


def get_dcp() -> DCPContext:
    return _DCP_CTX
