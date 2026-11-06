"""DCP cache ownership shared by worker initialization and Attention.

Blocks are interleaved by global ID; decode partials are gathered and
merged by log-sum-exp in layers/attention.py. No ReduceScatter backend.
"""


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
