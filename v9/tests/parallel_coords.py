# v8/parallel_coords.py
"""v8 混合并行的地基：进程多维坐标 + 组管理。

每个进程的完整坐标是一个向量（对齐 vLLM parallel_state 的设计）：
    coords = (dp_rank, tp_rank, cp_rank, ep_rank, pp_rank)
world_size = dp × tp × cp × ep × pp（各维独立切分，笛卡尔积）。

每一维对应一个进程组（通信域）：
    TP 组: 层内矩阵同步（allreduce）    —— 坐标除 tp 维外全相同
    EP 组: 专家分片聚合（allreduce）    —— 除 ep 维外全相同
    DCP 组: KV cache 分片合并（ag/rs） —— 除 cp 维外全相同
    DP 组: 请求副本（零通信）           —— 除 dp 维外全相同

用法:
    coords = init_coords(dp=1, tp=1, cp=2, ep=1, pp=1)
    coords["cp"]       # 本进程的 cp 维坐标
    coords.group("cp") # 本进程所属的 DCP 组
"""
import os

import torch
import torch.distributed as dist


class ParallelCoords:
    """进程的多维并行坐标。rank = 线性化（row-major）的坐标向量。"""

    # 维度顺序决定线性化: rank = dp*(tp*cp*ep*pp) + tp*(cp*ep*pp) + ...
    DIMS = ("dp", "tp", "cp", "ep", "pp")

    def __init__(self, world_rank: int, sizes: dict, device: str = "cuda"):
        self.sizes = sizes
        self.world_rank = world_rank
        self.world_size = 1
        for v in sizes.values():
            self.world_size *= v

        # 线性化 rank → 多维坐标（row-major 解码）
        r = world_rank
        self._coord = {}
        for dim in reversed(self.DIMS):        # 从最后维开始剥
            self._coord[dim] = r % sizes[dim]
            r //= sizes[dim]

        # 建组: 每维一个组（本维坐标变化、其余维固定的所有 rank）
        self._groups = {}
        for dim in self.DIMS:
            if sizes[dim] > 1:
                members = [rr for rr in range(self.world_size)
                           if all(self._decode(rr, sizes)[d] == self._coord[d]
                                  for d in self.DIMS if d != dim)]
                self._groups[dim] = dist.new_group(ranks=members)

    @staticmethod
    def _decode(rank: int, sizes: dict) -> dict:
        """纯函数版坐标解码（row-major，与 __init__ 的编码一致）。"""
        r, coord = rank, {}
        for dim in reversed(ParallelCoords.DIMS):
            coord[dim] = r % sizes[dim]
            r //= sizes[dim]
        return coord

    def __getitem__(self, dim: str) -> int:
        return self._coord[dim]

    def group(self, dim: str):
        """本进程在 dim 维的组（None = 该维不并行，单进程）。"""
        return self._groups.get(dim)

    def coord_vector(self) -> dict:
        return dict(self._coord)

    def __repr__(self):
        c = ",".join(f"{k}={v}" for k, v in self._coord.items())
        return f"Coords[{c}] (world={self.world_rank}/{self.world_size})"


def init_coords(dp=1, tp=1, cp=1, ep=1, pp=1, init_dist=False) -> ParallelCoords:
    """初始化坐标系统。
    init_dist=True 时自动 init_process_group（带 device_id——多 new_group 交错
    时 NCCL bootstrap 有竞态：不绑设备 → TCP wire-up 被拒 → 卡死第一个组通信，
    v8.2 实测 25 分钟死锁的根因）。"""
    sizes = {"dp": dp, "tp": tp, "cp": cp, "ep": ep, "pp": pp}
    if init_dist and not dist.is_initialized():
        rank = int(os.environ.get("RANK", "0"))
        dist.init_process_group(
            "nccl", device_id=torch.device(f"cuda:{torch.cuda.current_device()}"))
    if not dist.is_initialized():
        assert sum(sizes.values()) == 5, "单进程模式（所有维度=1）"
        return ParallelCoords(0, sizes)
    return ParallelCoords(dist.get_rank(), sizes)


# ============ 各维的通信原语 ============
