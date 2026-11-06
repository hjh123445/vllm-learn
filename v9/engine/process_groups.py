"""Canonical DP/PP/TP/CP topology; EP overlays DP x TP, never a fifth axis."""

from dataclasses import dataclass
from itertools import product


@dataclass(frozen=True)
class ParallelTopology:
    tp_size: int = 1
    dp_size: int = 1
    pp_size: int = 1
    cp_size: int = 1
    enable_expert_parallel: bool = False

    def __post_init__(self):
        if min(self.tp_size, self.dp_size, self.pp_size, self.cp_size) < 1:
            raise ValueError("parallel sizes must be positive")

    @property
    def local_world_size(self):
        return self.pp_size * self.tp_size * self.cp_size

    @property
    def world_size(self):
        return self.dp_size * self.local_world_size

    @property
    def ep_size(self):
        return self.dp_size * self.tp_size if self.enable_expert_parallel else 1

    def rank(self, d, p, t, c=0):
        return ((d * self.pp_size + p) * self.tp_size + t) * self.cp_size + c

    def coords(self, rank):
        if not 0 <= rank < self.world_size:
            raise ValueError(f"invalid rank {rank}")
        q, c = divmod(rank, self.cp_size)
        q, t = divmod(q, self.tp_size)
        d, p = divmod(q, self.pp_size)
        return d, p, t, c


def group_plan(tp_size=1, cp_size=1, dp_size=1, pp_size=1,
               enable_expert_parallel=False):
    topology = ParallelTopology(tp_size, dp_size, pp_size, cp_size,
                                enable_expert_parallel)
    sizes = (dp_size, pp_size, tp_size, cp_size)
    for name, axis in (("tp", 2), ("cp", 3), ("dp", 0)):
        if sizes[axis] == 1:
            continue
        fixed = [range(n) if i != axis else (0,)
                 for i, n in enumerate(sizes)]
        for coords in product(*fixed):
            members = []
            for value in range(sizes[axis]):
                point = list(coords)
                point[axis] = value
                members.append(topology.rank(*point))
            yield name, members
    if enable_expert_parallel and topology.ep_size > 1:
        for p, c in product(range(pp_size), range(cp_size)):
            yield "ep", [topology.rank(d, p, t, c)
                         for d in range(dp_size) for t in range(tp_size)]


def create_groups(rank, tp_size=1, cp_size=1, dp_size=1, pp_size=1,
                  enable_expert_parallel=False):
    import os
    import torch.distributed as dist
    groups = dict(tp=None, cp=None, dp=None, ep=None, ep2=None)
    # new_group is collective over the default group, including nonmembers.
    # Creating only one's own group gives incompatible rendezvous sequences.
    plans = list(group_plan(tp_size, cp_size, dp_size, pp_size,
                            enable_expert_parallel))
    for name, members in plans:
        group = dist.new_group(ranks=members)
        if rank in members:
            groups[name] = group
    # Optional duplicate EP communicator for the dispatch/combine pipeline
    # (VLLM_LEARN_EP_OVERLAP=1). Same membership, separate NCCL stream so two
    # all-to-allv can be in flight at once. Created symmetrically on all ranks.
    if os.getenv("VLLM_LEARN_EP_OVERLAP", "0") == "1":
        for name, members in plans:
            if name != "ep":
                continue
            group = dist.new_group(ranks=members)
            if rank in members:
                groups["ep2"] = group
    return groups


def create_pipeline_edge_groups(rank, inner_size, pp_size, dp_size=1):
    """Isolate each adjacent PP link from other links and world collectives.

    Every worker creates groups in the same order, including nonmembers.
    Two-rank groups avoid serializing unrelated P2P links on an eagerly
    initialized NCCL world communicator. Returns this rank's input/output link.
    """
    import torch.distributed as dist
    incoming, outgoing = None, None
    for replica in range(dp_size):
        for stage, inner in product(range(pp_size - 1), range(inner_size)):
            base = replica * pp_size * inner_size
            members = [base + stage * inner_size + inner,
                       base + (stage + 1) * inner_size + inner]
            group = dist.new_group(ranks=members)
            if rank == members[0]:
                outgoing = group
            elif rank == members[1]:
                incoming = group
    return incoming, outgoing
