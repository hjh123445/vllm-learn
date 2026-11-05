"""Canonical TP/CP/EP group order shared by every distributed worker."""

from itertools import product


def group_plan(tp_size, cp_size, ep_size, pp_size):
    sizes = (pp_size, ep_size, tp_size, cp_size)

    def rank(p, e, t, c):
        return ((p * ep_size + e) * tp_size + t) * cp_size + c

    for name, axis in (("tp", 2), ("cp", 3), ("ep", 1)):
        if sizes[axis] == 1:
            continue
        fixed = [range(n) if i != axis else (0,)
                 for i, n in enumerate(sizes)]
        for coords in product(*fixed):
            members = []
            for value in range(sizes[axis]):
                point = list(coords)
                point[axis] = value
                members.append(rank(*point))
            yield name, members


def create_groups(rank, tp_size, cp_size, ep_size, pp_size):
    import torch.distributed as dist
    groups = dict(tp=None, cp=None, ep=None)
    # new_group is collective over the default group, including nonmembers.
    # Creating only one's own group gives incompatible rendezvous sequences.
    for name, members in group_plan(tp_size, cp_size, ep_size, pp_size):
        group = dist.new_group(ranks=members)
        if rank in members:
            groups[name] = group
    return groups


def create_pipeline_edge_groups(rank, inner_size, pp_size):
    """Isolate each adjacent PP link from other links and world collectives.

    Every worker creates groups in the same order, including nonmembers.
    Two-rank groups avoid serializing unrelated P2P links on an eagerly
    initialized NCCL world communicator. Returns this rank's input/output link.
    """
    import torch.distributed as dist
    incoming, outgoing = None, None
    for stage in range(pp_size - 1):
        for inner in range(inner_size):
            members = [stage * inner_size + inner, (stage + 1) * inner_size + inner]
            group = dist.new_group(ranks=members)
            if rank == members[0]:
                outgoing = group
            elif rank == members[1]:
                incoming = group
    return incoming, outgoing
