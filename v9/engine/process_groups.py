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
