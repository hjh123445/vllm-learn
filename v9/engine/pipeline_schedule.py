"""Pipeline topology and offline scheduling utilities.

This module is CPU-only and intentionally independent of torch/NCCL. It makes
the scheduling policy testable without loading a model and keeps physical
placement separate from model-layer code.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class PipelineTopology:
    """Map logical PP stages to physical worker ranks.

    stage_to_worker may map multiple logical stages to one worker. That is the
    representation needed for interleaved PP. The current one-stage worker
    runtime accepts the common one-to-one mapping; the planner still validates
    and exposes multi-stage placements for a future local-stage executor.
    """

    pp_size: int
    workers: int
    stage_to_worker: tuple[int, ...]

    def __post_init__(self):
        if self.pp_size < 1 or self.workers < 1:
            raise ValueError("pp_size and workers must be positive")
        if len(self.stage_to_worker) != self.pp_size:
            raise ValueError("stage_to_worker length must equal pp_size")
        if any(w < 0 or w >= self.workers for w in self.stage_to_worker):
            raise ValueError("stage_to_worker contains an invalid worker")
        if set(self.stage_to_worker) != set(range(self.workers)):
            raise ValueError("every worker must own at least one stage")

    @classmethod
    def one_to_one(cls, pp_size: int):
        return cls(pp_size, pp_size, tuple(range(pp_size)))

    @classmethod
    def interleaved(cls, pp_size: int, workers: int):
        if pp_size < workers:
            raise ValueError("pp_size must be >= workers")
        mapping = tuple(stage % workers for stage in range(pp_size))
        return cls(pp_size, workers, mapping)

    def stages_for_worker(self, worker: int) -> tuple[int, ...]:
        return tuple(s for s, w in enumerate(self.stage_to_worker) if w == worker)

    @property
    def has_local_multiple_stages(self) -> bool:
        return any(len(self.stages_for_worker(w)) > 1
                   for w in range(self.workers))

    def worker_for_stage(self, stage: int) -> int:
        return self.stage_to_worker[stage]

    def as_dict(self):
        return {
            "pp_size": self.pp_size,
            "workers": self.workers,
            "stage_to_worker": list(self.stage_to_worker),
            "stages_per_worker": [
                list(self.stages_for_worker(w)) for w in range(self.workers)
            ],
            "has_local_multiple_stages": self.has_local_multiple_stages,
        }


@dataclass(frozen=True)
class PipelineAction:
    """One idealized forward action used to inspect a PP schedule."""

    kind: str
    stage: int
    microbatch_id: int
    lane: int
    clock: int


def build_forward_schedule(pp_size: int, microbatches: int, lanes: int = 2,
                           mode: str = "optimized") -> list[PipelineAction]:
    """Build an idealized forward-only schedule for tests and planning.

    This is not the worker runtime scheduler or a measured execution plan. It
    does not implement training scheduling or DualPipe.
    """
    p, m = int(pp_size), int(microbatches)
    if p < 1 or m < 0 or lanes < 1:
        raise ValueError("pp_size, microbatches and lanes must be valid")
    if mode not in ("baseline", "optimized"):
        raise ValueError("mode must be baseline or optimized")
    actions = []
    if mode == "baseline":
        clock = 0
        for mb in range(m):
            for stage in range(p):
                actions.append(PipelineAction("forward", stage, mb,
                                               mb % lanes, clock))
                clock += 1
        return actions
    # At ideal unit-duration compute, stage s starts mb m at s+m. Sorting by
    # clock exposes the overlap window and keeps stage order deterministic.
    for mb in range(m):
        for stage in range(p):
            actions.append(PipelineAction("forward", stage, mb,
                                           mb % lanes, stage + mb))
    return sorted(actions, key=lambda a: (a.clock, a.stage, a.microbatch_id))
