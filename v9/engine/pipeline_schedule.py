"""Pipeline topology, virtual-engine assignment, and trace utilities.

This module is CPU-only and intentionally independent of torch/NCCL. It makes
the scheduling policy testable without loading a model and keeps physical
placement separate from model-layer code.
"""

from dataclasses import dataclass, field


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


@dataclass
class VirtualEngineSlot:
    """One forward lane's ownership and in-flight accounting."""

    slot_id: int
    lane: int
    in_flight: int = 0
    submitted: int = 0
    completed: int = 0


class VirtualEnginePool:
    """Stable microbatch-to-lane assignment with explicit lifecycle."""

    def __init__(self, size: int):
        self.size = max(1, int(size))
        self.slots = [VirtualEngineSlot(i, i % self.size)
                      for i in range(self.size)]

    def assign(self, microbatch_id: int) -> VirtualEngineSlot:
        slot = self.slots[int(microbatch_id) % self.size]
        slot.in_flight += 1
        slot.submitted += 1
        return slot

    def complete(self, slot_id: int):
        slot = self.slots[int(slot_id) % self.size]
        if slot.in_flight <= 0:
            raise RuntimeError(f"virtual engine {slot_id} has no in-flight work")
        slot.in_flight -= 1
        slot.completed += 1

    def snapshot(self):
        return [
            {
                "slot_id": s.slot_id,
                "lane": s.lane,
                "in_flight": s.in_flight,
                "submitted": s.submitted,
                "completed": s.completed,
            }
            for s in self.slots
        ]


@dataclass
class TraceEvent:
    name: str
    ts: float
    end: float
    stage: int = -1
    microbatch_id: int = -1
    lane: int = -1
    category: str = "pipeline"
    args: dict = field(default_factory=dict)

    def as_chrome(self, pid: int = 0, tid: int = 0, origin: float = 0.0):
        return {
            "name": self.name,
            "cat": self.category,
            "ph": "X",
            "ts": (self.ts - origin) * 1e6,
            "dur": max(0.0, self.end - self.ts) * 1e6,
            "pid": pid,
            "tid": tid,
            "args": {
                "stage": self.stage,
                "microbatch_id": self.microbatch_id,
                "lane": self.lane,
                **self.args,
            },
        }


class PipelineTrace:
    """Collect host-side stage/transport events and export Chrome trace JSON."""

    def __init__(self, origin: float = 0.0):
        self.origin = origin
        self.events: list[TraceEvent] = []

    def add(self, name, start, end, *, stage=-1, microbatch_id=-1,
            lane=-1, category="pipeline", **args):
        self.events.append(TraceEvent(
            name, float(start), float(end), stage, int(microbatch_id),
            int(lane), category, args))

    def chrome_events(self, pid=0, tid=0):
        return [event.as_chrome(pid, tid, self.origin)
                for event in self.events]

    def as_dict(self, *, pid=0, tid=0, metadata=None):
        return {
            "displayTimeUnit": "ms",
            "traceEvents": self.chrome_events(pid, tid),
            "metadata": metadata or {},
        }

    def summary(self):
        by_name = {}
        for event in self.events:
            by_name.setdefault(event.name, 0.0)
            by_name[event.name] += max(0.0, event.end - event.ts)
        return by_name



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
    """Build the deterministic forward-only schedule used by trace planning.

    ``optimized`` emits a fill/steady/drain order where a stage can start
    microbatch ``m`` after its predecessor has produced it. ``baseline`` keeps
    the same dependency order but serializes each microbatch through all
    stages, which is useful as a control trace. This is an execution plan and
    does not pretend to implement training DualPipe backward priority.
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
