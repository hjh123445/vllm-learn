"""Pure tests for PP topology and the offline idealized planner."""

from engine.pipeline_schedule import (
    PipelineTopology, build_forward_schedule)


def test_interleaved_topology_can_describe_multiple_stages_per_worker():
    topology = PipelineTopology.interleaved(pp_size=4, workers=2)
    assert topology.stage_to_worker == (0, 1, 0, 1)
    assert topology.stages_for_worker(0) == (0, 2)
    assert topology.has_local_multiple_stages is True
    assert topology.as_dict()["stages_per_worker"] == [[0, 2], [1, 3]]


def test_forward_schedule_exposes_fill_and_overlap_clock():
    actions = build_forward_schedule(pp_size=2, microbatches=3)
    assert [(a.stage, a.microbatch_id, a.clock) for a in actions] == [
        (0, 0, 0), (0, 1, 1), (1, 0, 1),
        (0, 2, 2), (1, 1, 2), (1, 2, 3)]
