"""Pure tests for PP topology and virtual-engine bookkeeping."""

from engine.pipeline_schedule import (
    PipelineTopology, VirtualEnginePool, build_forward_schedule)


def test_interleaved_topology_can_describe_multiple_stages_per_worker():
    topology = PipelineTopology.interleaved(pp_size=4, workers=2)
    assert topology.stage_to_worker == (0, 1, 0, 1)
    assert topology.stages_for_worker(0) == (0, 2)
    assert topology.has_local_multiple_stages is True
    assert topology.as_dict()["stages_per_worker"] == [[0, 2], [1, 3]]


def test_virtual_engine_pool_lifecycle_is_explicit():
    pool = VirtualEnginePool(2)
    first = pool.assign(0)
    second = pool.assign(1)
    third = pool.assign(2)
    assert (first.slot_id, first.lane) == (0, 0)
    assert (second.slot_id, second.lane) == (1, 1)
    assert (third.slot_id, third.lane) == (0, 0)
    assert pool.snapshot()[0]["in_flight"] == 2
    pool.complete(0)
    pool.complete(1)
    pool.complete(0)
    assert all(row["in_flight"] == 0 for row in pool.snapshot())



def test_forward_schedule_exposes_fill_and_overlap_clock():
    actions = build_forward_schedule(pp_size=2, microbatches=3)
    assert [(a.stage, a.microbatch_id, a.clock) for a in actions] == [
        (0, 0, 0), (0, 1, 1), (1, 0, 1),
        (0, 2, 2), (1, 1, 2), (1, 2, 3)]
