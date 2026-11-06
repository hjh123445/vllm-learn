"""Pure CPU contracts: EP overlays DP x TP; idle DP ranks cannot skip a tick."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine.process_groups import ParallelTopology, group_plan
from engine.frontend import DPFrontend
from test_pipeline_batch_queue import core_for_test


@pytest.mark.parametrize("dp,tp,pp,cp", [(1, 8, 1, 1), (8, 1, 1, 1), (2, 2, 2, 1), (2, 2, 2, 2)])
def test_ep_size_and_membership(dp, tp, pp, cp):
    topology = ParallelTopology(tp, dp, pp, cp, True)
    assert topology.ep_size == dp * tp
    assert topology.world_size == dp * tp * pp * cp
    ep = [members for name, members in group_plan(tp, cp, dp, pp, True) if name == "ep"]
    assert len(ep) == pp * cp
    assert sorted(rank for group in ep for rank in group) == list(range(topology.world_size))
    for members in ep:
        coords = [topology.coords(rank) for rank in members]
        assert len({(p, c) for d, p, t, c in coords}) == 1
        assert {(d, t) for d, p, t, c in coords} == {(d, t) for d in range(dp) for t in range(tp)}
    for rank in range(topology.world_size):
        assert topology.rank(*topology.coords(rank)) == rank


def test_disabled_ep_has_no_extra_dimension_or_group():
    topology = ParallelTopology(tp_size=2, dp_size=2, pp_size=2)
    assert topology.ep_size == 1 and topology.world_size == 8
    assert all(name != "ep" for name, _ in group_plan(2, 1, 2, 2))


def test_old_independent_ep_rejected_before_loading_model():
    from engine.core import EngineCore
    with pytest.raises(TypeError, match="ep_size"):
        EngineCore("unused", tp_size=2, ep_size=2)
    with pytest.raises(TypeError, match="ep_size"):
        DPFrontend("unused", dp_size=2, tp_size=2, ep_size=2, enable_expert_parallel=True)


def test_idle_core_issues_empty_batch_and_can_finish_with_dummies_pending():
    core = core_for_test(pp=2)
    core._coupled_dp = True
    core.step()
    assert core.executor.messages[-1]["scheduled"] == []
    assert len(core.batch_queue) == 1 and core.is_finished()
    core.step()
    assert len(core.executor.messages) == 2
    assert core.batch_queue[0].batch_id == 1


class Replica:
    def __init__(self, finished):
        self.finished = finished
        self.commands = []

    def send(self, command):
        self.commands.append(command[0])

    def recv(self):
        return ("finished", self.finished) if self.commands[-1] == "finished" else ("step", [], [])


@pytest.mark.parametrize("states", [(True, False), (False, True), (True, True)])
def test_frontend_drives_idle_peers_until_globally_finished(states):
    frontend = DPFrontend.__new__(DPFrontend)
    frontend.dp_size, frontend._coupled = 2, True
    frontend._conns = [Replica(state) for state in states]
    frontend.last_step_events = []
    assert frontend.step_all() == {}
    expected = ["finished"] if all(states) else ["finished", "step"]
    assert all(replica.commands == expected for replica in frontend._conns)
