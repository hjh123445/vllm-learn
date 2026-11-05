"""CPU regression for request ids that collide between spawned DP replicas."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine.frontend import DPFrontend


class Replica:
    def __init__(self, tokens, events=None):
        self.tokens = tokens
        self.events = events or []

    def send(self, command):
        self.command = command[0]

    def recv(self):
        return {'add': ('seq', 0), 'finished': ('finished', False),
                'step': ('step', [(0, self.tokens)], self.events)}[self.command]


class FrontendTests(unittest.TestCase):
    def frontend(self):
        fe = DPFrontend.__new__(DPFrontend)
        fe.dp_size = 2
        fe._rr = fe._next_seq_id = 0
        fe._seq_core = {}
        fe._local_to_global = {}
        fe._conns = [Replica([10]), Replica([20])]
        return fe

    def test_colliding_local_ids(self):
        fe = self.frontend()
        a, rank_a = fe.add_request([1])
        b, rank_b = fe.add_request([2])
        self.assertEqual((rank_a, rank_b), (0, 1))
        self.assertNotEqual(a.seq_id, b.seq_id)
        self.assertEqual(fe.step_all(), {a.seq_id: [10], b.seq_id: [20]})
        self.assertFalse(fe._local_to_global)

    def test_worker_exception_propagates(self):
        fe = self.frontend()
        # Missing mapping simulates an invalid response from a replica.
        with self.assertRaisesRegex(RuntimeError, 'DP replica step failed'):
            fe.step_all()

    def test_step_events_are_namespaced_before_completion_removes_mapping(self):
        fe = self.frontend()
        fe._conns = [Replica([10], [{'seq_id': 0, 'step_end': 1.25}]),
                     Replica([20], [{'seq_id': 0, 'step_end': 2.5}])]
        a, _ = fe.add_request([1])
        b, _ = fe.add_request([2])
        fe.step_all()
        events = {e['seq_id']: e for e in fe.last_step_events}
        self.assertEqual(events[a.seq_id]['dp_rank'], 0)
        self.assertEqual(events[b.seq_id]['dp_rank'], 1)
        self.assertEqual(events[a.seq_id]['step_end'], 1.25)
        self.assertEqual(events[b.seq_id]['step_end'], 2.5)
        self.assertFalse(fe._local_to_global)


if __name__ == '__main__':
    unittest.main()
