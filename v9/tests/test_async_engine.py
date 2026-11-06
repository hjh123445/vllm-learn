"""CPU concurrency regression: exercise the adapter with an owner-checked core."""
import itertools
import pathlib
import sys
import threading
import time
import types
import unittest
from unittest.mock import patch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from engine.async_engine import AsyncEngine, EngineOverloaded
from samplingParams import SamplingParams

class FakeCore:
    gate = None
    fail = False
    def __init__(self, *args, **kwargs):
        self.config = types.SimpleNamespace(max_model_len=8192, eos=0)
        self.pending = {}
        self.ids = itertools.count()
        self.entered = threading.Event()
        self.last_step_had_work = False
        self.owner = None
        self.closed = False
    def owned(self):
        current = threading.get_ident()
        self.owner = self.owner or current
        assert self.owner == current, 'scheduler mutated from another thread'
    def add_request(self, prompt, sp):
        self.owned()
        seq = types.SimpleNamespace(seq_id=next(self.ids), completion_token_ids=[],
            max_tokens=sp.max_tokens, ignore_eos=True, is_finished=False)
        self.pending[seq.seq_id] = seq
        return seq
    def abort_request(self, sid):
        self.owned()
        self.pending.pop(sid, None)
    def is_finished(self):
        return not self.pending
    def step(self):
        self.owned()
        self.entered.set()
        if self.gate is not None:
            assert self.gate.wait(3), 'test step timed out'
        if self.fail:
            raise RuntimeError('injected worker failure')
        for sid, s in list(self.pending.items()):
            s.completion_token_ids.append(42)
            s.is_finished = len(s.completion_token_ids) == s.max_tokens
            if s.is_finished:
                del self.pending[sid]
        self.last_step_had_work = True
        time.sleep(.001)
    def shutdown(self):
        self.closed = True

class TestAsyncEngine(unittest.TestCase):
    def setUp(self):
        FakeCore.gate, FakeCore.fail = None, False
        tok = types.SimpleNamespace(encode=lambda *a, **k: [1,2])
        self.patches = patch.dict(sys.modules, {
            'transformers': types.SimpleNamespace(AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *a, **k: tok)),
            'engine.core': types.SimpleNamespace(EngineCore=FakeCore)})
        self.patches.start()
        self.engines = []
    def tearDown(self):
        if FakeCore.gate is not None:
            FakeCore.gate.set()
        for e in self.engines:
            e.stop()
        self.patches.stop()
    def engine(self, **kw):
        e = AsyncEngine('fake', **kw)
        self.engines.append(e)
        e.start()
        return e
    def consume(self, st):
        tokens = []
        while True:
            t = st.q.get(timeout=3)
            if t is None:
                return tokens
            tokens.append(t)
    def wait_empty(self, e):
        deadline=time.monotonic()+3
        while e.stats()['active'] and time.monotonic()<deadline:
            time.sleep(.002)
        self.assertEqual(e.stats()['active'], 0)
        self.assertFalse(e._seqs)
    def test_generate_and_repeated_cleanup(self):
        e=self.engine()
        for _ in range(50):
            self.assertEqual(e.generate([1],SamplingParams(max_tokens=3)),([42]*3,'length'))
        self.wait_empty(e)
        self.assertEqual(e.stats()['completed'],50)
    def test_admission_during_blocking_step_and_cancellation(self):
        FakeCore.gate=threading.Event()
        e=self.engine(max_pending_requests=2)
        _,a=e.add([1],SamplingParams(max_tokens=5))
        self.assertTrue(e.engine.entered.wait(3))
        start=time.monotonic()
        rid,b=e.add([2],SamplingParams(max_tokens=5))
        self.assertLess(time.monotonic()-start,.1)
        with self.assertRaises(EngineOverloaded):
            e.add([3],SamplingParams(max_tokens=1))
        e.abort(rid)
        FakeCore.gate.set()
        self.assertEqual(self.consume(b),[])
        self.assertEqual(b.finish_reason,'cancelled')
        self.assertEqual(self.consume(a),[42]*5)
        self.wait_empty(e)
    def test_slow_consumer_does_not_block_other_requests(self):
        e=self.engine(stream_buffer_tokens=2)
        _,slow=e.add([1],SamplingParams(max_tokens=10))
        self.assertEqual(e.generate([2],SamplingParams(max_tokens=1)),([42],'length'))
        deadline=time.monotonic()+3
        while not slow.finished and time.monotonic()<deadline: time.sleep(.002)
        self.assertIsInstance(slow.error,RuntimeError)
        self.assertLessEqual(slow.q.qsize(),3)
        self.consume(slow)
        self.wait_empty(e)
    def test_failure_propagates_and_stop_wakes_waiters(self):
        FakeCore.fail=True
        e=self.engine()
        with self.assertRaisesRegex(RuntimeError,'injected worker'):
            e.generate([1],SamplingParams(max_tokens=5))
        self.wait_empty(e)
        FakeCore.fail=False
        FakeCore.gate=threading.Event()
        e2=self.engine()
        _,st=e2.add([1],SamplingParams(max_tokens=100))
        self.assertTrue(e2.engine.entered.wait(3))
        FakeCore.gate.set();e2.stop()
        self.consume(st)
        self.assertTrue(st.finished)
        self.wait_empty(e2)
    def test_input_validation(self):
        e=self.engine()
        for ids in ([],[-1],[True]):
            with self.assertRaises(ValueError):e.add(ids,SamplingParams(max_tokens=1))

if __name__=='__main__': unittest.main()
