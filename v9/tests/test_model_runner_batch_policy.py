"""Batch-shape policy keeps request ordering and selects the intended GEMM M."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from engine.model_runner import ModelRunner


class _FakeModel:
    def __init__(self):
        self.batch_sizes = []

    def __call__(self, input_ids, positions, intermediate_tensors=None):
        self.batch_sizes.append(input_ids.shape[0])
        return input_ids.float().unsqueeze(-1)

    def compute_logits(self, hidden):
        return hidden


class _GreedySampler:
    def __call__(self, logits, temperatures):
        return logits[:, 0].long()


class ModelRunnerBatchPolicyTests(unittest.TestCase):
    def _runner(self, mode):
        runner = ModelRunner.__new__(ModelRunner)
        runner.config = SimpleNamespace(batch_execution_mode=mode,
                                        hf_config=SimpleNamespace(hidden_size=1))
        runner.device = "cpu"
        runner.dtype = torch.float32
        runner.model = _FakeModel()
        runner.sampler = _GreedySampler()
        runner.prepare_decode = lambda seqs: (
            torch.tensor([s.last_token for s in seqs]),
            torch.zeros(len(seqs), dtype=torch.long))
        return runner

    def _run(self, runner, pp_size=1, pp_rank=0):
        state = SimpleNamespace(pp_size=pp_size, pp_rank=pp_rank,
                                pipeline_lane=0, microbatch_id=0,
                                pipeline_warmup=False,
                                pipeline_transport=SimpleNamespace(recv_forward=lambda lane, **kw:
                                    (torch.zeros(kw['expected_shape']),
                                     torch.zeros(kw['expected_shape']),
                                     kw['expected_microbatch_id'], lane)))
        seqs = [SimpleNamespace(last_token=i, temperature=0.) for i in (7, 3, 9)]
        with patch("layers.parallel.STATE", state):
            result = runner.run(seqs, is_prefill=False)
        return result

    def test_batched_mode_uses_one_forward(self):
        runner = self._runner("batched")
        self.assertEqual(self._run(runner), [7, 3, 9])
        self.assertEqual(runner.model.batch_sizes, [3])

    def test_per_request_mode_preserves_order_on_last_pipeline_stage(self):
        runner = self._runner("per_request")
        self.assertEqual(self._run(runner, pp_size=2, pp_rank=1), [7, 3, 9])
        self.assertEqual(runner.model.batch_sizes, [1, 1, 1])


if __name__ == "__main__":
    unittest.main()
