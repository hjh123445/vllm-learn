"""Prefix-cache ablation must not reuse a previous request's KV blocks."""
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine.block_manager import BlockManager
from engine.sequence import Sequence


class CacheSwitchTests(unittest.TestCase):
    def cached_blocks(self, enabled):
        manager = BlockManager(16, 256, enable_prefix_caching=enabled)
        seq = Sequence(list(range(600)))
        manager.allocate(seq, 0)
        seq.num_scheduled_tokens = 600
        manager.hash_blocks(seq)
        manager.deallocate(seq)
        return manager.can_allocate(Sequence(list(range(600)))), manager

    def test_disabled_recomputes_all_prompt_tokens(self):
        hits, manager = self.cached_blocks(False)
        self.assertEqual(hits, 0)
        self.assertFalse(manager.hash_to_block_id)

    def test_default_still_reuses_full_blocks(self):
        hits, _ = self.cached_blocks(True)
        self.assertEqual(hits, 2)


if __name__ == '__main__':
    unittest.main()
