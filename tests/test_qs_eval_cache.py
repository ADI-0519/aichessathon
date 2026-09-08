from __future__ import annotations

import unittest
from pathlib import Path
from typing import Any

import numpy as np

from tools.search_diagnostics import load_engine_modules

REPOSITORY = Path(__file__).resolve().parents[1]
CHALLENGER = REPOSITORY / "challengers" / "exp_qs_eval_cache_v7ct"


class QuiescenceEvaluationCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.engine, cls.search = load_engine_modules(CHALLENGER)

    def test_zero_key_requires_a_valid_entry_and_collision_replaces(self) -> None:
        search: Any = self.search
        keys = np.zeros(4, dtype=np.uint64)
        scores = np.zeros(4, dtype=np.int32)
        valid = np.zeros(4, dtype=np.uint8)
        stats = np.zeros(search.STAT_COUNT, dtype=np.int64)
        calls: list[int] = []

        def fake_evaluate(*_arguments: object) -> int:
            calls.append(1)
            return 37 + len(calls)

        original = search.evaluate
        search.evaluate = fake_evaluate
        try:
            cached_qeval = search._cached_qeval.py_func
            unused = np.zeros(1, dtype=np.int64)
            self.assertEqual(
                cached_qeval(unused, unused, np.uint64(0), unused, keys, scores, valid, stats),
                38,
            )
            self.assertEqual(
                cached_qeval(unused, unused, np.uint64(0), unused, keys, scores, valid, stats),
                38,
            )
            self.assertEqual(
                cached_qeval(unused, unused, np.uint64(4), unused, keys, scores, valid, stats),
                39,
            )
            self.assertEqual(
                cached_qeval(unused, unused, np.uint64(0), unused, keys, scores, valid, stats),
                40,
            )
        finally:
            search.evaluate = original

        self.assertEqual(len(calls), 3)
        self.assertEqual(int(stats[search.STAT_Q_EVAL_PROBES]), 4)
        self.assertEqual(int(stats[search.STAT_Q_EVAL_HITS]), 1)

    def test_game_reset_invalidates_cache_entries(self) -> None:
        memory = self.search.SearchMemory.create(10)
        memory.q_eval_valid[7] = np.uint8(1)
        memory.q_eval_keys[7] = np.uint64(123)
        memory.q_eval_scores[7] = np.int32(-45)
        memory.clear()
        self.assertFalse(memory.q_eval_valid.any())


if __name__ == "__main__":
    unittest.main()
