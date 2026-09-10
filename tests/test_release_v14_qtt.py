from __future__ import annotations

import unittest
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from tools.search_diagnostics import load_engine_modules

REPOSITORY = Path(__file__).resolve().parents[1]
CHALLENGER = REPOSITORY / "challengers" / "exp_release_v14_qtt"


class ReleaseV14QttTests(unittest.TestCase):
    search: ClassVar[Any]

    @classmethod
    def setUpClass(cls) -> None:
        _, cls.search = load_engine_modules(CHALLENGER)

    def setUp(self) -> None:
        self.keys = np.zeros(8, dtype=np.uint64)
        self.data = np.zeros(
            (8, self.search.TT_FIELD_COUNT), dtype=np.int32
        )
        self.stats = np.zeros(self.search.STAT_COUNT, dtype=np.int64)

    def _store(
        self,
        key: int,
        score: int,
        bound: int,
        *,
        halfmove: int = 7,
        generation: int = 3,
    ) -> None:
        self.search._qtt_store.py_func(
            np.uint64(key),
            halfmove,
            score,
            bound,
            0,
            generation,
            self.keys,
            self.data,
            self.stats,
        )

    def _probe(
        self,
        key: int,
        *,
        halfmove: int = 7,
        alpha: int = -100,
        beta: int = 100,
    ) -> tuple[int, bool]:
        result = self.search._qtt_probe.py_func(
            np.uint64(key),
            halfmove,
            alpha,
            beta,
            0,
            self.keys,
            self.data,
            self.stats,
        )
        return int(result[0]), bool(result[1])

    def test_exact_entry_requires_matching_key_and_halfmove_clock(self) -> None:
        self._store(5, 42, self.search.TT_EXACT)
        self.assertEqual(self._probe(5), (42, True))
        self.assertEqual(self._probe(5, halfmove=8), (0, False))
        self.assertEqual(self._probe(13), (0, False))
        self.assertEqual(int(self.stats[self.search.STAT_QTT_STORES]), 1)
        self.assertEqual(int(self.stats[self.search.STAT_QTT_PROBES]), 3)
        self.assertEqual(int(self.stats[self.search.STAT_QTT_HITS]), 1)
        self.assertEqual(int(self.stats[self.search.STAT_QTT_CUTOFFS]), 1)

    def test_bounds_only_cut_off_when_the_window_allows_it(self) -> None:
        self._store(1, 80, self.search.TT_LOWER)
        self.assertEqual(self._probe(1, beta=70), (80, True))
        self.assertEqual(self._probe(1, beta=90), (0, False))

        self._store(2, -80, self.search.TT_UPPER)
        self.assertEqual(self._probe(2, alpha=-70), (-80, True))
        self.assertEqual(self._probe(2, alpha=-90), (0, False))

    def test_qsearch_never_replaces_same_key_main_search_entry(self) -> None:
        index = 5
        self.keys[index] = np.uint64(5)
        self.data[index, self.search.TT_SCORE] = np.int32(91)
        self.data[index, self.search.TT_DEPTH] = np.int32(4)
        self.data[index, self.search.TT_BOUND] = np.int32(self.search.TT_EXACT)
        self.data[index, self.search.TT_GENERATION] = np.int32(3)
        self.data[index, self.search.TT_HALFMOVE] = np.int32(7)

        self._store(5, -12, self.search.TT_UPPER)
        self.assertEqual(int(self.data[index, self.search.TT_DEPTH]), 4)
        self.assertEqual(int(self.data[index, self.search.TT_SCORE]), 91)
        self.assertEqual(int(self.stats[self.search.STAT_QTT_STORES]), 0)

    def test_collision_preserves_only_current_generation_main_entry(self) -> None:
        index = 5
        colliding_key = 13
        self.keys[index] = np.uint64(5)
        self.data[index, self.search.TT_DEPTH] = np.int32(4)
        self.data[index, self.search.TT_BOUND] = np.int32(self.search.TT_EXACT)
        self.data[index, self.search.TT_GENERATION] = np.int32(3)

        self._store(colliding_key, 20, self.search.TT_EXACT, generation=3)
        self.assertEqual(int(self.keys[index]), 5)

        self._store(colliding_key, 20, self.search.TT_EXACT, generation=4)
        self.assertEqual(int(self.keys[index]), colliding_key)
        self.assertEqual(int(self.data[index, self.search.TT_DEPTH]), 0)
        self.assertEqual(int(self.stats[self.search.STAT_QTT_STORES]), 1)


if __name__ == "__main__":
    unittest.main()
