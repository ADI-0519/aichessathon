from __future__ import annotations

import unittest
from pathlib import Path
from typing import Any, ClassVar

import chess
import numpy as np

from tools.search_diagnostics import load_engine_modules

REPOSITORY = Path(__file__).resolve().parents[1]
CHALLENGER = REPOSITORY / "challengers" / "exp_release_v15_search"


class ReleaseV15SearchTests(unittest.TestCase):
    engine: ClassVar[Any]
    search: ClassVar[Any]

    @classmethod
    def setUpClass(cls) -> None:
        cls.engine, cls.search = load_engine_modules(CHALLENGER)

    def setUp(self) -> None:
        self.keys = np.zeros(8, dtype=np.uint64)
        self.data = np.zeros(
            (8, self.search.TT_FIELD_COUNT),
            dtype=np.int32,
        )
        self.stats = np.zeros(self.search.STAT_COUNT, dtype=np.int64)

    def test_reverse_futility_never_prunes_singular_verification(self) -> None:
        allowed = self.search._reverse_futility_allowed
        self.assertTrue(allowed(0, True, self.search.V10_RFP_MAX_DEPTH, True))
        self.assertFalse(allowed(1, True, self.search.V10_RFP_MAX_DEPTH, True))
        self.assertFalse(allowed(0, False, self.search.V10_RFP_MAX_DEPTH, True))
        self.assertFalse(allowed(0, True, self.search.V10_RFP_MAX_DEPTH + 1, True))
        self.assertFalse(allowed(0, True, self.search.V10_RFP_MAX_DEPTH, False))

    def _entry(self, index: int, key: int, depth: int, generation: int) -> None:
        self.keys[index] = np.uint64(key)
        self.data[index, self.search.TT_DEPTH] = np.int32(depth)
        self.data[index, self.search.TT_BOUND] = np.int32(self.search.TT_EXACT)
        self.data[index, self.search.TT_GENERATION] = np.int32(generation)

    def test_bucket_probe_finds_either_slot_and_prefers_deeper_duplicate(self) -> None:
        self._entry(0, 8, 4, 1)
        self._entry(1, 8, 7, 1)
        probe = self.search._tt_probe_index.py_func
        self.assertEqual(probe(np.uint64(8), self.keys, self.data), 1)
        self.assertEqual(probe(np.uint64(16), self.keys, self.data), -1)

    def test_replacement_uses_empty_then_stale_shallow_slot(self) -> None:
        replacement = self.search._tt_replacement_index.py_func
        self._entry(0, 8, 9, 2)
        self.assertEqual(
            replacement(np.uint64(16), 4, 2, self.keys, self.data, self.stats),
            1,
        )
        self._entry(1, 24, 3, 1)
        self.assertEqual(
            replacement(np.uint64(16), 4, 2, self.keys, self.data, self.stats),
            1,
        )

    def test_same_key_shallow_result_does_not_replace_deep_current_entry(self) -> None:
        self._entry(0, 8, 12, 3)
        replacement = self.search._tt_replacement_index.py_func
        self.assertEqual(
            replacement(np.uint64(8), 4, 3, self.keys, self.data, self.stats),
            -1,
        )

    def test_capture_history_uses_bounded_signed_gravity(self) -> None:
        board = chess.Board("8/8/8/3p4/4P3/8/8/4K2k w - - 0 1")
        position = self.engine.position_from_board(board)
        move = next(
            move
            for move in self.engine.legal_moves(position)
            if self.engine.move_to_uci(move) == "e4d5"
        )
        table = np.zeros(
            (2, self.engine.PIECE_KIND_COUNT, 64, self.engine.PIECE_KIND_COUNT),
            dtype=np.int32,
        )
        update = self.search._bounded_capture_history_update.py_func
        update(position.pieces, self.engine.WHITE, move, 2_048, table)
        first = int(table[self.engine.WHITE, self.engine.PAWN, chess.D5, self.engine.PAWN])
        update(position.pieces, self.engine.WHITE, move, -2_048, table)
        second = int(table[self.engine.WHITE, self.engine.PAWN, chess.D5, self.engine.PAWN])
        self.assertGreater(first, 0)
        self.assertLess(second, first)
        self.assertTrue(np.abs(table).max() <= self.search.CAPTURE_HISTORY_LIMIT)


if __name__ == "__main__":
    unittest.main()
