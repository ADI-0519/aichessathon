from __future__ import annotations

import unittest
from pathlib import Path
from typing import Any

import numpy as np

from tools.search_diagnostics import load_engine_modules

REPOSITORY = Path(__file__).resolve().parents[1]
CHALLENGER = REPOSITORY / "challengers" / "exp_accumulator_freshness"


class AccumulatorFreshnessTests(unittest.TestCase):
    engine: Any
    search: Any
    nnue: Any

    @classmethod
    def setUpClass(cls) -> None:
        cls.engine, cls.search = load_engine_modules(CHALLENGER)
        cls.nnue = cls.search.nnue

    def _packed_move(self, position: Any, uci: str) -> int:
        engine: Any = self.engine
        for candidate in engine.legal_moves(position):
            packed = int(candidate)
            if engine.move_to_uci(packed) == uci:
                return packed
        self.fail(f"engine did not generate legal move {uci}")

    def test_update_refreshes_parent_skipped_after_king_bucket_crossing(self) -> None:
        engine: Any = self.engine
        nnue: Any = self.nnue
        position = engine.position_from_fen("7k/8/8/8/8/8/1K6/R7 w - - 0 1")
        parent = np.empty((2, nnue.ACCUMULATOR_ROW), dtype=np.int32)
        first_child = np.empty_like(parent)
        second_child = np.empty_like(parent)
        rebuilt = np.empty_like(parent)
        undo = np.empty(engine.UNDO_SIZE, dtype=np.int64)
        undo_key = np.empty(1, dtype=np.uint64)
        nnue.rebuild(position.pieces, parent)

        king_move = self._packed_move(position, "b2c3")
        nnue.update_for_move(
            position.pieces,
            position.state,
            king_move,
            parent,
            first_child,
        )
        self.assertEqual(first_child[0, nnue.BUCKET_SLOT], nnue.STALE)
        self.assertTrue(
            engine.make_move(
                position.pieces,
                position.state,
                position.key,
                king_move,
                undo,
                undo_key,
            )
        )

        # Deliberately skip evaluate()/refresh(), matching a TT or qeval-cache
        # return before the next incremental update.
        reply = self._packed_move(position, "h8g8")
        nnue.update_for_move(
            position.pieces,
            position.state,
            reply,
            first_child,
            second_child,
        )
        self.assertNotEqual(first_child[0, nnue.BUCKET_SLOT], nnue.STALE)
        self.assertTrue(
            engine.make_move(
                position.pieces,
                position.state,
                position.key,
                reply,
                undo,
                undo_key,
            )
        )
        nnue.rebuild(position.pieces, rebuilt)

        np.testing.assert_array_equal(second_child, rebuilt)


if __name__ == "__main__":
    unittest.main()
