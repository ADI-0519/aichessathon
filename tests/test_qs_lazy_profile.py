from __future__ import annotations

import unittest
from pathlib import Path
from typing import Any

import chess

from tools.search_diagnostics import load_engine_modules

REPOSITORY = Path(__file__).resolve().parents[1]
CHALLENGER = REPOSITORY / "challengers" / "exp_qs_lazy_profile_v7q"
TACTICAL_FEN = "r2q1rk1/bp3ppp/p1p2n2/P7/1P1P2b1/2QB1N2/5PPP/R1B1R1K1 w - - 1 20"


class QuiescenceLazyAccumulatorProfileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.engine, cls.search = load_engine_modules(CHALLENGER)

    def test_profile_accounts_for_every_qsearch_move(self) -> None:
        engine: Any = self.engine
        search: Any = self.search
        position = engine.position_from_board(chess.Board(TACTICAL_FEN))
        result = search.search_position(
            position,
            search.SearchMemory.create(10),
            node_limit=10_000,
            max_depth=search.MAX_DEPTH,
        )

        self.assertGreater(result.q_moves_considered, 0)
        self.assertEqual(result.q_accumulator_updates, result.q_moves_considered)
        self.assertEqual(
            result.q_pruned_after_update + result.q_children_searched,
            result.q_moves_considered,
        )
        self.assertLessEqual(result.q_check_saves, result.q_children_searched)


if __name__ == "__main__":
    unittest.main()
