from __future__ import annotations

import unittest
from typing import Any, ClassVar

import chess
import numpy as np

from tools.search_diagnostics import REPOSITORY, load_engine_modules


class SearchV10S1LeanTests(unittest.TestCase):
    engine: ClassVar[Any]
    search: ClassVar[Any]

    @classmethod
    def setUpClass(cls) -> None:
        cls.engine, cls.search = load_engine_modules(
            REPOSITORY / "challengers" / "exp_search_v10_s1lean"
        )

    def tearDown(self) -> None:
        self.search.configure_experiment("baseline")

    def test_profiles_keep_both_s1_mechanisms_isolated(self) -> None:
        self.assertEqual(
            set(self.search.available_profiles()),
            {"baseline", "v10", "history-v2", "quiet-see", "current"},
        )

        self.search.configure_experiment("v10")
        self.assertFalse(self.search.ENABLE_HISTORY_V2)
        self.assertFalse(self.search.ENABLE_QUIET_SEE_PRUNING)

        self.search.configure_experiment("history-v2")
        self.assertTrue(self.search.ENABLE_HISTORY_V2)
        self.assertFalse(self.search.ENABLE_QUIET_SEE_PRUNING)

        self.search.configure_experiment("quiet-see")
        self.assertFalse(self.search.ENABLE_HISTORY_V2)
        self.assertTrue(self.search.ENABLE_QUIET_SEE_PRUNING)

    def test_signed_history_update_moves_both_directions(self) -> None:
        history = np.zeros((2, 64, 64), dtype=np.int32)
        move = self.engine.pack_move(chess.A2, chess.A3, 0, 0)

        self.search._bounded_history_update.py_func(
            self.engine.WHITE, move, 512, history
        )
        positive = int(history[self.engine.WHITE, chess.A2, chess.A3])
        self.assertGreater(positive, 0)

        self.search._bounded_history_update.py_func(
            self.engine.WHITE, move, -1024, history
        )
        self.assertLess(
            int(history[self.engine.WHITE, chess.A2, chess.A3]), positive
        )

    def test_quiet_see_detects_an_undefended_rook_move(self) -> None:
        board = chess.Board("7k/8/8/8/1p6/8/R7/K7 w - - 0 1")
        position = self.engine.position_from_board(board)
        move = self.engine.pack_move(chess.A2, chess.A3, 0, 0)
        gains = np.empty(self.search.SEE_MAX_EXCHANGES, dtype=np.int32)

        score = self.search.static_exchange_eval.py_func(
            position.pieces, position.state, move, gains
        )
        self.assertEqual(score, -int(self.search.MG_VALUE[self.engine.ROOK]))


if __name__ == "__main__":
    unittest.main()
