from __future__ import annotations

import unittest
from pathlib import Path
from typing import Any

import chess
import numpy as np

from tools.search_diagnostics import REPOSITORY, load_engine_modules


class StableTimeoutTests(unittest.TestCase):
    def test_interrupted_iteration_keeps_last_completed_move(self) -> None:
        engine, search = load_engine_modules(
            Path(REPOSITORY) / "challengers" / "v6_stable_timeout"
        )
        position = engine.position_from_board(chess.Board())
        completed_move = 202
        partial_move = 303

        original_legal_moves = engine.legal_moves
        original_rebuild = search.nnue.rebuild
        original_search_root = search._search_root

        def legal_moves(_position: Any) -> np.ndarray[Any, np.dtype[np.int32]]:
            return np.asarray((101, completed_move), dtype=np.int32)

        def rebuild(
            _pieces: np.ndarray[Any, np.dtype[np.uint64]],
            accumulator: np.ndarray[Any, np.dtype[np.int32]],
        ) -> None:
            accumulator.fill(0)

        def search_root(*args: Any) -> tuple[int, int, bool, int]:
            depth = int(args[3])
            if depth == 1:
                return 10, 101, False, 0
            if depth == 2:
                return 20, completed_move, False, 0
            return 0, partial_move, True, partial_move

        engine.legal_moves = legal_moves
        search.nnue.rebuild = rebuild
        search._search_root = search_root
        try:
            result = search.search_position(
                position,
                search.SearchMemory.create(10),
                node_limit=100,
                max_depth=3,
            )
        finally:
            engine.legal_moves = original_legal_moves
            search.nnue.rebuild = original_rebuild
            search._search_root = original_search_root

        self.assertTrue(result.stopped)
        self.assertEqual(result.depth, 2)
        self.assertEqual(result.score, 20)
        self.assertEqual(result.move, completed_move)
        self.assertNotEqual(result.move, partial_move)


if __name__ == "__main__":
    unittest.main()
