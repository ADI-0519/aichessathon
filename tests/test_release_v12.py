from __future__ import annotations

import random
import runpy
import unittest
from pathlib import Path
from typing import Any, ClassVar

import chess
import numpy as np

from tools.search_diagnostics import load_engine_modules

REPOSITORY = Path(__file__).resolve().parents[1]
CANDIDATE = REPOSITORY / "challengers" / "exp_release_v12"


class ReleaseV12Tests(unittest.TestCase):
    engine: ClassVar[Any]
    search: ClassVar[Any]
    nnue: ClassVar[Any]

    @classmethod
    def setUpClass(cls) -> None:
        cls.engine, cls.search = load_engine_modules(CANDIDATE)
        cls.nnue = cls.search.nnue

    def _assert_post_move_update(self, board: chess.Board, move: chess.Move) -> None:
        position = self.engine.position_from_board(board)
        packed = next(
            int(candidate)
            for candidate in self.engine.legal_moves(position)
            if self.engine.move_to_uci(int(candidate)) == move.uci()
        )
        parent = np.empty((2, self.nnue.ACCUMULATOR_ROW), dtype=np.int32)
        eager = np.empty_like(parent)
        lazy = np.empty_like(parent)
        rebuilt = np.empty_like(parent)
        self.nnue.rebuild(position.pieces, parent)
        self.nnue.update_for_move(
            position.pieces, position.state, packed, parent, eager
        )

        working = position.copy()
        undo = np.empty(self.engine.UNDO_SIZE, dtype=np.int64)
        undo_key = np.empty(1, dtype=np.uint64)
        self.assertTrue(
            self.engine.make_move(
                working.pieces,
                working.state,
                working.key,
                packed,
                undo,
                undo_key,
            )
        )
        self.nnue.update_after_move(packed, undo, parent, lazy)
        np.testing.assert_array_equal(lazy, eager)

        self.nnue.refresh(working.pieces, lazy)
        self.nnue.rebuild(working.pieces, rebuilt)
        np.testing.assert_array_equal(lazy, rebuilt)

    def test_post_move_updates_cover_special_moves_and_bucket_crossing(self) -> None:
        cases = (
            (chess.STARTING_FEN, "e2e4"),
            ("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", "e5d6"),
            ("1r2k3/P7/8/8/8/8/8/4K3 w - - 0 1", "a7b8q"),
            ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1g1"),
            ("7k/8/8/8/8/8/1K6/8 w - - 0 1", "b2c3"),
        )
        for fen, uci in cases:
            with self.subTest(uci=uci):
                self._assert_post_move_update(
                    chess.Board(fen), chess.Move.from_uci(uci)
                )

    def test_post_move_updates_match_random_legal_sequences(self) -> None:
        generator = random.Random(0xA1C4E55A)
        checked = 0
        for _game in range(4):
            board = chess.Board()
            for _ply in range(32):
                legal = list(board.legal_moves)
                if not legal:
                    break
                move = generator.choice(legal)
                self._assert_post_move_update(board, move)
                board.push(move)
                checked += 1
        self.assertGreaterEqual(checked, 100)

    def test_adaptive_limits_are_ordered_and_reserve_safe(self) -> None:
        namespace = runpy.run_path(str(CANDIDATE / "time_manager.py"))
        move_time_limits = namespace["move_time_limits"]
        for time_left_ms in (101, 250, 1_000, 10_000, 120_000):
            limits = move_time_limits(time_left_ms, 20)
            self.assertLessEqual(limits.soft_ms, limits.normal_ms)
            self.assertLessEqual(limits.normal_ms, limits.hard_ms)
            self.assertLess(limits.hard_ms, time_left_ms)


if __name__ == "__main__":
    unittest.main()
