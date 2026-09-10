from __future__ import annotations

import random
import unittest
from pathlib import Path
from typing import Any

import chess
import numpy as np

from tools.search_diagnostics import load_engine_modules

REPOSITORY = Path(__file__).resolve().parents[1]
CHALLENGER = REPOSITORY / "challengers" / "exp_qs_lazy_accumulator_v8"


class QuiescenceLazyAccumulatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.engine, cls.search = load_engine_modules(CHALLENGER)
        cls.nnue = cls.search.nnue

    def _assert_move_update_matches_rebuild(self, board: chess.Board, move: chess.Move) -> None:
        engine: Any = self.engine
        nnue: Any = self.nnue
        position = engine.position_from_board(board)
        packed = next(
            (
                int(candidate)
                for candidate in engine.legal_moves(position)
                if engine.move_to_uci(int(candidate)) == move.uci()
            ),
            None,
        )
        self.assertIsNotNone(packed, f"engine did not generate legal move {move.uci()}")
        assert packed is not None

        parent = np.empty((2, nnue.ACCUMULATOR_SIZE), dtype=np.int32)
        before_move = np.empty_like(parent)
        after_move = np.empty_like(parent)
        rebuilt = np.empty_like(parent)
        nnue.rebuild(position.pieces, parent)
        nnue.update_for_move(position.pieces, position.state, packed, parent, before_move)

        working = position.copy()
        undo = np.empty(engine.UNDO_SIZE, dtype=np.int64)
        undo_key = np.empty(1, dtype=np.uint64)
        self.assertTrue(
            engine.make_move(working.pieces, working.state, working.key, packed, undo, undo_key)
        )
        nnue.update_after_move(packed, undo, parent, after_move)
        nnue.rebuild(working.pieces, rebuilt)

        np.testing.assert_array_equal(after_move, before_move)
        np.testing.assert_array_equal(after_move, rebuilt)

    def test_special_move_updates_match_full_rebuild(self) -> None:
        cases = (
            (chess.STARTING_FEN, "e2e4", "quiet move"),
            (
                "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1",
                "e7e5",
                "black quiet move",
            ),
            ("4k3/8/8/8/8/8/3p4/3RK3 w - - 0 1", "d1d2", "capture"),
            ("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", "e5d6", "en passant"),
            ("4k3/P7/8/8/8/8/8/4K3 w - - 0 1", "a7a8q", "promotion"),
            ("1r2k3/P7/8/8/8/8/8/4K3 w - - 0 1", "a7b8q", "capture promotion"),
            ("4k3/8/8/8/8/8/p7/4K3 b - - 0 1", "a2a1n", "black promotion"),
            (
                "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
                "e1g1",
                "kingside castling",
            ),
            (
                "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
                "e1c1",
                "queenside castling",
            ),
            (
                "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1",
                "e8c8",
                "black castling",
            ),
        )
        for fen, uci, label in cases:
            with self.subTest(label=label):
                self._assert_move_update_matches_rebuild(
                    chess.Board(fen), chess.Move.from_uci(uci)
                )

    def test_random_legal_sequences_match_full_rebuild(self) -> None:
        generator = random.Random(0xA1C4E55A)
        checked = 0
        for _game in range(6):
            board = chess.Board()
            for _ply in range(40):
                legal_moves = list(board.legal_moves)
                if not legal_moves:
                    break
                move = generator.choice(legal_moves)
                self._assert_move_update_matches_rebuild(board, move)
                checked += 1
                board.push(move)
        self.assertGreaterEqual(checked, 200)


if __name__ == "__main__":
    unittest.main()
