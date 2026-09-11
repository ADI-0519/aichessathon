from __future__ import annotations

import random
import unittest
from pathlib import Path
from typing import Any, ClassVar

import chess

from tools.search_diagnostics import load_engine_modules

REPOSITORY = Path(__file__).resolve().parents[1]
CHALLENGER = REPOSITORY / "challengers" / "exp_release_v17_searchmax"


class ReleaseV17SearchMaxTests(unittest.TestCase):
    engine: ClassVar[Any]
    search: ClassVar[Any]

    @classmethod
    def setUpClass(cls) -> None:
        cls.engine, cls.search = load_engine_modules(CHALLENGER)

    def test_searchmax_profile_is_explicit(self) -> None:
        self.assertTrue(self.search.ENABLE_DEFERRED_MOVE_GENERATION)
        self.assertEqual(self.search.LMR_REDUCTION_DIVISOR, 1.75)
        self.assertEqual(self.search.V10_RFP_MAX_DEPTH, 6)
        self.assertEqual(self.search.V10_QF_MAX_DEPTH, 4)
        self.assertEqual(self.search.V10_LMP_MAX_DEPTH, 4)
        self.assertEqual(self.search.DEFAULT_TT_BITS, 22)
        self.assertEqual(self.search.TT_BUCKET_SIZE, 2)
        self.assertEqual(self.search.TT_HALFMOVE_LIMIT, 100)

    def test_king_step_witness_matches_legal_moves(self) -> None:
        boards = [
            chess.Board(),
            chess.Board("7k/8/8/8/8/8/1K6/8 w - - 0 1"),
            chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1"),
            chess.Board("7k/5Q2/7K/8/8/8/8/8 b - - 0 1"),
        ]
        generator = random.Random(0xA1C4E55A)
        board = chess.Board()
        for _ in range(160):
            if board.is_game_over():
                board = chess.Board()
            move = generator.choice(list(board.legal_moves))
            board.push(move)
            if not board.is_check():
                boards.append(board.copy(stack=False))

        helper = self.search._king_has_legal_step
        for position in boards:
            if position.is_check():
                continue
            expected = any(
                position.piece_type_at(move.from_square) == chess.KING
                for move in position.legal_moves
            )
            compiled = self.engine.position_from_board(position)
            with self.subTest(fen=position.fen()):
                self.assertEqual(
                    bool(
                        helper(
                            compiled.pieces,
                            int(compiled.state[self.engine.STATE_SIDE]),
                        )
                    ),
                    expected,
                )


if __name__ == "__main__":
    unittest.main()
