from __future__ import annotations

import random
import unittest

import chess

from challengers.exp_qsearch_tacticals import engine as candidate
from current import engine as baseline


class QsearchTacticalGeneratorTests(unittest.TestCase):
    @staticmethod
    def _moves(module: object, board: chess.Board) -> tuple[list[str], bool]:
        position = module.position_from_board(board)  # type: ignore[attr-defined]
        moves, has_legal_move = module.legal_captures(position)  # type: ignore[attr-defined]
        return (
            [module.move_to_uci(int(move)) for move in moves],  # type: ignore[attr-defined]
            has_legal_move,
        )

    def assert_matches_reference(self, board: chess.Board) -> None:
        expected = [
            move.uci()
            for move in board.legal_moves
            if board.is_capture(move) or move.promotion is not None
        ]
        baseline_moves, baseline_has_legal = self._moves(baseline, board)
        candidate_moves, candidate_has_legal = self._moves(candidate, board)

        self.assertEqual(candidate_moves, baseline_moves, board.fen())
        self.assertEqual(set(candidate_moves), set(expected), board.fen())
        self.assertEqual(candidate_has_legal, baseline_has_legal, board.fen())
        self.assertEqual(candidate_has_legal, any(board.legal_moves), board.fen())

    def test_special_tactical_and_terminal_positions(self) -> None:
        positions = (
            chess.Board(),
            chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1"),
            chess.Board("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1"),
            chess.Board("4k3/P7/8/8/8/8/8/4K3 w - - 0 1"),
            chess.Board("1r2k3/P7/8/8/8/8/8/4K3 w - - 0 1"),
            chess.Board("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1"),
            chess.Board("4k3/8/8/8/8/2n5/3P4/4K3 w - - 0 1"),
        )
        for board in positions:
            with self.subTest(fen=board.fen()):
                self.assert_matches_reference(board)

    def test_random_legal_positions_preserve_exact_order_and_state(self) -> None:
        rng = random.Random(2026090901)
        board = chess.Board()
        for _ in range(400):
            if board.is_game_over(claim_draw=True):
                board.reset()
            before = board.fen(en_passant="fen")
            self.assert_matches_reference(board)
            self.assertEqual(board.fen(en_passant="fen"), before)
            board.push(rng.choice(list(board.legal_moves)))


if __name__ == "__main__":
    unittest.main()
