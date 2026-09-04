from __future__ import annotations

import unittest
from unittest.mock import patch

import chess

from challengers.numba_v1 import agent


class NumbaAgentTests(unittest.TestCase):
    def setUp(self) -> None:
        agent._game_board = None
        agent._position_history.clear()
        agent._memory.clear()

    def test_import_warmup_fits_initialization_allowance(self) -> None:
        self.assertLess(agent._warmup_elapsed_s, 75.0)

    def test_low_clock_returns_legal_move(self) -> None:
        board = chess.Board()
        move = chess.Move.from_uci(agent.get_move(board.fen(), 1))
        self.assertIn(move, board.legal_moves)

    def test_normal_search_returns_legal_move(self) -> None:
        board = chess.Board()
        move = chess.Move.from_uci(agent.get_move(board.fen(), 1_000))
        self.assertIn(move, board.legal_moves)

    def test_persistent_state_tracks_our_and_opponent_moves(self) -> None:
        board = chess.Board()
        our_move = chess.Move.from_uci(agent.get_move(board.fen(), 150))
        board.push(our_move)
        opponent_move = next(iter(board.legal_moves))
        board.push(opponent_move)

        reply = chess.Move.from_uci(agent.get_move(board.fen(), 150))
        self.assertIn(reply, board.legal_moves)
        self.assertEqual(len(agent._position_history), 4)

    def test_internal_failure_keeps_legal_fallback(self) -> None:
        board = chess.Board()
        with patch.object(agent, "_choose_move", side_effect=RuntimeError("boom")):
            move = chess.Move.from_uci(agent.get_move(board.fen(), 1_000))
        self.assertIn(move, board.legal_moves)

    def test_clock_schedule_always_preserves_a_reserve(self) -> None:
        for remaining in (1, 10, 100, 1_000, 10_000, 60_000, 120_000):
            with self.subTest(remaining=remaining):
                budget = agent._move_budget_ms(remaining)
                self.assertGreaterEqual(budget, 0)
                self.assertLess(budget, remaining)


if __name__ == "__main__":
    unittest.main()
