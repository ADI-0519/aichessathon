from __future__ import annotations

import random
import time
import unittest
from unittest.mock import patch

import chess

import agent


class AgentTests(unittest.TestCase):
    def setUp(self) -> None:
        agent._game_board = None
        agent._position_history.clear()
        agent._memory.clear()

    def test_import_and_warmup_fit_the_initialization_allowance(self) -> None:
        self.assertLess(agent._warmup_elapsed_s, 75.0)

    def test_move_is_legal_from_varied_positions(self) -> None:
        rng = random.Random(20260905)
        board = chess.Board()
        for _ in range(40):
            if board.is_game_over(claim_draw=True):
                board = chess.Board()
            board.push(rng.choice(list(board.legal_moves)))
            move = chess.Move.from_uci(agent.get_move(board.fen(), 200))
            self.assertIn(move, board.legal_moves, board.fen())
            agent._game_board = None
            agent._position_history.clear()

    def test_terminal_positions_return_null_move(self) -> None:
        for fen in ("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1", "7k/5Q2/5K2/8/8/8/8/8 b - - 0 1"):
            self.assertEqual(agent.get_move(fen, 5_000), "0000")

    def test_unexpected_internal_error_uses_legal_emergency_move(self) -> None:
        board = chess.Board()
        with patch.object(agent, "_choose_move", side_effect=RuntimeError("boom")):
            move = chess.Move.from_uci(agent.get_move(board.fen(), 5_000))
        self.assertIn(move, board.legal_moves)

    def test_low_clock_returns_promptly_and_legally(self) -> None:
        board = chess.Board()
        for remaining in (1, 50, 200):
            started = time.perf_counter()
            move = chess.Move.from_uci(agent.get_move(board.fen(), remaining))
            self.assertIn(move, board.legal_moves)
            self.assertLess(time.perf_counter() - started, 1.0)

    def test_clock_schedule_retains_a_reserve_over_a_full_game(self) -> None:
        remaining = 120_000
        for _ in range(300):
            budget = agent._move_budget_ms(remaining)
            self.assertGreaterEqual(budget, 0)
            self.assertLess(budget, remaining)
            remaining = remaining - budget + 500
        self.assertGreater(remaining, 0)

    def test_persistent_board_syncs_through_opponent_move(self) -> None:
        board = chess.Board()
        board.push(chess.Move.from_uci(agent.get_move(board.fen(), 300)))
        board.push(next(iter(board.legal_moves)))
        reply = chess.Move.from_uci(agent.get_move(board.fen(), 300))
        self.assertIn(reply, board.legal_moves)
        self.assertEqual(len(agent._position_history), 4)

    def test_unreachable_position_resets_rather_than_desyncing(self) -> None:
        agent.get_move(chess.Board().fen(), 300)
        # no single legal move reaches this from the start, so the sync must reset
        board = chess.Board("8/5pk1/6p1/3p4/3P4/5KP1/5P2/8 w - - 0 1")
        move = chess.Move.from_uci(agent.get_move(board.fen(), 300))
        self.assertIn(move, board.legal_moves)
        self.assertEqual(len(agent._position_history), 2)


if __name__ == "__main__":
    unittest.main()
