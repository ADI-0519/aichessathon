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
        agent._history.clear()
        agent._eval_cache.clear()
        agent._transposition_table.clear()

    def test_move_is_legal_from_varied_positions(self) -> None:
        fens = (
            chess.STARTING_FEN,
            "r3k2r/p1ppqpb1/bn2pnp1/2pP4/1p2P3/2N2N2/PPQ1BPPP/R1B1K2R w KQkq - 0 1",
            "8/5pk1/6p1/3p4/3P4/5KP1/5P2/8 w - - 0 1",
            "4k3/8/8/8/8/8/3p4/4K3 b - - 0 1",
        )
        for fen in fens:
            with self.subTest(fen=fen):
                board = chess.Board(fen)
                move = chess.Move.from_uci(agent.get_move(fen, 100))
                self.assertIn(move, board.legal_moves)

    def test_finds_mate_in_one(self) -> None:
        fen = "7k/5Q2/6K1/8/8/8/8/8 w - - 0 1"
        board = chess.Board(fen)
        move = chess.Move.from_uci(agent.get_move(fen, 1_000))
        board.push(move)
        self.assertTrue(board.is_checkmate())

    def test_search_timeout_restores_board(self) -> None:
        board = chess.Board()
        original_fen = board.fen()
        searcher = agent.Searcher(time.monotonic() - 1.0)
        fallback = next(iter(board.legal_moves))
        self.assertEqual(searcher.best_move(board, fallback), fallback)
        self.assertEqual(board.fen(), original_fen)

    def test_mate_score_table_round_trip(self) -> None:
        for score in (agent.MATE_SCORE - 7, -agent.MATE_SCORE + 9, 123, -456):
            stored = agent.Searcher._table_score(score, 5)
            self.assertEqual(agent.Searcher._search_score(stored, 5), score)

    def test_low_clock_keeps_reserve(self) -> None:
        for remaining in (0, 10, 99, 100, 500, 1_000, 120_000):
            with self.subTest(remaining=remaining):
                self.assertLess(agent._move_budget_ms(remaining), max(remaining, 1))

    def test_unexpected_internal_error_uses_legal_emergency_move(self) -> None:
        board = chess.Board()
        with patch.object(agent, "_choose_move", side_effect=RuntimeError("boom")):
            move = chess.Move.from_uci(agent.get_move(board.fen(), 10_000))
        self.assertIn(move, board.legal_moves)

    def test_terminal_positions_return_null_move(self) -> None:
        for fen in (
            "7k/6Q1/6K1/8/8/8/8/8 b - - 0 1",
            "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1",
        ):
            with self.subTest(fen=fen):
                self.assertEqual(agent.get_move(fen, 10_000), "0000")

    def test_persistent_board_syncs_through_opponent_move(self) -> None:
        board = chess.Board()
        first = chess.Move.from_uci(agent.get_move(board.fen(), 100))
        board.push(first)
        board.push(next(iter(board.legal_moves)))
        reply = chess.Move.from_uci(agent.get_move(board.fen(), 100))
        self.assertIn(reply, board.legal_moves)
        board.push(reply)
        self.assertIsNotNone(agent._game_board)
        self.assertEqual(agent._game_board.fen(), board.fen())

    def test_random_position_fuzz_always_returns_legal_move(self) -> None:
        rng = random.Random(20260904)
        for _ in range(250):
            board = chess.Board()
            for _ in range(rng.randrange(80)):
                if board.is_game_over():
                    break
                board.push(rng.choice(list(board.legal_moves)))
            if board.is_game_over():
                continue
            agent._game_board = None
            move = chess.Move.from_uci(agent.get_move(board.fen(), 0))
            self.assertIn(move, board.legal_moves)

    def test_official_clock_schedule_retains_time_for_300_moves(self) -> None:
        clock = 120_000
        for _ in range(300):
            budget = agent._move_budget_ms(clock)
            self.assertLess(budget, clock)
            clock = clock - budget + 500
        self.assertGreater(clock, 0)

    def test_transposition_table_survives_between_searchers(self) -> None:
        first = agent.Searcher(time.monotonic() + 1.0)
        key = ("sentinel", 0)
        entry = agent.TableEntry(4, 123, agent.EXACT, None)
        first.table[key] = entry
        second = agent.Searcher(time.monotonic() + 1.0)
        self.assertIs(second.table.get(key), entry)


if __name__ == "__main__":
    unittest.main()
