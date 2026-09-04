from __future__ import annotations

import pathlib
import random
import subprocess
import sys
import unittest

import chess
import numpy as np

from challengers.numba_v1 import engine, search
from harness.rules import INIT_BUDGET_S


class NumbaSearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        search.warmup()

    def test_evaluation_is_side_relative_and_values_material(self) -> None:
        white = engine.position_from_fen("4k3/8/8/8/8/8/8/Q3K3 w - - 0 1")
        black = engine.position_from_fen("4k3/8/8/8/8/8/8/Q3K3 b - - 0 1")
        self.assertGreater(search.evaluate(white.pieces, white.state), 800)
        self.assertLess(search.evaluate(black.pieces, black.state), -800)

        starting = engine.position_from_board(chess.Board())
        self.assertEqual(search.evaluate(starting.pieces, starting.state), 10)

    def test_static_exchange_evaluation_handles_special_and_defended_captures(self) -> None:
        cases = (
            ("r3k3/8/8/8/8/8/p7/R3K3 w - - 0 1", "a1a2", -400),
            ("4k3/8/4p3/3q4/4P3/8/8/4K3 w - - 0 1", "e4d5", 800),
            ("4k3/P7/8/8/8/8/8/4K3 w - - 0 1", "a7a8q", 800),
            ("1r2k3/P7/8/8/8/8/8/4K3 w - - 0 1", "a7b8q", 1_300),
            ("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", "e5d6", 100),
            ("4k3/3pr3/8/8/6B1/8/8/K2RR3 w - - 0 1", "g4d7", 100),
        )
        gains = np.empty(search.SEE_MAX_EXCHANGES, dtype=np.int32)
        for fen, uci, expected in cases:
            with self.subTest(fen=fen, uci=uci):
                position = engine.position_from_fen(fen)
                move = next(
                    int(candidate)
                    for candidate in engine.legal_moves(position)
                    if engine.move_to_uci(int(candidate)) == uci
                )
                self.assertEqual(
                    search.static_exchange_eval(
                        position.pieces, position.state, move, gains
                    ),
                    expected,
                )

    def test_finds_and_scores_mate_in_one(self) -> None:
        board = chess.Board("7k/5Q2/6K1/8/8/8/8/8 w - - 0 1")
        result = search.search_position(
            engine.position_from_board(board),
            search.SearchMemory.create(12),
            node_limit=20_000,
            max_depth=4,
        )
        move = chess.Move.from_uci(engine.move_to_uci(result.move))
        self.assertIn(move, board.legal_moves)
        board.push(move)
        self.assertTrue(board.is_checkmate())
        self.assertEqual(result.score, search.MATE_SCORE - 1)

    def test_captures_a_hanging_queen(self) -> None:
        board = chess.Board("4k3/8/8/8/8/8/q7/R3K3 w - - 0 1")
        result = search.search_position(
            engine.position_from_board(board),
            search.SearchMemory.create(12),
            node_limit=20_000,
            max_depth=5,
        )
        self.assertEqual(engine.move_to_uci(result.move), "a1a2")

    def test_fixed_node_search_is_deterministic(self) -> None:
        position = engine.position_from_board(chess.Board())
        first = search.search_position(
            position, search.SearchMemory.create(14), node_limit=20_000, max_depth=8
        )
        second = search.search_position(
            position, search.SearchMemory.create(14), node_limit=20_000, max_depth=8
        )
        self.assertEqual(first.move, second.move)
        self.assertEqual(first.score, second.score)
        self.assertEqual(first.depth, second.depth)
        self.assertEqual(first.nodes, second.nodes)
        self.assertEqual(first.lmr_reductions, second.lmr_reductions)
        self.assertEqual(first.lmr_researches, second.lmr_researches)
        self.assertGreater(first.lmr_reductions, 0)
        self.assertLessEqual(first.lmr_researches, first.lmr_reductions)

    def test_timed_stop_returns_promptly_with_a_legal_completed_move(self) -> None:
        board = chess.Board()
        result = search.search_position(
            engine.position_from_board(board),
            search.SearchMemory.create(12),
            time_limit_s=0.05,
            max_depth=search.MAX_DEPTH,
        )
        move = chess.Move.from_uci(engine.move_to_uci(result.move))
        self.assertIn(move, board.legal_moves)
        self.assertGreaterEqual(result.depth, 1)
        self.assertLess(result.elapsed_s, 0.30)

    def test_search_never_mutates_caller_position(self) -> None:
        position = engine.position_from_fen(
            "r3k2r/p1ppqpb1/bn2pnp1/2pP4/1p2P3/2N2N2/PPQ1BPPP/R3K2R w KQkq - 0 1"
        )
        pieces = position.pieces.copy()
        state = position.state.copy()
        key = position.key.copy()
        search.search_position(
            position, search.SearchMemory.create(12), node_limit=10_000, max_depth=8
        )
        np.testing.assert_array_equal(position.pieces, pieces)
        np.testing.assert_array_equal(position.state, state)
        np.testing.assert_array_equal(position.key, key)

    def test_returns_legal_moves_from_varied_positions(self) -> None:
        rng = random.Random(2026090406)
        board = chess.Board()
        for _ in range(20):
            if board.is_game_over(claim_draw=True):
                board.reset()
            result = search.search_position(
                engine.position_from_board(board),
                search.SearchMemory.create(10),
                node_limit=2_000,
                max_depth=6,
            )
            move = chess.Move.from_uci(engine.move_to_uci(result.move))
            self.assertIn(move, board.legal_moves, board.fen())
            board.push(move)
            if not board.is_game_over(claim_draw=True):
                board.push(rng.choice(list(board.legal_moves)))

    def test_mate_scores_round_trip_through_transposition_encoding(self) -> None:
        for ply in (0, 1, 17, 63):
            for score in (search.MATE_SCORE - 12, -search.MATE_SCORE + 12, 427, -935):
                with self.subTest(ply=ply, score=score):
                    stored = search._score_to_table(score, ply)
                    self.assertEqual(search._score_from_table(stored, ply), score)

    def test_cold_import_and_warmup_fit_the_initialization_allowance(self) -> None:
        # only a fresh process measures what the platform pays before move one
        package = pathlib.Path(__file__).resolve().parent.parent / "challengers" / "numba_v1"
        script = (
            "import time;"
            "started = time.perf_counter();"
            "import search;"
            "search.warmup();"
            "print(time.perf_counter() - started)"
        )
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=package,
            capture_output=True,
            text=True,
            timeout=INIT_BUDGET_S * 3,
            check=True,
        )
        elapsed = float(completed.stdout.strip().splitlines()[-1])
        self.assertLess(elapsed, INIT_BUDGET_S * 2 / 3)

    def test_quiet_position_still_searches_and_stalemate_returns_no_move(self) -> None:
        # position with no captures must not dead-end on the stand-pat score
        quiet = engine.position_from_fen("8/5pk1/6p1/3p4/3P4/5KP1/5P2/8 w - - 0 1")
        result = search.search_position(quiet, search.SearchMemory.create(), node_limit=20_000)
        self.assertIn(
            engine.move_to_uci(result.move),
            {move.uci() for move in engine.board_from_position(quiet).legal_moves},
        )

        stalemate = engine.position_from_fen("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
        self.assertEqual(
            search.search_position(
                stalemate, search.SearchMemory.create(), node_limit=1_000
            ).move,
            0,
        )


if __name__ == "__main__":
    unittest.main()
