from __future__ import annotations

import unittest

import chess
import numpy as np

from challengers.v4_dev_safe import agent, engine, search


class V4DevSafeTests(unittest.TestCase):
    def test_snapshot_excludes_null_move(self) -> None:
        self.assertFalse(hasattr(engine, "make_null_move"))
        self.assertFalse(hasattr(engine, "unmake_null_move"))

    def test_reference_perft_and_position_restoration(self) -> None:
        position = engine.position_from_board(chess.Board())
        pieces = position.pieces.copy()
        state = position.state.copy()
        key = position.key.copy()

        self.assertEqual(engine.perft(position, 4), 197_281)
        np.testing.assert_array_equal(position.pieces, pieces)
        np.testing.assert_array_equal(position.state, state)
        np.testing.assert_array_equal(position.key, key)

    def test_fixed_node_search_is_deterministic_and_legal(self) -> None:
        board = chess.Board(
            "r3k2r/p1ppqpb1/bn2pnp1/2pP4/1p2P3/2N2N2/PPQ1BPPP/R3K2R w KQkq - 0 1"
        )
        position = engine.position_from_board(board)
        first = search.search_position(
            position,
            search.SearchMemory.create(14),
            node_limit=30_000,
            max_depth=10,
        )
        second = search.search_position(
            position,
            search.SearchMemory.create(14),
            node_limit=30_000,
            max_depth=10,
        )

        self.assertEqual(first.move, second.move)
        self.assertEqual(first.score, second.score)
        self.assertEqual(first.depth, second.depth)
        self.assertEqual(first.nodes, second.nodes)
        move = chess.Move.from_uci(engine.move_to_uci(first.move))
        self.assertIn(move, board.legal_moves)

    def test_timed_search_uses_budget_and_returns_legal_move(self) -> None:
        board = chess.Board(
            "r1bqkb1r/pp2pppp/2np1n2/8/3NP3/2N5/PPP2PPP/R1BQKB1R w KQkq - 0 6"
        )
        budget_s = 0.2
        result = search.search_position(
            engine.position_from_board(board),
            search.SearchMemory.create(14),
            time_limit_s=budget_s,
        )

        move = chess.Move.from_uci(engine.move_to_uci(result.move))
        self.assertIn(move, board.legal_moves)
        self.assertTrue(result.stopped)
        self.assertGreater(result.elapsed_s, budget_s * 0.85)
        self.assertLess(result.elapsed_s, budget_s + 0.25)

    def test_agent_boundary_returns_legal_uci(self) -> None:
        board = chess.Board()
        move = chess.Move.from_uci(agent.get_move(board.fen(), 1_000))
        self.assertIn(move, board.legal_moves)


if __name__ == "__main__":
    unittest.main()
