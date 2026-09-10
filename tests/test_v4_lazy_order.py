from __future__ import annotations

import unittest

import chess
import numpy as np

from challengers.numba_v1 import engine as v3_engine
from challengers.numba_v1 import search as v3_search
from challengers.v4_lazy_order import engine, search


class V4LazyOrderingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        v3_search.warmup()
        search.warmup()

    def test_lazy_picker_matches_stable_descending_sort(self) -> None:
        moves = np.asarray((10, 11, 12, 13, 14, 15), dtype=np.int32)
        scores = np.asarray((4, 9, 9, -1, 4, 7), dtype=np.int32)

        for index in range(len(moves)):
            search._pick_next_move(moves, scores, index, len(moves))

        np.testing.assert_array_equal(scores, (9, 9, 7, 4, 4, -1))
        np.testing.assert_array_equal(moves, (11, 12, 15, 10, 14, 13))

    def test_fixed_node_search_matches_v3(self) -> None:
        fens = (
            chess.STARTING_FEN,
            "r3k2r/p1ppqpb1/bn2pnp1/2pP4/1p2P3/2N2N2/PPQ1BPPP/R3K2R w KQkq - 0 1",
            "8/p6k/1p1pp1pb/1P2p3/PB2PnBq/3P1P2/2Q5/5K2 w - - 3 43",
            "4k3/8/4p3/3q4/4P3/8/8/4K3 w - - 0 1",
        )
        for fen in fens:
            with self.subTest(fen=fen):
                board = chess.Board(fen)
                expected = v3_search.search_position(
                    v3_engine.position_from_board(board),
                    v3_search.SearchMemory.create(12),
                    node_limit=30_000,
                    max_depth=10,
                )
                actual = search.search_position(
                    engine.position_from_board(board),
                    search.SearchMemory.create(12),
                    node_limit=30_000,
                    max_depth=10,
                )

                self.assertEqual(
                    engine.move_to_uci(actual.move), v3_engine.move_to_uci(expected.move)
                )
                self.assertEqual(actual.score, expected.score)
                self.assertEqual(actual.depth, expected.depth)
                self.assertEqual(actual.nodes, expected.nodes)
                self.assertEqual(actual.qnodes, expected.qnodes)


if __name__ == "__main__":
    unittest.main()
