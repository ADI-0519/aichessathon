from __future__ import annotations

import unittest

import chess

from challengers.numba_v1 import engine as v3_engine
from challengers.numba_v1 import search as v3_search
from challengers.v4_lab import engine, search


class V4LabTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        search.configure_experiment("baseline")
        search.warmup()
        v3_search.warmup()

    def test_profiles_are_explicit_and_reconfiguration_after_warmup_is_rejected(self) -> None:
        self.assertEqual(
            search.available_profiles(),
            ("baseline", "no-lmr", "no-q-pruning", "check-extension"),
        )
        with self.assertRaisesRegex(RuntimeError, "before warmup"):
            search.configure_experiment("no-lmr")

    def test_baseline_profile_matches_frozen_v3(self) -> None:
        board = chess.Board(
            "8/p6k/1p1pp1pb/1P2p3/PB2PnBq/3P1P2/2Q5/5K2 w - - 3 43"
        )
        v3 = v3_search.search_position(
            v3_engine.position_from_board(board),
            v3_search.SearchMemory.create(12),
            node_limit=20_000,
            max_depth=8,
        )
        v4_baseline = search.search_position(
            engine.position_from_board(board),
            search.SearchMemory.create(12),
            node_limit=20_000,
            max_depth=8,
        )
        self.assertEqual(engine.move_to_uci(v4_baseline.move), v3_engine.move_to_uci(v3.move))
        self.assertEqual(v4_baseline.score, v3.score)
        self.assertEqual(v4_baseline.depth, v3.depth)
        self.assertEqual(v4_baseline.nodes, v3.nodes)


if __name__ == "__main__":
    unittest.main()
