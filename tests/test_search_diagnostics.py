from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import chess

from challengers.numba_v1 import engine, search
from tools.search_diagnostics import (
    DEFAULT_SUITE,
    analyze_root_moves,
    load_critical_positions,
    probe_node_limits,
)


class SearchDiagnosticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        search.warmup()

    def test_critical_suite_contains_legal_unique_positions(self) -> None:
        positions = load_critical_positions(DEFAULT_SUITE)
        self.assertEqual(len(positions), 4)
        self.assertEqual(len({position.identifier for position in positions}), 4)
        for position in positions:
            board = chess.Board(position.fen)
            legal = {move.uci() for move in board.legal_moves}
            self.assertIn(position.played_move, legal)
            self.assertIn(position.baseline_move, legal)
            self.assertIn(position.reference_move, legal)

    def test_invalid_suite_move_is_rejected(self) -> None:
        source = load_critical_positions(DEFAULT_SUITE)[0]
        invalid = asdict(source)
        invalid["id"] = invalid.pop("identifier")
        invalid["reference_move"] = "a1a1"
        payload = {"schema_version": 1, "positions": [invalid]}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "invalid.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "illegal reference_move"):
                load_critical_positions(path)

    def test_root_analysis_scores_every_move_without_mutating_board(self) -> None:
        board = chess.Board("4k3/8/8/8/8/8/q7/R3K3 w - - 0 1")
        original = board.fen()
        lines = analyze_root_moves(
            engine,
            search,
            board,
            depth=3,
            tt_bits=12,
            pv_plies=6,
        )
        self.assertEqual(len(lines), board.legal_moves.count())
        self.assertTrue(all(line.complete for line in lines))
        self.assertTrue(all(line.score is not None for line in lines))
        self.assertEqual(lines[0].move, "a1a2")
        self.assertEqual(lines[0].pv[0], lines[0].move)
        self.assertEqual(board.fen(), original)

        previous = search.INFINITY
        for line in lines:
            assert line.score is not None
            self.assertLessEqual(line.score, previous)
            previous = line.score
            replay = board.copy(stack=False)
            for uci in line.pv:
                move = chess.Move.from_uci(uci)
                self.assertIn(move, replay.legal_moves)
                replay.push(move)

    def test_fixed_node_probes_are_deterministic(self) -> None:
        board = chess.Board(
            "8/p6k/1p1pp1pb/1P2p3/PB2PnBq/3P1P2/2Q5/5K2 w - - 3 43"
        )
        first = probe_node_limits(engine, search, board, [5_000], 8, 12)
        second = probe_node_limits(engine, search, board, [5_000], 8, 12)
        self.assertEqual(first[0].move, second[0].move)
        self.assertEqual(first[0].score, second[0].score)
        self.assertEqual(first[0].depth, second[0].depth)
        self.assertEqual(first[0].nodes, second[0].nodes)
        self.assertGreater(first[0].nodes, 0)


if __name__ == "__main__":
    unittest.main()
