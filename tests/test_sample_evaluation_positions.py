from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import chess

from tools.sample_evaluation_positions import (
    SampledPosition,
    balanced_selection,
    collect_positions,
    position_stratum,
)


class EvaluationPositionSamplingTests(unittest.TestCase):
    def test_position_strata_cover_phase_and_tactical_state(self) -> None:
        self.assertEqual(position_stratum(chess.Board()), "opening-quiet")
        self.assertEqual(
            position_stratum(chess.Board("4k3/8/8/8/8/8/r7/R3K3 w - - 0 1")),
            "endgame-tactical",
        )

    def test_balanced_selection_is_deterministic_and_redistributes_quota(self) -> None:
        buckets = {
            "opening-quiet": [
                SampledPosition(3, "oq-3", chess.STARTING_FEN, "opening-quiet"),
                SampledPosition(1, "oq-1", chess.STARTING_FEN, "opening-quiet"),
            ],
            "endgame-quiet": [
                SampledPosition(2, "eq-2", "4k3/8/8/8/8/8/8/4K3 w - - 0 1", "endgame-quiet")
            ],
        }

        selected = balanced_selection(buckets, 3)

        self.assertEqual([item.identifier for item in selected], ["oq-1", "eq-2", "oq-3"])

    def test_pgn_collection_normalizes_clocks_and_groups_identifiers(self) -> None:
        pgn = """[Event \"sampling\"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 *
"""
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "games.pgn"
            path.write_text(pgn, encoding="utf-8")
            positions, games = collect_positions(
                [path], count=3, seed="test", min_ply=4, max_ply=12, stride=2
            )

        self.assertEqual(games, 1)
        self.assertEqual(len(positions), 3)
        self.assertTrue(all(item.identifier.startswith("g0000001-p") for item in positions))
        self.assertTrue(all(item.fen.endswith(" 0 1") for item in positions))


if __name__ == "__main__":
    unittest.main()
