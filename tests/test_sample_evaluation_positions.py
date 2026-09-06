from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import chess

from tools.sample_evaluation_positions import (
    SampledPosition,
    balanced_selection,
    build_evaluation_suite,
    collect_positions,
    position_stratum,
    source_format,
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
            positions, statistics = collect_positions(
                [path], count=3, seed="test", min_ply=4, max_ply=12, stride=2
            )

        self.assertEqual(statistics.records_read, 1)
        self.assertEqual(statistics.positions_examined, 5)
        self.assertEqual(len(positions), 3)
        self.assertTrue(all(item.identifier.startswith("g0000001-p") for item in positions))
        self.assertTrue(all(item.fen.endswith(" 0 1") for item in positions))

    def test_epd_and_fen_collection_deduplicates_and_skips_terminal_positions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            epd_path = root / "positions.epd"
            fen_path = root / "positions.fen"
            epd_path.write_text(
                "# a comment\n"
                f"{chess.STARTING_BOARD_FEN} w KQkq - id \"start\";\n"
                "4k3/8/8/8/8/8/8/4K3 w - - id \"terminal\";\n",
                encoding="utf-8",
            )
            fen_path.write_text(
                f"{chess.STARTING_FEN}\n"
                "4k3/8/8/8/8/8/r7/R3K3 w - - 7 42\n",
                encoding="utf-8",
            )
            positions, statistics = collect_positions(
                [epd_path, fen_path],
                count=2,
                seed="test",
                min_ply=12,
                max_ply=160,
                stride=4,
            )

        self.assertEqual(len(positions), 2)
        self.assertEqual(statistics.records_read, 4)
        self.assertEqual(statistics.positions_examined, 4)
        self.assertEqual(statistics.unique_positions, 2)
        self.assertEqual(statistics.duplicates_skipped, 1)
        self.assertEqual(statistics.terminal_positions_skipped, 1)
        self.assertEqual({item.identifier[0] for item in positions}, {"e", "f"})
        self.assertTrue(all(item.fen.endswith(" 0 1") for item in positions))

    def test_source_format_rejects_ambiguous_files(self) -> None:
        self.assertEqual(source_format(Path("positions.EPD")), "epd")
        with self.assertRaisesRegex(ValueError, "expected .pgn, .epd, or .fen"):
            source_format(Path("positions.txt"))

    def test_builder_writes_provenance_manifest_and_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "positions.fen"
            output = root / "generated" / "sample.epd"
            source.write_text(
                f"{chess.STARTING_FEN}\n"
                "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2\n",
                encoding="utf-8",
            )
            selected, manifest = build_evaluation_suite(
                [source],
                output,
                count=2,
                seed="manifest-test",
                min_ply=12,
                max_ply=160,
                stride=4,
                source_urls=["https://example.test/positions.fen"],
            )
            persisted = json.loads(
                output.with_suffix(".manifest.json").read_text(encoding="utf-8")
            )

            self.assertEqual(len(selected), 2)
            self.assertEqual(manifest, persisted)
            self.assertEqual(manifest["schema_version"], 2)
            self.assertEqual(persisted["sources"][0]["source_url"], "https://example.test/positions.fen")
            self.assertIn("sha256", persisted["sources"][0])
            self.assertIn("output_sha256", persisted)
            with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                build_evaluation_suite(
                    [source],
                    output,
                    count=2,
                    seed="manifest-test",
                    min_ply=12,
                    max_ply=160,
                    stride=4,
                )

    def test_collection_reports_invalid_epd_with_line_number(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "broken.epd"
            path.write_text("not an epd\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, r"broken\.epd:1: invalid EPD"):
                collect_positions(
                    [path], count=1, seed="test", min_ply=12, max_ply=160, stride=4
                )

    def test_builder_can_require_every_stratum(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "opening.fen"
            source.write_text(f"{chess.STARTING_FEN}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "does not meet --min-per-stratum"):
                build_evaluation_suite(
                    [source],
                    root / "sample.epd",
                    count=1,
                    seed="test",
                    min_ply=12,
                    max_ply=160,
                    stride=4,
                    min_per_stratum=1,
                )


if __name__ == "__main__":
    unittest.main()
