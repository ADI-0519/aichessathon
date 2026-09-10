from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import chess

from tools.backtest_core import load_epd, positions_from_fens, suite_digest
from tools.build_suite import build_suite, render_epd, select_positions


class BuildSuiteTests(unittest.TestCase):
    def test_hash_selection_is_independent_of_source_order(self) -> None:
        entries = []
        board = chess.Board()
        for index, move in enumerate(("e2e4", "e7e5", "g1f3", "b8c6"), start=1):
            board.push_uci(move)
            entries.append((f"position-{index}", board.fen()))
        forward = positions_from_fens(entries, split_seed="split")
        backward = positions_from_fens(reversed(entries), split_seed="split")
        first = select_positions(forward, count=2, seed="selection")
        second = select_positions(backward, count=2, seed="selection")
        self.assertEqual(
            [position.fen for position in first], [position.fen for position in second]
        )

    def test_epd_round_trip_preserves_identifiers_and_clock_state(self) -> None:
        fen = "8/5pk1/6p1/3p4/3P4/5KP1/5P2/8 w - - 17 42"
        positions = positions_from_fens((("ending", fen),), split_seed="split")
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "suite.epd"
            path.write_text(render_epd(positions, source_name="source.pgn"), encoding="utf-8")
            restored = load_epd(path, split_seed="split")
        self.assertEqual(restored[0].identifier, "ending")
        self.assertEqual(restored[0].fen, fen)

    def test_build_records_provenance_and_refuses_accidental_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "openings.pgn"
            source.write_text(
                '[Event "One"]\n[Result "*"]\n\n1. e4 e5 *\n\n'
                '[Event "Two"]\n[Result "*"]\n\n1. d4 d5 *\n\n'
                '[Event "Three"]\n[Result "*"]\n\n1. c4 e5 *\n',
                encoding="utf-8",
            )
            output = root / "sample.epd"
            manifest = build_suite(
                source,
                output,
                count=2,
                selection_seed="selection",
                split_seed="split",
                source_url="https://example.test/openings.pgn",
                force=False,
            )

            self.assertEqual(len(load_epd(output, split_seed="split")), 2)
            output_bytes = output.read_bytes()
            output_record = manifest["output"]
            self.assertIsInstance(output_record, dict)
            assert isinstance(output_record, dict)
            self.assertEqual(output_record["sha256"], hashlib.sha256(output_bytes).hexdigest())
            self.assertTrue(output.with_suffix(".manifest.json").is_file())
            with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                build_suite(
                    source,
                    output,
                    count=2,
                    selection_seed="selection",
                    split_seed="split",
                    source_url=None,
                    force=False,
                )

    def test_rejects_impossible_sample_size(self) -> None:
        positions = positions_from_fens((("start", chess.STARTING_FEN),), split_seed="split")
        with self.assertRaisesRegex(ValueError, "only 1"):
            select_positions(positions, count=2, seed="selection")

    def test_pinned_suite_matches_its_manifest(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        suite = repository / "benchmarks" / "suites" / "openings_8moves_v3_500.epd"
        manifest_path = suite.with_suffix(".manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        positions = load_epd(suite, split_seed=manifest["splits"]["seed"])
        canonical_payload = suite.read_text(encoding="utf-8").encode()

        self.assertEqual(len(positions), 500)
        self.assertEqual(
            Counter(position.split for position in positions),
            manifest["splits"]["counts"],
        )
        self.assertEqual(
            hashlib.sha256(canonical_payload).hexdigest(), manifest["output"]["sha256"]
        )
        self.assertEqual(len(canonical_payload), manifest["output"]["bytes"])
        self.assertEqual(suite_digest(positions), manifest["output"]["suite_digest"])
        self.assertTrue((suite.parent / "STOCKFISH_BOOKS_LICENSE.txt").is_file())


if __name__ == "__main__":
    unittest.main()
