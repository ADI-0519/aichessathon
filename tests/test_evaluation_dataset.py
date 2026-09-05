from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import chess

from tools.backtest_core import positions_from_fens
from tools.evaluation_dataset import (
    evaluation_group,
    evaluation_identity,
    load_labels,
    make_label,
    write_labels,
    write_manifest,
)


class EvaluationDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.position = positions_from_fens(
            (("initial", chess.STARTING_FEN),), split_seed="test"
        )[0]

    def test_label_records_residual_and_rejects_illegal_best_move(self) -> None:
        with self.assertRaises(ValueError):
            make_label(
                self.position,
                0,
                chess.Move.from_uci("a1a8"),
                baseline_cp=12,
            )

        label = make_label(
            self.position,
            25,
            chess.Move.from_uci("e2e4"),
            baseline_cp=10,
        )
        self.assertEqual(label.residual_cp, 15)

    def test_labels_round_trip_with_schema_and_digest(self) -> None:
        label = make_label(
            self.position,
            24,
            chess.Move.from_uci("e2e4"),
            baseline_cp=10,
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "labels.jsonl"
            digest = write_labels(path, [label])
            self.assertEqual(load_labels(path), [label])
            self.assertEqual(len(digest), 64)
            payload = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(payload["schema_version"], 2)
        self.assertEqual(payload["baseline_cp"], 10)

    def test_loader_rejects_duplicate_positions(self) -> None:
        label = make_label(self.position, 0, None, baseline_cp=0)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "labels.jsonl"
            line = json.dumps(label.as_dict(), sort_keys=True)
            path.write_text(f"{line}\n{line}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate normalized FEN"):
                load_labels(path)

    def test_evaluation_identity_ignores_clocks_and_sampler_group_uses_game(self) -> None:
        first = chess.Board()
        second = chess.Board()
        second.halfmove_clock = 37
        second.fullmove_number = 92

        self.assertEqual(evaluation_identity(first), evaluation_identity(second))
        self.assertEqual(
            evaluation_group("g0000042-p036-middlegame-quiet", first),
            "g0000042",
        )

    def test_manifest_records_teacher_baseline_and_relative_paths(self) -> None:
        label = make_label(
            self.position,
            -18,
            chess.Move.from_uci("d2d4"),
            baseline_cp=-4,
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            labels_path = root / "labels.jsonl"
            manifest_path = root / "labels.manifest.json"
            digest = write_labels(labels_path, [label])
            write_manifest(
                manifest_path,
                labels_path=labels_path,
                suite_path=root / "suite.epd",
                suite_digest="suite-digest",
                engine_path=root / "stockfish.exe",
                engine_digest="engine-digest",
                nodes=5_000,
                labels=[label],
                label_digest=digest,
                baseline_fingerprint={"sha256": "baseline-digest", "path": "."},
                split_seed="test-split",
                selection_count=1,
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        self.assertEqual(manifest["schema_version"], 2)
        self.assertEqual(manifest["target"], "teacher_score_cp_minus_v3_static_score_cp")
        self.assertEqual(manifest["teacher_nodes"], 5_000)
        self.assertEqual(manifest["baseline"]["sha256"], "baseline-digest")
        self.assertEqual(manifest["labels"], "labels.jsonl")
        self.assertEqual(manifest["split_seed"], "test-split")
        self.assertEqual(manifest["teacher_mate_count"], 0)
        self.assertTrue(manifest["complete"])
        self.assertTrue(manifest["clear_hash_each_position"])


if __name__ == "__main__":
    unittest.main()
