from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import chess

from tools.backtest_core import positions_from_fens
from tools.evaluation_dataset import make_label, write_labels, write_manifest
from tools.label_positions import _validate_resume, baseline_static_score


class LabelPositionTests(unittest.TestCase):
    def test_canonical_baseline_evaluates_a_position(self) -> None:
        score = baseline_static_score(chess.Board())
        self.assertIsInstance(score, int)

    def test_resume_requires_identical_inputs_and_intact_labels(self) -> None:
        position = positions_from_fens(
            (("initial", chess.STARTING_FEN),), split_seed="resume-test"
        )[0]
        label = make_label(position, 15, None, baseline_cp=4)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            labels_path = root / "labels.jsonl"
            manifest_path = root / "labels.manifest.json"
            digest = write_labels(labels_path, [label])
            write_manifest(
                manifest_path,
                labels_path=labels_path,
                suite_path=root / "suite.epd",
                suite_digest="suite-hash",
                engine_path=root / "stockfish.exe",
                engine_digest="teacher-hash",
                nodes=10_000,
                labels=[label],
                label_digest=digest,
                baseline_fingerprint={"sha256": "baseline-hash", "path": "."},
                split_seed="resume-test",
                selection_count=2,
            )

            _validate_resume(
                manifest_path,
                labels_path,
                suite_sha256="suite-hash",
                teacher_sha256="teacher-hash",
                baseline_sha256="baseline-hash",
                nodes=10_000,
                split_seed="resume-test",
                selection_count=2,
            )
            with self.assertRaisesRegex(ValueError, "teacher_nodes"):
                _validate_resume(
                    manifest_path,
                    labels_path,
                    suite_sha256="suite-hash",
                    teacher_sha256="teacher-hash",
                    baseline_sha256="baseline-hash",
                    nodes=20_000,
                    split_seed="resume-test",
                    selection_count=2,
                )


if __name__ == "__main__":
    unittest.main()
