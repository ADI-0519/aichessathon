from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import chess
import numpy as np

from tools.evaluation_dataset import EvaluationLabel, write_labels
from tools.evaluation_features import FEATURE_NAMES, MAX_PHASE
from tools.fit_hce import fit_labels, fit_linear_evaluator, write_fit


def synthetic_label(index: int, split: str, x: int, residual: int) -> EvaluationLabel:
    board = chess.Board()
    board.fullmove_number = index + 1
    features = [0] * len(FEATURE_NAMES)
    features[0] = x * MAX_PHASE
    return EvaluationLabel(
        identifier=f"position-{index}",
        group=f"game-{index}",
        fen=board.fen(),
        split=split,  # type: ignore[arg-type]
        score_cp=20 + residual,
        baseline_cp=20,
        teacher_mate=False,
        best_move=None,
        phase=MAX_PHASE,
        features=tuple(features),
    )


class HceFitTests(unittest.TestCase):
    def test_standardized_fit_recovers_simple_weights(self) -> None:
        features = np.asarray([[1, 0], [0, 1], [2, 1], [-1, 2]], dtype=np.float64)
        targets = np.asarray([10, -5, 15, -20], dtype=np.float64)

        weights, intercept = fit_linear_evaluator(features, targets, ridge=0.0)

        np.testing.assert_allclose(weights, (10.0, -5.0), atol=1e-8)
        self.assertAlmostEqual(intercept, 0.0)

    def test_fit_uses_residual_target_and_validation_selection(self) -> None:
        labels = [
            synthetic_label(0, "development", -2, -20),
            synthetic_label(1, "development", -1, -10),
            synthetic_label(2, "development", 1, 10),
            synthetic_label(3, "development", 2, 20),
            synthetic_label(4, "validation", -3, -30),
            synthetic_label(5, "validation", 3, 30),
            synthetic_label(6, "holdout", 4, 40),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            labels_path = root / "labels.jsonl"
            output_path = root / "fit.json"
            write_labels(labels_path, labels)
            result = fit_labels(
                labels_path,
                ridge_candidates=(0.0, 100.0),
                min_train_count=1,
            )
            write_fit(output_path, labels_path, result)
            artifact = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(result.ridge, 0.0)
        self.assertAlmostEqual(result.weights[0], 10.0)
        self.assertNotIn("holdout", result.metrics)
        self.assertEqual(artifact["target"], "teacher_score_cp_minus_v3_static_score_cp")
        self.assertEqual(artifact["validation_count"], 2)
        self.assertIn("integer", artifact["metrics"]["validation"])

    def test_fit_requires_enough_development_data_and_validation(self) -> None:
        labels = [synthetic_label(0, "development", 1, 10)]
        with tempfile.TemporaryDirectory() as temporary:
            labels_path = Path(temporary) / "labels.jsonl"
            write_labels(labels_path, labels)
            with self.assertRaisesRegex(ValueError, "at least 500"):
                fit_labels(labels_path)
            with self.assertRaisesRegex(ValueError, "validation"):
                fit_labels(labels_path, min_train_count=1)


if __name__ == "__main__":
    unittest.main()
