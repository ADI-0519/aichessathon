from __future__ import annotations

import unittest

import chess

from tools.evaluation_dataset import EvaluationLabel
from tools.evaluation_features import FEATURE_NAMES
from tools.relabel_evaluation_baseline import rebase_labels


class RelabelEvaluationBaselineTests(unittest.TestCase):
    def test_rebase_changes_only_the_baseline_score(self) -> None:
        label = EvaluationLabel(
            identifier="position-1",
            group="game-1",
            fen=chess.Board().fen(),
            split="development",
            score_cp=75,
            baseline_cp=20,
            teacher_mate=False,
            best_move="e2e4",
            phase=24,
            features=(0,) * len(FEATURE_NAMES),
        )

        result = rebase_labels([label], lambda fen: 50 if fen == label.fen else 0)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].baseline_cp, 50)
        self.assertEqual(result[0].residual_cp, 25)
        self.assertEqual(result[0].score_cp, label.score_cp)
        self.assertEqual(result[0].features, label.features)
        self.assertEqual(result[0].fen, label.fen)
        self.assertEqual(label.baseline_cp, 20)


if __name__ == "__main__":
    unittest.main()
