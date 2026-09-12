import unittest

from tools.evaluator_calibration import Sample, _is_holdout, build_report


class EvaluatorCalibrationTests(unittest.TestCase):
    def test_stable_holdout_assignment(self) -> None:
        fen = "8/8/8/8/8/8/4K3/6k1 w - - 0 1"
        self.assertEqual(_is_holdout(fen, 25), _is_holdout(fen, 25))

    def test_report_selects_best_blend_without_mutating_samples(self) -> None:
        samples = [
            Sample(f"game-{index}", f"fen-{index}", target, pieces, learned, handcrafted)
            for index, (target, pieces, learned, handcrafted) in enumerate(
                (
                    (100, 32, 100, -100),
                    (-100, 32, -100, 100),
                    (60, 12, 60, -60),
                    (-60, 12, -60, 60),
                    (25, 6, 25, -25),
                    (-25, 6, -25, 25),
                    (80, 20, 80, -80),
                    (-80, 20, -80, 80),
                    (40, 15, 40, -40),
                    (-40, 15, -40, 40),
                    (90, 28, 90, -90),
                    (-90, 28, -90, 90),
                )
            )
        ]
        report = build_report(
            samples,
            blend_weights=(0, 50, 100),
            piece_bands=((2, 32),),
            holdout_percent=50,
        )
        self.assertEqual(report["blend_holdout"]["100"]["rmse_cp"], 0.0)
        self.assertGreater(report["blend_holdout"]["0"]["rmse_cp"], 0.0)
        self.assertEqual(len(samples), 12)
