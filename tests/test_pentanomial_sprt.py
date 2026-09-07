from __future__ import annotations

import unittest

from tools.pentanomial_sprt import (
    elo_to_score,
    evaluate,
    log_likelihood_ratio,
    score_to_elo,
)


class PentanomialSprtTests(unittest.TestCase):
    def test_elo_score_round_trip(self) -> None:
        for elo in (-100.0, -20.0, 0.0, 20.0, 100.0):
            self.assertAlmostEqual(score_to_elo(elo_to_score(elo)), elo, places=9)

    def test_balanced_match_is_evidence_for_h0_over_positive_h1(self) -> None:
        verdict = evaluate((5, 10, 20, 10, 5), elo0=0.0, elo1=20.0)
        self.assertLess(verdict.llr, 0.0)
        self.assertEqual(verdict.decision, "continue")

    def test_historical_nnue100_result(self) -> None:
        verdict = evaluate((11, 15, 14, 5, 5), elo0=0.0, elo1=20.0)
        self.assertAlmostEqual(verdict.llr, -1.5956, places=3)
        self.assertEqual(verdict.decision, "continue")
        self.assertAlmostEqual(verdict.score, 0.39)

    def test_wald_boundaries_and_decisions(self) -> None:
        neutral = evaluate((0, 0, 1000, 0, 0), min_pairs=0)
        winning = evaluate((0, 0, 0, 0, 100), min_pairs=0)
        self.assertAlmostEqual(neutral.lower_bound, -2.944438979, places=6)
        self.assertAlmostEqual(neutral.upper_bound, 2.944438979, places=6)
        self.assertEqual(neutral.decision, "accept_h0")
        self.assertEqual(winning.decision, "accept_h1")

    def test_minimum_pairs_delays_a_boundary_decision(self) -> None:
        verdict = evaluate((0, 0, 0, 0, 100), min_pairs=250)
        self.assertGreater(verdict.llr, verdict.upper_bound)
        self.assertEqual(verdict.decision, "continue")

    def test_invalid_inputs_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            log_likelihood_ratio((1, 1, 1, 1, 1), 20.0, 0.0)
        with self.assertRaises(ValueError):
            evaluate((1, 2, 3))
        with self.assertRaises(ValueError):
            evaluate((1, -1, 1, 1, 1))
        with self.assertRaises(ValueError):
            evaluate((1, 1, 1, 1, 1), alpha=1.0)


if __name__ == "__main__":
    unittest.main()
