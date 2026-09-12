from __future__ import annotations

import runpy
import unittest
from itertools import pairwise
from pathlib import Path
from typing import Any

REPOSITORY = Path(__file__).resolve().parents[1]
CANDIDATE = REPOSITORY / "challengers" / "exp_finals_v18_core"


class FinalsV18TimeManagerTests(unittest.TestCase):
    namespace: dict[str, Any]

    @classmethod
    def setUpClass(cls) -> None:
        cls.namespace = runpy.run_path(str(CANDIDATE / "time_manager.py"))

    def test_horizon_declines_continuously_and_keeps_long_floor(self) -> None:
        estimate = self.namespace["estimated_moves_remaining"]
        horizons = [estimate(move, 32) for move in range(1, 161)]

        self.assertTrue(all(a >= b for a, b in pairwise(horizons)))
        self.assertEqual(horizons[-1], self.namespace["HORIZON_FLOOR_MOVES"])
        self.assertGreaterEqual(estimate(60, 10), 50)
        self.assertGreaterEqual(estimate(100, 6), 36)

    def test_piece_count_does_not_shorten_sparse_endgames(self) -> None:
        estimate = self.namespace["estimated_moves_remaining"]
        self.assertEqual(estimate(60, 32), estimate(60, 6))

    def test_invalid_piece_counts_are_rejected(self) -> None:
        estimate = self.namespace["estimated_moves_remaining"]
        for piece_count in (0, 1, 33):
            with self.subTest(piece_count=piece_count), self.assertRaises(ValueError):
                estimate(20, piece_count)

    def test_deadlines_are_ordered_and_reserve_safe(self) -> None:
        limits_for = self.namespace["move_time_limits"]
        for remaining in (50, 100, 500, 2_000, 12_000, 120_000):
            for fullmove in (1, 20, 60, 120):
                limits = limits_for(remaining, fullmove, 16)
                with self.subTest(remaining=remaining, fullmove=fullmove):
                    self.assertLessEqual(limits.soft_ms, limits.normal_ms)
                    self.assertLessEqual(limits.normal_ms, limits.hard_ms)
                    self.assertLess(limits.hard_ms, remaining)

    def test_new_opening_budget_is_more_conservative_than_v16(self) -> None:
        old = runpy.run_path(
            str(
                REPOSITORY
                / "challengers"
                / "exp_release_v16_search_bignet"
                / "time_manager.py"
            )
        )["move_time_limits"]
        new = self.namespace["move_time_limits"]

        self.assertLess(new(120_000, 8, 30).normal_ms, old(120_000, 8, 30).normal_ms)


if __name__ == "__main__":
    unittest.main()
