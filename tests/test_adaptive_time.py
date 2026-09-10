from __future__ import annotations

import unittest
from pathlib import Path
from typing import Any

from challengers.exp_adaptive_time.time_manager import (
    MAX_HARD_BUDGET_MS,
    move_budget_ms,
    move_time_limits,
)
from tools.search_diagnostics import load_engine_modules

REPOSITORY = Path(__file__).resolve().parents[1]
CHALLENGER = REPOSITORY / "challengers" / "exp_adaptive_time"


class AdaptiveTimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _, cls.search = load_engine_modules(CHALLENGER)

    def test_normal_budget_remains_the_reviewed_continuous_schedule(self) -> None:
        for remaining_ms, fullmove in ((120_000, 7), (60_000, 30), (41_496, 49)):
            limits = move_time_limits(remaining_ms, fullmove)
            self.assertEqual(limits.normal_ms, move_budget_ms(remaining_ms, fullmove))
            self.assertLess(limits.soft_ms, limits.normal_ms)
            self.assertGreater(limits.hard_ms, limits.normal_ms)
            self.assertLess(limits.hard_ms, remaining_ms)
            self.assertLessEqual(limits.hard_ms, MAX_HARD_BUDGET_MS)

    def test_deadlines_are_ordered_even_on_exhausted_clocks(self) -> None:
        for remaining_ms in (-1, 0, 200, 251, 500, 2_000, 10_000, 120_000):
            limits = move_time_limits(remaining_ms, 30)
            self.assertLessEqual(limits.soft_ms, limits.normal_ms)
            self.assertLessEqual(limits.normal_ms, limits.hard_ms)
            self.assertGreaterEqual(limits.soft_ms, 0)
            if limits.hard_ms:
                self.assertLess(limits.hard_ms, remaining_ms)

    def test_worst_case_hard_spending_retains_time(self) -> None:
        for base_ms, increment_ms in ((120_000, 500), (10_000, 100)):
            remaining_ms = base_ms
            for fullmove in range(9, 159):
                limits = move_time_limits(remaining_ms, fullmove)
                remaining_ms = remaining_ms - limits.hard_ms + increment_ms
                self.assertGreater(remaining_ms, 0)

    def test_iteration_policy_banks_time_only_after_stability(self) -> None:
        policy: Any = self.search._should_stop_after_iteration
        common = {
            "soft_limit_s": 2.0,
            "normal_limit_s": 3.0,
            "hard_limit_s": 5.0,
            "last_iteration_s": 0.4,
        }
        self.assertTrue(
            policy(elapsed_s=2.1, stable_iterations=2, unstable=False, **common)
        )
        self.assertFalse(
            policy(elapsed_s=2.1, stable_iterations=1, unstable=False, **common)
        )
        self.assertTrue(
            policy(elapsed_s=3.1, stable_iterations=0, unstable=False, **common)
        )
        self.assertFalse(
            policy(elapsed_s=3.1, stable_iterations=0, unstable=True, **common)
        )

    def test_iteration_policy_does_not_start_work_that_cannot_finish(self) -> None:
        policy: Any = self.search._should_stop_after_iteration
        self.assertTrue(
            policy(
                elapsed_s=4.5,
                soft_limit_s=2.0,
                normal_limit_s=3.0,
                hard_limit_s=5.0,
                stable_iterations=0,
                unstable=True,
                last_iteration_s=0.4,
            )
        )


if __name__ == "__main__":
    unittest.main()
