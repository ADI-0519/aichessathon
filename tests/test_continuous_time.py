from __future__ import annotations

import unittest

from challengers.v7_continuous_time.time_manager import (
    estimated_moves_remaining,
    move_budget_ms,
)


class ContinuousTimeTests(unittest.TestCase):
    def test_reference_budgets_match_the_reviewed_schedule(self) -> None:
        self.assertEqual(move_budget_ms(120_000, 7), 4_269)
        self.assertEqual(move_budget_ms(60_000, 30), 3_223)
        self.assertEqual(move_budget_ms(41_496, 49), 3_350)

    def test_schedule_has_no_old_threshold_cliffs(self) -> None:
        self.assertLessEqual(
            abs(move_budget_ms(60_000, 30) - move_budget_ms(59_999, 30)),
            1,
        )
        self.assertLessEqual(
            abs(move_budget_ms(10_000, 50) - move_budget_ms(9_999, 50)),
            1,
        )

    def test_budget_increases_with_time_and_game_progress(self) -> None:
        remaining = (500, 1_000, 5_000, 10_000, 30_000, 60_000, 120_000)
        budgets = [move_budget_ms(value, 30) for value in remaining]
        self.assertEqual(budgets, sorted(budgets))
        self.assertGreater(move_budget_ms(50_000, 49), move_budget_ms(50_000, 9))
        self.assertGreater(
            estimated_moves_remaining(9),
            estimated_moves_remaining(49),
        )

    def test_reserve_prevents_flag_in_long_clock_simulations(self) -> None:
        for base_ms, increment_ms in ((120_000, 500), (10_000, 100)):
            remaining_ms = base_ms
            for fullmove_number in range(9, 159):
                budget_ms = move_budget_ms(remaining_ms, fullmove_number)
                self.assertGreaterEqual(budget_ms, 0)
                self.assertLess(budget_ms, remaining_ms)
                remaining_ms = remaining_ms - budget_ms + increment_ms
                self.assertGreater(remaining_ms, 0)

    def test_nonpositive_and_exhausted_clocks_return_zero(self) -> None:
        self.assertEqual(move_budget_ms(0, 30), 0)
        self.assertEqual(move_budget_ms(-1, 30), 0)
        self.assertEqual(move_budget_ms(200, 30), 0)


if __name__ == "__main__":
    unittest.main()
