"""Continuous, move-aware clock allocation for the fixed tournament control."""

from __future__ import annotations

MAX_BUDGET_MS = 4_500

# A move where the search keeps changing its mind is worth more than the flat
# allocation gives it, and one that settled early is worth less. These bound how
# far either way the deepening loop may go; move_budget_ms stays the target it
# aims at, and hard_budget_ms is the ceiling it may not cross.
HARD_BUDGET_MULTIPLIER = 2
MAX_HARD_BUDGET_MS = 9_000
MIN_RESERVE_MS = 250
MAX_RESERVE_MS = 2_000


def estimated_moves_remaining(fullmove_number: int) -> int:
    """Estimate our remaining decisions without relying on hidden game state."""
    move_number = max(1, fullmove_number)
    return max(12, min(32, 32 - (2 * move_number) // 5))


def move_budget_ms(time_left_ms: int, fullmove_number: int) -> int:
    """Return a smooth search budget while retaining a hard clock reserve.

    The declining moves-to-go estimate spends progressively more of the clock
    in late middlegames and endings.  The small credit models part of the fixed
    500 ms increment, but tapers towards zero on a critically low clock.
    """
    if time_left_ms <= 0:
        return 0

    reserve_ms = max(
        MIN_RESERVE_MS,
        min(MAX_RESERVE_MS, time_left_ms // 16),
    )
    usable_ms = max(0, time_left_ms - reserve_ms)
    if usable_ms == 0:
        return 0

    moves_remaining = estimated_moves_remaining(fullmove_number)
    increment_credit_ms = 350 * time_left_ms // (time_left_ms + 5_000)
    target_ms = usable_ms // moves_remaining + increment_credit_ms
    return max(0, min(MAX_BUDGET_MS, usable_ms, target_ms))


def hard_budget_ms(time_left_ms: int, fullmove_number: int) -> int:
    """Return the most this move may take when the search is still unsettled.

    Never more than half of what is usable: one interesting position must not
    be able to spend the clock that the rest of the game needs.
    """
    target_ms = move_budget_ms(time_left_ms, fullmove_number)
    if target_ms <= 0:
        return 0
    reserve_ms = max(MIN_RESERVE_MS, min(MAX_RESERVE_MS, time_left_ms // 16))
    usable_ms = max(0, time_left_ms - reserve_ms)
    ceiling_ms = min(MAX_HARD_BUDGET_MS, usable_ms // 6)
    return max(target_ms, min(ceiling_ms, target_ms * HARD_BUDGET_MULTIPLIER))
