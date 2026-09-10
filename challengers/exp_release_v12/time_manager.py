"""Continuous, move-aware clock allocation for the fixed tournament control."""

from __future__ import annotations

from dataclasses import dataclass

MAX_BUDGET_MS = 4_500
MAX_HARD_BUDGET_MS = 8_000
MIN_RESERVE_MS = 250
MAX_RESERVE_MS = 2_000
SOFT_BUDGET_NUMERATOR = 3
SOFT_BUDGET_DENOMINATOR = 4
HARD_BUDGET_NUMERATOR = 9
HARD_BUDGET_DENOMINATOR = 5


@dataclass(frozen=True, slots=True)
class MoveTimeLimits:
    """Three search deadlines, all covered by the retained clock reserve."""

    soft_ms: int
    normal_ms: int
    hard_ms: int


def estimated_moves_remaining(fullmove_number: int) -> int:
    """Estimate our remaining decisions without relying on hidden game state."""
    move_number = max(1, fullmove_number)
    return max(12, min(32, 32 - (2 * move_number) // 5))


def move_time_limits(time_left_ms: int, fullmove_number: int) -> MoveTimeLimits:
    """Return adaptive deadlines while retaining a hard clock reserve.

    The declining moves-to-go estimate spends progressively more of the clock
    in late middlegames and endings.  The small credit models part of the fixed
    500 ms increment, but tapers towards zero on a critically low clock. Stable
    searches may stop at the soft limit; unstable searches may use the hard
    limit without touching the reserve.
    """
    if time_left_ms <= 0:
        return MoveTimeLimits(0, 0, 0)

    reserve_ms = max(
        MIN_RESERVE_MS,
        min(MAX_RESERVE_MS, time_left_ms // 16),
    )
    usable_ms = max(0, time_left_ms - reserve_ms)
    if usable_ms == 0:
        return MoveTimeLimits(0, 0, 0)

    moves_remaining = estimated_moves_remaining(fullmove_number)
    increment_credit_ms = 350 * time_left_ms // (time_left_ms + 5_000)
    target_ms = usable_ms // moves_remaining + increment_credit_ms
    normal_ms = max(0, min(MAX_BUDGET_MS, usable_ms, target_ms))
    soft_ms = max(
        1,
        normal_ms * SOFT_BUDGET_NUMERATOR // SOFT_BUDGET_DENOMINATOR,
    )
    hard_ms = min(
        MAX_HARD_BUDGET_MS,
        usable_ms,
        normal_ms * HARD_BUDGET_NUMERATOR // HARD_BUDGET_DENOMINATOR,
    )
    return MoveTimeLimits(soft_ms, normal_ms, max(normal_ms, hard_ms))


def move_budget_ms(time_left_ms: int, fullmove_number: int) -> int:
    """Return the normal deadline for compatibility with existing tooling."""
    return move_time_limits(time_left_ms, fullmove_number).normal_ms
