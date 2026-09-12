# Long-horizon adaptive clock allocation for the finals candidate.
from __future__ import annotations

from dataclasses import dataclass

MAX_NORMAL_BUDGET_MS = 4_500
MAX_HARD_BUDGET_MS = 8_000
MIN_RESERVE_MS = 250
MAX_RESERVE_MS = 2_000
HORIZON_BASE_MOVES = 78
HORIZON_FLOOR_MOVES = 36
HORIZON_DECAY_NUMERATOR = 2
HORIZON_DECAY_DENOMINATOR = 5
SOFT_BUDGET_NUMERATOR = 3
SOFT_BUDGET_DENOMINATOR = 4
HARD_BUDGET_NUMERATOR = 9
HARD_BUDGET_DENOMINATOR = 5


@dataclass(frozen=True, slots=True)
class MoveTimeLimits:
    soft_ms: int
    normal_ms: int
    hard_ms: int


def _move_number_horizon(fullmove_number: int) -> int:
    """Estimate remaining decisions without assuming short endgames.

    The qualifier games showed that piece count is a poor proxy for remaining
    length: several sparse positions continued for dozens of moves.  The
    horizon therefore declines continuously with game age and retains a large
    floor instead of stepping down whenever material crosses a boundary.
    """
    move_number = max(1, fullmove_number)
    elapsed_allowance = (
        HORIZON_DECAY_NUMERATOR * move_number // HORIZON_DECAY_DENOMINATOR
    )
    return max(HORIZON_FLOOR_MOVES, HORIZON_BASE_MOVES - elapsed_allowance)


def estimated_moves_remaining(fullmove_number: int, piece_count: int = 32) -> int:
    """Return the planning horizon for our remaining moves.

    ``piece_count`` remains part of the public helper signature for compatibility
    with the agent and diagnostics.  It is validated but deliberately does not
    shorten the horizon: sparse qualifier endings were often the longest games.
    """
    if not 2 <= piece_count <= 32:
        raise ValueError("piece_count must be between 2 and 32")
    return _move_number_horizon(fullmove_number)


def move_time_limits(
    time_left_ms: int,
    fullmove_number: int,
    piece_count: int = 32,
) -> MoveTimeLimits:
    if time_left_ms <= 0:
        return MoveTimeLimits(0, 0, 0)

    reserve_ms = max(
        MIN_RESERVE_MS,
        min(MAX_RESERVE_MS, time_left_ms // 16),
    )
    usable_ms = max(0, time_left_ms - reserve_ms)
    if usable_ms == 0:
        return MoveTimeLimits(0, 0, 0)

    moves_remaining = estimated_moves_remaining(fullmove_number, piece_count)
    increment_credit_ms = 350 * time_left_ms // (time_left_ms + 5_000)
    target_ms = usable_ms // moves_remaining + increment_credit_ms

    normal_ms = max(1, min(MAX_NORMAL_BUDGET_MS, usable_ms, target_ms))
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


def move_budget_ms(
    time_left_ms: int,
    fullmove_number: int,
    piece_count: int = 32,
) -> int:
    return move_time_limits(
        time_left_ms,
        fullmove_number,
        piece_count,
    ).normal_ms
