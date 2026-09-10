# Material-aware adaptive clock allocation for V13.
from __future__ import annotations

from dataclasses import dataclass

MAX_NORMAL_BUDGET_MS = 4_500
MAX_HARD_BUDGET_MS = 8_000
MIN_RESERVE_MS = 250
MAX_RESERVE_MS = 2_000
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
    move_number = max(1, fullmove_number)
    return max(12, min(32, 32 - (2 * move_number) // 5))


def _material_horizon(piece_count: int) -> int:
    pieces = max(2, min(32, piece_count))
    if pieces >= 26:
        return 40
    if pieces >= 20:
        return 34
    if pieces >= 14:
        return 32
    if pieces >= 8:
        return 24
    return 10


def estimated_moves_remaining(fullmove_number: int, piece_count: int = 32) -> int:
    material = _material_horizon(piece_count)
    move_number = _move_number_horizon(fullmove_number)
    if piece_count >= 14:
        return max(material, move_number)
    if fullmove_number < 35:
        return max(material, move_number)
    return material


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
