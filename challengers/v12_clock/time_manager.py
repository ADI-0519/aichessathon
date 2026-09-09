"""Continuous, move-aware clock allocation for the fixed tournament control."""

from __future__ import annotations

MAX_BUDGET_MS = 4_500
MIN_RESERVE_MS = 250
MAX_RESERVE_MS = 2_000


def estimated_moves_remaining(fullmove_number: int, piece_count: int = 32) -> int:
    """Estimate our remaining decisions from the material still on the board.

    The move number alone is a poor predictor and the old estimate bottomed out
    at twelve, so from move fifty it believed the game was nearly over however
    long it ran. Since the budget is the usable clock divided by this number, it
    *rose* to the cap as a game went long: rounds 80 and 82 reached moves 86 and
    87 spending the maximum from move 40, and both finished under seven seconds
    against opponents holding twenty-five.

    Material is measurable every move and predicts length much better. Medians
    over 3,439 positions from our own rated games:

        26-32 pieces -> 43 of our moves left
        20-25        -> 31
        14-19        -> 33
        8-13         -> 23
        2-7          -> 8

    A grinding ending with ten pieces still has around twenty of our moves in
    it, which is where the old estimate was worst and where the audit of rounds
    77 to 82 found sub-second moves costing 35.4 cp against 16.5 for moves given
    three seconds.
    """
    if piece_count >= 26:
        base = 40
    elif piece_count >= 14:
        base = 32
    elif piece_count >= 8:
        base = 24
    else:
        base = 10
    # Very long games keep going; never plan on fewer than a dozen more.
    return max(12, base - max(0, fullmove_number - 60) // 4)


def move_budget_ms(
    time_left_ms: int, fullmove_number: int, piece_count: int = 32
) -> int:
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

    moves_remaining = estimated_moves_remaining(fullmove_number, piece_count)
    # The half second per move is ours to spend; the old credit topped out at
    # 342 ms and so quietly banked part of every increment.
    increment_credit_ms = 450 * time_left_ms // (time_left_ms + 3_000)
    target_ms = usable_ms // moves_remaining + increment_credit_ms
    return max(0, min(MAX_BUDGET_MS, usable_ms, target_ms))
