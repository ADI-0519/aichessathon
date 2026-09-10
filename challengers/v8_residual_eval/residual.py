"""Compiled strategic correction fitted against the exact V5 static evaluator."""

from __future__ import annotations

import numpy as np
from numba import njit
from numpy.typing import NDArray

import engine

MAX_PHASE = 24
INTERCEPT = 20

# Feature order: knight/bishop/rook/queen mobility, king-zone pressure,
# hanging minor/major pieces, safe space, rook activity, blocked passers,
# minor outposts, and king-file exposure. Each has middlegame/endgame weights.
MG_WEIGHTS = np.asarray((3, 5, 10, 0, 12, 112, 66, 3, -12, 50, 21, 42), dtype=np.int32)
EG_WEIGHTS = np.asarray(
    (-20, -16, -21, -30, -9, 74, 129, -18, -12, 94, -23, -51),
    dtype=np.int32,
)
PHASE_WEIGHTS = np.asarray((0, 1, 1, 2, 4, 0), dtype=np.int32)


def _space_mask(files: range, ranks: tuple[int, ...]) -> np.uint64:
    value = 0
    for rank in ranks:
        for file_index in files:
            value |= 1 << (rank * 8 + file_index)
    return np.uint64(value)


SPACE_MASKS = np.asarray(
    (
        _space_mask(range(2, 6), (3, 4, 5)),
        _space_mask(range(2, 6), (2, 3, 4)),
    ),
    dtype=np.uint64,
)


@njit(cache=False, inline="always")
def _sliding_attacks(
    square: int,
    occupied: np.uint64,
    file_deltas: NDArray[np.int8],
    rank_deltas: NDArray[np.int8],
) -> np.uint64:
    attacks = np.uint64(0)
    square_file = square & 7
    square_rank = square >> 3
    for direction in range(4):
        file_index = square_file + int(file_deltas[direction])
        rank = square_rank + int(rank_deltas[direction])
        while 0 <= file_index < 8 and 0 <= rank < 8:
            target = rank * 8 + file_index
            target_bit = engine.bit(target)
            attacks |= target_bit
            if occupied & target_bit:
                break
            file_index += int(file_deltas[direction])
            rank += int(rank_deltas[direction])
    return attacks


@njit(cache=False, inline="always")
def _piece_attacks(
    pieces: NDArray[np.uint64],
    color: int,
    kind: int,
    square: int,
    occupied: np.uint64,
) -> np.uint64:
    if kind == engine.PAWN:
        return np.uint64(engine.PAWN_ATTACKS[color, square])
    if kind == engine.KNIGHT:
        return np.uint64(engine.KNIGHT_ATTACKS[square])
    if kind == engine.BISHOP:
        return _sliding_attacks(
            square,
            occupied,
            engine.BISHOP_FILE_DELTAS,
            engine.BISHOP_RANK_DELTAS,
        )
    if kind == engine.ROOK:
        return _sliding_attacks(
            square,
            occupied,
            engine.ROOK_FILE_DELTAS,
            engine.ROOK_RANK_DELTAS,
        )
    if kind == engine.QUEEN:
        return _sliding_attacks(
            square,
            occupied,
            engine.BISHOP_FILE_DELTAS,
            engine.BISHOP_RANK_DELTAS,
        ) | _sliding_attacks(
            square,
            occupied,
            engine.ROOK_FILE_DELTAS,
            engine.ROOK_RANK_DELTAS,
        )
    return np.uint64(engine.KING_ATTACKS[square])


@njit(cache=False, inline="always")
def _pawn_attacks(pieces: NDArray[np.uint64], color: int) -> np.uint64:
    attacks = np.uint64(0)
    pawns = pieces[engine.piece_index(color, engine.PAWN)]
    while pawns:
        square = engine.lsb_square(pawns)
        pawns ^= engine.bit(square)
        attacks |= engine.PAWN_ATTACKS[color, square]
    return attacks


@njit(cache=False)
def _side_features(
    pieces: NDArray[np.uint64],
    color: int,
    occupied: np.uint64,
) -> tuple[int, int, int, int, int, int, int, int, int, int, int, int]:
    enemy = engine.BLACK if color == engine.WHITE else engine.WHITE
    own_occupied = engine.occupancy_for(pieces, color)
    enemy_king = pieces[engine.piece_index(enemy, engine.KING)]
    king_zone = np.uint64(0)
    if enemy_king:
        enemy_king_square = engine.lsb_square(enemy_king)
        king_zone = engine.KING_ATTACKS[enemy_king_square] | enemy_king

    knight_mobility = 0
    bishop_mobility = 0
    rook_mobility = 0
    queen_mobility = 0
    king_pressure = 0
    hanging_minor = 0
    hanging_major = 0
    controlled = np.uint64(0)

    for kind in range(engine.PAWN, engine.KING):
        units = pieces[engine.piece_index(color, kind)]
        while units:
            square = engine.lsb_square(units)
            units ^= engine.bit(square)
            attacks = _piece_attacks(pieces, color, kind, square, occupied)
            controlled |= attacks
            king_pressure += engine.popcount(attacks & king_zone)
            mobility = engine.popcount(attacks & ~own_occupied)
            if kind == engine.KNIGHT:
                knight_mobility += mobility
            elif kind == engine.BISHOP:
                bishop_mobility += mobility
            elif kind == engine.ROOK:
                rook_mobility += mobility
            elif kind == engine.QUEEN:
                queen_mobility += mobility

            if kind == engine.KNIGHT or kind == engine.BISHOP:
                if engine.is_square_attacked(
                    pieces, square, enemy
                ) and not engine.is_square_attacked(pieces, square, color):
                    hanging_minor += 1
            elif (
                (kind == engine.ROOK or kind == engine.QUEEN)
                and engine.is_square_attacked(pieces, square, enemy)
                and not engine.is_square_attacked(pieces, square, color)
            ):
                hanging_major += 2 if kind == engine.QUEEN else 1

    enemy_pawn_attacks = _pawn_attacks(pieces, enemy)
    safe_space = engine.popcount(
        controlled & SPACE_MASKS[color] & ~own_occupied & ~enemy_pawn_attacks
    )

    rook_activity = 0
    rooks = pieces[engine.piece_index(color, engine.ROOK)]
    rook_scan = rooks
    seventh_rank = 6 if color == engine.WHITE else 1
    first_rook = -1
    second_rook = -1
    while rook_scan:
        square = engine.lsb_square(rook_scan)
        rook_scan ^= engine.bit(square)
        if first_rook < 0:
            first_rook = square
        elif second_rook < 0:
            second_rook = square
        if square >> 3 == seventh_rank:
            rook_activity += 2
    if first_rook >= 0 and second_rook >= 0:
        attacks = _piece_attacks(
            pieces, color, engine.ROOK, first_rook, occupied
        )
        if attacks & engine.bit(second_rook):
            rook_activity += 1

    pawns = pieces[engine.piece_index(color, engine.PAWN)]
    own_pawn_attacks = _pawn_attacks(pieces, color)
    blocked_passers = 0
    pawn_scan = pawns
    while pawn_scan:
        square = engine.lsb_square(pawn_scan)
        pawn_scan ^= engine.bit(square)
        file_index = square & 7
        rank = square >> 3
        passed = True
        enemy_pawns = pieces[engine.piece_index(enemy, engine.PAWN)]
        while enemy_pawns:
            enemy_square = engine.lsb_square(enemy_pawns)
            enemy_pawns ^= engine.bit(enemy_square)
            enemy_file = enemy_square & 7
            enemy_rank = enemy_square >> 3
            if abs(enemy_file - file_index) <= 1 and (
                (color == engine.WHITE and enemy_rank > rank)
                or (color == engine.BLACK and enemy_rank < rank)
            ):
                passed = False
                break
        front = square + (8 if color == engine.WHITE else -8)
        if passed and 0 <= front < 64 and occupied & engine.bit(front):
            blocked_passers += 1

    minor_outposts = 0
    for kind in range(engine.KNIGHT, engine.BISHOP + 1):
        minors = pieces[engine.piece_index(color, kind)]
        while minors:
            square = engine.lsb_square(minors)
            minors ^= engine.bit(square)
            rank = square >> 3
            enemy_half = rank >= 4 if color == engine.WHITE else rank <= 3
            square_bit = engine.bit(square)
            if (
                enemy_half
                and own_pawn_attacks & square_bit
                and not enemy_pawn_attacks & square_bit
            ):
                minor_outposts += 1

    king_file_exposure = 0
    king = pieces[engine.piece_index(color, engine.KING)]
    if king:
        king_file = engine.lsb_square(king) & 7
        for file_index in range(max(0, king_file - 1), min(7, king_file + 1) + 1):
            file_mask = np.uint64(0x0101010101010101) << np.uint64(file_index)
            if pawns & file_mask == 0:
                king_file_exposure += 1

    return (
        knight_mobility,
        bishop_mobility,
        rook_mobility,
        queen_mobility,
        king_pressure,
        hanging_minor,
        hanging_major,
        safe_space,
        rook_activity,
        blocked_passers,
        minor_outposts,
        king_file_exposure,
    )


@njit(cache=False, inline="always")
def correction(
    pieces: NDArray[np.uint64],
    state: NDArray[np.int64],
) -> int:
    """Return the fitted additive correction from the side to move."""
    phase = 0
    for color in range(2):
        for kind in range(engine.PIECE_KIND_COUNT):
            phase += engine.popcount(
                pieces[engine.piece_index(color, kind)]
            ) * int(PHASE_WEIGHTS[kind])
    phase = min(MAX_PHASE, phase)
    occupied = engine.all_occupancy(pieces)
    white = _side_features(pieces, engine.WHITE, occupied)
    black = _side_features(pieces, engine.BLACK, occupied)
    perspective = 1 if int(state[engine.STATE_SIDE]) == engine.WHITE else -1
    numerator = 0
    for index in range(len(MG_WEIGHTS)):
        liability = index in (5, 6, 9, 11)
        balance = black[index] - white[index] if liability else white[index] - black[index]
        value = perspective * balance
        numerator += (
            value * int(MG_WEIGHTS[index]) * phase
            + value * int(EG_WEIGHTS[index]) * (MAX_PHASE - phase)
        )
    return INTERCEPT + numerator // MAX_PHASE
