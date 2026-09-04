"""Numba-compatible bitboard state, legal move generation, hashing, and make/unmake.

This module deliberately implements only the board core. Search is added after
the representation passes perft, differential move-generation, and reversible
state tests. The production ``agent.py`` remains untouched until a complete
challenger clears the reliability and strength gates.
"""

from __future__ import annotations

from dataclasses import dataclass

import chess
import numpy as np
from numba import njit
from numpy.typing import NDArray

# Internal colors and piece kinds. Piece indices are color * 6 + kind.
WHITE = 0
BLACK = 1
PAWN = 0
KNIGHT = 1
BISHOP = 2
ROOK = 3
QUEEN = 4
KING = 5
PIECE_KIND_COUNT = 6
PIECE_BITBOARD_COUNT = 12
NO_PIECE = -1

A1 = 0
B1 = 1
C1 = 2
D1 = 3
E1 = 4
F1 = 5
G1 = 6
H1 = 7
A8 = 56
B8 = 57
C8 = 58
D8 = 59
E8 = 60
F8 = 61
G8 = 62
H8 = 63

# Scalar state slots.
STATE_SIDE = 0
STATE_CASTLING = 1
STATE_EP_SQUARE = 2
STATE_HALFMOVE = 3
STATE_FULLMOVE = 4
STATE_SIZE = 5

# Castling-right bits.
CASTLE_WHITE_KING = 1
CASTLE_WHITE_QUEEN = 2
CASTLE_BLACK_KING = 4
CASTLE_BLACK_QUEEN = 8

# Packed move: from[0:6], to[6:12], promotion kind[12:15], flags[15:20].
FROM_MASK = 0x3F
TO_SHIFT = 6
TO_MASK = 0x3F
PROMOTION_SHIFT = 12
PROMOTION_MASK = 0x7
FLAG_CAPTURE = 1 << 15
FLAG_DOUBLE_PAWN = 1 << 16
FLAG_EN_PASSANT = 1 << 17
FLAG_CASTLING = 1 << 18
FLAG_PROMOTION = 1 << 19

MAX_MOVES = 256

# Undo slots. Board bitboards are reversed from the move plus these scalars.
UNDO_CASTLING = 0
UNDO_EP_SQUARE = 1
UNDO_HALFMOVE = 2
UNDO_FULLMOVE = 3
UNDO_CAPTURED_PIECE = 4
UNDO_CAPTURED_SQUARE = 5
UNDO_MOVING_PIECE = 6
UNDO_SIZE = 7

ZOBRIST_SEED = 0xA1C4_E55A_7EED_2026
ZOBRIST_VALUE_COUNT = PIECE_BITBOARD_COUNT * 64 + 1 + 16 + 8

PROMOTION_KINDS = np.array((QUEEN, ROOK, BISHOP, KNIGHT), dtype=np.int8)
ROOK_FILE_DELTAS = np.array((1, -1, 0, 0), dtype=np.int8)
ROOK_RANK_DELTAS = np.array((0, 0, 1, -1), dtype=np.int8)
BISHOP_FILE_DELTAS = np.array((1, 1, -1, -1), dtype=np.int8)
BISHOP_RANK_DELTAS = np.array((1, -1, 1, -1), dtype=np.int8)


def _build_jump_attacks(deltas: tuple[tuple[int, int], ...]) -> NDArray[np.uint64]:
    attacks = np.zeros(64, dtype=np.uint64)
    for square in range(64):
        file_index = chess.square_file(square)
        rank = chess.square_rank(square)
        mask = 0
        for file_delta, rank_delta in deltas:
            target_file = file_index + file_delta
            target_rank = rank + rank_delta
            if 0 <= target_file < 8 and 0 <= target_rank < 8:
                mask |= 1 << chess.square(target_file, target_rank)
        attacks[square] = np.uint64(mask)
    return attacks


KNIGHT_ATTACKS = _build_jump_attacks(
    ((1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2))
)
KING_ATTACKS = _build_jump_attacks(
    ((1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1))
)
PAWN_ATTACKS = np.zeros((2, 64), dtype=np.uint64)
for _square in range(64):
    _file = chess.square_file(_square)
    _rank = chess.square_rank(_square)
    for _color, _rank_delta in ((WHITE, 1), (BLACK, -1)):
        _mask = 0
        for _file_delta in (-1, 1):
            _target_file = _file + _file_delta
            _target_rank = _rank + _rank_delta
            if 0 <= _target_file < 8 and 0 <= _target_rank < 8:
                _mask |= 1 << chess.square(_target_file, _target_rank)
        PAWN_ATTACKS[_color, _square] = np.uint64(_mask)


def _build_king_rays() -> NDArray[np.uint64]:
    rays = np.zeros(64, dtype=np.uint64)
    directions = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))
    for square in range(64):
        mask = 0
        for file_delta, rank_delta in directions:
            target_file = chess.square_file(square) + file_delta
            target_rank = chess.square_rank(square) + rank_delta
            while 0 <= target_file < 8 and 0 <= target_rank < 8:
                mask |= 1 << chess.square(target_file, target_rank)
                target_file += file_delta
                target_rank += rank_delta
        rays[square] = np.uint64(mask)
    return rays


KING_RAYS = _build_king_rays()


def _build_zobrist_values(count: int) -> NDArray[np.uint64]:
    """Build deterministic, non-zero SplitMix64 keys without global RNG state."""
    mask = (1 << 64) - 1
    value = ZOBRIST_SEED
    result = np.empty(count, dtype=np.uint64)
    for index in range(count):
        value = (value + 0x9E3779B97F4A7C15) & mask
        mixed = value
        mixed = ((mixed ^ (mixed >> 30)) * 0xBF58476D1CE4E5B9) & mask
        mixed = ((mixed ^ (mixed >> 27)) * 0x94D049BB133111EB) & mask
        mixed ^= mixed >> 31
        result[index] = np.uint64(mixed or 1)
    return result


_ZOBRIST_VALUES = _build_zobrist_values(ZOBRIST_VALUE_COUNT)
_zobrist_offset = 0
ZOBRIST_PIECES = _ZOBRIST_VALUES[_zobrist_offset : _zobrist_offset + 12 * 64].reshape(
    (12, 64)
)
_zobrist_offset += 12 * 64
ZOBRIST_SIDE = _ZOBRIST_VALUES[_zobrist_offset]
_zobrist_offset += 1
ZOBRIST_CASTLING = _ZOBRIST_VALUES[_zobrist_offset : _zobrist_offset + 16]
_zobrist_offset += 16
ZOBRIST_EP_FILE = _ZOBRIST_VALUES[_zobrist_offset : _zobrist_offset + 8]
DARK_SQUARES = np.uint64(chess.BB_DARK_SQUARES)
LIGHT_SQUARES = np.uint64(chess.BB_LIGHT_SQUARES)


@dataclass(slots=True)
class Position:
    """Mutable engine state passed into compiled functions."""

    pieces: NDArray[np.uint64]
    state: NDArray[np.int64]
    key: NDArray[np.uint64]

    def copy(self) -> Position:
        return Position(self.pieces.copy(), self.state.copy(), self.key.copy())


@njit(cache=False, inline="always")
def bit(square: int) -> np.uint64:
    return np.uint64(1) << np.uint64(square)


@njit(cache=False, inline="always")
def piece_index(color: int, kind: int) -> int:
    return color * PIECE_KIND_COUNT + kind


@njit(cache=False, inline="always")
def move_from(move: int) -> int:
    return move & FROM_MASK


@njit(cache=False, inline="always")
def move_to(move: int) -> int:
    return (move >> TO_SHIFT) & TO_MASK


@njit(cache=False, inline="always")
def move_promotion(move: int) -> int:
    return (move >> PROMOTION_SHIFT) & PROMOTION_MASK


@njit(cache=False, inline="always")
def move_flags(move: int) -> int:
    return move & ~((1 << 15) - 1)


@njit(cache=False, inline="always")
def pack_move(from_square: int, to_square: int, promotion: int, flags: int) -> np.int32:
    return np.int32(
        from_square | (to_square << TO_SHIFT) | (promotion << PROMOTION_SHIFT) | flags
    )


@njit(cache=False, inline="always")
def lsb_square(board: np.uint64) -> int:
    square = 0
    while board & np.uint64(1) == 0:
        board >>= np.uint64(1)
        square += 1
    return square


@njit(cache=False, inline="always")
def occupancy_for(pieces: NDArray[np.uint64], color: int) -> np.uint64:
    occupied = np.uint64(0)
    start = color * PIECE_KIND_COUNT
    for index in range(start, start + PIECE_KIND_COUNT):
        occupied |= pieces[index]
    return occupied


@njit(cache=False, inline="always")
def all_occupancy(pieces: NDArray[np.uint64]) -> np.uint64:
    occupied = np.uint64(0)
    for index in range(PIECE_BITBOARD_COUNT):
        occupied |= pieces[index]
    return occupied


@njit(cache=False, inline="always")
def piece_at(
    pieces: NDArray[np.uint64], square: int, first_index: int, last_index: int
) -> int:
    square_bit = bit(square)
    for index in range(first_index, last_index):
        if pieces[index] & square_bit:
            return index
    return NO_PIECE


@njit(cache=False)
def is_square_attacked(pieces: NDArray[np.uint64], square: int, by_color: int) -> bool:
    """Return whether ``square`` is attacked, including pinned attackers."""
    pawn_origins = PAWN_ATTACKS[BLACK if by_color == WHITE else WHITE, square]
    if pieces[piece_index(by_color, PAWN)] & pawn_origins:
        return True
    if pieces[piece_index(by_color, KNIGHT)] & KNIGHT_ATTACKS[square]:
        return True
    if pieces[piece_index(by_color, KING)] & KING_ATTACKS[square]:
        return True

    occupied = all_occupancy(pieces)
    square_file = square & 7
    square_rank = square >> 3

    for direction in range(4):
        file_index = square_file + int(ROOK_FILE_DELTAS[direction])
        rank = square_rank + int(ROOK_RANK_DELTAS[direction])
        while 0 <= file_index < 8 and 0 <= rank < 8:
            target = rank * 8 + file_index
            target_bit = bit(target)
            if occupied & target_bit:
                if (
                    pieces[piece_index(by_color, ROOK)]
                    | pieces[piece_index(by_color, QUEEN)]
                ) & target_bit:
                    return True
                break
            file_index += int(ROOK_FILE_DELTAS[direction])
            rank += int(ROOK_RANK_DELTAS[direction])

    for direction in range(4):
        file_index = square_file + int(BISHOP_FILE_DELTAS[direction])
        rank = square_rank + int(BISHOP_RANK_DELTAS[direction])
        while 0 <= file_index < 8 and 0 <= rank < 8:
            target = rank * 8 + file_index
            target_bit = bit(target)
            if occupied & target_bit:
                if (
                    pieces[piece_index(by_color, BISHOP)]
                    | pieces[piece_index(by_color, QUEEN)]
                ) & target_bit:
                    return True
                break
            file_index += int(BISHOP_FILE_DELTAS[direction])
            rank += int(BISHOP_RANK_DELTAS[direction])
    return False


@njit(cache=False)
def is_in_check(pieces: NDArray[np.uint64], color: int) -> bool:
    king = pieces[piece_index(color, KING)]
    if king == 0:
        return True
    return is_square_attacked(pieces, lsb_square(king), BLACK if color == WHITE else WHITE)


@njit(cache=False)
def has_legal_en_passant(
    pieces: NDArray[np.uint64], state: NDArray[np.int64]
) -> bool:
    """Return whether the FEN en-passant square changes the legal move set."""
    ep_square = int(state[STATE_EP_SQUARE])
    if ep_square < 0 or ep_square >= 64 or all_occupancy(pieces) & bit(ep_square):
        return False

    side = int(state[STATE_SIDE])
    enemy = BLACK if side == WHITE else WHITE
    captured_square = ep_square - 8 if side == WHITE else ep_square + 8
    if not 0 <= captured_square < 64:
        return False
    enemy_pawn = piece_index(enemy, PAWN)
    if pieces[enemy_pawn] & bit(captured_square) == 0:
        return False

    own_pawn = piece_index(side, PAWN)
    origins = pieces[own_pawn] & PAWN_ATTACKS[enemy, ep_square]
    while origins:
        from_square = lsb_square(origins)
        origins ^= bit(from_square)
        pieces[own_pawn] ^= bit(from_square) | bit(ep_square)
        pieces[enemy_pawn] ^= bit(captured_square)
        legal = not is_in_check(pieces, side)
        pieces[enemy_pawn] ^= bit(captured_square)
        pieces[own_pawn] ^= bit(from_square) | bit(ep_square)
        if legal:
            return True
    return False


@njit(cache=False, inline="always")
def _ep_hash_component(
    pieces: NDArray[np.uint64], state: NDArray[np.int64]
) -> np.uint64:
    ep_square = int(state[STATE_EP_SQUARE])
    if ep_square >= 0 and has_legal_en_passant(pieces, state):
        return np.uint64(ZOBRIST_EP_FILE[ep_square & 7])
    return np.uint64(0)


@njit(cache=False)
def recompute_zobrist(
    pieces: NDArray[np.uint64], state: NDArray[np.int64]
) -> np.uint64:
    """Compute the canonical position key, excluding move counters."""
    key = ZOBRIST_CASTLING[int(state[STATE_CASTLING]) & 15]
    if int(state[STATE_SIDE]) == BLACK:
        key ^= ZOBRIST_SIDE
    for piece in range(PIECE_BITBOARD_COUNT):
        occupied = pieces[piece]
        while occupied:
            square = lsb_square(occupied)
            occupied ^= bit(square)
            key ^= ZOBRIST_PIECES[piece, square]
    return np.uint64(key ^ _ep_hash_component(pieces, state))


@njit(cache=False, inline="always")
def popcount(board: np.uint64) -> int:
    count = 0
    while board:
        board &= board - np.uint64(1)
        count += 1
    return count


@njit(cache=False)
def _has_insufficient_material(pieces: NDArray[np.uint64], color: int) -> bool:
    enemy = BLACK if color == WHITE else WHITE
    own_occupied = occupancy_for(pieces, color)
    if own_occupied & (
        pieces[piece_index(color, PAWN)]
        | pieces[piece_index(color, ROOK)]
        | pieces[piece_index(color, QUEEN)]
    ):
        return False

    if pieces[piece_index(color, KNIGHT)]:
        enemy_non_king_queen = occupancy_for(pieces, enemy) & ~(
            pieces[piece_index(enemy, KING)] | pieces[piece_index(enemy, QUEEN)]
        )
        return popcount(own_occupied) <= 2 and enemy_non_king_queen == 0

    if pieces[piece_index(color, BISHOP)]:
        bishops = pieces[piece_index(WHITE, BISHOP)] | pieces[piece_index(BLACK, BISHOP)]
        bishops_share_color = bishops & DARK_SQUARES == 0 or bishops & LIGHT_SQUARES == 0
        pawns = pieces[piece_index(WHITE, PAWN)] | pieces[piece_index(BLACK, PAWN)]
        knights = pieces[piece_index(WHITE, KNIGHT)] | pieces[piece_index(BLACK, KNIGHT)]
        return bool(bishops_share_color and pawns == 0 and knights == 0)
    return True


@njit(cache=False)
def is_insufficient_material(pieces: NDArray[np.uint64]) -> bool:
    """Match python-chess's conservative automatic-material draw test."""
    return _has_insufficient_material(pieces, WHITE) and _has_insufficient_material(
        pieces, BLACK
    )


@njit(cache=False)
def is_repetition_draw(
    key: np.uint64,
    history: NDArray[np.uint64],
    history_count: int,
    root_history_count: int,
    halfmove_clock: int,
) -> bool:
    """Detect a threefold or a cycle already entered by the current search line.

    ``history`` includes the current position. Positions before
    ``root_history_count`` belong to the played game; later positions belong to
    the speculative search line. A repeat within the current line is scored as
    a draw to stop cycles, while game history alone requires three occurrences.
    """
    if history_count <= 1:
        return False
    last = history_count - 1
    minimum = max(0, last - halfmove_clock)
    repetitions = 1
    index = last - 2
    while index >= minimum:
        if history[index] == key:
            repetitions += 1
            if repetitions >= 3:
                return True
            if last >= root_history_count and index >= root_history_count - 1:
                return True
        index -= 2
    return False


@njit(cache=False)
def has_rule_draw(
    pieces: NDArray[np.uint64],
    state: NDArray[np.int64],
    key: np.uint64,
    history: NDArray[np.uint64],
    history_count: int,
    root_history_count: int,
) -> bool:
    """Return draw state that a non-terminal search node should score as zero."""
    return (
        int(state[STATE_HALFMOVE]) >= 100
        or is_insufficient_material(pieces)
        or is_repetition_draw(
            key,
            history,
            history_count,
            root_history_count,
            int(state[STATE_HALFMOVE]),
        )
    )


@njit(cache=False, inline="always")
def _append_slider_moves(
    moves: NDArray[np.int32],
    count: int,
    from_square: int,
    own_occupied: np.uint64,
    enemy_occupied: np.uint64,
    file_deltas: NDArray[np.int8],
    rank_deltas: NDArray[np.int8],
) -> int:
    from_file = from_square & 7
    from_rank = from_square >> 3
    for direction in range(len(file_deltas)):
        file_index = from_file + int(file_deltas[direction])
        rank = from_rank + int(rank_deltas[direction])
        while 0 <= file_index < 8 and 0 <= rank < 8:
            target = rank * 8 + file_index
            target_bit = bit(target)
            if own_occupied & target_bit:
                break
            flags = FLAG_CAPTURE if enemy_occupied & target_bit else 0
            moves[count] = pack_move(from_square, target, 0, flags)
            count += 1
            if enemy_occupied & target_bit:
                break
            file_index += int(file_deltas[direction])
            rank += int(rank_deltas[direction])
    return count


@njit(cache=False, inline="always")
def _can_castle(
    pieces: NDArray[np.uint64], state: NDArray[np.int64], color: int, kingside: bool
) -> bool:
    rights = int(state[STATE_CASTLING])
    occupied = all_occupancy(pieces)
    enemy = BLACK if color == WHITE else WHITE
    if color == WHITE:
        if pieces[piece_index(WHITE, KING)] & bit(E1) == 0:
            return False
        if kingside:
            if rights & CASTLE_WHITE_KING == 0:
                return False
            if pieces[piece_index(WHITE, ROOK)] & bit(H1) == 0:
                return False
            if occupied & (bit(F1) | bit(G1)):
                return False
            return not (
                is_square_attacked(pieces, E1, enemy)
                or is_square_attacked(pieces, F1, enemy)
                or is_square_attacked(pieces, G1, enemy)
            )
        if rights & CASTLE_WHITE_QUEEN == 0:
            return False
        if pieces[piece_index(WHITE, ROOK)] & bit(A1) == 0:
            return False
        if occupied & (bit(B1) | bit(C1) | bit(D1)):
            return False
        return not (
            is_square_attacked(pieces, E1, enemy)
            or is_square_attacked(pieces, D1, enemy)
            or is_square_attacked(pieces, C1, enemy)
        )

    if pieces[piece_index(BLACK, KING)] & bit(E8) == 0:
        return False
    if kingside:
        if rights & CASTLE_BLACK_KING == 0:
            return False
        if pieces[piece_index(BLACK, ROOK)] & bit(H8) == 0:
            return False
        if occupied & (bit(F8) | bit(G8)):
            return False
        return not (
            is_square_attacked(pieces, E8, enemy)
            or is_square_attacked(pieces, F8, enemy)
            or is_square_attacked(pieces, G8, enemy)
        )
    if rights & CASTLE_BLACK_QUEEN == 0:
        return False
    if pieces[piece_index(BLACK, ROOK)] & bit(A8) == 0:
        return False
    if occupied & (bit(B8) | bit(C8) | bit(D8)):
        return False
    return not (
        is_square_attacked(pieces, E8, enemy)
        or is_square_attacked(pieces, D8, enemy)
        or is_square_attacked(pieces, C8, enemy)
    )


@njit(cache=False)
def generate_pseudo_legal_moves(
    pieces: NDArray[np.uint64], state: NDArray[np.int64], moves: NDArray[np.int32]
) -> int:
    """Generate all pseudo-legal moves, with castling transit checks enforced."""
    side = int(state[STATE_SIDE])
    enemy = BLACK if side == WHITE else WHITE
    own_occupied = occupancy_for(pieces, side)
    enemy_occupied = occupancy_for(pieces, enemy)
    occupied = own_occupied | enemy_occupied
    count = 0

    pawns = pieces[piece_index(side, PAWN)]
    while pawns:
        from_square = lsb_square(pawns)
        pawns ^= bit(from_square)
        rank = from_square >> 3
        forward = from_square + (8 if side == WHITE else -8)
        promotion_rank = 6 if side == WHITE else 1
        start_rank = 1 if side == WHITE else 6
        if 0 <= forward < 64 and occupied & bit(forward) == 0:
            if rank == promotion_rank:
                for promotion in PROMOTION_KINDS:
                    moves[count] = pack_move(
                        from_square, forward, int(promotion), FLAG_PROMOTION
                    )
                    count += 1
            else:
                moves[count] = pack_move(from_square, forward, 0, 0)
                count += 1
                double_target = from_square + (16 if side == WHITE else -16)
                if rank == start_rank and occupied & bit(double_target) == 0:
                    moves[count] = pack_move(
                        from_square, double_target, 0, FLAG_DOUBLE_PAWN
                    )
                    count += 1

        captures = PAWN_ATTACKS[side, from_square]
        while captures:
            target = lsb_square(captures)
            captures ^= bit(target)
            is_en_passant = target == int(state[STATE_EP_SQUARE])
            if enemy_occupied & bit(target) == 0 and not is_en_passant:
                continue
            flags = FLAG_CAPTURE | (FLAG_EN_PASSANT if is_en_passant else 0)
            if rank == promotion_rank:
                for promotion in PROMOTION_KINDS:
                    moves[count] = pack_move(
                        from_square, target, int(promotion), flags | FLAG_PROMOTION
                    )
                    count += 1
            else:
                moves[count] = pack_move(from_square, target, 0, flags)
                count += 1

    knights = pieces[piece_index(side, KNIGHT)]
    while knights:
        from_square = lsb_square(knights)
        knights ^= bit(from_square)
        targets = KNIGHT_ATTACKS[from_square] & ~own_occupied
        while targets:
            target = lsb_square(targets)
            targets ^= bit(target)
            flags = FLAG_CAPTURE if enemy_occupied & bit(target) else 0
            moves[count] = pack_move(from_square, target, 0, flags)
            count += 1

    bishops = pieces[piece_index(side, BISHOP)]
    while bishops:
        from_square = lsb_square(bishops)
        bishops ^= bit(from_square)
        count = _append_slider_moves(
            moves,
            count,
            from_square,
            own_occupied,
            enemy_occupied,
            BISHOP_FILE_DELTAS,
            BISHOP_RANK_DELTAS,
        )

    rooks = pieces[piece_index(side, ROOK)]
    while rooks:
        from_square = lsb_square(rooks)
        rooks ^= bit(from_square)
        count = _append_slider_moves(
            moves,
            count,
            from_square,
            own_occupied,
            enemy_occupied,
            ROOK_FILE_DELTAS,
            ROOK_RANK_DELTAS,
        )

    queens = pieces[piece_index(side, QUEEN)]
    while queens:
        from_square = lsb_square(queens)
        queens ^= bit(from_square)
        count = _append_slider_moves(
            moves,
            count,
            from_square,
            own_occupied,
            enemy_occupied,
            ROOK_FILE_DELTAS,
            ROOK_RANK_DELTAS,
        )
        count = _append_slider_moves(
            moves,
            count,
            from_square,
            own_occupied,
            enemy_occupied,
            BISHOP_FILE_DELTAS,
            BISHOP_RANK_DELTAS,
        )

    kings = pieces[piece_index(side, KING)]
    if kings:
        from_square = lsb_square(kings)
        targets = KING_ATTACKS[from_square] & ~own_occupied
        while targets:
            target = lsb_square(targets)
            targets ^= bit(target)
            flags = FLAG_CAPTURE if enemy_occupied & bit(target) else 0
            moves[count] = pack_move(from_square, target, 0, flags)
            count += 1
        if _can_castle(pieces, state, side, True):
            target = G1 if side == WHITE else G8
            moves[count] = pack_move(from_square, target, 0, FLAG_CASTLING)
            count += 1
        if _can_castle(pieces, state, side, False):
            target = C1 if side == WHITE else C8
            moves[count] = pack_move(from_square, target, 0, FLAG_CASTLING)
            count += 1
    return count


@njit(cache=False)
def make_move(
    pieces: NDArray[np.uint64],
    state: NDArray[np.int64],
    key: NDArray[np.uint64],
    move: int,
    undo: NDArray[np.int64],
    undo_key: NDArray[np.uint64],
) -> bool:
    """Apply a pseudo-legal move and record exactly what unmake needs."""
    side = int(state[STATE_SIDE])
    enemy = BLACK if side == WHITE else WHITE
    from_square = move_from(move)
    to_square = move_to(move)
    flags = move_flags(move)

    moving_piece = piece_at(
        pieces,
        from_square,
        side * PIECE_KIND_COUNT,
        (side + 1) * PIECE_KIND_COUNT,
    )
    if moving_piece == NO_PIECE:
        return False

    undo_key[0] = key[0]
    next_key = (
        key[0]
        ^ _ep_hash_component(pieces, state)
        ^ ZOBRIST_CASTLING[int(state[STATE_CASTLING]) & 15]
        ^ ZOBRIST_SIDE
    )
    undo[UNDO_CASTLING] = state[STATE_CASTLING]
    undo[UNDO_EP_SQUARE] = state[STATE_EP_SQUARE]
    undo[UNDO_HALFMOVE] = state[STATE_HALFMOVE]
    undo[UNDO_FULLMOVE] = state[STATE_FULLMOVE]
    undo[UNDO_CAPTURED_PIECE] = NO_PIECE
    undo[UNDO_CAPTURED_SQUARE] = NO_PIECE
    undo[UNDO_MOVING_PIECE] = moving_piece

    captured_square = to_square
    if flags & FLAG_EN_PASSANT:
        captured_square = to_square - 8 if side == WHITE else to_square + 8
    captured_piece = piece_at(
        pieces,
        captured_square,
        enemy * PIECE_KIND_COUNT,
        (enemy + 1) * PIECE_KIND_COUNT,
    )
    if captured_piece != NO_PIECE:
        pieces[captured_piece] &= ~bit(captured_square)
        next_key ^= ZOBRIST_PIECES[captured_piece, captured_square]
        undo[UNDO_CAPTURED_PIECE] = captured_piece
        undo[UNDO_CAPTURED_SQUARE] = captured_square

    pieces[moving_piece] &= ~bit(from_square)
    next_key ^= ZOBRIST_PIECES[moving_piece, from_square]
    promotion = move_promotion(move)
    placed_piece = piece_index(side, promotion) if flags & FLAG_PROMOTION else moving_piece
    pieces[placed_piece] |= bit(to_square)
    next_key ^= ZOBRIST_PIECES[placed_piece, to_square]

    if flags & FLAG_CASTLING:
        if to_square == G1:
            pieces[piece_index(WHITE, ROOK)] ^= bit(H1) | bit(F1)
            next_key ^= ZOBRIST_PIECES[piece_index(WHITE, ROOK), H1]
            next_key ^= ZOBRIST_PIECES[piece_index(WHITE, ROOK), F1]
        elif to_square == C1:
            pieces[piece_index(WHITE, ROOK)] ^= bit(A1) | bit(D1)
            next_key ^= ZOBRIST_PIECES[piece_index(WHITE, ROOK), A1]
            next_key ^= ZOBRIST_PIECES[piece_index(WHITE, ROOK), D1]
        elif to_square == G8:
            pieces[piece_index(BLACK, ROOK)] ^= bit(H8) | bit(F8)
            next_key ^= ZOBRIST_PIECES[piece_index(BLACK, ROOK), H8]
            next_key ^= ZOBRIST_PIECES[piece_index(BLACK, ROOK), F8]
        else:
            pieces[piece_index(BLACK, ROOK)] ^= bit(A8) | bit(D8)
            next_key ^= ZOBRIST_PIECES[piece_index(BLACK, ROOK), A8]
            next_key ^= ZOBRIST_PIECES[piece_index(BLACK, ROOK), D8]

    rights = int(state[STATE_CASTLING])
    moving_kind = moving_piece % PIECE_KIND_COUNT
    if moving_kind == KING:
        rights &= ~(CASTLE_WHITE_KING | CASTLE_WHITE_QUEEN) if side == WHITE else ~(
            CASTLE_BLACK_KING | CASTLE_BLACK_QUEEN
        )
    if from_square == H1 or to_square == H1:
        rights &= ~CASTLE_WHITE_KING
    if from_square == A1 or to_square == A1:
        rights &= ~CASTLE_WHITE_QUEEN
    if from_square == H8 or to_square == H8:
        rights &= ~CASTLE_BLACK_KING
    if from_square == A8 or to_square == A8:
        rights &= ~CASTLE_BLACK_QUEEN
    state[STATE_CASTLING] = rights

    state[STATE_EP_SQUARE] = -1
    if moving_kind == PAWN and flags & FLAG_DOUBLE_PAWN:
        state[STATE_EP_SQUARE] = (from_square + to_square) // 2
    if moving_kind == PAWN or captured_piece != NO_PIECE:
        state[STATE_HALFMOVE] = 0
    else:
        state[STATE_HALFMOVE] += 1
    if side == BLACK:
        state[STATE_FULLMOVE] += 1
    state[STATE_SIDE] = enemy
    next_key ^= ZOBRIST_CASTLING[rights & 15]
    next_key ^= _ep_hash_component(pieces, state)
    key[0] = next_key
    return True


@njit(cache=False)
def unmake_move(
    pieces: NDArray[np.uint64],
    state: NDArray[np.int64],
    key: NDArray[np.uint64],
    move: int,
    undo: NDArray[np.int64],
    undo_key: NDArray[np.uint64],
) -> None:
    """Reverse a move using its compact undo record."""
    side = BLACK if int(state[STATE_SIDE]) == WHITE else WHITE
    from_square = move_from(move)
    to_square = move_to(move)
    flags = move_flags(move)
    moving_piece = int(undo[UNDO_MOVING_PIECE])
    promotion = move_promotion(move)
    placed_piece = piece_index(side, promotion) if flags & FLAG_PROMOTION else moving_piece

    pieces[placed_piece] &= ~bit(to_square)
    pieces[moving_piece] |= bit(from_square)

    if flags & FLAG_CASTLING:
        if to_square == G1:
            pieces[piece_index(WHITE, ROOK)] ^= bit(F1) | bit(H1)
        elif to_square == C1:
            pieces[piece_index(WHITE, ROOK)] ^= bit(D1) | bit(A1)
        elif to_square == G8:
            pieces[piece_index(BLACK, ROOK)] ^= bit(F8) | bit(H8)
        else:
            pieces[piece_index(BLACK, ROOK)] ^= bit(D8) | bit(A8)

    captured_piece = int(undo[UNDO_CAPTURED_PIECE])
    if captured_piece != NO_PIECE:
        pieces[captured_piece] |= bit(int(undo[UNDO_CAPTURED_SQUARE]))

    state[STATE_SIDE] = side
    state[STATE_CASTLING] = undo[UNDO_CASTLING]
    state[STATE_EP_SQUARE] = undo[UNDO_EP_SQUARE]
    state[STATE_HALFMOVE] = undo[UNDO_HALFMOVE]
    state[STATE_FULLMOVE] = undo[UNDO_FULLMOVE]
    key[0] = undo_key[0]


@njit(cache=False, inline="always")
def _king_legality_context(
    pieces: NDArray[np.uint64], state: NDArray[np.int64]
) -> tuple[int, np.uint64, bool]:
    side = int(state[STATE_SIDE])
    king = pieces[piece_index(side, KING)]
    if king == 0:
        return -1, np.uint64(0), True
    king_square = lsb_square(king)
    enemy = BLACK if side == WHITE else WHITE
    return king_square, KING_RAYS[king_square], is_square_attacked(pieces, king_square, enemy)


@njit(cache=False, inline="always")
def _needs_legality_test(
    move: int, king_square: int, king_rays: np.uint64, in_check: bool
) -> bool:
    # only vacating a line through the king can expose it, en passant vacates two
    if in_check:
        return True
    from_square = move_from(move)
    if from_square == king_square:
        return True
    if move_flags(move) & FLAG_EN_PASSANT:
        return True
    return king_rays & bit(from_square) != np.uint64(0)


@njit(cache=False)
def generate_legal_moves(
    pieces: NDArray[np.uint64],
    state: NDArray[np.int64],
    key: NDArray[np.uint64],
    legal_moves: NDArray[np.int32],
    pseudo_moves: NDArray[np.int32],
    scratch_undo: NDArray[np.int64],
    scratch_undo_key: NDArray[np.uint64],
) -> int:
    """Generate legal moves by making and checking every pseudo-legal move."""
    moving_side = int(state[STATE_SIDE])
    pseudo_count = generate_pseudo_legal_moves(pieces, state, pseudo_moves)
    king_square, king_rays, in_check = _king_legality_context(pieces, state)
    legal_count = 0
    for index in range(pseudo_count):
        move = int(pseudo_moves[index])
        if not _needs_legality_test(move, king_square, king_rays, in_check):
            legal_moves[legal_count] = np.int32(move)
            legal_count += 1
            continue
        if not make_move(pieces, state, key, move, scratch_undo, scratch_undo_key):
            continue
        legal = not is_in_check(pieces, moving_side)
        unmake_move(pieces, state, key, move, scratch_undo, scratch_undo_key)
        if legal:
            legal_moves[legal_count] = np.int32(move)
            legal_count += 1
    return legal_count


@njit(cache=False)
def generate_legal_captures(
    pieces: NDArray[np.uint64],
    state: NDArray[np.int64],
    key: NDArray[np.uint64],
    legal_moves: NDArray[np.int32],
    pseudo_moves: NDArray[np.int32],
    scratch_undo: NDArray[np.int64],
    scratch_undo_key: NDArray[np.uint64],
) -> int:
    # -1 not 0 (keeps stalemate distinct from pos with no captures)
    moving_side = int(state[STATE_SIDE])
    pseudo_count = generate_pseudo_legal_moves(pieces, state, pseudo_moves)
    king_square, king_rays, in_check = _king_legality_context(pieces, state)
    legal_count = 0
    saw_legal = False
    for index in range(pseudo_count):
        move = int(pseudo_moves[index])
        tactical = move_flags(move) & (FLAG_CAPTURE | FLAG_PROMOTION) != 0
        if saw_legal and not tactical:
            continue
        if not _needs_legality_test(move, king_square, king_rays, in_check):
            saw_legal = True
            if tactical:
                legal_moves[legal_count] = np.int32(move)
                legal_count += 1
            continue
        if not make_move(pieces, state, key, move, scratch_undo, scratch_undo_key):
            continue
        legal = not is_in_check(pieces, moving_side)
        unmake_move(pieces, state, key, move, scratch_undo, scratch_undo_key)
        if not legal:
            continue
        saw_legal = True
        if tactical:
            legal_moves[legal_count] = np.int32(move)
            legal_count += 1
    return legal_count if saw_legal else -1


@njit(cache=False)
def _perft(
    pieces: NDArray[np.uint64],
    state: NDArray[np.int64],
    key: NDArray[np.uint64],
    depth: int,
    legal_stack: NDArray[np.int32],
    pseudo_stack: NDArray[np.int32],
    undo_stack: NDArray[np.int64],
    undo_key_stack: NDArray[np.uint64],
    ply: int,
) -> np.int64:
    if depth == 0:
        return np.int64(1)
    count = generate_legal_moves(
        pieces,
        state,
        key,
        legal_stack[ply],
        pseudo_stack[ply],
        undo_stack[ply],
        undo_key_stack[ply],
    )
    if depth == 1:
        return np.int64(count)
    nodes = np.int64(0)
    for index in range(count):
        move = int(legal_stack[ply, index])
        make_move(pieces, state, key, move, undo_stack[ply], undo_key_stack[ply])
        nodes += _perft(
            pieces,
            state,
            key,
            depth - 1,
            legal_stack,
            pseudo_stack,
            undo_stack,
            undo_key_stack,
            ply + 1,
        )
        unmake_move(pieces, state, key, move, undo_stack[ply], undo_key_stack[ply])
    return nodes


def position_from_board(board: chess.Board) -> Position:
    pieces = np.zeros(PIECE_BITBOARD_COUNT, dtype=np.uint64)
    for color, chess_color in ((WHITE, chess.WHITE), (BLACK, chess.BLACK)):
        for kind in range(PIECE_KIND_COUNT):
            pieces[piece_index.py_func(color, kind)] = np.uint64(
                board.pieces_mask(kind + 1, chess_color)
            )
    rights = 0
    if board.has_kingside_castling_rights(chess.WHITE):
        rights |= CASTLE_WHITE_KING
    if board.has_queenside_castling_rights(chess.WHITE):
        rights |= CASTLE_WHITE_QUEEN
    if board.has_kingside_castling_rights(chess.BLACK):
        rights |= CASTLE_BLACK_KING
    if board.has_queenside_castling_rights(chess.BLACK):
        rights |= CASTLE_BLACK_QUEEN
    state = np.array(
        (
            WHITE if board.turn == chess.WHITE else BLACK,
            rights,
            board.ep_square if board.ep_square is not None else -1,
            board.halfmove_clock,
            board.fullmove_number,
        ),
        dtype=np.int64,
    )
    key = np.empty(1, dtype=np.uint64)
    key[0] = recompute_zobrist(pieces, state)
    return Position(pieces, state, key)


def position_from_fen(fen: str) -> Position:
    return position_from_board(chess.Board(fen))


def board_from_position(position: Position) -> chess.Board:
    board = chess.Board(None)
    for color, chess_color in ((WHITE, chess.WHITE), (BLACK, chess.BLACK)):
        for kind in range(PIECE_KIND_COUNT):
            mask = int(position.pieces[color * PIECE_KIND_COUNT + kind])
            for square in chess.scan_forward(mask):
                board.set_piece_at(square, chess.Piece(kind + 1, chess_color))
    board.turn = int(position.state[STATE_SIDE]) == WHITE
    rights = int(position.state[STATE_CASTLING])
    board.castling_rights = 0
    if rights & CASTLE_WHITE_KING:
        board.castling_rights |= chess.BB_H1
    if rights & CASTLE_WHITE_QUEEN:
        board.castling_rights |= chess.BB_A1
    if rights & CASTLE_BLACK_KING:
        board.castling_rights |= chess.BB_H8
    if rights & CASTLE_BLACK_QUEEN:
        board.castling_rights |= chess.BB_A8
    ep_square = int(position.state[STATE_EP_SQUARE])
    board.ep_square = ep_square if ep_square >= 0 else None
    board.halfmove_clock = int(position.state[STATE_HALFMOVE])
    board.fullmove_number = int(position.state[STATE_FULLMOVE])
    return board


def move_to_uci(move: int) -> str:
    text = chess.square_name(move_from.py_func(move)) + chess.square_name(move_to.py_func(move))
    if move_flags.py_func(move) & FLAG_PROMOTION:
        text += {KNIGHT: "n", BISHOP: "b", ROOK: "r", QUEEN: "q"}[
            move_promotion.py_func(move)
        ]
    return text


def legal_moves(position: Position) -> NDArray[np.int32]:
    legal_buffer = np.empty(MAX_MOVES, dtype=np.int32)
    pseudo_buffer = np.empty(MAX_MOVES, dtype=np.int32)
    undo = np.empty(UNDO_SIZE, dtype=np.int64)
    undo_key = np.empty(1, dtype=np.uint64)
    count = generate_legal_moves(
        position.pieces,
        position.state,
        position.key,
        legal_buffer,
        pseudo_buffer,
        undo,
        undo_key,
    )
    return legal_buffer[:count].copy()


def legal_moves_uci(position: Position) -> set[str]:
    return {move_to_uci(int(move)) for move in legal_moves(position)}


def legal_captures(position: Position) -> tuple[NDArray[np.int32], bool]:
    legal_buffer = np.empty(MAX_MOVES, dtype=np.int32)
    pseudo_buffer = np.empty(MAX_MOVES, dtype=np.int32)
    undo = np.empty(UNDO_SIZE, dtype=np.int64)
    undo_key = np.empty(1, dtype=np.uint64)
    count = generate_legal_captures(
        position.pieces,
        position.state,
        position.key,
        legal_buffer,
        pseudo_buffer,
        undo,
        undo_key,
    )
    if count < 0:
        return legal_buffer[:0].copy(), False
    return legal_buffer[:count].copy(), True


def perft(position: Position, depth: int) -> int:
    if depth < 0:
        raise ValueError("depth must be non-negative")
    levels = max(1, depth)
    legal_stack = np.empty((levels, MAX_MOVES), dtype=np.int32)
    pseudo_stack = np.empty((levels, MAX_MOVES), dtype=np.int32)
    undo_stack = np.empty((levels, UNDO_SIZE), dtype=np.int64)
    undo_key_stack = np.empty((levels, 1), dtype=np.uint64)
    working = position.copy()
    return int(
        _perft(
            working.pieces,
            working.state,
            working.key,
            depth,
            legal_stack,
            pseudo_stack,
            undo_stack,
            undo_key_stack,
            0,
        )
    )


def warmup() -> None:
    """Compile every current hot-path signature outside a future match clock."""
    position = position_from_board(chess.Board())
    legal_moves(position)
    legal_captures(position)
    perft(position, 1)
