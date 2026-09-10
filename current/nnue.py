from __future__ import annotations

from pathlib import Path

import engine
import numpy as np
from numba import njit
from numpy.typing import NDArray

BASE_FEATURE_COUNT = 12 * 64
KING_BUCKET_COUNT = 16
FEATURE_COUNT = KING_BUCKET_COUNT * BASE_FEATURE_COUNT
ACCUMULATOR_SIZE = 128
HIDDEN_SIZE = 32
FORMAT_VERSION = 2
INPUT_SCALE = 2_048
WEIGHT_SCALE = 2_048

# row carries its bucket after the values (STALE if king left it)
BUCKET_SLOT = ACCUMULATOR_SIZE
ACCUMULATOR_ROW = ACCUMULATOR_SIZE + 1
STALE = -1


def _king_bucket(square: int) -> int:
    return ((square >> 3) >> 1) * 4 + ((square & 7) >> 1)


KING_BUCKETS = np.array([_king_bucket(square) for square in range(64)], dtype=np.int32)


def _load_model() -> tuple[
    float,
    NDArray[np.float32],
    NDArray[np.float32],
    NDArray[np.float32],
    NDArray[np.float32],
    NDArray[np.float32],
    float,
]:
    path = Path(__file__).with_name("weights") / "model.npz"
    with np.load(path, allow_pickle=False) as archive:
        version = int(archive["format_version"])
        if version != FORMAT_VERSION:
            raise RuntimeError(f"unsupported neural model format: {version}")

        cp_scale = float(archive["cp_scale"])
        feature_weights = np.ascontiguousarray(archive["feature_weights"], dtype=np.float32)
        accumulator_bias = np.ascontiguousarray(archive["accumulator_bias"], dtype=np.float32)
        hidden_weights = np.ascontiguousarray(archive["hidden_weights"], dtype=np.float32)
        hidden_bias = np.ascontiguousarray(archive["hidden_bias"], dtype=np.float32)
        output_weights = np.ascontiguousarray(archive["output_weights"], dtype=np.float32)
        output_bias_array = np.asarray(archive["output_bias"], dtype=np.float32)

    expected = {
        "feature_weights": (FEATURE_COUNT, ACCUMULATOR_SIZE),
        "accumulator_bias": (ACCUMULATOR_SIZE,),
        "hidden_weights": (HIDDEN_SIZE, 2 * ACCUMULATOR_SIZE),
        "hidden_bias": (HIDDEN_SIZE,),
        "output_weights": (1, HIDDEN_SIZE),
        "output_bias": (1,),
    }
    actual = {
        "feature_weights": feature_weights.shape,
        "accumulator_bias": accumulator_bias.shape,
        "hidden_weights": hidden_weights.shape,
        "hidden_bias": hidden_bias.shape,
        "output_weights": output_weights.shape,
        "output_bias": output_bias_array.shape,
    }
    for name, shape in expected.items():
        if actual[name] != shape:
            raise RuntimeError(f"invalid {name} shape: {actual[name]}, expected {shape}")

    arrays = (
        feature_weights,
        accumulator_bias,
        hidden_weights,
        hidden_bias,
        output_weights,
        output_bias_array,
    )
    if not np.isfinite(cp_scale) or cp_scale <= 0.0:
        raise RuntimeError("model cp_scale must be finite and positive")
    if not all(np.isfinite(array).all() for array in arrays):
        raise RuntimeError("model contains a non-finite weight")

    return (
        cp_scale,
        feature_weights,
        accumulator_bias,
        hidden_weights,
        hidden_bias,
        output_weights,
        float(output_bias_array[0]),
    )


(
    CP_SCALE,
    FEATURE_WEIGHTS,
    ACCUMULATOR_BIAS,
    HIDDEN_WEIGHTS,
    HIDDEN_BIAS,
    OUTPUT_WEIGHTS,
    OUTPUT_BIAS,
) = _load_model()

FEATURE_WEIGHTS_Q = np.rint(FEATURE_WEIGHTS * INPUT_SCALE).astype(np.int16)
ACCUMULATOR_BIAS_Q = np.rint(ACCUMULATOR_BIAS * INPUT_SCALE).astype(np.int32)
HIDDEN_WEIGHTS_Q = np.rint(HIDDEN_WEIGHTS * WEIGHT_SCALE).astype(np.int16)
HIDDEN_BIAS_Q = np.rint(HIDDEN_BIAS * INPUT_SCALE * WEIGHT_SCALE).astype(np.int32)
OUTPUT_WEIGHTS_Q = np.rint(OUTPUT_WEIGHTS * WEIGHT_SCALE).astype(np.int16)
OUTPUT_BIAS_Q = int(np.rint(OUTPUT_BIAS * INPUT_SCALE * WEIGHT_SCALE))


@njit(cache=False, inline="always")
def _oriented_base(piece: int, square: int) -> int:
    swapped_piece = (piece + engine.PIECE_KIND_COUNT) % engine.PIECE_BITBOARD_COUNT
    return swapped_piece * 64 + (square ^ 56)


def _oriented_base_reference(piece: int, square: int) -> int:
    swapped_piece = (piece + engine.PIECE_KIND_COUNT) % engine.PIECE_BITBOARD_COUNT
    return swapped_piece * 64 + (square ^ 56)


@njit(cache=False, inline="always")
def white_bucket_of(pieces: NDArray[np.uint64]) -> int:
    square = engine.lsb_square(pieces[engine.piece_index(engine.WHITE, engine.KING)])
    return int(KING_BUCKETS[square])


@njit(cache=False, inline="always")
def black_bucket_of(pieces: NDArray[np.uint64]) -> int:
    square = engine.lsb_square(pieces[engine.piece_index(engine.BLACK, engine.KING)])
    return int(KING_BUCKETS[square ^ 56])


@njit(cache=False, inline="always")
def _apply_feature(
    accumulators: NDArray[np.int32],
    piece: int,
    square: int,
    sign: int,
    white_bucket: int,
    black_bucket: int,
) -> None:
    white_feature = white_bucket * BASE_FEATURE_COUNT + piece * 64 + square
    black_feature = black_bucket * BASE_FEATURE_COUNT + _oriented_base(piece, square)
    for column in range(ACCUMULATOR_SIZE):
        accumulators[0, column] += sign * FEATURE_WEIGHTS_Q[white_feature, column]
        accumulators[1, column] += sign * FEATURE_WEIGHTS_Q[black_feature, column]


@njit(cache=False)
def rebuild_perspective(
    pieces: NDArray[np.uint64],
    accumulators: NDArray[np.int32],
    perspective: int,
    bucket: int,
) -> None:
    for column in range(ACCUMULATOR_SIZE):
        accumulators[perspective, column] = ACCUMULATOR_BIAS_Q[column]
    offset = bucket * BASE_FEATURE_COUNT
    for piece in range(engine.PIECE_BITBOARD_COUNT):
        occupied = pieces[piece]
        while occupied:
            square = engine.lsb_square(occupied)
            occupied ^= engine.bit(square)
            if perspective == 0:
                feature = offset + piece * 64 + square
            else:
                feature = offset + _oriented_base(piece, square)
            for column in range(ACCUMULATOR_SIZE):
                accumulators[perspective, column] += FEATURE_WEIGHTS_Q[feature, column]
    accumulators[perspective, BUCKET_SLOT] = bucket


@njit(cache=False)
def rebuild(pieces: NDArray[np.uint64], accumulators: NDArray[np.int32]) -> None:
    rebuild_perspective(pieces, accumulators, 0, white_bucket_of(pieces))
    rebuild_perspective(pieces, accumulators, 1, black_bucket_of(pieces))


@njit(cache=False)
def _update_from_metadata(
    move: int,
    moving_piece: int,
    captured_piece: int,
    captured_square: int,
    parent: NDArray[np.int32],
    child: NDArray[np.int32],
) -> None:
    """Apply one move to a known-fresh accumulator using board undo metadata."""
    for perspective in range(2):
        for column in range(ACCUMULATOR_ROW):
            child[perspective, column] = parent[perspective, column]

    white_bucket = int(parent[0, BUCKET_SLOT])
    black_bucket = int(parent[1, BUCKET_SLOT])

    side = moving_piece // engine.PIECE_KIND_COUNT
    from_square = engine.move_from(move)
    to_square = engine.move_to(move)
    flags = engine.move_flags(move)
    _apply_feature(child, moving_piece, from_square, -1, white_bucket, black_bucket)

    placed_piece = moving_piece
    if flags & engine.FLAG_PROMOTION:
        placed_piece = engine.piece_index(side, engine.move_promotion(move))
    _apply_feature(child, placed_piece, to_square, 1, white_bucket, black_bucket)

    if captured_piece != engine.NO_PIECE:
        _apply_feature(
            child, captured_piece, captured_square, -1, white_bucket, black_bucket
        )

    if flags & engine.FLAG_CASTLING:
        rook_piece = engine.piece_index(side, engine.ROOK)
        if to_square == engine.G1:
            rook_from, rook_to = engine.H1, engine.F1
        elif to_square == engine.C1:
            rook_from, rook_to = engine.A1, engine.D1
        elif to_square == engine.G8:
            rook_from, rook_to = engine.H8, engine.F8
        else:
            rook_from, rook_to = engine.A8, engine.D8
        _apply_feature(child, rook_piece, rook_from, -1, white_bucket, black_bucket)
        _apply_feature(child, rook_piece, rook_to, 1, white_bucket, black_bucket)

    # king leaving its region invalidates perspective's whole row
    if moving_piece == engine.piece_index(side, engine.KING):
        if side == engine.WHITE:
            if int(KING_BUCKETS[to_square]) != white_bucket:
                child[0, BUCKET_SLOT] = STALE
        else:
            if int(KING_BUCKETS[to_square ^ 56]) != black_bucket:
                child[1, BUCKET_SLOT] = STALE


@njit(cache=False)
def update_for_move(
    pieces: NDArray[np.uint64],
    state: NDArray[np.int64],
    move: int,
    parent: NDArray[np.int32],
    child: NDArray[np.int32],
) -> None:
    """Build a child accumulator before making ``move`` on the board."""
    # A TT return or qeval-cache hit can skip evaluation after a king crosses
    # a bucket. Incremental deltas are valid only relative to a fresh parent.
    refresh(pieces, parent)

    side = int(state[engine.STATE_SIDE])
    enemy = engine.BLACK if side == engine.WHITE else engine.WHITE
    from_square = engine.move_from(move)
    to_square = engine.move_to(move)
    flags = engine.move_flags(move)
    moving_piece = engine.piece_at(
        pieces,
        from_square,
        side * engine.PIECE_KIND_COUNT,
        (side + 1) * engine.PIECE_KIND_COUNT,
    )
    captured_square = to_square
    if flags & engine.FLAG_EN_PASSANT:
        captured_square = to_square - 8 if side == engine.WHITE else to_square + 8
    captured_piece = engine.piece_at(
        pieces,
        captured_square,
        enemy * engine.PIECE_KIND_COUNT,
        (enemy + 1) * engine.PIECE_KIND_COUNT,
    )
    _update_from_metadata(
        move,
        moving_piece,
        captured_piece,
        captured_square,
        parent,
        child,
    )


@njit(cache=False)
def update_after_move(
    move: int,
    undo: NDArray[np.int64],
    parent: NDArray[np.int32],
    child: NDArray[np.int32],
) -> None:
    """Build a child accumulator after ``move`` using its populated undo row."""
    _update_from_metadata(
        move,
        int(undo[engine.UNDO_MOVING_PIECE]),
        int(undo[engine.UNDO_CAPTURED_PIECE]),
        int(undo[engine.UNDO_CAPTURED_SQUARE]),
        parent,
        child,
    )


@njit(cache=False, inline="always")
def _round_divide(value: int, divisor: int) -> int:
    if value >= 0:
        return (value + divisor // 2) // divisor
    return -((-value + divisor // 2) // divisor)


@njit(cache=False)
def refresh(pieces: NDArray[np.uint64], accumulators: NDArray[np.int32]) -> None:
    if accumulators[0, BUCKET_SLOT] == STALE:
        rebuild_perspective(pieces, accumulators, 0, white_bucket_of(pieces))
    if accumulators[1, BUCKET_SLOT] == STALE:
        rebuild_perspective(pieces, accumulators, 1, black_bucket_of(pieces))


@njit(cache=False)
def evaluate(
    pieces: NDArray[np.uint64], accumulators: NDArray[np.int32], side: int
) -> int:
    refresh(pieces, accumulators)
    own = side
    opponent = engine.BLACK if side == engine.WHITE else engine.WHITE
    output = OUTPUT_BIAS_Q
    for unit in range(HIDDEN_SIZE):
        value = int(HIDDEN_BIAS_Q[unit])
        for column in range(ACCUMULATOR_SIZE):
            own_value = min(INPUT_SCALE, max(0, int(accumulators[own, column])))
            opponent_value = min(INPUT_SCALE, max(0, int(accumulators[opponent, column])))
            value += int(HIDDEN_WEIGHTS_Q[unit, column]) * own_value
            value += (
                int(HIDDEN_WEIGHTS_Q[unit, ACCUMULATOR_SIZE + column]) * opponent_value
            )
        hidden = min(INPUT_SCALE, max(0, _round_divide(value, WEIGHT_SCALE)))
        output += int(OUTPUT_WEIGHTS_Q[0, unit]) * hidden
    scaled_output = output * int(CP_SCALE)
    return _round_divide(scaled_output, INPUT_SCALE * WEIGHT_SCALE)


def evaluate_reference(pieces: NDArray[np.uint64], side: int) -> float:
    white_square = int(np.log2(int(pieces[engine.piece_index(engine.WHITE, engine.KING)])))
    black_square = int(np.log2(int(pieces[engine.piece_index(engine.BLACK, engine.KING)])))
    white_offset = int(KING_BUCKETS[white_square]) * BASE_FEATURE_COUNT
    black_offset = int(KING_BUCKETS[black_square ^ 56]) * BASE_FEATURE_COUNT
    accumulators = np.repeat(ACCUMULATOR_BIAS[None, :], 2, axis=0)
    for piece in range(engine.PIECE_BITBOARD_COUNT):
        occupied = int(pieces[piece])
        while occupied:
            least_bit = occupied & -occupied
            square = least_bit.bit_length() - 1
            occupied ^= least_bit
            accumulators[0] += FEATURE_WEIGHTS[white_offset + piece * 64 + square]
            accumulators[1] += FEATURE_WEIGHTS[
                black_offset + _oriented_base_reference(piece, square)
            ]
    own = side
    opponent = engine.BLACK if side == engine.WHITE else engine.WHITE
    inputs = np.concatenate((accumulators[own], accumulators[opponent]))
    hidden = np.clip(HIDDEN_WEIGHTS @ np.clip(inputs, 0.0, 1.0) + HIDDEN_BIAS, 0.0, 1.0)
    return float((OUTPUT_WEIGHTS @ hidden)[0] + OUTPUT_BIAS) * CP_SCALE


@njit(cache=False)
def benchmark_evaluations(
    pieces: NDArray[np.uint64], accumulators: NDArray[np.int32], iterations: int
) -> int:
    checksum = 0
    for index in range(iterations):
        checksum += evaluate(pieces, accumulators, index & 1)
    return checksum


def warmup() -> None:
    pieces = np.zeros(engine.PIECE_BITBOARD_COUNT, dtype=np.uint64)
    pieces[engine.piece_index(engine.WHITE, engine.KING)] = np.uint64(1)
    pieces[engine.piece_index(engine.BLACK, engine.KING)] = np.uint64(1) << np.uint64(63)
    accumulators = np.empty((2, ACCUMULATOR_ROW), dtype=np.int32)
    rebuild(pieces, accumulators)
    refresh(pieces, accumulators)
    evaluate(pieces, accumulators, engine.WHITE)
