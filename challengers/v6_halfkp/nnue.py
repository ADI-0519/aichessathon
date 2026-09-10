"""Incremental king-conditioned evaluator for the V6 HalfKP experiment.

Every active feature combines the friendly king square with one oriented
piece-square. Ordinary moves update both accumulators incrementally. A king
move rebuilds only that king's perspective; the opposite perspective keeps its
king context and receives the usual piece delta.

The model archive contains quantized arrays only. This keeps the submission
comfortably below its size limit and avoids retaining redundant float weights
in memory during games.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numba import njit
from numpy.typing import NDArray

import engine

PIECE_BUCKETS = 12
KING_SQUARES = 64
SQUARES = 64
FEATURE_COUNT = KING_SQUARES * PIECE_BUCKETS * SQUARES
FORMAT_VERSION = 2


def _load_model() -> tuple[
    float,
    int,
    int,
    NDArray[np.int16],
    NDArray[np.int32],
    NDArray[np.int16],
    NDArray[np.int32],
    NDArray[np.int16],
    int,
]:
    path = Path(__file__).with_name("weights") / "model.npz"
    with np.load(path, allow_pickle=False) as archive:
        version = int(archive["format_version"])
        if version != FORMAT_VERSION:
            raise RuntimeError(f"unsupported king-conditioned model format: {version}")
        cp_scale = float(archive["cp_scale"])
        input_scale = int(archive["input_scale"])
        weight_scale = int(archive["weight_scale"])
        feature_weights = np.ascontiguousarray(archive["feature_weights_q"])
        accumulator_bias = np.ascontiguousarray(archive["accumulator_bias_q"])
        hidden_weights = np.ascontiguousarray(archive["hidden_weights_q"])
        hidden_bias = np.ascontiguousarray(archive["hidden_bias_q"])
        output_weights = np.ascontiguousarray(archive["output_weights_q"])
        output_bias_array = np.ascontiguousarray(archive["output_bias_q"])

    if feature_weights.dtype != np.int16 or feature_weights.ndim != 2:
        raise RuntimeError("feature_weights_q must be a two-dimensional int16 array")
    accumulator_size = feature_weights.shape[1]
    hidden_size = hidden_bias.shape[0]
    expected = {
        "feature_weights_q": (FEATURE_COUNT, accumulator_size),
        "accumulator_bias_q": (accumulator_size,),
        "hidden_weights_q": (hidden_size, 2 * accumulator_size),
        "hidden_bias_q": (hidden_size,),
        "output_weights_q": (1, hidden_size),
        "output_bias_q": (1,),
    }
    actual = {
        "feature_weights_q": feature_weights.shape,
        "accumulator_bias_q": accumulator_bias.shape,
        "hidden_weights_q": hidden_weights.shape,
        "hidden_bias_q": hidden_bias.shape,
        "output_weights_q": output_weights.shape,
        "output_bias_q": output_bias_array.shape,
    }
    dtypes = {
        "accumulator_bias_q": accumulator_bias.dtype,
        "hidden_weights_q": hidden_weights.dtype,
        "hidden_bias_q": hidden_bias.dtype,
        "output_weights_q": output_weights.dtype,
        "output_bias_q": output_bias_array.dtype,
    }
    expected_dtypes = {
        "accumulator_bias_q": np.dtype(np.int32),
        "hidden_weights_q": np.dtype(np.int16),
        "hidden_bias_q": np.dtype(np.int32),
        "output_weights_q": np.dtype(np.int16),
        "output_bias_q": np.dtype(np.int64),
    }
    for name, shape in expected.items():
        if actual[name] != shape:
            raise RuntimeError(f"invalid {name} shape: {actual[name]}, expected {shape}")
    for name, dtype in expected_dtypes.items():
        if dtypes[name] != dtype:
            raise RuntimeError(f"invalid {name} dtype: {dtypes[name]}, expected {dtype}")
    if not np.isfinite(cp_scale) or cp_scale <= 0.0:
        raise RuntimeError("cp_scale must be finite and positive")
    if input_scale <= 0 or weight_scale <= 0:
        raise RuntimeError("quantization scales must be positive")
    if accumulator_size <= 0 or hidden_size <= 0:
        raise RuntimeError("model layers must be non-empty")

    return (
        cp_scale,
        input_scale,
        weight_scale,
        feature_weights,
        accumulator_bias,
        hidden_weights,
        hidden_bias,
        output_weights,
        int(output_bias_array[0]),
    )


(
    CP_SCALE,
    INPUT_SCALE,
    WEIGHT_SCALE,
    FEATURE_WEIGHTS_Q,
    ACCUMULATOR_BIAS_Q,
    HIDDEN_WEIGHTS_Q,
    HIDDEN_BIAS_Q,
    OUTPUT_WEIGHTS_Q,
    OUTPUT_BIAS_Q,
) = _load_model()

ACCUMULATOR_SIZE = FEATURE_WEIGHTS_Q.shape[1]
HIDDEN_SIZE = HIDDEN_BIAS_Q.shape[0]


@njit(cache=False, inline="always")
def _oriented_piece(piece: int, perspective: int) -> int:
    if perspective == engine.WHITE:
        return piece
    return (piece + engine.PIECE_KIND_COUNT) % engine.PIECE_BITBOARD_COUNT


@njit(cache=False, inline="always")
def _oriented_square(square: int, perspective: int) -> int:
    return square if perspective == engine.WHITE else square ^ 56


@njit(cache=False, inline="always")
def _feature(piece: int, square: int, king_square: int, perspective: int) -> int:
    oriented_piece = _oriented_piece(piece, perspective)
    oriented_square = _oriented_square(square, perspective)
    return (king_square * PIECE_BUCKETS + oriented_piece) * SQUARES + oriented_square


@njit(cache=False, inline="always")
def _king_square(pieces: NDArray[np.uint64], perspective: int) -> int:
    king = pieces[engine.piece_index(perspective, engine.KING)]
    if king == np.uint64(0):
        return 0
    return _oriented_square(engine.lsb_square(king), perspective)


@njit(cache=False, inline="always")
def _reset_accumulator(accumulator: NDArray[np.int32]) -> None:
    for column in range(ACCUMULATOR_SIZE):
        accumulator[column] = ACCUMULATOR_BIAS_Q[column]


@njit(cache=False, inline="always")
def _apply_feature(
    accumulator: NDArray[np.int32], feature: int, sign: int
) -> None:
    for column in range(ACCUMULATOR_SIZE):
        accumulator[column] += sign * int(FEATURE_WEIGHTS_Q[feature, column])


@njit(cache=False)
def rebuild(
    pieces: NDArray[np.uint64], accumulators: NDArray[np.int32]
) -> None:
    """Rebuild both king-conditioned accumulators from a position."""
    for perspective in range(2):
        _reset_accumulator(accumulators[perspective])
        king_square = _king_square(pieces, perspective)
        for piece in range(engine.PIECE_BITBOARD_COUNT):
            occupied = pieces[piece]
            while occupied:
                square = engine.lsb_square(occupied)
                occupied ^= engine.bit(square)
                _apply_feature(
                    accumulators[perspective],
                    _feature(piece, square, king_square, perspective),
                    1,
                )


@njit(cache=False, inline="always")
def _castling_rook_squares(to_square: int) -> tuple[int, int]:
    if to_square == engine.G1:
        return engine.H1, engine.F1
    if to_square == engine.C1:
        return engine.A1, engine.D1
    if to_square == engine.G8:
        return engine.H8, engine.F8
    return engine.A8, engine.D8


@njit(cache=False)
def _rebuild_after_king_move(
    pieces: NDArray[np.uint64],
    move: int,
    perspective: int,
    moving_piece: int,
    captured_piece: int,
    captured_square: int,
    accumulator: NDArray[np.int32],
) -> None:
    """Rebuild one perspective by projecting the king move onto the board."""
    from_square = engine.move_from(move)
    to_square = engine.move_to(move)
    flags = engine.move_flags(move)
    king_square = _oriented_square(to_square, perspective)
    rook_from = -1
    rook_to = -1
    rook_piece = engine.piece_index(perspective, engine.ROOK)
    if flags & engine.FLAG_CASTLING:
        rook_from, rook_to = _castling_rook_squares(to_square)

    _reset_accumulator(accumulator)
    for piece in range(engine.PIECE_BITBOARD_COUNT):
        occupied = pieces[piece]
        while occupied:
            square = engine.lsb_square(occupied)
            occupied ^= engine.bit(square)
            if piece == captured_piece and square == captured_square:
                continue
            projected_square = square
            if piece == moving_piece and square == from_square:
                projected_square = to_square
            elif piece == rook_piece and square == rook_from:
                projected_square = rook_to
            _apply_feature(
                accumulator,
                _feature(piece, projected_square, king_square, perspective),
                1,
            )


@njit(cache=False)
def update_for_move(
    pieces: NDArray[np.uint64],
    state: NDArray[np.int64],
    move: int,
    parent: NDArray[np.int32],
    child: NDArray[np.int32],
) -> None:
    """Derive child accumulators from the board immediately before ``move``."""
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
    placed_piece = moving_piece
    if flags & engine.FLAG_PROMOTION:
        placed_piece = engine.piece_index(side, engine.move_promotion(move))

    captured_square = to_square
    if flags & engine.FLAG_EN_PASSANT:
        captured_square = to_square - 8 if side == engine.WHITE else to_square + 8
    captured_piece = engine.piece_at(
        pieces,
        captured_square,
        enemy * engine.PIECE_KIND_COUNT,
        (enemy + 1) * engine.PIECE_KIND_COUNT,
    )
    king_move = moving_piece == engine.piece_index(side, engine.KING)

    for perspective in range(2):
        if king_move and perspective == side:
            _rebuild_after_king_move(
                pieces,
                move,
                perspective,
                moving_piece,
                captured_piece,
                captured_square,
                child[perspective],
            )
            continue

        for column in range(ACCUMULATOR_SIZE):
            child[perspective, column] = parent[perspective, column]
        king_square = _king_square(pieces, perspective)
        _apply_feature(
            child[perspective],
            _feature(moving_piece, from_square, king_square, perspective),
            -1,
        )
        _apply_feature(
            child[perspective],
            _feature(placed_piece, to_square, king_square, perspective),
            1,
        )
        if captured_piece != engine.NO_PIECE:
            _apply_feature(
                child[perspective],
                _feature(captured_piece, captured_square, king_square, perspective),
                -1,
            )
        if flags & engine.FLAG_CASTLING:
            rook_from, rook_to = _castling_rook_squares(to_square)
            rook_piece = engine.piece_index(side, engine.ROOK)
            _apply_feature(
                child[perspective],
                _feature(rook_piece, rook_from, king_square, perspective),
                -1,
            )
            _apply_feature(
                child[perspective],
                _feature(rook_piece, rook_to, king_square, perspective),
                1,
            )


@njit(cache=False, inline="always")
def _round_divide(value: int, divisor: int) -> int:
    if value >= 0:
        return (value + divisor // 2) // divisor
    return -((-value + divisor // 2) // divisor)


@njit(cache=False)
def evaluate(accumulators: NDArray[np.int32], side: int) -> int:
    """Evaluate from the side-to-move perspective in centipawns."""
    own = side
    opponent = engine.BLACK if side == engine.WHITE else engine.WHITE
    output = OUTPUT_BIAS_Q
    for unit in range(HIDDEN_SIZE):
        value = int(HIDDEN_BIAS_Q[unit])
        for column in range(ACCUMULATOR_SIZE):
            own_value = min(INPUT_SCALE, max(0, int(accumulators[own, column])))
            opponent_value = min(
                INPUT_SCALE, max(0, int(accumulators[opponent, column]))
            )
            value += int(HIDDEN_WEIGHTS_Q[unit, column]) * own_value
            value += (
                int(HIDDEN_WEIGHTS_Q[unit, ACCUMULATOR_SIZE + column])
                * opponent_value
            )
        hidden = min(INPUT_SCALE, max(0, _round_divide(value, WEIGHT_SCALE)))
        output += int(OUTPUT_WEIGHTS_Q[0, unit]) * hidden
    scaled_output = output * int(CP_SCALE)
    return _round_divide(scaled_output, INPUT_SCALE * WEIGHT_SCALE)


@njit(cache=False)
def benchmark_evaluations(
    accumulators: NDArray[np.int32], iterations: int
) -> int:
    """Run the dense head in compiled code and return a live checksum."""
    checksum = 0
    for index in range(iterations):
        checksum += evaluate(accumulators, index & 1)
    return checksum


def evaluate_reference(pieces: NDArray[np.uint64], side: int) -> float:
    """Readable NumPy evaluation of the exported quantized graph."""
    accumulators_q = np.empty((2, ACCUMULATOR_SIZE), dtype=np.int32)
    rebuild(pieces, accumulators_q)
    accumulators = accumulators_q.astype(np.float32) / INPUT_SCALE
    own = side
    opponent = engine.BLACK if side == engine.WHITE else engine.WHITE
    inputs = np.concatenate((accumulators[own], accumulators[opponent]))
    hidden_weights = HIDDEN_WEIGHTS_Q.astype(np.float32) / WEIGHT_SCALE
    hidden_bias = HIDDEN_BIAS_Q.astype(np.float32) / (INPUT_SCALE * WEIGHT_SCALE)
    hidden = np.clip(
        hidden_weights @ np.clip(inputs, 0.0, 1.0) + hidden_bias,
        0.0,
        1.0,
    )
    output_weights = OUTPUT_WEIGHTS_Q.astype(np.float32) / WEIGHT_SCALE
    output_bias = OUTPUT_BIAS_Q / (INPUT_SCALE * WEIGHT_SCALE)
    return float((output_weights @ hidden)[0] + output_bias) * CP_SCALE


def warmup() -> None:
    """Compile the standalone rebuild and dense inference signatures."""
    pieces = np.zeros(engine.PIECE_BITBOARD_COUNT, dtype=np.uint64)
    accumulators = np.empty((2, ACCUMULATOR_SIZE), dtype=np.int32)
    rebuild(pieces, accumulators)
    evaluate(accumulators, engine.WHITE)
