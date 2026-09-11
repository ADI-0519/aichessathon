from __future__ import annotations

from pathlib import Path

import engine
import numpy as np
from numba import njit
from numpy.typing import NDArray

BASE_FEATURE_COUNT = 12 * 64
KING_BUCKET_COUNT = 16
FEATURE_COUNT = KING_BUCKET_COUNT * BASE_FEATURE_COUNT
FORMAT_V2 = 2
FORMAT_V3 = 3
V11_ARCHITECTURE = "kingnet_v11_big_pairwise_dual_material"
INPUT_SCALE = 2_048
WEIGHT_SCALE = 2_048
ACTIVATION_SCALE = 2_048
STALE = -1


def _king_bucket(square: int) -> int:
    return ((square >> 3) >> 1) * 4 + ((square & 7) >> 1)


KING_BUCKETS = np.array([_king_bucket(square) for square in range(64)], dtype=np.int32)


def _array(archive: np.lib.npyio.NpzFile, name: str) -> NDArray[np.float32]:
    if name not in archive:
        raise RuntimeError(f"neural model is missing {name}")
    value = np.ascontiguousarray(archive[name], dtype=np.float32)
    if not np.isfinite(value).all():
        raise RuntimeError(f"neural model contains a non-finite {name}")
    return value


def _scalar_text(archive: np.lib.npyio.NpzFile, name: str) -> str:
    if name not in archive:
        raise RuntimeError(f"neural model is missing {name}")
    value = np.asarray(archive[name])
    if value.shape != ():
        raise RuntimeError(f"neural model {name} must be a scalar")
    return str(value.item())


def _require_shape(name: str, actual: tuple[int, ...], expected: tuple[int, ...]) -> None:
    if actual != expected:
        raise RuntimeError(f"invalid {name} shape: {actual}, expected {expected}")


def _load_model() -> dict[str, object]:
    path = Path(__file__).with_name("weights") / "model.npz"
    with np.load(path, allow_pickle=False) as archive:
        version = int(np.asarray(archive["format_version"]).item())
        if version not in (FORMAT_V2, FORMAT_V3):
            raise RuntimeError(f"unsupported neural model format: {version}")
        cp_scale = float(np.asarray(archive["cp_scale"]).item())
        stored_feature_dtype = np.asarray(archive["feature_weights"]).dtype
        feature = _array(archive, "feature_weights")
        accumulator_bias = _array(archive, "accumulator_bias")
        hidden = _array(archive, "hidden_weights")
        hidden_bias = _array(archive, "hidden_bias")

        if feature.ndim != 2:
            raise RuntimeError("feature_weights must be a matrix")
        accumulator = feature.shape[1]
        _require_shape("feature_weights", feature.shape, (FEATURE_COUNT, accumulator))
        _require_shape("accumulator_bias", accumulator_bias.shape, (accumulator,))
        if accumulator <= 0:
            raise RuntimeError("neural model accumulator width must be positive")

        if version == FORMAT_V2:
            if accumulator != 128:
                raise RuntimeError("format-v2 KingNet requires a 128-wide accumulator")
            _require_shape("hidden_weights", hidden.shape, (32, 2 * accumulator))
            _require_shape("hidden_bias", hidden_bias.shape, (32,))
            output_relu = _array(archive, "output_weights")
            output_square = np.zeros_like(output_relu)
            output_bias = _array(archive, "output_bias")
            _require_shape("output_weights", output_relu.shape, (1, 32))
            _require_shape("output_bias", output_bias.shape, (1,))
            pairwise_width = 0
            piece_head_map = np.zeros(33, dtype=np.int32)
            head_count = 1
            hidden_size = 32
        else:
            if _scalar_text(archive, "architecture") != V11_ARCHITECTURE:
                raise RuntimeError("unsupported format-v3 neural architecture")
            feature_storage = _scalar_text(archive, "feature_storage")
            if feature_storage not in {"float16", "float32"}:
                raise RuntimeError("unsupported format-v3 feature storage")
            expected_dtype = np.dtype(feature_storage)
            if stored_feature_dtype != expected_dtype:
                raise RuntimeError(
                    "feature_weights dtype does not match declared feature_storage"
                )
            runtime_input_scale = int(np.asarray(archive["runtime_input_scale"]).item())
            if runtime_input_scale != INPUT_SCALE:
                raise RuntimeError("format-v3 runtime_input_scale is incompatible")
            pairwise_width = int(np.asarray(archive["pairwise_width"]).item())
            if pairwise_width <= 0 or 2 * pairwise_width > accumulator:
                raise RuntimeError("invalid pairwise_width for accumulator")
            if "piece_head_map" not in archive:
                raise RuntimeError("neural model is missing piece_head_map")
            piece_head_map = np.ascontiguousarray(archive["piece_head_map"], dtype=np.int32)
            _require_shape("piece_head_map", piece_head_map.shape, (33,))
            if np.any(piece_head_map < 0):
                raise RuntimeError("piece_head_map contains a negative head id")
            legal_heads = sorted(set(int(value) for value in piece_head_map[2:]))
            if legal_heads != list(range(len(legal_heads))):
                raise RuntimeError("piece_head_map legal head ids must be contiguous from zero")
            head_count = len(legal_heads)
            if hidden.ndim != 3:
                raise RuntimeError("format-v3 hidden_weights must have three dimensions")
            hidden_size = hidden.shape[1]
            _require_shape(
                "hidden_weights",
                hidden.shape,
                (head_count, hidden_size, 2 * pairwise_width),
            )
            _require_shape("hidden_bias", hidden_bias.shape, (head_count, hidden_size))
            output_relu = _array(archive, "output_relu_weights")
            output_square = _array(archive, "output_clipped_square_weights")
            output_bias = _array(archive, "output_bias")
            _require_shape("output_relu_weights", output_relu.shape, (head_count, hidden_size))
            _require_shape(
                "output_clipped_square_weights",
                output_square.shape,
                (head_count, hidden_size),
            )
            _require_shape("output_bias", output_bias.shape, (head_count,))

    if not np.isfinite(cp_scale) or cp_scale <= 0.0:
        raise RuntimeError("model cp_scale must be finite and positive")
    return {
        "version": version,
        "cp_scale": cp_scale,
        "feature": feature,
        "accumulator_bias": accumulator_bias,
        "accumulator": accumulator,
        "pairwise_width": pairwise_width,
        "piece_head_map": piece_head_map,
        "head_count": head_count,
        "hidden_size": hidden_size,
        "hidden": hidden,
        "hidden_bias": hidden_bias,
        "output_relu": output_relu,
        "output_square": output_square,
        "output_bias": output_bias,
    }


_MODEL = _load_model()
FORMAT_VERSION = int(_MODEL["version"])
CP_SCALE = float(_MODEL["cp_scale"])
ACCUMULATOR_SIZE = int(_MODEL["accumulator"])
PAIRWISE_WIDTH = int(_MODEL["pairwise_width"])
HEAD_COUNT = int(_MODEL["head_count"])
HIDDEN_SIZE = int(_MODEL["hidden_size"])
PIECE_HEAD_MAP = np.asarray(_MODEL["piece_head_map"], dtype=np.int32)
FEATURE_WEIGHTS = np.asarray(_MODEL["feature"], dtype=np.float32)
ACCUMULATOR_BIAS = np.asarray(_MODEL["accumulator_bias"], dtype=np.float32)
HIDDEN_WEIGHTS = np.asarray(_MODEL["hidden"], dtype=np.float32)
HIDDEN_BIAS = np.asarray(_MODEL["hidden_bias"], dtype=np.float32)
OUTPUT_RELU_WEIGHTS = np.asarray(_MODEL["output_relu"], dtype=np.float32)
OUTPUT_SQUARE_WEIGHTS = np.asarray(_MODEL["output_square"], dtype=np.float32)
OUTPUT_BIAS = np.asarray(_MODEL["output_bias"], dtype=np.float32)

BUCKET_SLOT = ACCUMULATOR_SIZE
COUNT_SLOT = ACCUMULATOR_SIZE + 1
ACCUMULATOR_ROW = ACCUMULATOR_SIZE + (2 if FORMAT_VERSION == FORMAT_V3 else 1)

def _quantize(
    values: NDArray[np.float32], scale: int, dtype: type[np.signedinteger]
) -> NDArray[np.signedinteger]:
    rounded = np.rint(values * scale)
    limits = np.iinfo(dtype)
    if np.any(rounded < limits.min) or np.any(rounded > limits.max):
        raise RuntimeError(f"neural model values overflow {np.dtype(dtype).name}")
    return np.ascontiguousarray(rounded.astype(dtype))


FEATURE_WEIGHTS_Q = _quantize(FEATURE_WEIGHTS, INPUT_SCALE, np.int16)
ACCUMULATOR_BIAS_Q = _quantize(ACCUMULATOR_BIAS, INPUT_SCALE, np.int32)
HIDDEN_WEIGHTS_Q = _quantize(HIDDEN_WEIGHTS, WEIGHT_SCALE, np.int16)
if FORMAT_VERSION == FORMAT_V2:
    HIDDEN_BIAS_Q = _quantize(
        HIDDEN_BIAS, INPUT_SCALE * WEIGHT_SCALE, np.int32
    )
else:
    HIDDEN_BIAS_Q = _quantize(
        HIDDEN_BIAS, INPUT_SCALE * WEIGHT_SCALE, np.int64
    )
OUTPUT_RELU_WEIGHTS_Q = _quantize(OUTPUT_RELU_WEIGHTS, WEIGHT_SCALE, np.int16)
OUTPUT_SQUARE_WEIGHTS_Q = _quantize(OUTPUT_SQUARE_WEIGHTS, WEIGHT_SCALE, np.int16)
OUTPUT_BIAS_Q = _quantize(
    OUTPUT_BIAS, ACTIVATION_SCALE * WEIGHT_SCALE, np.int64
)
# (head, column, unit) so the hidden loop's innermost index is contiguous.
# Format 2 keeps a two-dimensional weight matrix and never reads this; numba
# still needs a concretely typed array to compile the module against.
if FORMAT_VERSION == FORMAT_V3:
    HIDDEN_WEIGHTS_T_Q = np.ascontiguousarray(HIDDEN_WEIGHTS_Q.transpose(0, 2, 1))
else:
    HIDDEN_WEIGHTS_T_Q = np.zeros((1, 1, 1), dtype=np.int16)


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
def _piece_count(pieces: NDArray[np.uint64]) -> int:
    count = 0
    for piece in range(engine.PIECE_BITBOARD_COUNT):
        count += engine.popcount(pieces[piece])
    return count


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
    if FORMAT_VERSION == FORMAT_V3:
        count = _piece_count(pieces)
        accumulators[0, COUNT_SLOT] = count
        accumulators[1, COUNT_SLOT] = count


@njit(cache=False)
def refresh(pieces: NDArray[np.uint64], accumulators: NDArray[np.int32]) -> None:
    if accumulators[0, BUCKET_SLOT] == STALE:
        rebuild_perspective(pieces, accumulators, 0, white_bucket_of(pieces))
    if accumulators[1, BUCKET_SLOT] == STALE:
        rebuild_perspective(pieces, accumulators, 1, black_bucket_of(pieces))
    if FORMAT_VERSION == FORMAT_V3 and accumulators[0, COUNT_SLOT] < 2:
        count = _piece_count(pieces)
        accumulators[0, COUNT_SLOT] = count
        accumulators[1, COUNT_SLOT] = count


@njit(cache=False)
def update_for_move(
    pieces: NDArray[np.uint64],
    state: NDArray[np.int64],
    move: int,
    parent: NDArray[np.int32],
    child: NDArray[np.int32],
) -> None:
    # Format 3 must not propagate an invalid bucket. Search may reach a child
    # without evaluating its parent, especially after a qeval-cache hit.
    if FORMAT_VERSION == FORMAT_V3:
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
        move, moving_piece, captured_piece, captured_square, parent, child
    )


@njit(cache=False)
def _update_from_metadata(
    move: int,
    moving_piece: int,
    captured_piece: int,
    captured_square: int,
    parent: NDArray[np.int32],
    child: NDArray[np.int32],
) -> None:
    """Apply one move to a known-fresh parent accumulator.

    Takes the moved/captured pieces rather than looking them up, so it works
    both before the move (update_for_move derives them from the board) and
    after it (update_after_move reads them out of the populated undo row).
    """
    for perspective in range(2):
        for column in range(ACCUMULATOR_ROW):
            child[perspective, column] = parent[perspective, column]

    white_bucket = int(parent[0, BUCKET_SLOT])
    black_bucket = int(parent[1, BUCKET_SLOT])
    # Post-move there is no side-to-move to read, but the piece index carries
    # its colour.
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
        _apply_feature(child, captured_piece, captured_square, -1, white_bucket, black_bucket)
        if FORMAT_VERSION == FORMAT_V3:
            child[0, COUNT_SLOT] -= 1
            child[1, COUNT_SLOT] -= 1

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

    if moving_piece == engine.piece_index(side, engine.KING):
        if side == engine.WHITE:
            if int(KING_BUCKETS[to_square]) != white_bucket:
                child[0, BUCKET_SLOT] = STALE
        elif int(KING_BUCKETS[to_square ^ 56]) != black_bucket:
            child[1, BUCKET_SLOT] = STALE


@njit(cache=False)
def update_after_move(
    move: int,
    undo: NDArray[np.int64],
    parent: NDArray[np.int32],
    child: NDArray[np.int32],
) -> None:
    """Build a child accumulator after ``move``, from its populated undo row.

    V14 delays the feature deltas until it knows the child will actually be
    searched, which means this runs *after* make_move.  That matters for
    format 3: update_for_move() can call refresh(pieces, parent) because
    ``pieces`` is still the parent position, but here ``pieces`` already
    describes the child, so refreshing the parent with it would rebuild the
    parent's accumulator from the wrong board.

    So do not refresh.  If the parent is not fresh, its bucket is the STALE
    sentinel and using it as a feature offset would read garbage weights --
    hand the staleness to the child instead and let evaluate() rebuild it from
    its own position.  A rebuild costs more than an incremental update, but
    only in the rare case, and it is correct.
    """
    stale = parent[0, BUCKET_SLOT] == STALE or parent[1, BUCKET_SLOT] == STALE
    if FORMAT_VERSION == FORMAT_V3 and parent[0, COUNT_SLOT] < 2:
        stale = True
    if stale:
        child[0, BUCKET_SLOT] = STALE
        child[1, BUCKET_SLOT] = STALE
        if FORMAT_VERSION == FORMAT_V3:
            child[0, COUNT_SLOT] = 0
            child[1, COUNT_SLOT] = 0
        return

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
def _evaluate_v2(accumulators: NDArray[np.int32], side: int) -> int:
    opponent = engine.BLACK if side == engine.WHITE else engine.WHITE
    output = int(OUTPUT_BIAS_Q[0])
    for unit in range(HIDDEN_SIZE):
        value = int(HIDDEN_BIAS_Q[unit])
        for column in range(ACCUMULATOR_SIZE):
            own_value = min(INPUT_SCALE, max(0, int(accumulators[side, column])))
            opponent_value = min(
                INPUT_SCALE, max(0, int(accumulators[opponent, column]))
            )
            value += int(HIDDEN_WEIGHTS_Q[unit, column]) * own_value
            value += (
                int(HIDDEN_WEIGHTS_Q[unit, ACCUMULATOR_SIZE + column])
                * opponent_value
            )
        hidden = min(INPUT_SCALE, max(0, _round_divide(value, WEIGHT_SCALE)))
        output += int(OUTPUT_RELU_WEIGHTS_Q[0, unit]) * hidden
    return _round_divide(output * int(CP_SCALE), INPUT_SCALE * WEIGHT_SCALE)


@njit(cache=False, inline="always")
def _pair_value(accumulators: NDArray[np.int32], perspective: int, column: int) -> int:
    # Two clipped activations, each at INPUT_SCALE, multiplied. Their raw
    # product carries INPUT_SCALE squared -- up to 4.2 million -- which forces
    # every downstream multiply-accumulate into int64 and stops LLVM
    # vectorising the hidden layer at all. Dividing back to INPUT_SCALE keeps
    # the pair in [0, 2048], so the products stay narrow and the inner loop
    # vectorises like the format-2 path does. The rounding this costs is a
    # quantum of the same size the accumulator already carries.
    left = min(INPUT_SCALE, max(0, int(accumulators[perspective, column])))
    right = min(
        INPUT_SCALE,
        max(0, int(accumulators[perspective, PAIRWISE_WIDTH + column])),
    )
    return _round_divide(left * right, INPUT_SCALE)


@njit(cache=False)
def _evaluate_v3(accumulators: NDArray[np.int32], side: int) -> int:
    opponent = engine.BLACK if side == engine.WHITE else engine.WHITE
    count = min(32, max(2, int(accumulators[0, COUNT_SLOT])))
    head = int(PIECE_HEAD_MAP[count])

    # Column-outer, unit-inner. The pair values do not depend on the hidden
    # unit, so the original unit-outer order recomputed all 256 of them 32
    # times over. Weights are pre-transposed to (head, column, unit) so this
    # inner loop walks memory contiguously.
    values = np.empty(HIDDEN_SIZE, dtype=np.int64)
    for unit in range(HIDDEN_SIZE):
        values[unit] = int(HIDDEN_BIAS_Q[head, unit])
    for column in range(PAIRWISE_WIDTH):
        own = _pair_value(accumulators, side, column)
        other = _pair_value(accumulators, opponent, column)
        for unit in range(HIDDEN_SIZE):
            values[unit] += int(HIDDEN_WEIGHTS_T_Q[head, column, unit]) * own
            values[unit] += (
                int(HIDDEN_WEIGHTS_T_Q[head, PAIRWISE_WIDTH + column, unit]) * other
            )

    output = int(OUTPUT_BIAS_Q[head])
    for unit in range(HIDDEN_SIZE):
        preactivation = _round_divide(values[unit], WEIGHT_SCALE)
        relu = max(0, preactivation)
        clipped = min(ACTIVATION_SCALE, relu)
        clipped_square = _round_divide(clipped * clipped, ACTIVATION_SCALE)
        output += int(OUTPUT_RELU_WEIGHTS_Q[head, unit]) * relu
        output += int(OUTPUT_SQUARE_WEIGHTS_Q[head, unit]) * clipped_square

    return _round_divide(output * int(CP_SCALE), ACTIVATION_SCALE * WEIGHT_SCALE)


@njit(cache=False)
def evaluate(
    pieces: NDArray[np.uint64], accumulators: NDArray[np.int32], side: int
) -> int:
    refresh(pieces, accumulators)
    if FORMAT_VERSION == FORMAT_V2:
        return _evaluate_v2(accumulators, side)
    return _evaluate_v3(accumulators, side)


def _reference_accumulators(pieces: NDArray[np.uint64]) -> NDArray[np.float32]:
    white_king = int(pieces[engine.piece_index(engine.WHITE, engine.KING)])
    black_king = int(pieces[engine.piece_index(engine.BLACK, engine.KING)])
    white_square = white_king.bit_length() - 1
    black_square = black_king.bit_length() - 1
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
    return accumulators


def evaluate_reference(pieces: NDArray[np.uint64], side: int) -> float:
    accumulators = _reference_accumulators(pieces)
    opponent = engine.BLACK if side == engine.WHITE else engine.WHITE
    if FORMAT_VERSION == FORMAT_V2:
        inputs = np.concatenate((accumulators[side], accumulators[opponent]))
        hidden = np.clip(
            HIDDEN_WEIGHTS @ np.clip(inputs, 0.0, 1.0) + HIDDEN_BIAS,
            0.0,
            1.0,
        )
        return float((OUTPUT_RELU_WEIGHTS @ hidden)[0] + OUTPUT_BIAS[0]) * CP_SCALE

    width = PAIRWISE_WIDTH
    activated = np.clip(accumulators, 0.0, 1.0)
    own_pair = activated[side, :width] * activated[side, width : 2 * width]
    opponent_pair = (
        activated[opponent, :width] * activated[opponent, width : 2 * width]
    )
    pairwise = np.concatenate((own_pair, opponent_pair))
    count = min(32, max(2, sum(int(value).bit_count() for value in pieces)))
    head = int(PIECE_HEAD_MAP[count])
    preactivation = HIDDEN_WEIGHTS[head] @ pairwise + HIDDEN_BIAS[head]
    relu = np.maximum(preactivation, 0.0)
    clipped_square = np.clip(preactivation, 0.0, 1.0) ** 2
    output = float(OUTPUT_BIAS[head])
    output += float(OUTPUT_RELU_WEIGHTS[head] @ relu)
    output += float(OUTPUT_SQUARE_WEIGHTS[head] @ clipped_square)
    return output * CP_SCALE


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
