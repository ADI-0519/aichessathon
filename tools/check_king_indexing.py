import random
import sys
from pathlib import Path

import chess

# challenger carries its own bucket map, so compare the two arrays directly
PIPELINE = Path(sys.argv[1] if len(sys.argv) > 1 else "scratch/nnue/pipeline").resolve()
CHALLENGER = Path(sys.argv[2] if len(sys.argv) > 2 else "challengers/numba_v1").resolve()

sys.path.insert(0, str(PIPELINE))
sys.path.insert(0, str(CHALLENGER))

from king_features import BASE_FEATURE_COUNT, KING_BUCKETS, OWN_KING_SLOT  # noqa: E402

import engine  # noqa: E402
from tools.nnue_features import canonical_indices, orient_index  # noqa: E402

print(f"pipeline {PIPELINE}")
print(f"challenger {CHALLENGER}")


def trainer_features(board: chess.Board) -> tuple[frozenset[int], frozenset[int]]:
    canonical = canonical_indices(board)
    white = [orient_index(index, chess.WHITE) for index in canonical]
    black = [orient_index(index, chess.BLACK) for index in canonical]
    sides = []
    for oriented in (white, black):
        king = [index for index in oriented if index // 64 == OWN_KING_SLOT]
        assert len(king) == 1, king
        bucket = int(KING_BUCKETS[king[0] % 64])
        sides.append(frozenset(bucket * BASE_FEATURE_COUNT + index for index in oriented))
    return sides[0], sides[1]


def runtime_features(board: chess.Board) -> tuple[frozenset[int], frozenset[int]]:
    position = engine.position_from_board(board)
    pieces = position.pieces
    def lsb(bitboard: int) -> int:
        return (bitboard & -bitboard).bit_length() - 1

    white_king = lsb(int(pieces[engine.piece_index(engine.WHITE, engine.KING)]))
    black_king = lsb(int(pieces[engine.piece_index(engine.BLACK, engine.KING)]))
    white_offset = int(KING_BUCKETS[white_king]) * BASE_FEATURE_COUNT
    black_offset = int(KING_BUCKETS[black_king ^ 56]) * BASE_FEATURE_COUNT
    white: set[int] = set()
    black: set[int] = set()
    for piece in range(engine.PIECE_BITBOARD_COUNT):
        occupied = int(pieces[piece])
        while occupied:
            least = occupied & -occupied
            square = least.bit_length() - 1
            occupied ^= least
            white.add(white_offset + piece * 64 + square)
            swapped = (piece + engine.PIECE_KIND_COUNT) % engine.PIECE_BITBOARD_COUNT
            black.add(black_offset + swapped * 64 + (square ^ 56))
    return frozenset(white), frozenset(black)


try:
    import nnue
    import numpy as np

    same = np.array_equal(np.asarray(nnue.KING_BUCKETS), np.asarray(KING_BUCKETS))
    print(f"runtime KING_BUCKETS matches pipeline: {same}")
    print(
        f"runtime FEATURE_COUNT {nnue.FEATURE_COUNT}, accumulator {nnue.ACCUMULATOR_SIZE}"
    )
    if not same:
        raise SystemExit("runtime and pipeline disagree on the bucket map")
except ModuleNotFoundError:
    print("challenger has no nnue module; skipping the bucket-map comparison")

print("engine piece indexing:")
for colour, name in ((engine.WHITE, "white"), (engine.BLACK, "black")):
    slots = [engine.piece_index(colour, kind) for kind in range(engine.PIECE_KIND_COUNT)]
    print(f"  {name}: {slots}")
print(f"  PIECE_BITBOARD_COUNT={engine.PIECE_BITBOARD_COUNT} KING={engine.KING}")

random.seed(20260906)
checked = 0
mismatches = 0
board = chess.Board()
for _game in range(120):
    board.reset()
    for _ in range(random.randint(0, 70)):
        moves = list(board.legal_moves)
        if not moves:
            break
        board.push(random.choice(moves))
    if board.king(chess.WHITE) is None or board.king(chess.BLACK) is None:
        continue
    expected = trainer_features(board)
    actual = runtime_features(board)
    if expected != actual:
        mismatches += 1
        if mismatches == 1:
            print(f"\nfirst mismatch at {board.fen()}")
            for side, exp, act in (
                ("white", expected[0], actual[0]),
                ("black", expected[1], actual[1]),
            ):
                if exp != act:
                    print(f"  {side}: trainer-only {sorted(exp - act)[:6]}")
                    print(f"  {side}: runtime-only {sorted(act - exp)[:6]}")
    checked += 1

print(f"\n{checked} positions checked, {mismatches} mismatches")
buckets_seen = {int(KING_BUCKETS[square]) for square in range(64)}
print(f"bucket ids in use: {sorted(buckets_seen)}")
