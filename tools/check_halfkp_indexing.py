import random
import sys
from pathlib import Path

import chess

PIPELINE = Path("scratch/nnue/pipeline64").resolve()
CHALLENGER = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(PIPELINE))
sys.path.insert(0, str(CHALLENGER))

import nnue  # noqa: E402
from king_features import BASE_FEATURE_COUNT, KING_BUCKETS, OWN_KING_SLOT  # noqa: E402

import engine  # noqa: E402
from tools.nnue_features import canonical_indices, orient_index  # noqa: E402

print(f"pipeline {PIPELINE}")
print(f"challenger {CHALLENGER}")
print(f"runtime FEATURE_COUNT {nnue.FEATURE_COUNT}, accumulator {nnue.ACCUMULATOR_SIZE}")


def trainer_features(board):
    canonical = canonical_indices(board)
    out = []
    for colour in (chess.WHITE, chess.BLACK):
        oriented = [orient_index(i, colour) for i in canonical]
        king = [i for i in oriented if i // 64 == OWN_KING_SLOT]
        assert len(king) == 1, king
        bucket = int(KING_BUCKETS[king[0] % 64])
        out.append(frozenset(bucket * BASE_FEATURE_COUNT + i for i in oriented))
    return out[0], out[1]


def halfkp_features(board):
    position = engine.position_from_board(board)
    pieces = position.pieces
    out = []
    for perspective in (engine.WHITE, engine.BLACK):
        king_square = nnue._king_square(pieces, perspective)
        feats = set()
        for piece in range(engine.PIECE_BITBOARD_COUNT):
            occupied = int(pieces[piece])
            while occupied:
                least = occupied & -occupied
                square = least.bit_length() - 1
                occupied ^= least
                feats.add(int(nnue._feature(piece, square, king_square, perspective)))
        out.append(frozenset(feats))
    return out[0], out[1]


random.seed(20260907)
board = chess.Board()
checked = mismatches = 0
kings_seen = set()
for _ in range(1200):
    board.reset()
    for _ in range(random.randint(0, 190)):
        moves = list(board.legal_moves)
        if not moves:
            break
        board.push(random.choice(moves))
    if board.king(chess.WHITE) is None or board.king(chess.BLACK) is None:
        continue
    expected = trainer_features(board)
    actual = halfkp_features(board)
    kings_seen.add(board.king(chess.WHITE))
    if expected != actual:
        mismatches += 1
        if mismatches == 1:
            print(f"\nfirst mismatch at {board.fen()}")
            for name, e, a in (
                ("white", expected[0], actual[0]),
                ("black", expected[1], actual[1]),
            ):
                if e != a:
                    print(f"  {name} trainer-only {sorted(e - a)[:6]}")
                    print(f"  {name} halfkp-only  {sorted(a - e)[:6]}")
    checked += 1

print(f"\n{checked} positions checked, {mismatches} mismatches")
print(f"distinct white king squares exercised: {len(kings_seen)}")
