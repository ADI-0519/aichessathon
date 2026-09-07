import random
import sys
from pathlib import Path

import chess
import numpy as np

# width and bucket slot are read off the challenger, so any config works here
CHALLENGER = Path(sys.argv[1] if len(sys.argv) > 1 else "scratch/v7/kingnet").resolve()
sys.path.insert(0, str(CHALLENGER))
print(f"challenger {CHALLENGER}")

import nnue  # noqa: E402

import engine  # noqa: E402

nnue.warmup()
print(
    f"buckets {nnue.KING_BUCKET_COUNT}, accumulator {nnue.ACCUMULATOR_SIZE}, "
    f"features {nnue.FEATURE_COUNT}"
)

random.seed(20260906)


def accumulators() -> np.ndarray:
    return np.empty((2, nnue.ACCUMULATOR_ROW), dtype=np.int32)


def rebuilt(position) -> np.ndarray:
    rows = accumulators()
    nnue.rebuild(position.pieces, rows)
    return rows


incremental_checks = 0
incremental_errors = 0
bucket_changes = 0
king_moves = 0
stale_seen = 0

board = chess.Board()
for _game in range(60):
    board.reset()
    position = engine.position_from_board(board)
    live = rebuilt(position)

    for _ in range(90):
        moves = list(board.legal_moves)
        if not moves:
            break
        move = random.choice(moves)

        parent_position = engine.position_from_board(board)
        internal = None
        for candidate in engine.legal_moves(parent_position):
            if engine.move_to_uci(int(candidate)) == move.uci():
                internal = int(candidate)
                break
        if internal is None:
            break

        child = accumulators()
        nnue.update_for_move(
            parent_position.pieces, parent_position.state, internal, live, child
        )
        was_stale = int(child[0, nnue.BUCKET_SLOT]) == nnue.STALE or (
            int(child[1, nnue.BUCKET_SLOT]) == nnue.STALE
        )
        if board.piece_at(move.from_square).piece_type == chess.KING:
            king_moves += 1
            if was_stale:
                bucket_changes += 1
        if was_stale:
            stale_seen += 1

        board.push(move)
        child_position = engine.position_from_board(board)

        # a stale row must survive repair and still match a full rebuild
        nnue.refresh(child_position.pieces, child)
        reference = rebuilt(child_position)
        if not np.array_equal(child, reference):
            incremental_errors += 1
            if incremental_errors == 1:
                print(f"first drift after {move.uci()} at {board.fen()}")
                delta = child[:, : nnue.ACCUMULATOR_SIZE] - reference[:, : nnue.ACCUMULATOR_SIZE]
                print(f"  max column delta {np.abs(delta).max()}")
                print(
                    f"  buckets child {child[:, nnue.BUCKET_SLOT]} "
                    f"ref {reference[:, nnue.BUCKET_SLOT]}"
                )
        incremental_checks += 1
        live = child

print(f"incremental vs rebuild: {incremental_checks} checks, {incremental_errors} mismatches")
print(f"king moves seen {king_moves}, of which crossed a bucket {bucket_changes}")
print(f"rows marked stale {stale_seen}")

worst = 0.0
checked = 0
board.reset()
for _game in range(40):
    board.reset()
    for _ in range(random.randint(0, 60)):
        moves = list(board.legal_moves)
        if not moves:
            break
        board.push(random.choice(moves))
    if board.king(chess.WHITE) is None or board.king(chess.BLACK) is None:
        continue
    position = engine.position_from_board(board)
    rows = rebuilt(position)
    for side in (engine.WHITE, engine.BLACK):
        integer = nnue.evaluate(position.pieces, rows, side)
        reference = nnue.evaluate_reference(position.pieces, side)
        worst = max(worst, abs(integer - reference))
        checked += 1
print(f"quantisation: {checked} comparisons, worst gap {worst:.2f} cp")
