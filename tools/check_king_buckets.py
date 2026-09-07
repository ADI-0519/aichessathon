import random
import sys
from pathlib import Path

import chess
import numpy as np
import torch

PIPELINE = Path(sys.argv[1] if len(sys.argv) > 1 else "scratch/nnue/pipeline").resolve()
sys.path.insert(0, str(PIPELINE))
print(f"pipeline {PIPELINE}")

from king_features import (  # noqa: E402
    BASE_FEATURE_COUNT,
    KING_BUCKETS,
    OWN_KING_SLOT,
    PADDING_INDEX,
)
from train_king_factored import apply_king_bucket, black_perspective  # noqa: E402

from tools.nnue_features import padded_indices  # noqa: E402


def expected(indices: np.ndarray, base_padding: int) -> np.ndarray:
    out = np.empty_like(indices)
    for row in range(indices.shape[0]):
        live = [int(v) for v in indices[row] if int(v) != base_padding]
        kings = [v for v in live if v // 64 == OWN_KING_SLOT]
        assert len(kings) == 1, kings
        bucket = int(KING_BUCKETS[kings[0] % 64])
        for column, value in enumerate(indices[row]):
            value = int(value)
            out[row, column] = (
                PADDING_INDEX
                if value == base_padding
                else bucket * BASE_FEATURE_COUNT + value
            )
    return out


random.seed(20260906)
boards = []
board = chess.Board()
for _ in range(64):
    board.reset()
    for _ in range(random.randint(0, 60)):
        moves = list(board.legal_moves)
        if not moves:
            break
        board.push(random.choice(moves))
    if board.king(chess.WHITE) is not None and board.king(chess.BLACK) is not None:
        boards.append(board.copy())

packed = np.stack([padded_indices(b)[0] for b in boards]).astype(np.int64)
canonical = torch.from_numpy(packed)
base_padding = 12 * 64

white = apply_king_bucket(canonical).numpy()
black = apply_king_bucket(black_perspective(canonical)).numpy()

white_expected = expected(packed, base_padding)
black_np = black_perspective(canonical).numpy()
black_expected = expected(black_np, base_padding)

print(f"boards {len(boards)}")
print(f"white perspective matches: {np.array_equal(white, white_expected)}")
print(f"black perspective matches: {np.array_equal(black, black_expected)}")
print(f"index range {white.min()}..{white.max()} (padding {PADDING_INDEX})")
live = white[white != PADDING_INDEX]
print(f"live index range {live.min()}..{live.max()}, must be < {PADDING_INDEX}")
