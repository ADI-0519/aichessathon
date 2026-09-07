"""King-conditioned sparse features for the next learned evaluator.

The existing packed dataset stores absolute-colour piece-square indices.  This
module turns those indices into a view conditioned on the friendly king square:

``(friendly king square, relationship/piece type, piece square)``.

Both kings remain in the piece set.  That is a small extension of conventional
HalfKP which preserves king-to-king geometry and, importantly, lets us lift the
current V5 piece-square network into this representation exactly before
training.  Equivalent colour-flipped positions still share every feature.
"""

from __future__ import annotations

import chess
import numpy as np
from numpy.typing import NDArray

from tools.nnue_features import MAX_PIECES, canonical_indices, orient_index

PIECE_BUCKETS = 12
KING_SQUARES = 64
SQUARES = 64
FEATURE_COUNT = KING_SQUARES * PIECE_BUCKETS * SQUARES
PADDING_INDEX = FEATURE_COUNT


def feature_index(piece_index: int, king_square: int) -> int:
    """Condition one oriented piece-square index on the friendly king."""
    if not 0 <= piece_index < PIECE_BUCKETS * SQUARES:
        raise ValueError("piece-square index is outside the canonical range")
    if not 0 <= king_square < KING_SQUARES:
        raise ValueError("king square is outside 0..63")
    return king_square * PIECE_BUCKETS * SQUARES + piece_index


def perspective_indices(
    board: chess.Board, perspective: chess.Color
) -> tuple[int, ...]:
    """Return king-conditioned features from one player's perspective."""
    king = board.king(perspective)
    if king is None:
        raise ValueError("position is missing the perspective king")
    oriented_king = king if perspective == chess.WHITE else king ^ 56
    return tuple(
        feature_index(orient_index(index, perspective), oriented_king)
        for index in canonical_indices(board)
    )


def padded_perspective_indices(
    board: chess.Board, perspective: chess.Color
) -> tuple[NDArray[np.uint16], int]:
    """Encode one perspective as a fixed-width vector for diagnostics."""
    features = perspective_indices(board, perspective)
    encoded = np.full(MAX_PIECES, PADDING_INDEX, dtype=np.uint16)
    encoded[: len(features)] = features
    return encoded, len(features)
