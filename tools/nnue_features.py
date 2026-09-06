"""Sparse, colour-symmetric piece-square features for a learned evaluator.

The runtime model sees the board from both players' perspectives.  A feature is
``(relationship, piece type, oriented square)`` where relationship is either
own or opposing.  This keeps the representation small (12 * 64 = 768 inputs)
and makes colour symmetry explicit instead of asking the network to learn it.
"""

from __future__ import annotations

import chess
import numpy as np
from numpy.typing import NDArray

FEATURE_COUNT = 12 * 64
MAX_PIECES = 32
PADDING_INDEX = FEATURE_COUNT


def canonical_indices(board: chess.Board) -> tuple[int, ...]:
    """Return absolute-colour piece-square indices in stable square order."""
    values: list[int] = []
    for square in chess.scan_forward(board.occupied):
        piece = board.piece_at(square)
        if piece is None:  # Defensive: ``occupied`` and ``piece_at`` should agree.
            raise ValueError(f"occupied square {square} contains no piece")
        colour_offset = 0 if piece.color == chess.WHITE else 6
        piece_slot = colour_offset + piece.piece_type - 1
        values.append(piece_slot * 64 + square)
    if not values or len(values) > MAX_PIECES:
        raise ValueError(f"position has unsupported piece count: {len(values)}")
    return tuple(values)


def orient_index(index: int, perspective: chess.Color) -> int:
    """Orient one canonical index for ``perspective``.

    White's representation is canonical.  From Black's perspective colours are
    swapped and ranks are mirrored, so equivalent positions share features.
    """
    if not 0 <= index < FEATURE_COUNT:
        raise ValueError(f"feature index outside 0..{FEATURE_COUNT - 1}: {index}")
    if perspective == chess.WHITE:
        return index
    piece_slot, square = divmod(index, 64)
    return ((piece_slot + 6) % 12) * 64 + (square ^ 56)


def oriented_indices(
    canonical: tuple[int, ...], perspective: chess.Color
) -> tuple[int, ...]:
    """Return all piece-square inputs as seen by one player."""
    return tuple(orient_index(index, perspective) for index in canonical)


def padded_indices(board: chess.Board) -> tuple[NDArray[np.uint16], int]:
    """Encode a board into a fixed-width canonical vector and piece count."""
    canonical = canonical_indices(board)
    encoded = np.full(MAX_PIECES, PADDING_INDEX, dtype=np.uint16)
    encoded[: len(canonical)] = canonical
    return encoded, len(canonical)


def flip_board_colours(board: chess.Board) -> chess.Board:
    """Mirror ranks, swap colours, and preserve the equivalent side to move.

    This helper exists for representation tests and diagnostics.  Castling and
    en-passant state are handled by :meth:`chess.Board.mirror`.
    """
    return board.mirror()
