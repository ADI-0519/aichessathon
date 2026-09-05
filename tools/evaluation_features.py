"""Phase-aware strategic features for residual evaluation tuning.

The extractor intentionally avoids material and piece-square terms already in
V3. A fitted model should explain V3's remaining error, then be added to the
existing evaluator rather than replace it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import chess

MAX_PHASE = 24
RAW_FEATURE_NAMES = (
    "knight_mobility",
    "bishop_mobility",
    "rook_mobility",
    "queen_mobility",
    "king_zone_pressure",
    "hanging_minor",
    "hanging_major",
    "safe_space",
    "rook_activity",
    "blocked_passer",
    "minor_outpost",
    "king_file_exposure",
)
FEATURE_NAMES = tuple(
    channel
    for name in RAW_FEATURE_NAMES
    for channel in (f"{name}_mg", f"{name}_eg")
)

_PHASE_WEIGHTS = {
    chess.KNIGHT: 1,
    chess.BISHOP: 1,
    chess.ROOK: 2,
    chess.QUEEN: 4,
}
_MINOR_TYPES = (chess.KNIGHT, chess.BISHOP)
_MAJOR_TYPES = (chess.ROOK, chess.QUEEN)
_CENTER_FILES = (
    chess.BB_FILES[2] | chess.BB_FILES[3] | chess.BB_FILES[4] | chess.BB_FILES[5]
)
_SPACE_MASKS = {
    chess.WHITE: _CENTER_FILES
    & (chess.BB_RANKS[3] | chess.BB_RANKS[4] | chess.BB_RANKS[5]),
    chess.BLACK: _CENTER_FILES
    & (chess.BB_RANKS[2] | chess.BB_RANKS[3] | chess.BB_RANKS[4]),
}


@dataclass(frozen=True, slots=True)
class EvaluationFeatures:
    """Tapered integer components from the side-to-move perspective.

    Each raw balance produces ``raw * phase`` and
    ``raw * (MAX_PHASE - phase)``. The fitter divides these components by
    ``MAX_PHASE`` so exported weights remain centipawns per raw feature unit.
    """

    phase: int
    raw_values: tuple[int, ...]
    values: tuple[int, ...]

    def __post_init__(self) -> None:
        if not 0 <= self.phase <= MAX_PHASE:
            raise ValueError("phase is outside the tapered range")
        if len(self.raw_values) != len(RAW_FEATURE_NAMES):
            raise ValueError("raw feature vector length does not match schema")
        if len(self.values) != len(FEATURE_NAMES):
            raise ValueError("feature vector length does not match schema")

    def as_dict(self) -> dict[str, int]:
        return dict(zip(FEATURE_NAMES, self.values, strict=True))

    def raw_as_dict(self) -> dict[str, int]:
        return dict(zip(RAW_FEATURE_NAMES, self.raw_values, strict=True))


def extract_features(board: chess.Board) -> EvaluationFeatures:
    """Extract reproducible features without mutating ``board``."""
    phase = game_phase(board)
    raw_white = (
        _balance(board, lambda value, color: _mobility(value, color, chess.KNIGHT)),
        _balance(board, lambda value, color: _mobility(value, color, chess.BISHOP)),
        _balance(board, lambda value, color: _mobility(value, color, chess.ROOK)),
        _balance(board, lambda value, color: _mobility(value, color, chess.QUEEN)),
        _balance(board, _king_zone_pressure),
        _liability_balance(
            board, lambda value, color: _hanging(value, color, _MINOR_TYPES)
        ),
        _liability_balance(
            board, lambda value, color: _hanging(value, color, _MAJOR_TYPES)
        ),
        _balance(board, _safe_space),
        _balance(board, _rook_activity),
        _liability_balance(board, _blocked_passers),
        _balance(board, _minor_outposts),
        _liability_balance(board, _king_file_exposure),
    )
    perspective = 1 if board.turn == chess.WHITE else -1
    raw_values = tuple(perspective * value for value in raw_white)
    tapered: list[int] = []
    for value in raw_values:
        tapered.extend((value * phase, value * (MAX_PHASE - phase)))
    return EvaluationFeatures(phase, raw_values, tuple(tapered))


def game_phase(board: chess.Board) -> int:
    """Return the conventional capped 0..24 non-pawn-material phase."""
    phase = sum(
        len(board.pieces(piece_type, color)) * weight
        for piece_type, weight in _PHASE_WEIGHTS.items()
        for color in chess.COLORS
    )
    return min(MAX_PHASE, phase)


def _balance(board: chess.Board, feature: Callable[[chess.Board, chess.Color], int]) -> int:
    return feature(board, chess.WHITE) - feature(board, chess.BLACK)


def _liability_balance(
    board: chess.Board, feature: Callable[[chess.Board, chess.Color], int]
) -> int:
    """Convert per-side liabilities into a positive-is-good White balance."""
    return feature(board, chess.BLACK) - feature(board, chess.WHITE)


def _mobility(board: chess.Board, color: chess.Color, piece_type: chess.PieceType) -> int:
    own = board.occupied_co[color]
    return sum(
        chess.popcount(int(board.attacks(square)) & ~own)
        for square in board.pieces(piece_type, color)
    )


def _king_zone_pressure(board: chess.Board, color: chess.Color) -> int:
    enemy_king = board.king(not color)
    if enemy_king is None:
        return 0
    zone = chess.BB_KING_ATTACKS[enemy_king] | chess.BB_SQUARES[enemy_king]
    return sum(
        chess.popcount(int(board.attacks(square)) & zone)
        for piece_type in range(chess.PAWN, chess.KING)
        for square in board.pieces(piece_type, color)
    )


def _hanging(
    board: chess.Board,
    color: chess.Color,
    piece_types: tuple[chess.PieceType, ...],
) -> int:
    hanging = 0
    for piece_type in piece_types:
        unit = 2 if piece_type == chess.QUEEN else 1
        for square in board.pieces(piece_type, color):
            if board.attackers(not color, square) and not board.attackers(color, square):
                hanging += unit
    return hanging


def _pawn_attacks(board: chess.Board, color: chess.Color) -> chess.Bitboard:
    attacks = chess.BB_EMPTY
    for square in board.pieces(chess.PAWN, color):
        attacks |= int(board.attacks(square))
    return attacks


def _safe_space(board: chess.Board, color: chess.Color) -> int:
    controlled = chess.BB_EMPTY
    for piece_type in range(chess.PAWN, chess.KING):
        for square in board.pieces(piece_type, color):
            controlled |= int(board.attacks(square))
    safe = controlled & _SPACE_MASKS[color] & ~board.occupied_co[color]
    safe &= ~_pawn_attacks(board, not color)
    return chess.popcount(safe)


def _rook_activity(board: chess.Board, color: chess.Color) -> int:
    rooks = tuple(board.pieces(chess.ROOK, color))
    seventh_rank = 6 if color == chess.WHITE else 1
    activity = 2 * sum(chess.square_rank(square) == seventh_rank for square in rooks)
    if len(rooks) >= 2 and rooks[1] in board.attacks(rooks[0]):
        activity += 1
    return activity


def _is_passed_pawn(board: chess.Board, square: chess.Square, color: chess.Color) -> bool:
    file_index = chess.square_file(square)
    rank = chess.square_rank(square)
    for enemy in board.pieces(chess.PAWN, not color):
        if abs(chess.square_file(enemy) - file_index) > 1:
            continue
        enemy_rank = chess.square_rank(enemy)
        if (color == chess.WHITE and enemy_rank > rank) or (
            color == chess.BLACK and enemy_rank < rank
        ):
            return False
    return True


def _blocked_passers(board: chess.Board, color: chess.Color) -> int:
    direction = 8 if color == chess.WHITE else -8
    blocked = 0
    for square in board.pieces(chess.PAWN, color):
        front = square + direction
        if (
            _is_passed_pawn(board, square, color)
            and 0 <= front < 64
            and board.piece_at(front)
        ):
            blocked += 1
    return blocked


def _minor_outposts(board: chess.Board, color: chess.Color) -> int:
    own_pawn_attacks = _pawn_attacks(board, color)
    enemy_pawn_attacks = _pawn_attacks(board, not color)
    outposts = 0
    for piece_type in _MINOR_TYPES:
        for square in board.pieces(piece_type, color):
            rank = chess.square_rank(square)
            in_enemy_half = rank >= 4 if color == chess.WHITE else rank <= 3
            square_mask = chess.BB_SQUARES[square]
            if (
                in_enemy_half
                and own_pawn_attacks & square_mask
                and not enemy_pawn_attacks & square_mask
            ):
                outposts += 1
    return outposts


def _king_file_exposure(board: chess.Board, color: chess.Color) -> int:
    king = board.king(color)
    if king is None:
        return 0
    king_file = chess.square_file(king)
    pawns = board.pieces(chess.PAWN, color)
    return sum(
        not bool(pawns & chess.BB_FILES[file_index])
        for file_index in range(max(0, king_file - 1), min(7, king_file + 1) + 1)
    )
