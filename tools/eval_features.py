from __future__ import annotations

import chess
import numpy as np
from numpy.typing import NDArray

import engine
import search

# features 0-383 are 6 kinds x 64 squares from white side, black piece mirrors its rank
F_BISHOP_PAIR = 384
F_DOUBLED = 385
F_ISOLATED = 386
F_PASSED_RANK = 387
F_PASSED_RANK_SQUARED = 388
F_ROOK_SEMI_OPEN = 389
F_ROOK_OPEN = 390
F_KING_SHIELD = 391
FEATURE_COUNT = 392

MG_PINNED = (F_PASSED_RANK_SQUARED,)
EG_PINNED = (F_PASSED_RANK, F_KING_SHIELD)

SCALAR_WEIGHTS = {
    F_BISHOP_PAIR: (32, 42),
    F_DOUBLED: (-11, -14),
    F_ISOLATED: (-10, -8),
    F_PASSED_RANK: (7, 0),
    F_PASSED_RANK_SQUARED: (0, 5),
    F_ROOK_SEMI_OPEN: (12, 8),
    F_ROOK_OPEN: (10, 6),
    F_KING_SHIELD: (9, 0),
}
TEMPO = 10


def baseline_weights() -> tuple[NDArray[np.int64], NDArray[np.int64], int]:
    middlegame = np.zeros(FEATURE_COUNT, dtype=np.int64)
    endgame = np.zeros(FEATURE_COUNT, dtype=np.int64)
    for kind in range(engine.PIECE_KIND_COUNT):
        index = engine.piece_index(engine.WHITE, kind)
        for square in range(64):
            middlegame[kind * 64 + square] = search.MG_VALUE[kind] + search.MG_PST[index, square]
            endgame[kind * 64 + square] = search.EG_VALUE[kind] + search.EG_PST[index, square]
    for feature, (mg, eg) in SCALAR_WEIGHTS.items():
        middlegame[feature] = mg
        endgame[feature] = eg
    return middlegame, endgame, TEMPO


def extract(board: chess.Board) -> tuple[NDArray[np.int16], int]:
    features = np.zeros(FEATURE_COUNT, dtype=np.int16)
    phase = 0
    # engine kinds and colours are 0-based ints, python-chess wants 1-based kinds and bools
    for color in (engine.WHITE, engine.BLACK):
        white = color == engine.WHITE
        sign = 1 if white else -1
        mirror = 0 if white else 56
        pawns = board.pieces_mask(chess.PAWN, white)
        enemy_pawns = board.pieces_mask(chess.PAWN, not white)

        for kind in range(engine.PIECE_KIND_COUNT):
            occupied = board.pieces_mask(kind + 1, white)
            phase += chess.popcount(occupied) * int(search.PHASE_VALUE[kind])
            for square in chess.scan_forward(occupied):
                features[kind * 64 + (square ^ mirror)] += sign

        if chess.popcount(board.pieces_mask(chess.BISHOP, white)) >= 2:
            features[F_BISHOP_PAIR] += sign

        for square in chess.scan_forward(pawns):
            file_index = chess.square_file(square)
            rank = chess.square_rank(square)
            relative_rank = rank if white else 7 - rank
            if chess.popcount(pawns & chess.BB_FILES[file_index]) > 1:
                features[F_DOUBLED] += sign
            neighbours = 0
            if file_index > 0:
                neighbours |= pawns & chess.BB_FILES[file_index - 1]
            if file_index < 7:
                neighbours |= pawns & chess.BB_FILES[file_index + 1]
            if not neighbours:
                features[F_ISOLATED] += sign
            if not enemy_pawns & int(search.PASSED_MASKS[color, square]):
                features[F_PASSED_RANK] += sign * relative_rank
                features[F_PASSED_RANK_SQUARED] += sign * relative_rank * relative_rank

        for square in chess.scan_forward(board.pieces_mask(chess.ROOK, white)):
            file_index = chess.square_file(square)
            if not pawns & chess.BB_FILES[file_index]:
                features[F_ROOK_SEMI_OPEN] += sign
                if not enemy_pawns & chess.BB_FILES[file_index]:
                    features[F_ROOK_OPEN] += sign

        king_square = board.king(white)
        if king_square is not None:
            king_file = chess.square_file(king_square)
            shield_rank = chess.square_rank(king_square) + (1 if white else -1)
            if 0 <= shield_rank < 8:
                for shield_file in range(max(0, king_file - 1), min(7, king_file + 1) + 1):
                    if pawns & chess.BB_SQUARES[shield_rank * 8 + shield_file]:
                        features[F_KING_SHIELD] += sign

    return features, phase


def score(
    features: NDArray[np.int16],
    phase: int,
    middlegame_weights: NDArray[np.int64],
    endgame_weights: NDArray[np.int64],
    tempo: int,
    white_to_move: bool,
) -> int:
    # integers throughout, including the floor (otherwise stops reproducing evaluate() exactly)
    phase = min(phase, search.MAX_PHASE)
    middlegame = int(features.astype(np.int64) @ middlegame_weights)
    endgame = int(features.astype(np.int64) @ endgame_weights)
    tapered = (middlegame * phase + endgame * (search.MAX_PHASE - phase)) // search.MAX_PHASE
    tapered += tempo if white_to_move else -tempo
    return tapered if white_to_move else -tapered
