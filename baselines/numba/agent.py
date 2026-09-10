"""Same search as baselines/minimax, with the evaluation jitted by numba.

The pattern worth copying is the warm-up call at the bottom of this file. numba compiles on
first call, and that first call costs far more than the move it is part of. Running it at
import spends the compile inside the 90 second init budget instead of on your clock. numba
compiles per signature, so warm every jitted function with the argument types it will really
see. An int32 warm-up does not help a float64 call later.
"""

import math
import os
import random

import chess
import numpy as np
from numba import njit

RNG = random.Random(os.environ.get("HARNESS_SEED", "69"))

PIECE_VALUE = np.array([0, 100, 320, 330, 500, 900, 0], dtype=np.int32)
MOBILITY_WEIGHT = 4
MATE = 1e6


@njit(cache=False)
def evaluate(pieces: np.ndarray, mine: np.ndarray, mobility: int) -> int:
    material = 0
    for square in range(64):
        piece = pieces[square]
        if piece == 0:
            continue
        value = PIECE_VALUE[piece]
        material += value if mine[square] else -value
    return material + MOBILITY_WEIGHT * mobility


def encode(board: chess.Board) -> tuple[np.ndarray, np.ndarray]:
    pieces = np.zeros(64, dtype=np.int32)
    mine = np.zeros(64, dtype=np.bool_)
    for square, piece in board.piece_map().items():
        pieces[square] = piece.piece_type
        mine[square] = piece.color == board.turn
    return pieces, mine


def negamax(board: chess.Board, depth: int) -> float:
    moves = list(board.legal_moves)
    if not moves:
        return -MATE if board.is_check() else 0.0
    if depth == 0:
        pieces, mine = encode(board)
        return float(evaluate(pieces, mine, len(moves)))
    best = -math.inf
    for move in moves:
        board.push(move)
        best = max(best, -negamax(board, depth - 1))
        board.pop()
    return best


def get_move(fen: str, time_left_ms: int) -> str:
    board = chess.Board(fen)
    best_score = -math.inf
    best: list[chess.Move] = []
    for move in board.legal_moves:
        board.push(move)
        score = -negamax(board, 1)
        board.pop()
        if score > best_score:
            best_score = score
            best = [move]
        elif score == best_score:
            best.append(move)
    return RNG.choice(best).uci()


evaluate(*encode(chess.Board()), 20)
