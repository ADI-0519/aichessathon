import math
import os
import random

import chess

RNG = random.Random(os.environ.get("HARNESS_SEED", "69"))

PIECE_VALUE = {
    chess.PAWN: 100.0,
    chess.KNIGHT: 320.0,
    chess.BISHOP: 330.0,
    chess.ROOK: 500.0,
    chess.QUEEN: 900.0,
}
MOBILITY_WEIGHT = 4.0
MATE = 1e6


def evaluate(board: chess.Board, mobility: int) -> float:
    mover = board.turn
    material = sum(
        value * (len(board.pieces(piece, mover)) - len(board.pieces(piece, not mover)))
        for piece, value in PIECE_VALUE.items()
    )
    return material + MOBILITY_WEIGHT * mobility


def negamax(board: chess.Board, depth: int) -> float:
    moves = list(board.legal_moves)
    if not moves:
        return -MATE if board.is_check() else 0.0
    if depth == 0:
        return evaluate(board, len(moves))
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
