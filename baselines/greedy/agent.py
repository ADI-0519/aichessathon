import os
import random

import chess

RNG = random.Random(os.environ.get("HARNESS_SEED", "69"))

PIECE_VALUE = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
}
MATE = 10**6


def material(board: chess.Board, side: chess.Color) -> int:
    return sum(
        value * (len(board.pieces(piece, side)) - len(board.pieces(piece, not side)))
        for piece, value in PIECE_VALUE.items()
    )


def get_move(fen: str, time_left_ms: int) -> str:
    board = chess.Board(fen)
    mover = board.turn
    best_score = -MATE
    best: list[chess.Move] = []
    for move in board.legal_moves:
        board.push(move)
        score = MATE if board.is_checkmate() else material(board, mover)
        board.pop()
        if score > best_score:
            best_score = score
            best = [move]
        elif score == best_score:
            best.append(move)
    return RNG.choice(best).uci()
