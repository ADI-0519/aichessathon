import os
import random

import chess

RNG = random.Random(os.environ.get("HARNESS_SEED", "69"))


def get_move(fen: str, time_left_ms: int) -> str:
    return RNG.choice(list(chess.Board(fen).legal_moves)).uci()
