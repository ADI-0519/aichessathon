"""Measure how an agent's choice changes with search time on one PGN position."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import chess
import chess.pgn

import agent


class ProfilingSearcher(agent.Searcher):
    def __init__(self, deadline: float) -> None:
        super().__init__(deadline)
        self.completed_depth = 0

    def search_root(
        self,
        board: chess.Board,
        depth: int,
        alpha: int,
        beta: int,
        preferred: chess.Move,
    ) -> tuple[int, chess.Move]:
        result = super().search_root(board, depth, alpha, beta, preferred)
        self.completed_depth = max(self.completed_depth, depth)
        return result


def selected_position(path: Path, color: chess.Color, fullmove: int) -> chess.Board:
    with path.open(encoding="utf-8") as handle:
        game = chess.pgn.read_game(handle)
    if game is None:
        raise ValueError(f"no game in {path}")
    board = game.board()
    for node in game.mainline():
        if board.turn == color and board.fullmove_number == fullmove:
            return board
        board.push(node.move)
    raise ValueError(f"position {fullmove} for {chess.COLOR_NAMES[color]} not found")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--color", required=True, choices=("white", "black"))
    parser.add_argument("--fullmove", required=True, type=int)
    parser.add_argument("--seconds", default="1,2,4,8,16")
    parser.add_argument("pgn", type=Path)
    args = parser.parse_args()

    color = chess.WHITE if args.color == "white" else chess.BLACK
    board = selected_position(args.pgn, color, args.fullmove)
    fallback = next(iter(board.legal_moves))
    print(board.fen())
    for seconds in (float(value) for value in args.seconds.split(",")):
        agent._eval_cache.clear()
        agent._transposition_table.clear()
        searcher = ProfilingSearcher(time.monotonic() + seconds)
        started = time.monotonic()
        move = searcher.best_move(board, fallback)
        elapsed = time.monotonic() - started
        print(
            f"limit={seconds:>5.1f}s elapsed={elapsed:>6.3f}s "
            f"depth={searcher.completed_depth} nodes={searcher.nodes} "
            f"qnodes={searcher.qnodes} move={move.uci()}"
        )


if __name__ == "__main__":
    main()
