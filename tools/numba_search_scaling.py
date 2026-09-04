"""Measure deterministic fixed-node scaling for the compiled challenger."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import chess
import chess.pgn

from challengers.numba_v1 import engine, search


def selected_pgn_position(path: Path, color: chess.Color, fullmove: int) -> chess.Board:
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


def positive_int_list(text: str) -> list[int]:
    values = [int(value) for value in text.split(",")]
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--fen", default=chess.STARTING_FEN)
    source.add_argument("--pgn", type=Path)
    parser.add_argument("--color", choices=("white", "black"))
    parser.add_argument("--fullmove", type=int)
    parser.add_argument(
        "--nodes", type=positive_int_list, default=positive_int_list("10000,50000,200000")
    )
    parser.add_argument("--max-depth", type=int, default=search.MAX_DEPTH)
    args = parser.parse_args()

    if args.pgn is not None:
        if args.color is None or args.fullmove is None:
            parser.error("--pgn requires --color and --fullmove")
        color = chess.WHITE if args.color == "white" else chess.BLACK
        board = selected_pgn_position(args.pgn, color, args.fullmove)
    else:
        board = chess.Board(args.fen)

    compile_started = time.perf_counter()
    search.warmup()
    compile_s = time.perf_counter() - compile_started
    print(f"fen={board.fen(en_passant='fen')}")
    print(f"cold_warmup={compile_s:.3f}s")
    print("limit      d      nodes    q%    elapsed      nps move     score  tt-hit  lmr/re")
    for node_limit in args.nodes:
        result = search.search_position(
            engine.position_from_board(board),
            search.SearchMemory.create(),
            node_limit=node_limit,
            max_depth=args.max_depth,
        )
        nps = result.nodes / result.elapsed_s if result.elapsed_s else 0.0
        qshare = result.qnodes / result.nodes if result.nodes else 0.0
        print(
            f"{node_limit:>8,} {result.depth:>6} {result.nodes:>10,} {qshare:>5.1%} "
            f"{result.elapsed_s:>8.3f}s {nps:>8,.0f} "
            f"{engine.move_to_uci(result.move):<8} {result.score:>6} "
            f"{result.tt_hits:>7,} {result.lmr_reductions:>4,}/{result.lmr_researches:<3,}"
        )


if __name__ == "__main__":
    main()
