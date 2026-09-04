"""Analyze played moves in PGN files with a local UCI engine.

This is a development-only diagnostic. Nothing from the engine or this script is
included in the competition submission.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import chess
import chess.engine
import chess.pgn


def score_cp(score: chess.engine.PovScore, color: chess.Color) -> int:
    """Convert an engine score to centipawns from ``color``'s perspective."""
    value = score.pov(color).score(mate_score=100_000)
    if value is None:
        raise RuntimeError("engine returned an unscored position")
    return value


def analyze_game(
    engine: chess.engine.SimpleEngine,
    path: Path,
    nodes: int,
    focus: chess.Color | None,
) -> None:
    with path.open(encoding="utf-8") as handle:
        game = chess.pgn.read_game(handle)
    if game is None:
        raise ValueError(f"no PGN game found in {path}")

    board = game.board()
    print(f"\n===== {path.name} =====")
    print(f"start: {board.fen()}")
    focus_name = chess.COLOR_NAMES[focus] if focus is not None else "both"
    print(f"result: {game.headers.get('Result')}  focus: {focus_name}")
    print("ply move    played  best    before   after    loss  pv")

    previous_clock = {chess.WHITE: 120.0, chess.BLACK: 120.0}
    for ply, node in enumerate(game.mainline(), start=1):
        move = node.move
        mover = board.turn
        san = board.san(move)
        before = engine.analyse(board, chess.engine.Limit(nodes=nodes))
        best_move = before["pv"][0]
        before_cp = score_cp(before["score"], mover)

        board.push(move)
        after = engine.analyse(board, chess.engine.Limit(nodes=nodes))
        after_cp = score_cp(after["score"], mover)
        loss = max(0, before_cp - after_cp)

        clock = node.clock()
        think = "?"
        if clock is not None:
            spent = previous_clock[mover] + 0.5 - clock
            previous_clock[mover] = clock
            think = f"{max(0.0, spent):.2f}s"

        if focus is None or mover == focus or loss >= 100:
            pv_board = node.parent.board() if node.parent is not None else game.board()
            pv_text: list[str] = []
            for pv_move in before["pv"][:5]:
                if pv_move not in pv_board.legal_moves:
                    break
                pv_text.append(pv_board.san(pv_move))
                pv_board.push(pv_move)
            marker = "!!" if loss < 20 else "?!" if loss < 80 else "?" if loss < 200 else "??"
            print(
                f"{ply:>3} {san:<7} {move.uci():<7} {best_move.uci():<7} "
                f"{before_cp:>7} {after_cp:>7} {loss:>6} {marker:<2} "
                f"{think:>6}  {' '.join(pv_text)}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True, type=Path)
    parser.add_argument("--nodes", type=int, default=100_000)
    parser.add_argument(
        "--focus",
        choices=("white", "black", "both"),
        default="both",
        help="Print every move for this side; opponent blunders >= 100 cp are still shown.",
    )
    parser.add_argument("pgn", nargs="+", type=Path)
    args = parser.parse_args()

    focus = None
    if args.focus != "both":
        focus = args.focus == "white"

    engine = chess.engine.SimpleEngine.popen_uci(str(args.engine))
    try:
        engine.configure({"Threads": 1, "Hash": 128})
        for path in args.pgn:
            analyze_game(engine, path, args.nodes, focus)
    finally:
        engine.quit()


if __name__ == "__main__":
    main()
