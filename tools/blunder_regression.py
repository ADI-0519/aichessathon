from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from types import ModuleType

import chess
import chess.engine
import chess.pgn

BLUNDERS = ((12, "Rxc3"), (13, "Nxd5"), (14, "Qc3"), (16, "Nxc4"), (17, "h5"), (20, "f6"))


def _load_agent(package: Path) -> ModuleType:
    sys.path.insert(0, str(package.resolve()))
    return importlib.import_module("agent")


def _positions(pgn: Path) -> dict[int, tuple[chess.Board, chess.Move]]:
    with pgn.open(encoding="utf-8") as handle:
        game = chess.pgn.read_game(handle)
    if game is None:
        raise SystemExit(f"no game in {pgn}")
    found: dict[int, tuple[chess.Board, chess.Move]] = {}
    wanted = {fullmove for fullmove, _ in BLUNDERS}
    board = game.board()
    for node in game.mainline():
        if board.turn == chess.BLACK and board.fullmove_number in wanted:
            found[board.fullmove_number] = (board.copy(stack=False), node.move)
        board.push(node.move)
    return found


def main() -> None:
    parser = argparse.ArgumentParser(
        description="replay the round-4 losing decisions and ask a build what it plays now"
    )
    parser.add_argument("--package", type=Path, default=Path("challengers/numba_v1"))
    parser.add_argument("--pgn", type=Path, default=Path("scratch/ladder/pgn/round4.pgn"))
    parser.add_argument("--time-left-ms", type=int, default=120_000)
    parser.add_argument("--engine", type=Path, help="score the choice against the blunder")
    parser.add_argument("--nodes", type=int, default=200_000)
    parser.add_argument("--label", default="")
    arguments = parser.parse_args()

    positions = _positions(arguments.pgn)
    agent = _load_agent(arguments.package)
    analyser = None
    if arguments.engine is not None:
        analyser = chess.engine.SimpleEngine.popen_uci(str(arguments.engine.resolve()))
        analyser.configure({"Threads": 1, "Hash": 128})

    label = arguments.label or arguments.package
    print(f"package={label} time_left_ms={arguments.time_left_ms} pgn={arguments.pgn}")
    header = f"{'move':>5}  {'played':<8}{'chosen':<8}{'repeats?':<10}"
    print(header + ("  played cp   chosen cp   verdict" if analyser else ""))

    repeated = 0
    improved = 0
    for fullmove, san in BLUNDERS:
        board, played = positions[fullmove]
        chosen = chess.Move.from_uci(agent.get_move(board.fen(), arguments.time_left_ms))
        if chosen not in board.legal_moves:
            raise SystemExit(f"illegal move {chosen} at move {fullmove}")
        repeats = chosen == played
        repeated += repeats
        row = (
            f"{fullmove:>5}  {san:<8}{board.san(chosen):<8}"
            f"{'REPEATED' if repeats else 'avoided':<10}"
        )
        if analyser is not None:
            scores = []
            for move in (played, chosen):
                board.push(move)
                # a fresh game token forces ucinewgame, no hash carries over
                info = analyser.analyse(
                    board, chess.engine.Limit(nodes=arguments.nodes), game=object()
                )
                scores.append(info["score"].pov(chess.BLACK).score(mate_score=10_000))
                board.pop()
            played_cp, chosen_cp = scores
            better = chosen_cp > played_cp and not repeats
            improved += better
            verdict = "same" if repeats else ("better" if better else "not better")
            row += f"  {played_cp:>9}   {chosen_cp:>9}   {verdict}"
        print(row)

    print(f"\nrepeated {repeated} of {len(BLUNDERS)} recorded blunders")
    if analyser is not None:
        print(f"strictly better than the played move in {improved} of {len(BLUNDERS)} positions")
        analyser.quit()


if __name__ == "__main__":
    main()
