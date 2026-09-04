"""Play paired games against a local fixed-node Stockfish benchmark.

This is development tooling only. The Stockfish executable and this adapter are
never included in ``submission.zip``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

import chess
import chess.engine

from harness.referee import FAILED_TERMINATIONS, Outcome, play_match
from harness.sandbox import Agent, local
from tools.paired_arena import positions


class StockfishAgent:
    """Small adapter implementing the referee's agent interface."""

    def __init__(self, executable: Path, nodes: int) -> None:
        self.executable = executable
        self.nodes = nodes
        self.engine: chess.engine.SimpleEngine | None = None
        self.stderr_tail = ""

    def start(self, init_budget_s: float) -> None:
        del init_budget_s
        self.engine = chess.engine.SimpleEngine.popen_uci(str(self.executable))
        self.engine.configure({"Threads": 1, "Hash": 128})

    def move(self, fen: str, time_left_ms: int) -> str:
        del time_left_ms
        if self.engine is None:
            raise RuntimeError("Stockfish benchmark moved before start")
        result = self.engine.play(chess.Board(fen), chess.engine.Limit(nodes=self.nodes))
        if result.move is None:
            return "0000"
        return result.move.uci()

    def stop(self) -> None:
        if self.engine is not None:
            self.engine.quit()
            self.engine = None


def candidate_result(outcome: Outcome, candidate_is_white: bool) -> str:
    if outcome.result in {"draw", "void"}:
        return "="
    won = (outcome.result == "white") == candidate_is_white
    return "+" if won else "-"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, default=Path("."))
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--nodes", type=int, required=True)
    parser.add_argument("--base-ms", type=int, default=3_000)
    parser.add_argument("--increment-ms", type=int, default=50)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--positions", type=int)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--no-curated", action="store_true")
    parser.add_argument("--pgn-dir", type=Path)
    args = parser.parse_args()

    if args.nodes <= 0:
        parser.error("--nodes must be positive")
    if args.offset < 0:
        parser.error("--offset must be non-negative")
    if not args.engine.is_file():
        parser.error(f"engine not found: {args.engine}")
    if args.pgn_dir is not None:
        args.pgn_dir.mkdir(parents=True, exist_ok=True)

    candidate = args.candidate.resolve()
    suite = positions(args.positions, args.seed, not args.no_curated)[args.offset :]
    suite = suite[: args.limit]
    if not suite:
        parser.error("--offset selects no positions")
    totals = {"+": 0, "=": 0, "-": 0}
    failures: dict[str, int] = {}
    game_number = 0

    for position_number, fen in enumerate(suite, start=1):
        for candidate_is_white in (True, False):
            game_number += 1
            stockfish = StockfishAgent(args.engine.resolve(), args.nodes)
            candidate_agent = local(candidate)
            benchmark_agent = cast(Agent, stockfish)
            white: Agent
            black: Agent
            if candidate_is_white:
                white, black = candidate_agent, benchmark_agent
            else:
                white, black = benchmark_agent, candidate_agent
            outcome = play_match(
                white,
                black,
                args.base_ms,
                args.increment_ms,
                start_fen=fen,
            )
            marker = candidate_result(outcome, candidate_is_white)
            totals[marker] += 1
            if outcome.termination in FAILED_TERMINATIONS:
                failures[outcome.termination] = failures.get(outcome.termination, 0) + 1
            if args.pgn_dir is not None:
                name = f"sf{args.nodes}-game-{game_number:02d}-{marker}.pgn"
                (args.pgn_dir / name).write_text(outcome.pgn, encoding="utf-8")
            print(
                f"game {game_number}/{len(suite) * 2}, position {position_number}, "
                f"candidate {'white' if candidate_is_white else 'black'}: "
                f"{marker} by {outcome.termination}"
            )

    games = sum(totals.values())
    score = (totals["+"] + totals["="] / 2) / games
    print(
        f"\nStockfish {args.nodes} nodes/move: "
        f"+{totals['+']} ={totals['=']} -{totals['-']}, score {score:.1%}"
    )
    if failures:
        raise SystemExit(
            "candidate failures: "
            + ", ".join(f"{reason} {count}" for reason, count in failures.items())
        )


if __name__ == "__main__":
    main()
