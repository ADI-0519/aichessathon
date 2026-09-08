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
from harness.sandbox import Agent, AgentFailure, local
from tools.paired_arena import positions

STOCKFISH_THREADS = 1
STOCKFISH_HASH_MB = 128


class StockfishAgent:
    """Adapter that makes a local UCI Stockfish process look like a harness Agent.

    The official harness suspends agents between turns. Stockfish is driven
    synchronously with ``ponder=False`` (python-chess's default), so once
    ``engine.play()`` returns it is idle waiting for the next UCI command.
    ``suspend()`` and ``resume()`` can therefore be no-ops here.
    """

    def __init__(self, executable: Path, nodes: int) -> None:
        self.executable = executable.resolve()
        self.nodes = nodes
        self.name = f"stockfish-{nodes}"
        self.stderr_log = ""
        self.engine: chess.engine.SimpleEngine | None = None

    def start(self, init_budget_s: float) -> None:
        """Start Stockfish and force the benchmark to one search thread."""
        if self.engine is not None:
            raise RuntimeError("Stockfish benchmark started twice")

        engine: chess.engine.SimpleEngine | None = None

        try:
            engine = chess.engine.SimpleEngine.popen_uci(
                [str(self.executable)],
                timeout=max(0.1, init_budget_s),
            )
            engine.configure(
                {
                    "Threads": STOCKFISH_THREADS,
                    "Hash": STOCKFISH_HASH_MB,
                }
            )
        except Exception as exc:
            self._append_log(
                f"Stockfish init failed: {type(exc).__name__}: {exc}"
            )

            if engine is not None:
                try:
                    engine.close()
                except Exception:
                    pass

            raise AgentFailure("init") from exc

        self.engine = engine

    def suspend(self) -> None:
        """No-op: fixed-node Stockfish does not ponder between ``play()`` calls."""

    def resume(self) -> None:
        """No-op counterpart required by the harness referee."""

    def move(self, fen: str, time_left_ms: int) -> str:
        """Return Stockfish's fixed-node move for ``fen``.

        ``time_left_ms`` belongs to the referee clock. Stockfish strength in this
        benchmark is intentionally controlled only by ``self.nodes`` so runs are
        comparable across candidate versions.
        """
        del time_left_ms

        if self.engine is None:
            raise AgentFailure("crash")

        # Parse outside the engine exception handler. The referee should only ever
        # send valid FENs; an invalid one is a tooling bug rather than a Stockfish
        # process failure.
        board = chess.Board(fen)

        try:
            result = self.engine.play(
                board,
                chess.engine.Limit(nodes=self.nodes),
                ponder=False,
            )
        except Exception as exc:
            self._append_log(
                f"Stockfish move failed: {type(exc).__name__}: {exc}"
            )
            raise AgentFailure("crash") from exc

        # The referee checks terminal positions before asking an agent to move.
        # Therefore a missing Stockfish move here is an engine/protocol failure,
        # not a legitimate "0000" response.
        if result.move is None:
            self._append_log(
                "Stockfish returned no move in a non-terminal referee position"
            )
            raise AgentFailure("crash")

        return result.move.uci()

    def stop(self) -> None:
        """Stop Stockfish without allowing cleanup errors to mask the game result."""
        engine = self.engine
        self.engine = None

        if engine is None:
            return

        try:
            engine.quit()
        except Exception as exc:
            self._append_log(
                f"Stockfish shutdown failed: {type(exc).__name__}: {exc}"
            )

            try:
                engine.close()
            except Exception:
                pass

    def _append_log(self, message: str) -> None:
        self.stderr_log = f"{self.stderr_log}\n{message}".strip()


def candidate_result(outcome: Outcome, candidate_is_white: bool) -> str:
    """Return +/=/- for scored games and ! for a void game."""
    if outcome.result == "void":
        return "!"

    if outcome.result == "draw":
        return "="

    won = (outcome.result == "white") == candidate_is_white
    return "+" if won else "-"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate",
        type=Path,
        default=Path("current"),
    )
    parser.add_argument(
        "--engine",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--nodes",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--base-ms",
        type=int,
        default=3_000,
    )
    parser.add_argument(
        "--increment-ms",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--limit",
        type=int,
    )
    parser.add_argument(
        "--pgn-dir",
        type=Path,
    )

    args = parser.parse_args()

    if args.nodes <= 0:
        parser.error("--nodes must be positive")

    if args.base_ms <= 0:
        parser.error("--base-ms must be positive")

    if args.increment_ms < 0:
        parser.error("--increment-ms must be non-negative")

    if args.offset < 0:
        parser.error("--offset must be non-negative")

    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")

    engine_path = args.engine.resolve()
    if not engine_path.is_file():
        parser.error(f"engine not found: {engine_path}")

    candidate = args.candidate.resolve()
    if not candidate.is_dir():
        parser.error(f"candidate directory not found: {candidate}")

    if not (candidate / "agent.py").is_file():
        parser.error(f"candidate has no agent.py: {candidate}")

    if args.pgn_dir is not None:
        args.pgn_dir.mkdir(parents=True, exist_ok=True)

    suite = positions()[args.offset :]

    if args.limit is not None:
        suite = suite[: args.limit]

    if not suite:
        parser.error("--offset selects no positions")

    totals = {
        "+": 0,
        "=": 0,
        "-": 0,
        "!": 0,
    }

    failures: dict[str, int] = {}
    game_number = 0

    for position_number, fen in enumerate(suite, start=1):
        for candidate_is_white in (True, False):
            game_number += 1

            stockfish = StockfishAgent(
                engine_path,
                args.nodes,
            )

            candidate_agent = local(
                candidate,
                seed=game_number,
            )

            # StockfishAgent deliberately implements the same runtime interface as
            # harness.sandbox.Agent, but it is not a subclass because it is an
            # in-process adapter around python-chess's UCI wrapper.
            benchmark_agent = cast(Agent, stockfish)

            white: Agent
            black: Agent

            if candidate_is_white:
                white = candidate_agent
                black = benchmark_agent
            else:
                white = benchmark_agent
                black = candidate_agent

            outcome = play_match(
                white,
                black,
                args.base_ms,
                args.increment_ms,
                start_fen=fen,
            )

            marker = candidate_result(
                outcome,
                candidate_is_white,
            )

            totals[marker] += 1

            if outcome.termination in FAILED_TERMINATIONS:
                failures[outcome.termination] = (
                    failures.get(outcome.termination, 0) + 1
                )

            if args.pgn_dir is not None:
                result_name = {
                    "+": "win",
                    "=": "draw",
                    "-": "loss",
                    "!": "void",
                }[marker]

                name = (
                    f"sf{args.nodes}-"
                    f"game-{game_number:03d}-"
                    f"{result_name}.pgn"
                )

                (args.pgn_dir / name).write_text(
                    outcome.pgn + "\n",
                    encoding="utf-8",
                )

            print(
                f"game {game_number}/{len(suite) * 2}, "
                f"position {position_number}, "
                f"candidate "
                f"{'white' if candidate_is_white else 'black'}: "
                f"{marker} by {outcome.termination}"
            )

    scored_games = (
        totals["+"]
        + totals["="]
        + totals["-"]
    )

    if scored_games:
        score = (
            totals["+"]
            + totals["="] / 2
        ) / scored_games

        print(
            f"\nStockfish {args.nodes} nodes/move: "
            f"+{totals['+']} "
            f"={totals['=']} "
            f"-{totals['-']}, "
            f"score {score:.1%} "
            f"over {scored_games} scored games"
        )
    else:
        print(
            f"\nStockfish {args.nodes} nodes/move: "
            "no scored games"
        )

    if totals["!"]:
        print(
            f"void games: {totals['!']}"
        )

    if failures:
        raise SystemExit(
            "benchmark run contained failed games: "
            + ", ".join(
                f"{reason} {count}"
                for reason, count in sorted(failures.items())
            )
        )


if __name__ == "__main__":
    main()