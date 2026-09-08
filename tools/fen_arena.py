from __future__ import annotations

import argparse
from pathlib import Path

from harness.referee import FAILED_TERMINATIONS, play_match
from harness.sandbox import local


def main() -> None:
    parser = argparse.ArgumentParser(
        description="paired arena over an explicit position list, printing paired_arena's lines"
    )
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--opponent", type=Path, required=True)
    parser.add_argument("--fens", type=Path, required=True)
    parser.add_argument("--base-ms", type=int, default=10_000)
    parser.add_argument("--increment-ms", type=int, default=100)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--pgn-out", type=Path)
    arguments = parser.parse_args()

    suite = [
        line.strip() for line in arguments.fens.read_text().splitlines() if line.strip()
    ][arguments.offset :]
    if arguments.limit is not None:
        suite = suite[: arguments.limit]
    if not suite:
        parser.error("--offset selects no positions")

    candidate = arguments.candidate.resolve()
    opponent = arguments.opponent.resolve()
    handle = arguments.pgn_out.open("w", encoding="utf-8") if arguments.pgn_out else None

    game_number = 0
    failures: dict[str, int] = {}
    for position_number, fen in enumerate(suite, start=1):
        for candidate_is_white in (True, False):
            game_number += 1
            white = candidate if candidate_is_white else opponent
            black = opponent if candidate_is_white else candidate
            outcome = play_match(
                local(white),
                local(black),
                arguments.base_ms,
                arguments.increment_ms,
                start_fen=fen,
            )
            if outcome.result in {"draw", "void"}:
                marker = "="
            elif (outcome.result == "white") == candidate_is_white:
                marker = "+"
            else:
                marker = "-"
            if outcome.termination in FAILED_TERMINATIONS:
                failures[outcome.termination] = failures.get(outcome.termination, 0) + 1
            if handle is not None:
                handle.write(outcome.pgn + "\n\n")
                handle.flush()
            print(
                f"game {game_number}/{len(suite) * 2}, position {position_number}, "
                f"candidate {'white' if candidate_is_white else 'black'}: "
                f"{marker} by {outcome.termination}",
                flush=True,
            )
    if handle is not None:
        handle.close()
    if failures:
        raise SystemExit(
            "candidate failures: " + ", ".join(f"{k} {v}" for k, v in failures.items())
        )


if __name__ == "__main__":
    main()
