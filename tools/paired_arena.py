"""Paired, varied-position matches without changing the official harness."""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import chess

from harness.referee import FAILED_TERMINATIONS, play_match
from harness.sandbox import local

OPENING_LINES = (
    ("e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6"),
    ("d2d4", "d7d5", "c2c4", "e7e6", "b1c3", "g8f6"),
    ("e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4", "g8f6", "b1c3"),
    ("c2c4", "e7e5", "b1c3", "g8f6", "g2g3", "d7d5", "c4d5", "f6d5", "f1g2"),
    ("g1f3", "d7d5", "g2g3", "c7c5", "f1g2", "b8c6", "e1g1", "e7e5", "d2d3"),
)

FIXED_FENS = (
    "r3k2r/p1ppqpb1/bn2pnp1/2pP4/1p2P3/2N2N2/PPQ1BPPP/R1B1K2R w KQkq - 0 1",
    "8/5pk1/6p1/3p4/3P4/5KP1/5P2/8 w - - 0 1",
    "8/8/8/3k4/8/3K4/4P3/8 w - - 0 1",
)


def positions(extra: int = 6, seed: int = 20260904) -> list[str]:
    """The fixed suite, plus ``extra`` random positions drawn from ``seed``.

    Rated games start from curated positions the platform does not publish, so a
    suite that is only openings measures the wrong thing. The random positions are
    the cheap stand-in; raising ``extra`` is how the sample size goes up.
    """
    result = [chess.STARTING_FEN]
    for line in OPENING_LINES:
        board = chess.Board()
        for uci in line:
            board.push_uci(uci)
        result.append(board.fen())
    result.extend(FIXED_FENS)

    rng = random.Random(seed)
    while len(result) < len(OPENING_LINES) + len(FIXED_FENS) + 1 + extra:
        board = chess.Board()
        for _ in range(rng.choice((12, 16, 20, 24, 28, 32, 36, 40, 44))):
            if board.is_game_over(claim_draw=True):
                break
            board.push(rng.choice(list(board.legal_moves)))
        if not board.is_game_over(claim_draw=True):
            result.append(board.fen())
    return result


def elo(score: float) -> float:
    """Convert a score fraction to an Elo difference, clamped at the extremes."""
    if score <= 0.001:
        return -800.0
    if score >= 0.999:
        return 800.0
    return -400.0 * math.log10(1.0 / score - 1.0)


def report(results: list[float]) -> str:
    """Score, 95% interval and Elo, so a run says whether it resolved anything."""
    total = len(results)
    score = sum(results) / total
    if total < 2:
        return f"score {score:.1%} over {total} games"
    mean = score
    variance = sum((value - mean) ** 2 for value in results) / (total - 1)
    error = 1.96 * math.sqrt(variance / total)
    low, high = max(0.0, score - error), min(1.0, score + error)
    verdict = (
        "stronger" if low > 0.5 else "weaker" if high < 0.5 else "NOT RESOLVED at this sample size"
    )
    return (
        f"score {score:.1%} +/- {error:.1%} over {total} games "
        f"(95% CI {low:.1%} to {high:.1%})\n"
        f"Elo {elo(score):+.0f} (95% CI {elo(low):+.0f} to {elo(high):+.0f})\n"
        f"verdict: candidate is {verdict}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, default=Path("."))
    parser.add_argument("--opponent", type=Path, required=True)
    parser.add_argument("--base-ms", type=int, default=1_000)
    parser.add_argument("--increment-ms", type=int, default=50)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--extra-positions",
        type=int,
        default=6,
        help="random positions added to the fixed suite; each is played twice",
    )
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument(
        "--pgn-dir",
        type=Path,
        help="write each game as a PGN here, for tools/blunder_audit.py",
    )
    arguments = parser.parse_args()

    if arguments.offset < 0:
        parser.error("--offset must be non-negative")
    if arguments.pgn_dir is not None:
        arguments.pgn_dir.mkdir(parents=True, exist_ok=True)

    candidate = arguments.candidate.resolve()
    opponent = arguments.opponent.resolve()
    if arguments.pgn_dir is not None:
        arguments.pgn_dir.mkdir(parents=True, exist_ok=True)
    suite = positions(arguments.extra_positions, arguments.seed)[arguments.offset :]
    suite = suite[: arguments.limit]
    if not suite:
        parser.error("--offset selects no positions")
    wins = draws = losses = 0
    results: list[float] = []
    failures: dict[str, int] = {}

    game_number = 0
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
            candidate_won = (outcome.result == "white") == candidate_is_white
            if outcome.result in {"draw", "void"}:
                draws += 1
                results.append(0.5)
                marker = "="
            elif candidate_won:
                wins += 1
                results.append(1.0)
                marker = "+"
            else:
                losses += 1
                results.append(0.0)
                marker = "-"
            if arguments.pgn_dir is not None:
                colour = "w" if candidate_is_white else "b"
                destination = arguments.pgn_dir / f"game{game_number:03d}{colour}.pgn"
                destination.write_text(outcome.pgn + "\n", encoding="utf-8")
            if outcome.termination in FAILED_TERMINATIONS:
                failures[outcome.termination] = failures.get(outcome.termination, 0) + 1
            if arguments.pgn_dir is not None:
                name = f"game-{game_number:02d}-{marker}.pgn"
                (arguments.pgn_dir / name).write_text(outcome.pgn, encoding="utf-8")
            print(
                f"game {game_number}/{len(suite) * 2}, position {position_number}, "
                f"candidate {'white' if candidate_is_white else 'black'}: "
                f"{marker} by {outcome.termination}"
            )

    print(f"\n+{wins} ={draws} -{losses}")
    print(report(results))
    if failures:
        raise SystemExit(
            "candidate failures: " + ", ".join(f"{key} {value}" for key, value in failures.items())
        )


if __name__ == "__main__":
    main()
