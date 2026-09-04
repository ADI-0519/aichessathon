"""Paired, varied-position matches without changing the official harness."""

from __future__ import annotations

import argparse
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


def positions() -> list[str]:
    result = [chess.STARTING_FEN]
    for line in OPENING_LINES:
        board = chess.Board()
        for uci in line:
            board.push_uci(uci)
        result.append(board.fen())
    result.extend(FIXED_FENS)

    rng = random.Random(20260904)
    for target_plies in (14, 20, 26, 32, 38, 44):
        board = chess.Board()
        for _ in range(target_plies):
            if board.is_game_over(claim_draw=True):
                break
            board.push(rng.choice(list(board.legal_moves)))
        if not board.is_game_over(claim_draw=True):
            result.append(board.fen())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, default=Path("."))
    parser.add_argument("--opponent", type=Path, required=True)
    parser.add_argument("--base-ms", type=int, default=1_000)
    parser.add_argument("--increment-ms", type=int, default=50)
    parser.add_argument("--limit", type=int)
    arguments = parser.parse_args()

    candidate = arguments.candidate.resolve()
    opponent = arguments.opponent.resolve()
    suite = positions()[: arguments.limit]
    wins = draws = losses = 0
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
                marker = "="
            elif candidate_won:
                wins += 1
                marker = "+"
            else:
                losses += 1
                marker = "-"
            if outcome.termination in FAILED_TERMINATIONS:
                failures[outcome.termination] = failures.get(outcome.termination, 0) + 1
            print(
                f"game {game_number}/{len(suite) * 2}, position {position_number}, "
                f"candidate {'white' if candidate_is_white else 'black'}: "
                f"{marker} by {outcome.termination}"
            )

    total = wins + draws + losses
    score = (wins + draws / 2) / total
    print(f"\n+{wins} ={draws} -{losses}, score {score:.1%} over {total} paired games")
    if failures:
        raise SystemExit(
            "candidate failures: " + ", ".join(f"{key} {value}" for key, value in failures.items())
        )


if __name__ == "__main__":
    main()
