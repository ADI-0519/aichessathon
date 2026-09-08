from __future__ import annotations

import argparse
import random
import statistics
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import chess
import chess.engine

ENGINE = Path("scratch/bin/stockfish/stockfish-ubuntu-x86-64-avx2")


def _score(argument: tuple[list[str], int, str]) -> list[int]:
    fens, nodes, engine_path = argument
    with chess.engine.SimpleEngine.popen_uci(engine_path) as sf:
        sf.configure({"Threads": 1, "Hash": 64})
        return [
            sf.analyse(chess.Board(fen), chess.engine.Limit(nodes=nodes))["score"]
            .white()
            .score(mate_score=30_000)
            for fen in fens
        ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="keep the positions stockfish agrees are close to level"
    )
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit-cp", type=int, default=50)
    parser.add_argument("--count", type=int, default=72)
    parser.add_argument("--nodes", type=int, default=2_000_000)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--exclude", type=Path, help="a suite whose positions must not recur")
    arguments = parser.parse_args()

    banned: set[str] = set()
    if arguments.exclude:
        banned = {
            line.strip() for line in arguments.exclude.read_text().splitlines() if line.strip()
        }

    fens = [line.strip() for line in arguments.candidates.read_text().splitlines() if line.strip()]
    size = (len(fens) + arguments.workers - 1) // arguments.workers
    chunks = [fens[start : start + size] for start in range(0, len(fens), size)]
    engine_path = str(ENGINE.resolve())
    with ProcessPoolExecutor(max_workers=arguments.workers) as pool:
        scores = [
            value
            for chunk in pool.map(_score, [(c, arguments.nodes, engine_path) for c in chunks])
            for value in chunk
        ]

    level = [
        (fen, score) for fen, score in zip(fens, scores, strict=True)
        if abs(score) <= arguments.limit_cp and fen not in banned
    ]
    print(f"{len(fens)} candidates scored, {len(level)} within {arguments.limit_cp} cp of level")
    if banned:
        print(f"  {len(banned)} positions excluded as already used")
    if len(level) < arguments.count:
        raise SystemExit(f"only {len(level)} level positions, need {arguments.count}")

    rng = random.Random(arguments.seed)
    chosen = rng.sample(level, arguments.count)
    chosen.sort(key=lambda row: row[0])
    arguments.out.write_text("\n".join(fen for fen, _ in chosen) + "\n")

    magnitudes = [abs(score) for _, score in chosen]
    white = sum(1 for fen, _ in chosen if fen.split()[1] == "w")
    print(f"\nwrote {len(chosen)} positions to {arguments.out}")
    print(f"  mean |eval|   {statistics.mean(magnitudes):.0f} cp")
    print(f"  median |eval| {statistics.median(magnitudes):.0f} cp")
    print(f"  max  |eval|   {max(magnitudes)} cp")
    print(f"  {white} with white to move, {len(chosen) - white} with black")


if __name__ == "__main__":
    main()
