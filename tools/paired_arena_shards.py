from __future__ import annotations

import argparse
import math
import re
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

from harness.referee import FAILED_TERMINATIONS
from tools.paired_arena import positions

GAME_LINE = re.compile(
    r"^game \d+/\d+, position (\d+), candidate (white|black): ([+=-]) by (\S+)$"
)
POINTS = {"+": 1.0, "=": 0.5, "-": 0.0}


def _shard_bounds(total: int, shards: int) -> list[tuple[int, int]]:
    bounds = []
    start = 0
    for index in range(shards):
        size = total // shards + (1 if index < total % shards else 0)
        if size:
            bounds.append((start, size))
        start += size
    return bounds


def _aggregate(log_dir: Path, label: str) -> None:
    markers: Counter[str] = Counter()
    terminations: Counter[str] = Counter()
    failures: Counter[str] = Counter()
    by_position: defaultdict[int, list[float]] = defaultdict(list)

    for log in sorted(log_dir.glob("shard-*.log")):
        offset = int(log.stem.split("-")[1])
        for line in log.read_text().splitlines():
            match = GAME_LINE.match(line)
            if match is None:
                continue
            marker, termination = match.group(3), match.group(4)
            markers[marker] += 1
            terminations[termination] += 1
            if termination in FAILED_TERMINATIONS:
                failures[termination] += 1
            by_position[offset + int(match.group(1))].append(POINTS[marker])

    games = sum(markers.values())
    if not games:
        raise SystemExit(f"no games parsed from {log_dir}")
    # both games of a pair share a position, so spread is across positions
    pairs = [statistics.mean(scores) for scores in by_position.values() if len(scores) == 2]
    score = (markers["+"] + markers["="] / 2) / games

    print(f"\n{label}")
    print(
        f"+{markers['+']} ={markers['=']} -{markers['-']}, score {score:.1%} "
        f"over {games} paired games from {len(by_position)} positions"
    )
    if len(pairs) > 1:
        error = statistics.stdev(pairs) / math.sqrt(len(pairs))
        print(
            f"paired standard error {error:.1%}, "
            f"95% interval {score - 1.96 * error:.1%} to {score + 1.96 * error:.1%} "
            f"over {len(pairs)} complete pairs"
        )
    print("terminations: " + ", ".join(f"{k} {v}" for k, v in sorted(terminations.items())))
    if failures:
        print("FAILED terminations: " + ", ".join(f"{k} {v}" for k, v in failures.items()))
    else:
        print("FAILED terminations: none")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="run one paired arena as disjoint shards and aggregate the result"
    )
    parser.add_argument("--candidate", type=Path, default=Path("."))
    parser.add_argument("--opponent", type=Path)
    parser.add_argument("--base-ms", type=int, default=10_000)
    parser.add_argument("--increment-ms", type=int, default=100)
    parser.add_argument("--positions", type=int)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--no-curated", action="store_true")
    parser.add_argument("--shards", type=int, default=10)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument(
        "--module",
        default="tools.paired_arena",
        help="tools.stockfish_arena prints the same per-game lines",
    )
    parser.add_argument(
        "--extra",
        nargs=argparse.REMAINDER,
        default=[],
        help="everything after this goes to --module, so give it last",
    )
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="re-derive the score and spread from an existing --log-dir",
    )
    arguments = parser.parse_args()

    against = arguments.opponent if arguments.opponent else " ".join(arguments.extra)
    label = (
        f"candidate {arguments.candidate} vs {against} at "
        f"{arguments.base_ms}ms+{arguments.increment_ms}ms, seed {arguments.seed}"
    )
    if arguments.aggregate_only:
        _aggregate(arguments.log_dir, label)
        return

    if arguments.positions is None:
        parser.error("--positions is required unless --aggregate-only")
    if arguments.opponent is None and not arguments.extra:
        parser.error("give --opponent, or --extra for a module that names its own")
    if arguments.shards <= 0:
        parser.error("--shards must be positive")
    suite = positions(arguments.positions, arguments.seed, not arguments.no_curated)
    if len(suite) < arguments.positions:
        parser.error(f"suite yielded only {len(suite)} positions")

    arguments.log_dir.mkdir(parents=True, exist_ok=True)
    for stale in arguments.log_dir.glob("shard-*.log"):
        stale.unlink()

    shared = [
        "--candidate", str(arguments.candidate),
        "--base-ms", str(arguments.base_ms),
        "--increment-ms", str(arguments.increment_ms),
        "--positions", str(arguments.positions),
        "--seed", str(arguments.seed),
    ]
    if arguments.opponent is not None:
        shared += ["--opponent", str(arguments.opponent)]
    shared += arguments.extra
    if arguments.no_curated:
        shared.append("--no-curated")

    # Each shard writes straight into its own log rather than into a pipe the
    # parent drains at exit.  A long run is then inspectable while it is still
    # going, and a run that is killed keeps the games it already played.
    running = []
    handles = []
    for offset, limit in _shard_bounds(arguments.positions, arguments.shards):
        command = [sys.executable, "-m", arguments.module, *shared]
        command += ["--offset", str(offset), "--limit", str(limit)]
        handle = (arguments.log_dir / f"shard-{offset:03d}.log").open(
            "w", encoding="utf-8"
        )
        handles.append(handle)
        running.append((offset, subprocess.Popen(command, stdout=handle, text=True)))
    print(f"{len(running)} shards over {arguments.positions} positions", flush=True)

    try:
        for offset, process in running:
            process.wait()
            print(
                f"shard at position {offset} exited {process.returncode}", flush=True
            )
    finally:
        for handle in handles:
            handle.close()

    _aggregate(arguments.log_dir, label)


if __name__ == "__main__":
    main()
