from __future__ import annotations

import argparse
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from harness.referee import FAILED_TERMINATIONS

GAME_LINE = re.compile(
    r"^game \d+/\d+, position (\d+), candidate (white|black): ([+=-]) by (\S+)$"
)
POINTS = {"+": 1.0, "=": 0.5, "-": 0.0}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="aggregate fen_arena shard logs the way paired_arena_shards does"
    )
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--label", default="")
    arguments = parser.parse_args()

    markers: Counter[str] = Counter()
    terminations: Counter[str] = Counter()
    failures: Counter[str] = Counter()
    by_position: defaultdict[int, list[float]] = defaultdict(list)

    for log in sorted(arguments.log_dir.glob("shard-*.log")):
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
        raise SystemExit(f"no games parsed from {arguments.log_dir}")
    # both games of a pair share position, spread is across positions
    pairs = [statistics.mean(scores) for scores in by_position.values() if len(scores) == 2]
    score = (markers["+"] + markers["="] / 2) / games

    print(f"\n{arguments.label}")
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


if __name__ == "__main__":
    main()
