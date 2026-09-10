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
    clean_markers: Counter[str] = Counter()
    clean_by_position: defaultdict[int, list[float]] = defaultdict(list)

    for log in sorted(arguments.log_dir.glob("shard-*.log")):
        offset = int(log.stem.split("-")[1])
        for line in log.read_text().splitlines():
            match = GAME_LINE.match(line)
            if match is None:
                continue
            marker, termination = match.group(3), match.group(4)
            markers[marker] += 1
            terminations[termination] += 1
            position = offset + int(match.group(1))
            by_position[position].append(POINTS[marker])
            if termination in FAILED_TERMINATIONS:
                # marker tells which side the failure cost the game
                failures[f"{termination} scored {marker}"] += 1
                continue
            clean_markers[marker] += 1
            clean_by_position[position].append(POINTS[marker])

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
    if not failures:
        print("FAILED terminations: none")
        return

    print("FAILED terminations: " + ", ".join(f"{k} {v}" for k, v in failures.items()))
    # a game opponent lost to crash/flag isn't a game we won
    clean_games = sum(clean_markers.values())
    if not clean_games:
        return
    clean_pairs = [
        statistics.mean(scores) for scores in clean_by_position.values() if len(scores) == 2
    ]
    clean_score = (clean_markers["+"] + clean_markers["="] / 2) / clean_games
    print(
        f"excluding them: +{clean_markers['+']} ={clean_markers['=']} -{clean_markers['-']}, "
        f"score {clean_score:.1%} over {clean_games} games"
    )
    if len(clean_pairs) > 1:
        clean_error = statistics.stdev(clean_pairs) / math.sqrt(len(clean_pairs))
        print(
            f"  paired standard error {clean_error:.1%}, "
            f"95% interval {clean_score - 1.96 * clean_error:.1%} "
            f"to {clean_score + 1.96 * clean_error:.1%} "
            f"over {len(clean_pairs)} complete pairs"
        )


if __name__ == "__main__":
    main()
