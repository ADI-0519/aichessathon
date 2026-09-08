"""Replay the mistakes we actually made in rated games and see if a build fixes them.

Every case is a position from a rated game where Stockfish says the move we
played lost at least a set number of centipawns, and names a better one. A build
is scored by how often it now plays Stockfish's move instead of ours.

**This suite is underpowered and cannot tune.** Scoring the chosen moves in
centipawns with Stockfish and comparing V7 against V8 over the 71 cases gave a
mean paired difference of 12.7 cp, a bootstrap 95% interval of -301.5 to +317.1,
V8 better on 30 positions and worse on 31, and a sign test at p = 0.60. Per
position loss runs from 0 to 2000 cp, so the paired differences have a standard
deviation near 1330 cp and resolving an effect this size would need tens of
thousands of positions. Use it to catch a build that has broken badly. Do not
use it to choose between builds, and do not read a few positions off it as
evidence -- paired games remain the only thing that settles a change here.

Scoring is by exact match against Stockfish's first choice, which is crude: a
move that gives up fifteen centipawns counts the same as one that gives up four
hundred, and both land in "other". Read the fixed/repeated split as a coarse
signal and score the chosen moves in centipawns when the answer matters. On the
first run of this suite the two builds differed by 1.4 points on "fixed" while
differing by 8.5 on "repeated", which the exact-match metric cannot interpret.

This is a screen, not a verdict. It is biased towards positions the current
engine gets wrong, so a build that fixes many of them has not thereby been shown
stronger overall -- only that it repairs known failures. Paired games still
decide promotion. What this buys is speed: minutes against the two hours an
arena costs, which makes it usable while tuning.

    uv run python -m tools.mistake_suite --engine challengers/v8_search
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPOSITORY = Path(__file__).resolve().parent.parent
DEFAULT_SUITE = REPOSITORY / "benchmarks" / "suites" / "sf_rated_mistakes.json"

_WORKER = '''
import json, sys, pathlib
sys.path.insert(0, sys.argv[1])
import agent
cases = json.loads(pathlib.Path(sys.argv[2]).read_text())["cases"]
budget = int(sys.argv[3])
rows = []
for case in cases:
    move = agent.get_move(case["fen"], budget)
    rows.append({
        "round": case["round"],
        "loss": case["loss"],
        "move": move,
        "outcome": (
            "fixed" if move == case["best"]
            else "repeated" if move == case["played"]
            else "other"
        ),
    })
print(json.dumps(rows))
'''


def run_suite(engine: Path, suite: Path, budget_ms: int) -> list[dict[str, Any]]:
    """Run one engine directory over the suite in its own process.

    A separate process keeps each build's compiled search and module globals off
    the next one, which matters because these are compile-time configurations.
    """
    completed = subprocess.run(
        [sys.executable, "-c", _WORKER, str(engine.resolve()), str(suite), str(budget_ms)],
        capture_output=True,
        text=True,
        cwd=REPOSITORY,
        check=False,
    )
    if completed.returncode != 0:
        raise SystemExit(f"{engine} failed:\n{completed.stderr[-2000:]}")
    return list(json.loads(completed.stdout.strip().splitlines()[-1]))


def report(engine: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Print and return the fixed/repeated split, including the severe cases."""
    total = len(rows)
    counts = {key: sum(1 for r in rows if r["outcome"] == key) for key in
              ("fixed", "repeated", "other")}
    severe = [r for r in rows if r["loss"] >= 300]
    severe_fixed = sum(1 for r in severe if r["outcome"] == "fixed")
    print(f"{engine}")
    print(f"  cases {total}")
    for key in ("fixed", "repeated", "other"):
        print(f"  {key:9} {counts[key]:3}  {counts[key] / max(total, 1):6.1%}")
    if severe:
        print(f"  of the {len(severe)} worst (>=300 cp): {severe_fixed} fixed "
              f"({severe_fixed / len(severe):.1%})")
    return {"engine": engine.as_posix(), "cases": total, **counts,
            "severe": len(severe), "severe_fixed": severe_fixed}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, action="append", required=True)
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--budget-ms", type=int, default=100_000)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if not arguments.suite.is_file():
        parser.error(f"suite not found: {arguments.suite}")

    summaries = []
    for engine in arguments.engine:
        rows = run_suite(engine, arguments.suite, arguments.budget_ms)
        summaries.append(report(engine, rows))
    if arguments.output:
        arguments.output.write_text(json.dumps(summaries, indent=2) + "\n")
        print(f"wrote {arguments.output}")


if __name__ == "__main__":
    main()
