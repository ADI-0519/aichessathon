"""Check that two engine packages search an identical tree at a fixed node count.

Adapted from the dev branch (e6b1c3b).  A change that only reorders *work* --
deferring move generation past cutoffs that do not need the moves, say --
must reproduce the baseline's move, score, depth, node, qnode, TT-hit and
beta-cutoff counts exactly at a fixed node budget.  Any mismatch means it
changed the search, not just its cost.  Wall-clock time per package is
reported too, as a first read on speed; it is load-sensitive, so re-measure
interleaved before quoting it.

    uv run python -m tools.fixed_node_equivalence \
        --baseline challengers/exp_v14_bignet --candidate <package> \
        --fens benchmarks/suites/balanced_openings_v1.epd --nodes 150000

Positions are read with tools.paired_arena.load_suite (FEN or EPD, '#' comments).
Exits non-zero on any mismatch.
"""

from __future__ import annotations

import argparse
import importlib
import sys
import time
from pathlib import Path
from types import ModuleType

import chess

from tools.paired_arena import load_suite

FIELDS = ("move", "score", "depth", "nodes", "qnodes", "tt_hits", "beta_cutoffs")


def _load(package: Path) -> tuple[ModuleType, ModuleType]:
    # each package ships its own engine+search+nnue, so drop the previous import
    for name in ("engine", "search", "nnue", "time_manager"):
        sys.modules.pop(name, None)
    sys.path.insert(0, str(package.resolve()))
    engine = importlib.import_module("engine")
    search = importlib.import_module("search")
    sys.path.pop(0)
    return engine, search


def _run(package: Path, fens: list[str], nodes: int) -> tuple[list[tuple[object, ...]], float]:
    engine, search = _load(package)
    search.warmup()
    memory = search.SearchMemory.create()
    results: list[tuple[object, ...]] = []
    elapsed = 0.0
    for fen in fens:
        # a table shared across positions would make the comparison order-dependent
        memory.clear()
        position = engine.position_from_board(chess.Board(fen))
        started = time.perf_counter()
        result = search.search_position(position, memory, node_limit=nodes)
        elapsed += time.perf_counter() - started
        results.append(
            (
                engine.move_to_uci(result.move),
                result.score,
                result.depth,
                result.nodes,
                result.qnodes,
                result.tt_hits,
                result.beta_cutoffs,
            )
        )
    return results, elapsed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="check two packages search an identical tree at a fixed node count"
    )
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument(
        "--fens", type=Path, default=Path("benchmarks/suites/balanced_openings_v1.epd")
    )
    parser.add_argument("--positions", type=int, default=24)
    parser.add_argument("--nodes", type=int, default=150_000)
    arguments = parser.parse_args()

    fens = load_suite(arguments.fens)[: arguments.positions]
    baseline, baseline_s = _run(arguments.baseline, fens, arguments.nodes)
    candidate, candidate_s = _run(arguments.candidate, fens, arguments.nodes)

    mismatches = 0
    for fen, left, right in zip(fens, baseline, candidate, strict=True):
        if left == right:
            continue
        mismatches += 1
        differing = [
            f"{field} {a} vs {b}" for field, a, b in zip(FIELDS, left, right, strict=True) if a != b
        ]
        print(f"MISMATCH {fen}\n  " + ", ".join(differing))

    total_nodes = sum(int(row[3]) for row in baseline)
    print(f"{len(fens)} positions at {arguments.nodes:,} nodes, {mismatches} mismatches")
    print(
        f"search time: baseline {baseline_s:.2f}s, candidate {candidate_s:.2f}s "
        f"({baseline_s / candidate_s:.3f}x), {total_nodes:,} baseline nodes"
    )
    if mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
