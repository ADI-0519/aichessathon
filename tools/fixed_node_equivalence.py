from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from types import ModuleType

import chess

FIELDS = ("move", "score", "depth", "nodes", "qnodes", "tt_hits", "beta_cutoffs")


def _load(package: Path) -> tuple[ModuleType, ModuleType]:
    # each package ships its own engine+search, so drop the previous import
    for name in ("engine", "search", "nnue"):
        sys.modules.pop(name, None)
    sys.path.insert(0, str(package.resolve()))
    engine = importlib.import_module("engine")
    search = importlib.import_module("search")
    sys.path.pop(0)
    return engine, search


def _run(package: Path, fens: list[str], nodes: int) -> list[tuple[object, ...]]:
    engine, search = _load(package)
    search.warmup()
    memory = search.SearchMemory.create()
    results: list[tuple[object, ...]] = []
    for fen in fens:
        # shared table across positions would make the comparison order-dependent
        memory.clear()
        result = search.search_position(
            engine.position_from_board(chess.Board(fen)), memory, node_limit=nodes
        )
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
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="check two packages search an identical tree at a fixed node count"
    )
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--fens", type=Path, default=Path("tools/suites/level72.fens"))
    parser.add_argument("--positions", type=int, default=24)
    parser.add_argument("--nodes", type=int, default=150_000)
    arguments = parser.parse_args()

    fens = [
        line.split(";")[0].strip()
        for line in arguments.fens.read_text().splitlines()
        if line.strip()
    ][: arguments.positions]

    baseline = _run(arguments.baseline, fens, arguments.nodes)
    candidate = _run(arguments.candidate, fens, arguments.nodes)

    mismatches = 0
    for fen, left, right in zip(fens, baseline, candidate, strict=True):
        if left == right:
            continue
        mismatches += 1
        differing = [
            f"{field} {a} vs {b}" for field, a, b in zip(FIELDS, left, right, strict=True) if a != b
        ]
        print(f"MISMATCH {fen}\n  " + ", ".join(differing))

    print(f"{len(fens)} positions at {arguments.nodes:,} nodes, {mismatches} mismatches")
    if mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
