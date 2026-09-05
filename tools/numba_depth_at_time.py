from __future__ import annotations

import argparse
import importlib
import statistics
import sys
import time
from pathlib import Path
from types import ModuleType

import chess

# a spread of phases and branching factors, disjoint from the arena openings
POSITIONS = (
    ("start", chess.STARTING_FEN),
    ("kiwipete", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"),
    ("round4", "r2q1rk1/1Qp1bppp/2np4/p7/2BPn3/5N2/PP3PPP/R1B2RK1 b - - 0 12"),
    ("open-sicilian", "r1bqkb1r/pp2pppp/2np1n2/8/3NP3/2N5/PPP2PPP/R1BQKB1R w KQkq - 0 6"),
    ("closed-french", "r1bqk2r/pp1nbppp/4pn2/2ppP3/3P4/2N1BN2/PPP2PPP/R2QKB1R w KQkq - 0 8"),
    ("queenless", "r4rk1/pp3ppp/2n1p3/3p4/3P4/2N1P3/PP3PPP/R4RK1 w - - 0 14"),
    ("rook-endgame", "8/5pk1/6p1/3p4/3P4/5KP1/5P2/8 w - - 0 1"),
    ("sharp-tactic", "r1b1k2r/ppppqppp/2n2n2/2b5/2B1P3/2N2N2/PPPP1PPP/R1BQ1RK1 w kq - 6 7"),
)


def _load(package: Path) -> tuple[ModuleType, ModuleType]:
    # mirrors how the platform runner puts an agent directory on the path
    sys.path.insert(0, str(package.resolve()))
    return importlib.import_module("engine"), importlib.import_module("search")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="completed depth per move at a fixed wall-clock budget"
    )
    parser.add_argument("--package", type=Path, default=Path("challengers/numba_v1"))
    parser.add_argument(
        "--seconds",
        type=lambda text: [float(value) for value in text.split(",")],
        default=[0.5, 4.0],
        help="the agent spends about 0.5s per move at 10s+0.1s and 4.1s at 120s+0.5s",
    )
    parser.add_argument("--label", default="")
    arguments = parser.parse_args()

    engine, search = _load(arguments.package)
    started = time.perf_counter()
    search.warmup()
    warmup_s = time.perf_counter() - started
    print(f"package={arguments.label or arguments.package} warmup={warmup_s:.1f}s")

    for budget in arguments.seconds:
        depths: list[int] = []
        total_nodes = 0
        total_elapsed = 0.0
        print(f"\n--- {budget:g}s per move")
        print(f"{'position':<16}{'depth':>6}{'nodes':>12}{'nps':>10}{'used':>7}  move")
        for name, fen in POSITIONS:
            result = search.search_position(
                engine.position_from_board(chess.Board(fen)),
                search.SearchMemory.create(),
                time_limit_s=budget,
            )
            depths.append(result.depth)
            total_nodes += result.nodes
            total_elapsed += result.elapsed_s
            nps = result.nodes / result.elapsed_s if result.elapsed_s else 0.0
            print(
                f"{name:<16}{result.depth:>6}{result.nodes:>12,}{nps:>10,.0f}"
                f"{result.elapsed_s / budget:>6.0%}  "
                f"{engine.move_to_uci(result.move)}"
            )
        overall = total_nodes / total_elapsed if total_elapsed else 0.0
        print(
            f"mean depth {statistics.mean(depths):.2f}, total depth {sum(depths)}, "
            f"{overall:,.0f} nodes/s, "
            f"{total_elapsed / (budget * len(POSITIONS)):.0%} of budget used "
            f"over {len(POSITIONS)} positions"
        )


if __name__ == "__main__":
    main()
