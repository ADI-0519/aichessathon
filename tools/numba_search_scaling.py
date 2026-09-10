"""Measure repeatable fixed-node scaling for any compiled engine build."""

from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import chess
import chess.pgn

from tools.backtest_core import atomic_write_json, fingerprint_agent
from tools.cli import positive_int
from tools.search_diagnostics import load_critical_positions, load_engine_modules

REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_ENGINE_ROOT = REPOSITORY / "current"


@dataclass(frozen=True, slots=True)
class Probe:
    """One fresh-memory fixed-node search."""

    position_id: str
    node_limit: int
    repeat: int
    move: str
    score: int
    depth: int
    nodes: int
    qnodes: int
    elapsed_s: float
    nps: float
    tt_hits: int
    tt_cutoffs: int
    beta_cutoffs: int
    lmr_reductions: int
    lmr_researches: int
    q_eval_probes: int
    q_eval_hits: int
    q_moves_considered: int
    q_accumulator_updates: int
    q_pruned_after_update: int
    q_check_saves: int
    q_children_searched: int
    qtt_probes: int
    qtt_hits: int
    qtt_cutoffs: int
    qtt_stores: int


def selected_pgn_position(path: Path, color: chess.Color, fullmove: int) -> chess.Board:
    with path.open(encoding="utf-8") as handle:
        game = chess.pgn.read_game(handle)
    if game is None:
        raise ValueError(f"no game in {path}")
    board = game.board()
    for node in game.mainline():
        if board.turn == color and board.fullmove_number == fullmove:
            return board
        board.push(node.move)
    raise ValueError(f"position {fullmove} for {chess.COLOR_NAMES[color]} not found")


def positive_int_list(text: str) -> list[int]:
    try:
        values = [int(value) for value in text.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return values


def run_probe(
    engine: Any,
    search: Any,
    board: chess.Board,
    *,
    position_id: str,
    node_limit: int,
    repeat: int,
    max_depth: int,
) -> Probe:
    result = search.search_position(
        engine.position_from_board(board),
        search.SearchMemory.create(),
        node_limit=node_limit,
        max_depth=max_depth,
    )
    nps = result.nodes / result.elapsed_s if result.elapsed_s else 0.0
    return Probe(
        position_id=position_id,
        node_limit=node_limit,
        repeat=repeat,
        move=engine.move_to_uci(result.move),
        score=result.score,
        depth=result.depth,
        nodes=result.nodes,
        qnodes=result.qnodes,
        elapsed_s=result.elapsed_s,
        nps=nps,
        tt_hits=result.tt_hits,
        tt_cutoffs=result.tt_cutoffs,
        beta_cutoffs=result.beta_cutoffs,
        lmr_reductions=result.lmr_reductions,
        lmr_researches=result.lmr_researches,
        q_eval_probes=getattr(result, "q_eval_probes", 0),
        q_eval_hits=getattr(result, "q_eval_hits", 0),
        q_moves_considered=getattr(result, "q_moves_considered", 0),
        q_accumulator_updates=getattr(result, "q_accumulator_updates", 0),
        q_pruned_after_update=getattr(result, "q_pruned_after_update", 0),
        q_check_saves=getattr(result, "q_check_saves", 0),
        q_children_searched=getattr(result, "q_children_searched", 0),
        qtt_probes=getattr(result, "qtt_probes", 0),
        qtt_hits=getattr(result, "qtt_hits", 0),
        qtt_cutoffs=getattr(result, "qtt_cutoffs", 0),
        qtt_stores=getattr(result, "qtt_stores", 0),
    )


def assert_deterministic(probes: list[Probe]) -> None:
    """Reject comparisons whose fixed-node chess result changes between repeats."""
    signatures = {
        (
            probe.move,
            probe.score,
            probe.depth,
            probe.nodes,
            probe.qnodes,
            probe.tt_hits,
            probe.tt_cutoffs,
            probe.beta_cutoffs,
            probe.lmr_reductions,
            probe.lmr_researches,
            probe.q_eval_probes,
            probe.q_eval_hits,
            probe.q_moves_considered,
            probe.q_accumulator_updates,
            probe.q_pruned_after_update,
            probe.q_check_saves,
            probe.q_children_searched,
            probe.qtt_probes,
            probe.qtt_hits,
            probe.qtt_cutoffs,
            probe.qtt_stores,
        )
        for probe in probes
    }
    if len(signatures) != 1:
        raise RuntimeError(
            f"fixed-node search was not deterministic at limit {probes[0].node_limit}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine-root", type=Path, default=DEFAULT_ENGINE_ROOT)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--fen", default=chess.STARTING_FEN)
    source.add_argument("--pgn", type=Path)
    source.add_argument("--suite", type=Path, help="critical-position JSON suite")
    parser.add_argument("--color", choices=("white", "black"))
    parser.add_argument("--fullmove", type=positive_int)
    parser.add_argument(
        "--nodes",
        type=positive_int_list,
        default=positive_int_list("10000,50000,200000"),
    )
    parser.add_argument("--repeats", type=positive_int, default=3)
    parser.add_argument("--max-depth", type=positive_int)
    parser.add_argument("--output", type=Path, help="optional JSON report path")
    return parser


def main() -> None:
    parser = build_parser()
    arguments = parser.parse_args()
    positions: list[tuple[str, chess.Board]]
    if arguments.suite is not None:
        if arguments.color is not None or arguments.fullmove is not None:
            parser.error("--color and --fullmove require --pgn")
        positions = [
            (case.identifier, chess.Board(case.fen))
            for case in load_critical_positions(arguments.suite)
        ]
    elif arguments.pgn is not None:
        if arguments.color is None or arguments.fullmove is None:
            parser.error("--pgn requires --color and --fullmove")
        color = chess.WHITE if arguments.color == "white" else chess.BLACK
        positions = [
            (
                f"{arguments.pgn.name}:{arguments.color}:{arguments.fullmove}",
                selected_pgn_position(arguments.pgn, color, arguments.fullmove),
            )
        ]
    else:
        if arguments.color is not None or arguments.fullmove is not None:
            parser.error("--color and --fullmove require --pgn")
        positions = [("fen", chess.Board(arguments.fen))]

    engine_root = arguments.engine_root.resolve()
    engine, search = load_engine_modules(engine_root)
    max_depth = arguments.max_depth or int(search.MAX_DEPTH)

    compile_started = time.perf_counter()
    search.warmup()
    warmup_s = time.perf_counter() - compile_started
    print(f"engine={engine_root}")
    print(f"positions={len(positions)}")
    print(f"cold_warmup={warmup_s:.3f}s, repeats={arguments.repeats}")
    print(
        "limit      d      nodes    q% median nps      range move     score "
        "qeval-hit qacc-waste"
    )

    all_probes: list[Probe] = []
    aggregates: list[dict[str, object]] = []
    for position_id, board in positions:
        print(f"\n[{position_id}] {board.fen(en_passant='fen')}")
        for node_limit in arguments.nodes:
            probes = [
                run_probe(
                    engine,
                    search,
                    board,
                    position_id=position_id,
                    node_limit=node_limit,
                    repeat=repeat,
                    max_depth=max_depth,
                )
                for repeat in range(1, arguments.repeats + 1)
            ]
            assert_deterministic(probes)
            all_probes.extend(probes)
            first = probes[0]
            nps_values = [probe.nps for probe in probes]
            median_nps = statistics.median(nps_values)
            qshare = first.qnodes / first.nodes if first.nodes else 0.0
            q_eval_hit_rate = (
                first.q_eval_hits / first.q_eval_probes if first.q_eval_probes else None
            )
            qtt_hit_rate = (
                first.qtt_hits / first.qtt_probes if first.qtt_probes else None
            )
            qtt_cutoff_rate = (
                first.qtt_cutoffs / first.qtt_probes if first.qtt_probes else None
            )
            q_accumulator_waste_rate = (
                first.q_pruned_after_update / first.q_accumulator_updates
                if first.q_accumulator_updates
                else None
            )
            if first.q_accumulator_updates and (
                first.q_accumulator_updates != first.q_moves_considered
                or first.q_pruned_after_update + first.q_children_searched
                != first.q_moves_considered
            ):
                raise RuntimeError("inconsistent qsearch accumulator profile counters")
            aggregates.append(
                {
                    "position_id": position_id,
                    "node_limit": node_limit,
                    "move": first.move,
                    "score": first.score,
                    "depth": first.depth,
                    "nodes": first.nodes,
                    "qnodes": first.qnodes,
                    "qshare": qshare,
                    "median_nps": median_nps,
                    "minimum_nps": min(nps_values),
                    "maximum_nps": max(nps_values),
                    "tt_hits": first.tt_hits,
                    "tt_cutoffs": first.tt_cutoffs,
                    "beta_cutoffs": first.beta_cutoffs,
                    "lmr_reductions": first.lmr_reductions,
                    "lmr_researches": first.lmr_researches,
                    "q_eval_probes": first.q_eval_probes,
                    "q_eval_hits": first.q_eval_hits,
                    "q_eval_hit_rate": q_eval_hit_rate,
                    "qtt_probes": first.qtt_probes,
                    "qtt_hits": first.qtt_hits,
                    "qtt_hit_rate": qtt_hit_rate,
                    "qtt_cutoffs": first.qtt_cutoffs,
                    "qtt_cutoff_rate": qtt_cutoff_rate,
                    "qtt_stores": first.qtt_stores,
                    "q_moves_considered": first.q_moves_considered,
                    "q_accumulator_updates": first.q_accumulator_updates,
                    "q_pruned_after_update": first.q_pruned_after_update,
                    "q_check_saves": first.q_check_saves,
                    "q_children_searched": first.q_children_searched,
                    "q_accumulator_waste_rate": q_accumulator_waste_rate,
                }
            )
            hit_text = (
                f"{q_eval_hit_rate:>8.1%}" if q_eval_hit_rate is not None else "       -"
            )
            waste_text = (
                f"{q_accumulator_waste_rate:>10.1%}"
                if q_accumulator_waste_rate is not None
                else "         -"
            )
            print(
                f"{node_limit:>8,} {first.depth:>6} {first.nodes:>10,} {qshare:>5.1%} "
                f"{median_nps:>10,.0f} {min(nps_values):>8,.0f}..{max(nps_values):<8,.0f} "
                f"{first.move:<8} {first.score:>6} {hit_text} {waste_text}"
            )

    overall: list[dict[str, object]] = []
    for node_limit in arguments.nodes:
        repeat_rates: list[float] = []
        for repeat in range(1, arguments.repeats + 1):
            selected = [
                probe
                for probe in all_probes
                if probe.node_limit == node_limit and probe.repeat == repeat
            ]
            total_elapsed = sum(probe.elapsed_s for probe in selected)
            repeat_rates.append(sum(probe.nodes for probe in selected) / total_elapsed)
        overall.append(
            {
                "node_limit": node_limit,
                "median_nps": statistics.median(repeat_rates),
                "minimum_nps": min(repeat_rates),
                "maximum_nps": max(repeat_rates),
            }
        )

    if arguments.output is not None:
        report = {
            "schema_version": 1,
            "engine": fingerprint_agent(engine_root),
            "configuration": {
                "positions": [
                    {"id": position_id, "fen": board.fen(en_passant="fen")}
                    for position_id, board in positions
                ],
                "node_limits": arguments.nodes,
                "repeats": arguments.repeats,
                "max_depth": max_depth,
            },
            "warmup_s": warmup_s,
            "aggregates": aggregates,
            "overall": overall,
            "probes": [asdict(probe) for probe in all_probes],
        }
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(arguments.output, report)
        print(f"report={arguments.output.resolve()}")


if __name__ == "__main__":
    main()
