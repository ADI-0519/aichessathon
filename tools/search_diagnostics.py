"""Inspect root choices and principal variations in the compiled chess engine.

This development-only tool deliberately lives outside the packaged agent. It can
load any source-compatible engine directory containing ``engine.py`` and
``search.py``, making comparisons between frozen and experimental builds easy.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import chess
import numpy as np

from tools.cli import nonnegative_int, positive_int

REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_SUITE = REPOSITORY / "benchmarks" / "suites" / "v3_critical_positions.json"


@dataclass(frozen=True, slots=True)
class CriticalPosition:
    """A known position, the move V3 played, and an independently checked target."""

    identifier: str
    label: str
    fen: str
    played_move: str
    reference_move: str
    baseline_move: str
    diagnosis: str


@dataclass(frozen=True, slots=True)
class NodeProbe:
    """One completed iterative-deepening probe at a deterministic node limit."""

    node_limit: int
    move: str
    score: int
    depth: int
    nodes: int
    qnodes: int
    elapsed_s: float
    tt_hits: int
    tt_cutoffs: int
    lmr_reductions: int
    lmr_researches: int
    memory_mode: str


@dataclass(frozen=True, slots=True)
class RootLine:
    """A root move searched independently to an exact nominal depth."""

    move: str
    score: int | None
    static_score: int
    depth: int
    complete: bool
    nodes: int
    qnodes: int
    tt_hits: int
    tt_cutoffs: int
    lmr_reductions: int
    lmr_researches: int
    pv: tuple[str, ...]


def _module_from_path(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_engine_modules(root: Path) -> tuple[Any, Any]:
    """Load one engine/search pair without mixing it with an already imported build."""
    resolved = root.resolve()
    engine_path = resolved / "engine.py"
    nnue_path = resolved / "nnue.py"
    search_path = resolved / "search.py"
    for path in (engine_path, search_path):
        if not path.is_file():
            raise ValueError(f"engine root is missing {path.name}: {resolved}")

    token = hashlib.sha256(str(resolved).encode()).hexdigest()[:12]
    engine_module = _module_from_path(f"_diagnostic_engine_{token}", engine_path)

    # Production modules use top-level sibling imports because the submission
    # directory is placed first on sys.path. Supply the matching isolated
    # modules only while their dependants load; their global references remain
    # correct after the aliases are restored.
    previous_engine = sys.modules.get("engine")
    previous_nnue = sys.modules.get("nnue")
    sys.modules["engine"] = engine_module
    try:
        if nnue_path.is_file():
            nnue_module = _module_from_path(f"_diagnostic_nnue_{token}", nnue_path)
            sys.modules["nnue"] = nnue_module
        search_module = _module_from_path(f"_diagnostic_search_{token}", search_path)
    finally:
        if previous_engine is None:
            del sys.modules["engine"]
        else:
            sys.modules["engine"] = previous_engine
        if previous_nnue is None:
            sys.modules.pop("nnue", None)
        else:
            sys.modules["nnue"] = previous_nnue
    return cast(Any, engine_module), cast(Any, search_module)


def load_critical_positions(path: Path) -> list[CriticalPosition]:
    """Read and validate a critical-position suite."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError(f"unsupported critical-position schema in {path}")
    raw_positions = payload.get("positions")
    if not isinstance(raw_positions, list) or not raw_positions:
        raise ValueError(f"critical-position suite is empty: {path}")

    positions: list[CriticalPosition] = []
    seen: set[str] = set()
    required = {
        "id",
        "label",
        "fen",
        "played_move",
        "reference_move",
        "baseline_move",
        "diagnosis",
    }
    for index, raw in enumerate(raw_positions, start=1):
        if not isinstance(raw, dict) or not required.issubset(raw):
            missing = required - set(raw) if isinstance(raw, dict) else required
            raise ValueError(f"position {index} is missing fields: {sorted(missing)}")
        position = CriticalPosition(
            identifier=str(raw["id"]),
            label=str(raw["label"]),
            fen=str(raw["fen"]),
            played_move=str(raw["played_move"]),
            reference_move=str(raw["reference_move"]),
            baseline_move=str(raw["baseline_move"]),
            diagnosis=str(raw["diagnosis"]),
        )
        if position.identifier in seen:
            raise ValueError(f"duplicate critical-position id: {position.identifier}")
        seen.add(position.identifier)
        board = chess.Board(position.fen)
        legal = {move.uci() for move in board.legal_moves}
        for field, move in (
            ("played_move", position.played_move),
            ("reference_move", position.reference_move),
            ("baseline_move", position.baseline_move),
        ):
            if move not in legal:
                raise ValueError(f"{position.identifier} has illegal {field}: {move}")
        positions.append(position)
    return positions


def _parse_node_limits(value: str) -> list[int]:
    try:
        limits = [int(item.strip()) for item in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("nodes must be comma-separated integers") from error
    if not limits or any(limit <= 0 for limit in limits):
        raise argparse.ArgumentTypeError("nodes must be comma-separated positive integers")
    return limits


def probe_node_limits(
    engine_module: Any,
    search_module: Any,
    board: chess.Board,
    node_limits: list[int],
    max_depth: int,
    tt_bits: int,
    memory_mode: str = "fresh",
) -> list[NodeProbe]:
    """Run independent fixed-node searches so results are deterministic and comparable."""
    if memory_mode not in {"fresh", "persistent"}:
        raise ValueError("memory_mode must be 'fresh' or 'persistent'")
    probes: list[NodeProbe] = []
    position = engine_module.position_from_board(board)
    persistent_memory = search_module.SearchMemory.create(tt_bits)
    for node_limit in node_limits:
        memory = (
            search_module.SearchMemory.create(tt_bits)
            if memory_mode == "fresh"
            else persistent_memory
        )
        result = search_module.search_position(
            position,
            memory,
            node_limit=node_limit,
            max_depth=max_depth,
        )
        probes.append(
            NodeProbe(
                node_limit=node_limit,
                move=engine_module.move_to_uci(result.move),
                score=int(result.score),
                depth=int(result.depth),
                nodes=int(result.nodes),
                qnodes=int(result.qnodes),
                elapsed_s=float(result.elapsed_s),
                tt_hits=int(result.tt_hits),
                tt_cutoffs=int(result.tt_cutoffs),
                lmr_reductions=int(result.lmr_reductions),
                lmr_researches=int(result.lmr_researches),
                memory_mode=memory_mode,
            )
        )
    return probes


def _principal_variation(
    engine_module: Any,
    search_module: Any,
    position: Any,
    memory: Any,
    first_move: int,
    max_plies: int,
) -> tuple[str, ...]:
    """Follow legal transposition-table moves from a searched root move."""
    working = position.copy()
    move = first_move
    variation: list[str] = []
    seen_keys = {int(working.key[0])}
    for _ in range(max_plies):
        legal = {int(candidate) for candidate in engine_module.legal_moves(working)}
        if move not in legal:
            break
        variation.append(engine_module.move_to_uci(move))
        undo = np.empty(engine_module.UNDO_SIZE, dtype=np.int64)
        undo_key = np.empty(1, dtype=np.uint64)
        if not engine_module.make_move(
            working.pieces, working.state, working.key, move, undo, undo_key
        ):
            break
        current_key = int(working.key[0])
        if current_key in seen_keys:
            break
        seen_keys.add(current_key)
        tt_index = current_key & (len(memory.tt_keys) - 1)
        if (
            int(memory.tt_keys[tt_index]) != current_key
            or int(memory.tt_data[tt_index, search_module.TT_BOUND]) == search_module.TT_EMPTY
        ):
            break
        move = int(memory.tt_data[tt_index, search_module.TT_MOVE])
        if move == 0:
            break
    return tuple(variation)


def analyze_root_moves(
    engine_module: Any,
    search_module: Any,
    board: chess.Board,
    *,
    depth: int,
    node_limit: int = 0,
    tt_bits: int = 18,
    pv_plies: int = 12,
) -> list[RootLine]:
    """Score every legal root move independently at one nominal depth.

    Fresh search memory for every move prevents a previous candidate from
    affecting the next candidate's score. An optional node limit applies to
    each move separately; incomplete lines are retained and sorted last.
    """
    if depth <= 0:
        raise ValueError("depth must be positive")
    if depth > search_module.MAX_DEPTH:
        raise ValueError(f"depth cannot exceed {search_module.MAX_DEPTH}")
    if node_limit < 0:
        raise ValueError("node_limit cannot be negative")
    if pv_plies <= 0:
        raise ValueError("pv_plies must be positive")

    root = engine_module.position_from_board(board)
    root_moves = [int(move) for move in engine_module.legal_moves(root)]
    lines: list[RootLine] = []
    for root_move in root_moves:
        working = root.copy()
        history, history_count = search_module._history_buffer(working, None)
        root_history_count = history_count
        legal_stack = np.empty(
            (search_module.MAX_PLY, engine_module.MAX_MOVES), dtype=np.int32
        )
        pseudo_stack = np.empty_like(legal_stack)
        undo_stack = np.empty(
            (search_module.MAX_PLY, engine_module.UNDO_SIZE), dtype=np.int64
        )
        undo_key_stack = np.empty((search_module.MAX_PLY, 1), dtype=np.uint64)
        accumulator_stack = None
        if hasattr(search_module, "nnue"):
            accumulator_stack = np.empty(
                (
                    search_module.MAX_PLY,
                    2,
                    search_module.nnue.ACCUMULATOR_SIZE,
                ),
                dtype=np.int32,
            )
            search_module.nnue.rebuild(root.pieces, accumulator_stack[0])
        score_stack = np.empty_like(legal_stack)
        see_gain_stack = np.empty(
            (search_module.MAX_PLY, search_module.SEE_MAX_EXCHANGES), dtype=np.int32
        )
        killers = np.zeros((search_module.MAX_PLY, 2), dtype=np.int32)
        stop = np.zeros(1, dtype=np.uint8)
        stats = np.zeros(search_module.STAT_COUNT, dtype=np.int64)
        memory = search_module.SearchMemory.create(tt_bits)
        generation = memory.next_generation()

        if accumulator_stack is not None:
            search_module.nnue.update_for_move(
                working.pieces,
                working.state,
                root_move,
                accumulator_stack[0],
                accumulator_stack[1],
            )
        if not engine_module.make_move(
            working.pieces,
            working.state,
            working.key,
            root_move,
            undo_stack[0],
            undo_key_stack[0],
        ):
            raise RuntimeError(f"generated root move became illegal: {root_move}")
        if accumulator_stack is None:
            static_score = -int(search_module.evaluate(working.pieces, working.state))
        else:
            static_score = -int(
                search_module.evaluate(
                    working.pieces,
                    working.state,
                    accumulator_stack[1],
                )
            )
        history[history_count] = working.key[0]
        negamax_arguments = [
            working.pieces,
            working.state,
            working.key,
            depth - 1,
        ]
        if hasattr(search_module, "configure_experiment"):
            negamax_arguments.append(0)
        negamax_arguments.extend(
            [
                -search_module.INFINITY,
                search_module.INFINITY,
                1,
            ]
        )
        negamax_function = getattr(search_module._negamax, "py_func", search_module._negamax)
        if "allow_null" in inspect.signature(negamax_function).parameters:
            negamax_arguments.append(True)
        negamax_arguments.extend(
            [
                history,
                history_count + 1,
                root_history_count,
                legal_stack,
                pseudo_stack,
                undo_stack,
                undo_key_stack,
            ]
        )
        if accumulator_stack is not None:
            negamax_arguments.append(accumulator_stack)
        negamax_arguments.extend(
            [
                score_stack,
                see_gain_stack,
                killers,
                memory.quiet_history,
                memory.tt_keys,
                memory.tt_data,
                generation,
                stop,
                node_limit,
                stats,
            ]
        )
        child_score, aborted = search_module._negamax(*negamax_arguments)
        engine_module.unmake_move(
            working.pieces,
            working.state,
            working.key,
            root_move,
            undo_stack[0],
            undo_key_stack[0],
        )
        score = None if aborted else -int(child_score)
        pv = _principal_variation(
            engine_module, search_module, root, memory, root_move, pv_plies
        )
        lines.append(
            RootLine(
                move=engine_module.move_to_uci(root_move),
                score=score,
                static_score=static_score,
                depth=depth,
                complete=not aborted,
                nodes=int(stats[search_module.STAT_NODES]),
                qnodes=int(stats[search_module.STAT_QNODES]),
                tt_hits=int(stats[search_module.STAT_TT_HITS]),
                tt_cutoffs=int(stats[search_module.STAT_TT_CUTOFFS]),
                lmr_reductions=int(stats[search_module.STAT_LMR_REDUCTIONS]),
                lmr_researches=int(stats[search_module.STAT_LMR_RESEARCHES]),
                pv=pv,
            )
        )

    return sorted(
        lines,
        key=lambda line: (
            line.complete,
            line.score if line.score is not None else -search_module.INFINITY,
            line.move,
        ),
        reverse=True,
    )


def _case_payload(
    case: CriticalPosition,
    probes: list[NodeProbe],
    root_lines: list[RootLine],
) -> dict[str, object]:
    return {
        "case": asdict(case),
        "node_probes": [asdict(probe) for probe in probes],
        "root_lines": [asdict(line) for line in root_lines],
    }


def _print_case(
    case: CriticalPosition,
    probes: list[NodeProbe],
    root_lines: list[RootLine],
    top: int,
) -> None:
    board = chess.Board(case.fen)
    print(f"\n{case.identifier}: {case.label}")
    print(f"fen: {case.fen}")
    print(
        f"side: {chess.COLOR_NAMES[board.turn]}  played: {case.played_move}  "
        f"baseline: {case.baseline_move}  reference: {case.reference_move}"
    )
    print(f"diagnosis: {case.diagnosis}")
    print(
        "\nnode limit depth      nodes    q%  elapsed move     score  "
        "tt cut  lmr/re memory     target"
    )
    for probe in probes:
        qshare = probe.qnodes / probe.nodes if probe.nodes else 0.0
        target = "reference" if probe.move == case.reference_move else ""
        print(
            f"{probe.node_limit:>10,} {probe.depth:>5} {probe.nodes:>10,} "
            f"{qshare:>5.1%} {probe.elapsed_s:>7.3f}s {probe.move:<8} "
            f"{probe.score:>6} {probe.tt_cutoffs:>7,} "
            f"{probe.lmr_reductions:>4,}/{probe.lmr_researches:<3,} "
            f"{probe.memory_mode:<10} {target}"
        )

    if root_lines:
        print("\nrank move      score static      nodes    q%  tt cut  lmr/re pv")
    for rank, line in enumerate(root_lines[:top], start=1):
        qshare = line.qnodes / line.nodes if line.nodes else 0.0
        score = str(line.score) if line.score is not None else "incomplete"
        pv = " ".join(line.pv)
        print(
            f"{rank:>4} {line.move:<8} {score:>7} {line.static_score:>6} "
            f"{line.nodes:>10,} {qshare:>5.1%} {line.tt_cutoffs:>7,} "
            f"{line.lmr_reductions:>4,}/{line.lmr_researches:<3,} {pv}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--case", help="Critical-position id from --suite, or comma-separated ids"
    )
    source.add_argument("--all-cases", action="store_true")
    source.add_argument("--fen")
    parser.add_argument("--reference-move", help="Optional UCI target used with --fen")
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--engine-root", type=Path, default=REPOSITORY)
    parser.add_argument(
        "--profile",
        default="baseline",
        help="Compile-time profile when the selected engine supports experiments",
    )
    parser.add_argument("--memory-mode", choices=("fresh", "persistent"), default="fresh")
    parser.add_argument(
        "--nodes", type=_parse_node_limits, default=_parse_node_limits("25000,100000,300000")
    )
    parser.add_argument("--max-depth", type=positive_int, default=32)
    parser.add_argument("--root-depth", type=positive_int, default=5)
    parser.add_argument("--root-node-limit", type=nonnegative_int, default=0)
    parser.add_argument("--tt-bits", type=positive_int, default=18)
    parser.add_argument("--pv-plies", type=positive_int, default=12)
    parser.add_argument("--top", type=positive_int, default=8)
    parser.add_argument("--format", choices=("text", "json"), default="text")
    args = parser.parse_args()

    if args.fen is not None:
        board = chess.Board(args.fen)
        legal = {move.uci() for move in board.legal_moves}
        reference = args.reference_move or ""
        if reference and reference not in legal:
            parser.error(f"--reference-move is illegal in --fen: {reference}")
        cases = [
            CriticalPosition(
                identifier="custom",
                label="Custom FEN",
                fen=board.fen(),
                played_move="",
                reference_move=reference,
                baseline_move="",
                diagnosis="Ad hoc diagnostic position.",
            )
        ]
    else:
        cases = load_critical_positions(args.suite)
        if args.case is not None:
            requested = {identifier.strip() for identifier in args.case.split(",")}
            requested.discard("")
            selected = [case for case in cases if case.identifier in requested]
            missing = requested - {case.identifier for case in selected}
            if missing:
                parser.error(f"unknown case id(s): {', '.join(sorted(missing))}")
            cases = selected

    engine_module, search_module = load_engine_modules(args.engine_root)
    if hasattr(search_module, "configure_experiment"):
        try:
            search_module.configure_experiment(args.profile)
        except (RuntimeError, ValueError) as error:
            parser.error(str(error))
    elif args.profile != "baseline":
        parser.error("the selected engine supports only the baseline profile")
    warmup_started = time.perf_counter()
    search_module.warmup()
    warmup_s = time.perf_counter() - warmup_started
    payloads: list[dict[str, object]] = []
    if args.format == "text":
        print(f"engine root: {args.engine_root.resolve()}")
        print(f"profile: {args.profile}  memory: {args.memory_mode}")
        print(f"warmup: {warmup_s:.3f}s")

    for case in cases:
        board = chess.Board(case.fen)
        probes = probe_node_limits(
            engine_module,
            search_module,
            board,
            args.nodes,
            args.max_depth,
            args.tt_bits,
            args.memory_mode,
        )
        root_lines = analyze_root_moves(
            engine_module,
            search_module,
            board,
            depth=args.root_depth,
            node_limit=args.root_node_limit,
            tt_bits=args.tt_bits,
            pv_plies=args.pv_plies,
        )
        payloads.append(_case_payload(case, probes, root_lines))
        if args.format == "text":
            _print_case(case, probes, root_lines, args.top)

    if args.format == "json":
        print(
            json.dumps(
                {
                    "engine_root": str(args.engine_root.resolve()),
                    "profile": args.profile,
                    "memory_mode": args.memory_mode,
                    "warmup_s": warmup_s,
                    "positions": payloads,
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
