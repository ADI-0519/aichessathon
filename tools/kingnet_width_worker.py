"""Fresh-process worker for the V11-BIG width-cost benchmark."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

for _variable in (
    "MKL_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
):
    os.environ[_variable] = "1"


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return parsed


def _packed_move(engine: Any, position: Any) -> int:
    moves = engine.legal_moves(position)
    if len(moves) == 0:
        raise RuntimeError("benchmark position has no legal moves")
    return int(moves[0])


def run(arguments: argparse.Namespace) -> dict[str, Any]:
    engine_root = arguments.engine_root.resolve()
    if not (engine_root / "agent.py").is_file():
        raise ValueError(f"engine root has no agent.py: {engine_root}")
    sys.path.insert(0, str(engine_root))
    import_started = time.perf_counter()
    agent = importlib.import_module("agent")
    agent_import_s = time.perf_counter() - import_started

    engine = agent.engine
    search = agent.search
    nnue = search.nnue
    np = agent.np
    board = agent.chess.Board(arguments.fen)
    position = engine.position_from_board(board)
    accumulator = np.empty((2, nnue.ACCUMULATOR_ROW), dtype=np.int32)
    child = np.empty_like(accumulator)
    nnue.rebuild(position.pieces, accumulator)

    nnue.benchmark_evaluations(position.pieces, accumulator, 1)
    evaluation_started = time.perf_counter()
    evaluation_checksum = nnue.benchmark_evaluations(
        position.pieces,
        accumulator,
        arguments.evaluation_iterations,
    )
    evaluation_s = time.perf_counter() - evaluation_started

    move = _packed_move(engine, position)
    nnue.update_for_move(position.pieces, position.state, move, accumulator, child)
    update_started = time.perf_counter()
    for _ in range(arguments.update_iterations):
        nnue.update_for_move(position.pieces, position.state, move, accumulator, child)
    update_s = time.perf_counter() - update_started

    fixed = search.search_position(
        position,
        search.SearchMemory.create(),
        node_limit=arguments.nodes,
    )
    timed = search.search_position(
        position,
        search.SearchMemory.create(),
        time_limit_s=arguments.wall_time_s,
    )
    reported_warmup_s = float(agent._warmup_elapsed_s)
    return {
        "format_version": int(nnue.FORMAT_VERSION),
        "accumulator": int(nnue.ACCUMULATOR_SIZE),
        "pairwise_width": int(nnue.PAIRWISE_WIDTH),
        "agent_import_s": agent_import_s,
        "reported_jit_warmup_s": reported_warmup_s,
        "non_warmup_import_s": max(0.0, agent_import_s - reported_warmup_s),
        "total_init_s": agent_import_s,
        "evaluation_iterations": arguments.evaluation_iterations,
        "evaluation_checksum": int(evaluation_checksum),
        "evaluation_s": evaluation_s,
        "evaluations_per_s": arguments.evaluation_iterations / evaluation_s,
        "update_iterations": arguments.update_iterations,
        "update_s": update_s,
        "updates_per_s": arguments.update_iterations / update_s,
        "fixed_search": {
            "move": engine.move_to_uci(fixed.move),
            "score": int(fixed.score),
            "depth": int(fixed.depth),
            "nodes": int(fixed.nodes),
            "qnodes": int(fixed.qnodes),
            "elapsed_s": float(fixed.elapsed_s),
            "nps": fixed.nodes / fixed.elapsed_s,
        },
        "timed_search": {
            "move": engine.move_to_uci(timed.move),
            "score": int(timed.score),
            "completed_depth": int(timed.depth),
            "nodes": int(timed.nodes),
            "qnodes": int(timed.qnodes),
            "elapsed_s": float(timed.elapsed_s),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine-root", type=Path, required=True)
    parser.add_argument("--fen", required=True)
    parser.add_argument("--nodes", type=_positive_int, required=True)
    parser.add_argument("--evaluation-iterations", type=_positive_int, required=True)
    parser.add_argument("--update-iterations", type=_positive_int, required=True)
    parser.add_argument("--wall-time-s", type=float, required=True)
    arguments = parser.parse_args()
    if arguments.wall_time_s <= 0.0:
        parser.error("--wall-time-s must be positive")
    try:
        result = run(arguments)
    except (RuntimeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
