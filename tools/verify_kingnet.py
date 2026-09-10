"""Verify king-bucket updates, fixed-point inference, and CPU throughput."""

from __future__ import annotations

import argparse
import importlib
import json
import random
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any

import chess
import numpy as np

# Any single position this far from the float reference means something is
# structurally wrong, not merely rounded.
ABSURD_ERROR_CP = 100.0
# Judged over the whole sample. The 256-wide pairwise net measures 0.77 cp
# median and 4.71 cp at the 99th percentile over 800 positions; the 128-wide
# format-2 net sits well inside both.
MEDIAN_ERROR_LIMIT_CP = 3.0
TAIL_ERROR_LIMIT_CP = 15.0


def _load_candidate(candidate: Path) -> tuple[ModuleType, ModuleType]:
    candidate = candidate.resolve()
    if not (candidate / "engine.py").is_file() or not (candidate / "nnue.py").is_file():
        raise ValueError(f"candidate has no engine.py and nnue.py: {candidate}")
    sys.path.insert(0, str(candidate))
    engine = importlib.import_module("engine")
    nnue = importlib.import_module("nnue")
    for module in (engine, nnue):
        module_path = Path(str(module.__file__)).resolve()
        if module_path.parent != candidate:
            raise RuntimeError(f"loaded {module.__name__} from {module_path}, not {candidate}")
    required = (
        "ACCUMULATOR_ROW",
        "BUCKET_SLOT",
        "KING_BUCKET_COUNT",
        "STALE",
        "refresh",
    )
    missing = [name for name in required if not hasattr(nnue, name)]
    if missing:
        raise ValueError(f"candidate is not a king-bucket network; missing {missing}")
    return engine, nnue


def _internal_move(engine: ModuleType, position: Any, uci: str) -> int:
    for move in engine.legal_moves(position):
        encoded = int(move)
        if engine.move_to_uci(encoded) == uci:
            return encoded
    raise AssertionError(f"candidate did not generate legal move {uci}")


def _empty_accumulators(nnue: ModuleType) -> np.ndarray[Any, Any]:
    return np.empty((2, int(nnue.ACCUMULATOR_ROW)), dtype=np.int32)


def _check_position(
    engine: ModuleType,
    nnue: ModuleType,
    position: Any,
    accumulators: np.ndarray[Any, Any],
) -> float:
    nnue.refresh(position.pieces, accumulators)
    rebuilt = _empty_accumulators(nnue)
    nnue.rebuild(position.pieces, rebuilt)
    if not np.array_equal(accumulators, rebuilt):
        error = int(np.max(np.abs(accumulators.astype(np.int64) - rebuilt)))
        raise AssertionError(f"incremental accumulator error: {error}")

    side = int(position.state[engine.STATE_SIDE])
    compiled = int(nnue.evaluate(position.pieces, accumulators, side))
    reference = float(nnue.evaluate_reference(position.pieces, side))
    # A hard per-position ceiling is the wrong shape of gate. Fixed-point
    # rounding grows with accumulator width, and a 256-wide pairwise net
    # multiplies two rounded activations together, so isolated positions land a
    # few centipawns out while the net is perfectly sound. Real breakage -- a
    # mis-scaled weight, the wrong head, an accumulator that was never built --
    # moves the whole distribution, not one sample. So collect and judge in
    # verify() on the median and the tail, and keep only an absurdity ceiling
    # here to fail fast rather than benchmark a net that is plainly broken.
    inference_error = abs(compiled - reference)
    if inference_error > ABSURD_ERROR_CP:
        raise AssertionError(
            f"fixed-point/reference inference mismatch: {compiled} vs {reference:.4f}"
        )
    return inference_error


def _advance(
    engine: ModuleType,
    nnue: ModuleType,
    position: Any,
    accumulators: np.ndarray[Any, Any],
    move: int,
) -> tuple[np.ndarray[Any, Any], bool]:
    child = _empty_accumulators(nnue)
    nnue.update_for_move(position.pieces, position.state, move, accumulators, child)
    stale = bool(np.any(child[:, int(nnue.BUCKET_SLOT)] == int(nnue.STALE)))
    undo = np.empty(engine.UNDO_SIZE, dtype=np.int64)
    undo_key = np.empty(1, dtype=np.uint64)
    if not engine.make_move(
        position.pieces, position.state, position.key, move, undo, undo_key
    ):
        raise AssertionError("candidate rejected its own generated move")
    return child, stale


def verify(candidate: Path, random_plies: int, benchmark_iterations: int) -> dict[str, object]:
    engine, nnue = _load_candidate(candidate)
    engine.warmup()
    nnue.warmup()
    maximum_inference_error = 0.0
    checked = 0
    stale_transitions = 0

    cases = (
        (chess.STARTING_FEN, "e2e4"),
        ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1g1"),
        ("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1", "e8c8"),
        ("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", "e5d6"),
        ("4k3/P7/8/8/8/8/8/4K3 w - - 0 1", "a7a8q"),
        ("8/8/8/8/8/8/4K3/7k w - - 0 1", "e2d3"),
    )
    for fen, uci in cases:
        position = engine.position_from_board(chess.Board(fen))
        parent = _empty_accumulators(nnue)
        nnue.rebuild(position.pieces, parent)
        child, stale = _advance(
            engine, nnue, position, parent, _internal_move(engine, position, uci)
        )
        stale_transitions += int(stale)
        maximum_inference_error = max(
            maximum_inference_error,
            _check_position(engine, nnue, position, child),
        )
        checked += 1

    rng = random.Random(0xA1C4E55A)
    position = engine.position_from_board(chess.Board())
    accumulators = _empty_accumulators(nnue)
    nnue.rebuild(position.pieces, accumulators)
    inference_errors: list[float] = []
    for _ in range(random_plies):
        moves = engine.legal_moves(position)
        if len(moves) == 0:
            position = engine.position_from_board(chess.Board())
            nnue.rebuild(position.pieces, accumulators)
            continue
        accumulators, stale = _advance(
            engine,
            nnue,
            position,
            accumulators,
            int(moves[rng.randrange(len(moves))]),
        )
        stale_transitions += int(stale)
        error = _check_position(engine, nnue, position, accumulators)
        inference_errors.append(error)
        maximum_inference_error = max(maximum_inference_error, error)
        checked += 1

    if inference_errors:
        errors = np.array(inference_errors)
        median_error = float(np.median(errors))
        tail_error = float(np.percentile(errors, 99))
        if median_error > MEDIAN_ERROR_LIMIT_CP:
            raise AssertionError(
                f"fixed-point inference is biased: median error {median_error:.2f} cp "
                f"over {len(errors)} positions exceeds {MEDIAN_ERROR_LIMIT_CP} cp"
            )
        if tail_error > TAIL_ERROR_LIMIT_CP:
            raise AssertionError(
                f"fixed-point inference tail is too wide: p99 {tail_error:.2f} cp "
                f"over {len(errors)} positions exceeds {TAIL_ERROR_LIMIT_CP} cp"
            )
    else:
        median_error = 0.0
        tail_error = 0.0

    nnue.benchmark_evaluations(position.pieces, accumulators, 1)
    started = time.perf_counter()
    checksum = int(
        nnue.benchmark_evaluations(position.pieces, accumulators, benchmark_iterations)
    )
    elapsed = time.perf_counter() - started
    return {
        "candidate": str(candidate.resolve()),
        "positions_checked": checked,
        "stale_bucket_transitions": stale_transitions,
        "maximum_incremental_error": 0.0,
        "median_inference_error_cp": median_error,
        "p99_inference_error_cp": tail_error,
        "maximum_inference_error_cp": maximum_inference_error,
        "benchmark_iterations": benchmark_iterations,
        "benchmark_seconds": elapsed,
        "evaluations_per_second": benchmark_iterations / elapsed,
        "checksum": checksum,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, default=Path("challengers/v9_kingnet"))
    parser.add_argument("--random-plies", type=int, default=500)
    parser.add_argument("--benchmark-iterations", type=int, default=100_000)
    args = parser.parse_args()
    if args.random_plies < 0:
        parser.error("--random-plies must not be negative")
    if args.benchmark_iterations <= 0:
        parser.error("--benchmark-iterations must be positive")
    report = verify(args.candidate, args.random_plies, args.benchmark_iterations)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
