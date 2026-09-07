"""Verify king-conditioned feature parity, updates, inference, and speed."""

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

from tools.halfkp_features import perspective_indices


def _load_candidate(candidate: Path) -> tuple[ModuleType, ModuleType]:
    candidate = candidate.resolve()
    sys.path.insert(0, str(candidate))
    engine = importlib.import_module("engine")
    nnue = importlib.import_module("nnue")
    for module in (engine, nnue):
        module_path = Path(str(module.__file__)).resolve()
        if module_path.parent != candidate:
            raise RuntimeError(f"loaded {module.__name__} from {module_path}, not {candidate}")
    return engine, nnue


def _internal_move(engine: ModuleType, position: Any, uci: str) -> int:
    for move in engine.legal_moves(position):
        encoded = int(move)
        if engine.move_to_uci(encoded) == uci:
            return encoded
    raise AssertionError(f"candidate did not generate legal move {uci}")


def _feature_reference(nnue: ModuleType, board: chess.Board) -> np.ndarray[Any, Any]:
    expected = np.repeat(nnue.ACCUMULATOR_BIAS_Q[None, :], 2, axis=0).astype(
        np.int32
    )
    for perspective, colour in enumerate((chess.WHITE, chess.BLACK)):
        for feature in perspective_indices(board, colour):
            expected[perspective] += nnue.FEATURE_WEIGHTS_Q[feature]
    return expected


def _check_position(
    engine: ModuleType,
    nnue: ModuleType,
    board: chess.Board,
    position: Any,
    accumulators: np.ndarray[Any, Any],
) -> tuple[float, float]:
    rebuilt = np.empty_like(accumulators)
    nnue.rebuild(position.pieces, rebuilt)
    incremental_error = float(np.max(np.abs(accumulators - rebuilt)))
    if incremental_error != 0.0:
        raise AssertionError(f"incremental accumulator error: {incremental_error}")

    expected = _feature_reference(nnue, board)
    feature_error = float(np.max(np.abs(rebuilt - expected)))
    if feature_error != 0.0:
        raise AssertionError(f"Python/Numba feature error: {feature_error}")

    side = int(position.state[engine.STATE_SIDE])
    compiled = int(nnue.evaluate(accumulators, side))
    reference = float(nnue.evaluate_reference(position.pieces, side))
    inference_error = abs(compiled - reference)
    if inference_error > 8.0:
        raise AssertionError(
            f"fixed-point/reference inference mismatch: {compiled} vs {reference:.4f}"
        )
    return feature_error, inference_error


def verify(candidate: Path, random_plies: int, benchmark_iterations: int) -> dict[str, object]:
    engine, nnue = _load_candidate(candidate)
    nnue.warmup()
    max_feature_error = 0.0
    max_inference_error = 0.0
    checked = 0

    cases = (
        (chess.STARTING_FEN, "e2e4"),
        ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1g1"),
        ("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1", "e8c8"),
        ("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", "e5d6"),
        ("4k3/P7/8/8/8/8/8/4K3 w - - 0 1", "a7a8q"),
        ("4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1", "e4d5"),
        ("7k/8/8/8/8/8/4p3/4K3 w - - 0 1", "e1e2"),
    )
    for fen, uci in cases:
        board = chess.Board(fen)
        position = engine.position_from_board(board)
        parent = np.empty((2, nnue.ACCUMULATOR_SIZE), dtype=np.int32)
        child = np.empty_like(parent)
        nnue.rebuild(position.pieces, parent)
        move = _internal_move(engine, position, uci)
        nnue.update_for_move(position.pieces, position.state, move, parent, child)
        undo = np.empty(engine.UNDO_SIZE, dtype=np.int64)
        undo_key = np.empty(1, dtype=np.uint64)
        if not engine.make_move(
            position.pieces, position.state, position.key, move, undo, undo_key
        ):
            raise AssertionError(f"candidate rejected generated move {uci}")
        board.push(chess.Move.from_uci(uci))
        feature_error, inference_error = _check_position(
            engine, nnue, board, position, child
        )
        max_feature_error = max(max_feature_error, feature_error)
        max_inference_error = max(max_inference_error, inference_error)
        checked += 1

    rng = random.Random(0xA1C4E55A)
    board = chess.Board()
    position = engine.position_from_board(board)
    accumulators = np.empty((2, nnue.ACCUMULATOR_SIZE), dtype=np.int32)
    nnue.rebuild(position.pieces, accumulators)
    for _ in range(random_plies):
        moves = engine.legal_moves(position)
        if len(moves) == 0:
            board = chess.Board()
            position = engine.position_from_board(board)
            nnue.rebuild(position.pieces, accumulators)
            continue
        move = int(moves[rng.randrange(len(moves))])
        uci = engine.move_to_uci(move)
        child = np.empty_like(accumulators)
        nnue.update_for_move(
            position.pieces, position.state, move, accumulators, child
        )
        undo = np.empty(engine.UNDO_SIZE, dtype=np.int64)
        undo_key = np.empty(1, dtype=np.uint64)
        if not engine.make_move(
            position.pieces, position.state, position.key, move, undo, undo_key
        ):
            raise AssertionError("candidate rejected its own generated move")
        board.push(chess.Move.from_uci(uci))
        accumulators = child
        feature_error, inference_error = _check_position(
            engine, nnue, board, position, accumulators
        )
        max_feature_error = max(max_feature_error, feature_error)
        max_inference_error = max(max_inference_error, inference_error)
        checked += 1

    nnue.benchmark_evaluations(accumulators, 1)
    started = time.perf_counter()
    checksum = int(nnue.benchmark_evaluations(accumulators, benchmark_iterations))
    elapsed = time.perf_counter() - started
    return {
        "candidate": str(candidate.resolve()),
        "positions_checked": checked,
        "maximum_feature_error": max_feature_error,
        "maximum_incremental_error": 0.0,
        "maximum_inference_error_cp": max_inference_error,
        "benchmark_iterations": benchmark_iterations,
        "benchmark_seconds": elapsed,
        "evaluations_per_second": benchmark_iterations / elapsed,
        "checksum": checksum,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate", type=Path, default=Path("challengers/v6_halfkp")
    )
    parser.add_argument("--random-plies", type=int, default=500)
    parser.add_argument("--benchmark-iterations", type=int, default=100_000)
    args = parser.parse_args()
    if args.random_plies < 0:
        parser.error("--random-plies must not be negative")
    if args.benchmark_iterations <= 0:
        parser.error("--benchmark-iterations must be positive")
    report = verify(args.candidate, args.random_plies, args.benchmark_iterations)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
