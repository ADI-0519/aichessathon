"""Verify V5 incremental features, inference parity, and CPU throughput."""

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
import torch


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


def _torch_reference(nnue: ModuleType, pieces: np.ndarray[Any, Any], side: int) -> float:
    indices: list[int] = []
    for piece in range(12):
        occupied = int(pieces[piece])
        while occupied:
            least_bit = occupied & -occupied
            square = least_bit.bit_length() - 1
            occupied ^= least_bit
            indices.append(piece * 64 + square)

    canonical = torch.tensor(indices, dtype=torch.long)
    swapped_piece = (torch.div(canonical, 64, rounding_mode="floor") + 6).remainder(12)
    black_indices = swapped_piece * 64 + torch.bitwise_xor(canonical.remainder(64), 56)
    feature_weights = torch.from_numpy(nnue.FEATURE_WEIGHTS)
    bias = torch.from_numpy(nnue.ACCUMULATOR_BIAS)
    white = feature_weights[canonical].sum(dim=0) + bias
    black = feature_weights[black_indices].sum(dim=0) + bias
    own, opponent = (white, black) if side == 0 else (black, white)
    inputs = torch.cat((own, opponent)).clamp(0.0, 1.0)
    hidden = (
        torch.from_numpy(nnue.HIDDEN_WEIGHTS) @ inputs
        + torch.from_numpy(nnue.HIDDEN_BIAS)
    ).clamp(0.0, 1.0)
    output = (
        torch.from_numpy(nnue.OUTPUT_WEIGHTS) @ hidden
        + torch.tensor([nnue.OUTPUT_BIAS], dtype=torch.float32)
    )
    return float(output[0]) * float(nnue.CP_SCALE)


def _check_position(
    engine: ModuleType,
    nnue: ModuleType,
    position: Any,
    accumulators: np.ndarray[Any, Any],
    *,
    tolerance: float,
) -> tuple[float, float]:
    rebuilt = np.empty_like(accumulators)
    nnue.rebuild(position.pieces, rebuilt)
    error = float(np.max(np.abs(accumulators - rebuilt)))
    if error > tolerance:
        raise AssertionError(f"incremental accumulator error {error:.8f} > {tolerance}")

    side = int(position.state[engine.STATE_SIDE])
    compiled = int(nnue.evaluate(accumulators, side))
    numpy_score = float(nnue.evaluate_reference(position.pieces, side))
    torch_score = _torch_reference(nnue, position.pieces, side)
    if abs(numpy_score - torch_score) > 0.01:
        raise AssertionError(
            f"NumPy/Torch inference mismatch: {numpy_score:.5f} vs {torch_score:.5f}"
        )
    quantization_error = abs(compiled - torch_score)
    if quantization_error > 8.0:
        raise AssertionError(
            f"Numba/Torch inference mismatch: {compiled} vs {torch_score:.5f}"
        )
    return error, quantization_error


def verify(candidate: Path, random_plies: int, benchmark_iterations: int) -> dict[str, object]:
    engine, nnue = _load_candidate(candidate)
    engine.warmup()
    nnue.warmup()
    tolerance = 5e-5
    max_error = 0.0
    max_quantization_error = 0.0
    checked = 0

    cases = (
        (chess.STARTING_FEN, "e2e4"),
        ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1g1"),
        ("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1", "e8c8"),
        ("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", "e5d6"),
        ("4k3/P7/8/8/8/8/8/4K3 w - - 0 1", "a7a8q"),
        ("4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1", "e4d5"),
    )
    for fen, uci in cases:
        position = engine.position_from_board(chess.Board(fen))
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
        accumulator_error, quantization_error = _check_position(
            engine, nnue, position, child, tolerance=tolerance
        )
        max_error = max(max_error, accumulator_error)
        max_quantization_error = max(max_quantization_error, quantization_error)
        checked += 1

    rng = random.Random(0xA1C4E55A)
    position = engine.position_from_board(chess.Board())
    accumulators = np.empty((2, nnue.ACCUMULATOR_SIZE), dtype=np.int32)
    nnue.rebuild(position.pieces, accumulators)
    for _ in range(random_plies):
        moves = engine.legal_moves(position)
        if len(moves) == 0:
            position = engine.position_from_board(chess.Board())
            nnue.rebuild(position.pieces, accumulators)
            continue
        move = int(moves[rng.randrange(len(moves))])
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
        accumulators = child
        accumulator_error, quantization_error = _check_position(
            engine, nnue, position, accumulators, tolerance=tolerance
        )
        max_error = max(max_error, accumulator_error)
        max_quantization_error = max(max_quantization_error, quantization_error)
        checked += 1

    nnue.benchmark_evaluations(accumulators, 1)
    started = time.perf_counter()
    checksum = int(nnue.benchmark_evaluations(accumulators, benchmark_iterations))
    elapsed = time.perf_counter() - started
    return {
        "candidate": str(candidate.resolve()),
        "positions_checked": checked,
        "maximum_accumulator_error": max_error,
        "maximum_quantization_error_cp": max_quantization_error,
        "benchmark_iterations": benchmark_iterations,
        "benchmark_seconds": elapsed,
        "evaluations_per_second": benchmark_iterations / elapsed,
        "checksum": checksum,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, default=Path("challengers/v5_nnue"))
    parser.add_argument("--random-plies", type=int, default=500)
    parser.add_argument("--benchmark-iterations", type=int, default=100_000)
    arguments = parser.parse_args()
    if arguments.random_plies < 0:
        parser.error("--random-plies must not be negative")
    if arguments.benchmark_iterations <= 0:
        parser.error("--benchmark-iterations must be positive")
    report = verify(
        arguments.candidate,
        arguments.random_plies,
        arguments.benchmark_iterations,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
