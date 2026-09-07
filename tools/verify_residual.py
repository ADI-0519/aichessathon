"""Verify a compiled residual evaluator against its Python feature model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import chess
import numpy as np

from tools.evaluation_dataset import load_labels
from tools.evaluation_features import FEATURE_NAMES, MAX_PHASE, extract_features
from tools.search_diagnostics import load_engine_modules


def expected_correction(board: chess.Board, artifact: dict[str, Any]) -> int:
    """Evaluate the exported integer linear model through Python features."""
    names = tuple(str(value) for value in artifact["feature_names"])
    if names != FEATURE_NAMES:
        raise ValueError("artifact feature schema does not match the Python extractor")
    weights = np.asarray(artifact["integer_weights"], dtype=np.int64)
    if weights.shape != (len(FEATURE_NAMES),):
        raise ValueError("artifact has the wrong number of integer weights")
    values = np.asarray(extract_features(board).values, dtype=np.int64)
    return int(artifact["integer_intercept"]) + int(values @ weights) // MAX_PHASE


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine-root", required=True, type=Path)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit cannot be negative")

    artifact = json.loads(args.artifact.read_text(encoding="utf-8"))
    labels = load_labels(args.labels)
    if args.limit:
        labels = labels[: args.limit]
    engine, search = load_engine_modules(args.engine_root)
    if not hasattr(search, "residual"):
        parser.error("selected engine does not expose a residual evaluator")

    mismatches = 0
    maximum_difference = 0
    first_mismatch: tuple[str, int, int] | None = None
    for label in labels:
        board = chess.Board(label.fen)
        position = engine.position_from_board(board)
        expected = expected_correction(board, artifact)
        actual = int(search.residual.correction(position.pieces, position.state))
        difference = abs(actual - expected)
        maximum_difference = max(maximum_difference, difference)
        if difference:
            mismatches += 1
            if first_mismatch is None:
                first_mismatch = (label.identifier, expected, actual)

    print(f"positions: {len(labels):,}")
    print(f"mismatches: {mismatches:,}")
    print(f"maximum difference: {maximum_difference} cp")
    if first_mismatch is not None:
        identifier, expected, actual = first_mismatch
        print(f"first mismatch: {identifier}: expected {expected}, actual {actual}")
    if mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
