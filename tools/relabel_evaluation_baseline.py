"""Rebase existing teacher labels onto an exact runtime evaluator.

Teacher searches are expensive and do not need to be repeated when the local
baseline changes.  This development-only tool preserves every teacher score,
split, feature vector, and provenance identifier while replacing ``baseline_cp``
with the static score produced by a selected source-compatible agent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import chess
import numpy as np

from tools.backtest_core import atomic_write_text, fingerprint_agent
from tools.evaluation_dataset import EvaluationLabel, load_labels, write_labels
from tools.search_diagnostics import load_engine_modules


def rebase_labels(
    labels: list[EvaluationLabel],
    evaluate_fen: Callable[[str], int],
) -> list[EvaluationLabel]:
    """Return labels with only their runtime baseline scores replaced."""
    return [replace(label, baseline_cp=int(evaluate_fen(label.fen))) for label in labels]


class RuntimeStaticEvaluator:
    """Evaluate FENs through one loaded agent's production static evaluator."""

    def __init__(self, engine_root: Path) -> None:
        self.engine, self.search = load_engine_modules(engine_root)
        if not hasattr(self.search, "nnue"):
            raise ValueError("selected baseline does not expose an NNUE accumulator")
        size = int(self.search.nnue.ACCUMULATOR_SIZE)
        self.accumulators = np.empty((2, size), dtype=np.int32)

    def __call__(self, fen: str) -> int:
        board = chess.Board(fen)
        position = self.engine.position_from_board(board)
        self.search.nnue.rebuild(position.pieces, self.accumulators)
        return int(
            self.search.evaluate(
                position.pieces,
                position.state,
                self.accumulators,
            )
        )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest_path(output: Path) -> Path:
    return output.with_suffix(".manifest.json")


def _write_manifest(
    output: Path,
    source: Path,
    engine_root: Path,
    labels: list[EvaluationLabel],
    original: list[EvaluationLabel],
) -> None:
    differences = np.asarray(
        [new.baseline_cp - old.baseline_cp for new, old in zip(labels, original, strict=True)],
        dtype=np.float64,
    )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "operation": "replace_baseline_static_evaluation",
        "source": str(source.resolve()),
        "source_sha256": _sha256(source),
        "output": str(output.resolve()),
        "output_sha256": _sha256(output),
        "baseline": fingerprint_agent(engine_root),
        "count": len(labels),
        "teacher_scores_changed": False,
        "feature_vectors_changed": False,
        "baseline_difference_cp": {
            "mean": float(np.mean(differences)),
            "mae": float(np.mean(np.abs(differences))),
            "maximum_absolute": int(np.max(np.abs(differences))),
        },
    }
    atomic_write_text(
        _manifest_path(output),
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--engine-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    if not args.labels.is_file():
        parser.error(f"label file does not exist: {args.labels}")
    if args.output.resolve() == args.labels.resolve():
        parser.error("--output must not overwrite the source labels")
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")

    original = load_labels(args.labels)
    evaluator = RuntimeStaticEvaluator(args.engine_root)
    rebased = rebase_labels(original, evaluator)
    write_labels(args.output, rebased)
    _write_manifest(args.output, args.labels, args.engine_root, rebased, original)
    print(f"wrote {args.output} ({len(rebased):,} positions)")
    print(f"manifest: {_manifest_path(args.output)}")


if __name__ == "__main__":
    main()
