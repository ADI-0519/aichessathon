"""Fit a phase-aware additive correction to the frozen V3 evaluator."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tools.backtest_core import atomic_write_text
from tools.cli import positive_int
from tools.evaluation_dataset import EvaluationLabel, load_labels
from tools.evaluation_features import FEATURE_NAMES, MAX_PHASE

MAX_WEIGHT = 1_000
DEFAULT_RIDGES = (0.01, 0.1, 1.0, 10.0, 100.0)

Metric = dict[str, float | int]
SplitMetrics = dict[str, dict[str, Metric]]


@dataclass(frozen=True, slots=True)
class FitResult:
    """Selected residual model and float/integer split diagnostics."""

    feature_names: tuple[str, ...]
    weights: tuple[float, ...]
    integer_weights: tuple[int, ...]
    intercept: float
    integer_intercept: int
    ridge: float
    train_count: int
    validation_count: int
    excluded_mate_count: int
    metrics: SplitMetrics
    holdout_reported: bool


def fit_linear_evaluator(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    ridge: float,
    integer_limit: int = MAX_WEIGHT,
) -> tuple[np.ndarray, float]:
    """Fit standardized ridge regression without forming normal equations."""
    if features.ndim != 2 or targets.ndim != 1:
        raise ValueError("features must be 2-D and targets must be 1-D")
    if features.shape[0] != targets.shape[0] or features.shape[0] == 0:
        raise ValueError("features and targets must contain the same non-zero number of rows")
    if ridge < 0:
        raise ValueError("ridge must be non-negative")
    if integer_limit <= 0:
        raise ValueError("integer_limit must be positive")

    means = np.mean(features, axis=0)
    scales = np.std(features, axis=0)
    scales = np.where(scales > 1e-12, scales, 1.0)
    standardized = (features - means) / scales
    centered_targets = targets - np.mean(targets)

    design = standardized
    response = centered_targets
    if ridge > 0:
        penalty = np.sqrt(ridge) * np.eye(features.shape[1], dtype=np.float64)
        design = np.vstack((standardized, penalty))
        response = np.concatenate((centered_targets, np.zeros(features.shape[1])))

    standardized_weights = np.linalg.lstsq(design, response, rcond=None)[0]
    weights = np.clip(standardized_weights / scales, -integer_limit, integer_limit)
    intercept = float(np.mean(targets) - means @ weights)
    return weights, intercept


def regression_metrics(
    features: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
    intercept: float,
) -> Metric:
    predictions = intercept + features @ weights
    errors = predictions - targets
    return {
        "count": len(targets),
        "rmse_cp": float(np.sqrt(np.mean(errors * errors))),
        "mae_cp": float(np.mean(np.abs(errors))),
        "bias_cp": float(np.mean(errors)),
    }


def _matrix(labels: list[EvaluationLabel]) -> tuple[np.ndarray, np.ndarray]:
    # Kept local to make the phase denominator impossible to omit in one split.
    features = np.asarray([label.features for label in labels], dtype=np.float64) / MAX_PHASE
    targets = np.asarray([label.residual_cp for label in labels], dtype=np.float64)
    return features, targets


def _all_metrics(
    features: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
    intercept: float,
) -> dict[str, Metric]:
    integer_weights = np.rint(weights)
    integer_intercept = float(np.rint(intercept))
    return {
        "zero": regression_metrics(
            features, targets, np.zeros(features.shape[1]), 0.0
        ),
        "float": regression_metrics(features, targets, weights, intercept),
        "integer": regression_metrics(
            features, targets, integer_weights, integer_intercept
        ),
    }


def fit_labels(
    labels_path: Path,
    *,
    ridge_candidates: tuple[float, ...] = DEFAULT_RIDGES,
    integer_limit: int = MAX_WEIGHT,
    min_train_count: int = 500,
    require_validation: bool = True,
    include_holdout: bool = False,
) -> FitResult:
    """Select ridge on validation and fit a residual model on development only."""
    all_labels = load_labels(labels_path)
    labels = [label for label in all_labels if not label.teacher_mate]
    if any(len(label.features) != len(FEATURE_NAMES) for label in labels):
        raise ValueError("label feature schema does not match the current extractor")
    if not ridge_candidates or any(value < 0 for value in ridge_candidates):
        raise ValueError("ridge candidates must be a non-empty set of non-negative values")
    if min_train_count <= 0:
        raise ValueError("minimum training count must be positive")

    development = [label for label in labels if label.split == "development"]
    validation = [label for label in labels if label.split == "validation"]
    if len(development) < min_train_count:
        raise ValueError(
            f"development split has {len(development)} labels; at least {min_train_count} required"
        )
    if require_validation and not validation:
        raise ValueError("a non-empty validation split is required for model selection")

    train_features, train_targets = _matrix(development)
    selection_labels = validation or development
    selection_features, selection_targets = _matrix(selection_labels)
    candidates: list[tuple[float, float, np.ndarray, float]] = []
    for ridge in sorted(set(ridge_candidates)):
        weights, intercept = fit_linear_evaluator(
            train_features,
            train_targets,
            ridge=ridge,
            integer_limit=integer_limit,
        )
        metric = regression_metrics(
            selection_features,
            selection_targets,
            np.rint(weights),
            float(np.rint(intercept)),
        )
        candidates.append((float(metric["rmse_cp"]), ridge, weights, intercept))
    _, selected_ridge, weights, intercept = min(candidates, key=lambda item: (item[0], item[1]))

    metrics: SplitMetrics = {}
    visible_splits = ("development", "validation", "holdout")
    for split in visible_splits:
        if split == "holdout" and not include_holdout:
            continue
        selected = [label for label in labels if label.split == split]
        if not selected:
            continue
        features, targets = _matrix(selected)
        metrics[split] = _all_metrics(features, targets, weights, intercept)

    integer_weights = tuple(int(np.rint(value)) for value in weights)
    return FitResult(
        feature_names=tuple(FEATURE_NAMES),
        weights=tuple(float(value) for value in weights),
        integer_weights=integer_weights,
        intercept=float(intercept),
        integer_intercept=int(np.rint(intercept)),
        ridge=selected_ridge,
        train_count=len(development),
        validation_count=len(validation),
        excluded_mate_count=len(all_labels) - len(labels),
        metrics=metrics,
        holdout_reported=include_holdout,
    )


def write_fit(
    path: Path,
    labels_path: Path,
    result: FitResult,
    *,
    target_name: str = "teacher_score_cp_minus_v3_static_score_cp",
) -> None:
    if not target_name.strip():
        raise ValueError("target name must not be empty")
    payload = {
        "schema_version": 2,
        "target": target_name,
        "labels_sha256": hashlib.sha256(labels_path.read_bytes()).hexdigest(),
        "feature_names": list(result.feature_names),
        "taper_denominator": MAX_PHASE,
        "weights": list(result.weights),
        "integer_weights": list(result.integer_weights),
        "intercept": result.intercept,
        "integer_intercept": result.integer_intercept,
        "selected_ridge": result.ridge,
        "train_split": "development",
        "train_count": result.train_count,
        "validation_count": result.validation_count,
        "excluded_mate_count": result.excluded_mate_count,
        "holdout_reported": result.holdout_reported,
        "metrics": result.metrics,
    }
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _ridge_grid(value: str) -> tuple[float, ...]:
    try:
        values = tuple(float(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "ridge grid must contain comma-separated numbers"
        ) from error
    if not values or any(item < 0 for item in values):
        raise argparse.ArgumentTypeError("ridge grid values must be non-negative")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ridge-grid", type=_ridge_grid, default=DEFAULT_RIDGES)
    parser.add_argument("--integer-limit", type=positive_int, default=MAX_WEIGHT)
    parser.add_argument("--min-train-count", type=positive_int, default=500)
    parser.add_argument("--allow-no-validation", action="store_true")
    parser.add_argument("--report-holdout", action="store_true")
    parser.add_argument(
        "--target-name",
        default="teacher_score_cp_minus_v3_static_score_cp",
        help="Provenance label for the residual target stored in the artifact.",
    )
    args = parser.parse_args()
    if not args.labels.is_file():
        parser.error(f"label file not found: {args.labels}")

    result = fit_labels(
        args.labels,
        ridge_candidates=args.ridge_grid,
        integer_limit=args.integer_limit,
        min_train_count=args.min_train_count,
        require_validation=not args.allow_no_validation,
        include_holdout=args.report_holdout,
    )
    write_fit(args.output, args.labels, result, target_name=args.target_name)
    print(
        f"wrote {args.output} ({result.train_count} development, "
        f"{result.validation_count} validation, ridge {result.ridge:g})"
    )
    for split, variants in result.metrics.items():
        metric = variants["integer"]
        print(
            f"{split}: integer residual RMSE {metric['rmse_cp']:.1f} cp, "
            f"MAE {metric['mae_cp']:.1f} cp"
        )


if __name__ == "__main__":
    main()
