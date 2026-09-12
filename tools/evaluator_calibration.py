"""Measure evaluator quality and scale by material phase.

The input is one or more JSON reports produced by ``analyze_pgn_stockfish``.
This tool is diagnostic-only: it never modifies an engine or its weights.
Positions are de-duplicated by FEN and assigned to calibration/holdout sets by
a stable hash, so repeated runs and additional input files remain comparable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tools.search_diagnostics import accumulator_row_size, load_engine_modules

DEFAULT_ENGINE_ROOT = Path("current")
DEFAULT_BLEND_WEIGHTS = (0, 25, 50, 75, 100)
DEFAULT_PIECE_BANDS = ((2, 8), (9, 12), (13, 16), (17, 24), (25, 32))


@dataclass(frozen=True, slots=True)
class Sample:
    group: str
    fen: str
    target: int
    pieces: int
    learned: int
    handcrafted: int


def _parse_ints(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if not parsed:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return parsed


def _parse_bands(value: str) -> tuple[tuple[int, int], ...]:
    bands: list[tuple[int, int]] = []
    try:
        for item in value.split(","):
            lower_text, upper_text = item.strip().split("-", maxsplit=1)
            lower, upper = int(lower_text), int(upper_text)
            if lower < 2 or upper > 32 or lower > upper:
                raise ValueError
            bands.append((lower, upper))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "bands must be comma-separated ranges within 2-32"
        ) from error
    if not bands:
        raise argparse.ArgumentTypeError("expected at least one material band")
    return tuple(bands)


def _iter_moves(paths: Iterable[Path]) -> Iterable[tuple[str, dict[str, Any]]]:
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        games = payload.get("games")
        if not isinstance(games, list):
            raise ValueError(f"analysis report has no games list: {path}")
        for game_index, game in enumerate(games):
            if not isinstance(game, dict):
                continue
            headers = game.get("headers")
            file_name = game.get("file")
            if isinstance(headers, dict):
                identity = "|".join(
                    str(headers.get(field, ""))
                    for field in ("Date", "Round", "White", "Black")
                )
            else:
                identity = ""
            group = identity or (str(file_name) if file_name else f"{path}:{game_index}")
            moves = game.get("moves")
            if not isinstance(moves, list):
                continue
            for move in moves:
                if isinstance(move, dict):
                    yield group, move


def load_samples(
    paths: Iterable[Path],
    engine_root: Path,
    *,
    selected_only: bool,
    max_abs_cp: int,
) -> list[Sample]:
    engine, search = load_engine_modules(engine_root)
    nnue = search.nnue
    row_size = accumulator_row_size(nnue)
    samples: list[Sample] = []
    seen: set[str] = set()

    for group, move in _iter_moves(paths):
        if selected_only and not bool(move.get("selected")):
            continue
        fen = move.get("fen")
        before = move.get("before")
        if not isinstance(fen, str) or not isinstance(before, dict):
            continue
        target = before.get("cp")
        if not isinstance(target, int) or abs(target) > max_abs_cp or fen in seen:
            continue
        position = engine.position_from_fen(fen)
        accumulator = np.empty((2, row_size), dtype=np.int32)
        nnue.rebuild(position.pieces, accumulator)
        learned = int(
            nnue.evaluate(
                position.pieces,
                accumulator,
                int(position.state[engine.STATE_SIDE]),
            )
        )
        handcrafted = int(search.handcrafted_evaluate(position.pieces, position.state))
        piece_count = sum(int(mask).bit_count() for mask in position.pieces)
        samples.append(Sample(group, fen, target, piece_count, learned, handcrafted))
        seen.add(fen)
    if not samples:
        raise ValueError("no usable centipawn-labelled positions found")
    return samples


def _is_holdout(group: str, holdout_percent: int) -> bool:
    digest = hashlib.sha256(group.encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:4], "little") % 100
    return bucket < holdout_percent


def _metrics(
    targets: np.ndarray[Any, np.dtype[np.float64]],
    predictions: np.ndarray[Any, np.dtype[np.float64]],
) -> dict[str, float | int]:
    errors = predictions - targets
    return {
        "count": int(targets.size),
        "mae_cp": float(np.mean(np.abs(errors))),
        "rmse_cp": float(math.sqrt(float(np.mean(errors * errors)))),
        "bias_cp": float(np.mean(errors)),
    }


def _blend_predictions(samples: list[Sample], weight: int) -> np.ndarray[Any, np.dtype[np.float64]]:
    learned = np.asarray([sample.learned for sample in samples], dtype=np.float64)
    handcrafted = np.asarray([sample.handcrafted for sample in samples], dtype=np.float64)
    return (weight * learned + (100 - weight) * handcrafted) / 100.0


def _fit_scale(samples: list[Sample]) -> float:
    targets = np.asarray([sample.target for sample in samples], dtype=np.float64)
    learned = np.asarray([sample.learned for sample in samples], dtype=np.float64)
    denominator = float(learned @ learned)
    return float(learned @ targets) / denominator if denominator > 0.0 else 1.0


def _band_name(lower: int, upper: int) -> str:
    return f"{lower}-{upper}"


def build_report(
    samples: list[Sample],
    *,
    blend_weights: tuple[int, ...],
    piece_bands: tuple[tuple[int, int], ...],
    holdout_percent: int,
) -> dict[str, Any]:
    calibration = [s for s in samples if not _is_holdout(s.group, holdout_percent)]
    holdout = [s for s in samples if _is_holdout(s.group, holdout_percent)]
    if not calibration or not holdout:
        raise ValueError("stable split produced an empty calibration or holdout set")

    report: dict[str, Any] = {
        "schema_version": 1,
        "split": {
            "method": "sha256-game-identity",
            "holdout_percent": holdout_percent,
            "calibration_count": len(calibration),
            "holdout_count": len(holdout),
        },
        "blend_holdout": {},
        "bands": {},
    }
    holdout_targets = np.asarray([sample.target for sample in holdout], dtype=np.float64)
    for weight in blend_weights:
        report["blend_holdout"][str(weight)] = _metrics(
            holdout_targets, _blend_predictions(holdout, weight)
        )

    for lower, upper in piece_bands:
        band_calibration = [s for s in calibration if lower <= s.pieces <= upper]
        band_holdout = [s for s in holdout if lower <= s.pieces <= upper]
        band: dict[str, Any] = {
            "calibration_count": len(band_calibration),
            "holdout_count": len(band_holdout),
        }
        if band_calibration and band_holdout:
            scale = _fit_scale(band_calibration)
            targets = np.asarray([sample.target for sample in band_holdout], dtype=np.float64)
            learned = np.asarray([sample.learned for sample in band_holdout], dtype=np.float64)
            band["learned_scale"] = scale
            band["raw_learned"] = _metrics(targets, learned)
            band["scaled_learned"] = _metrics(targets, learned * scale)
        report["bands"][_band_name(lower, upper)] = band
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--engine-root", type=Path, default=DEFAULT_ENGINE_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selected-only", action="store_true")
    parser.add_argument("--max-abs-cp", type=int, default=1500)
    parser.add_argument("--holdout-percent", type=int, default=25)
    parser.add_argument("--blend-weights", type=_parse_ints, default=DEFAULT_BLEND_WEIGHTS)
    parser.add_argument(
        "--piece-bands",
        type=_parse_bands,
        default=DEFAULT_PIECE_BANDS,
        help="inclusive ranges, for example 2-8,9-12,13-16,17-24,25-32",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.max_abs_cp <= 0:
        raise SystemExit("--max-abs-cp must be positive")
    if not 1 <= args.holdout_percent <= 99:
        raise SystemExit("--holdout-percent must be within 1-99")
    if any(weight < 0 or weight > 100 for weight in args.blend_weights):
        raise SystemExit("--blend-weights must stay within 0-100")
    samples = load_samples(
        args.reports,
        args.engine_root,
        selected_only=args.selected_only,
        max_abs_cp=args.max_abs_cp,
    )
    report = build_report(
        samples,
        blend_weights=args.blend_weights,
        piece_bands=args.piece_bands,
        holdout_percent=args.holdout_percent,
    )
    report["engine_root"] = str(args.engine_root)
    report["inputs"] = [str(path) for path in args.reports]
    report["selected_only"] = bool(args.selected_only)
    report["max_abs_cp"] = int(args.max_abs_cp)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["blend_holdout"], indent=2))
    print(f"report={args.output}")


if __name__ == "__main__":
    main()
