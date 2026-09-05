"""Provenance-tracked residual-evaluation labels stored as JSONL."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import chess

from tools.backtest_core import Split, SuitePosition, atomic_write_text
from tools.evaluation_features import FEATURE_NAMES, extract_features

SCHEMA_VERSION = 2
MATE_LABEL_CP = 30_000
VALID_SPLITS: frozenset[str] = frozenset(("development", "validation", "holdout"))


@dataclass(frozen=True, slots=True)
class EvaluationLabel:
    """One teacher label paired with the exact V3 score it should correct."""

    identifier: str
    group: str
    fen: str
    split: Split
    score_cp: int
    baseline_cp: int
    teacher_mate: bool
    best_move: str | None
    phase: int
    features: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.split not in VALID_SPLITS:
            raise ValueError(f"unknown dataset split: {self.split!r}")
        if len(self.features) != len(FEATURE_NAMES):
            raise ValueError("label feature vector has the wrong length")
        if not -MATE_LABEL_CP <= self.score_cp <= MATE_LABEL_CP:
            raise ValueError("teacher score exceeds the configured mate bound")
        if not self.group:
            raise ValueError("dataset group must not be empty")
        if not 0 <= self.phase <= 24:
            raise ValueError("phase is outside the tapered range")
        board = chess.Board(self.fen)
        if self.best_move is not None:
            move = chess.Move.from_uci(self.best_move)
            if move not in board.legal_moves:
                raise ValueError(f"best move is illegal for {self.identifier}")

    @property
    def residual_cp(self) -> int:
        """The additive correction target for V3's static evaluation."""
        return self.score_cp - self.baseline_cp

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["schema_version"] = SCHEMA_VERSION
        value["features"] = list(self.features)
        return value

    @classmethod
    def from_dict(cls, value: object) -> EvaluationLabel:
        if not isinstance(value, dict):
            raise ValueError("label must be a JSON object")
        if int(value.get("schema_version", -1)) != SCHEMA_VERSION:
            raise ValueError(
                f"label schema is not version {SCHEMA_VERSION}; regenerate the dataset"
            )
        try:
            split = str(value["split"])
            if split not in VALID_SPLITS:
                raise ValueError(f"unknown dataset split: {split!r}")
            return cls(
                identifier=str(value["identifier"]),
                group=str(value["group"]),
                fen=str(value["fen"]),
                split=split,  # type: ignore[arg-type]
                score_cp=int(value["score_cp"]),
                baseline_cp=int(value["baseline_cp"]),
                teacher_mate=bool(value["teacher_mate"]),
                best_move=(
                    None if value.get("best_move") is None else str(value["best_move"])
                ),
                phase=int(value["phase"]),
                features=tuple(int(item) for item in value["features"]),
            )
        except KeyError as error:
            raise ValueError(f"label is missing {error.args[0]!r}") from error


def make_label(
    position: SuitePosition,
    score_cp: int,
    best_move: chess.Move | None,
    *,
    baseline_cp: int,
    teacher_mate: bool = False,
) -> EvaluationLabel:
    board = chess.Board(position.fen)
    extracted = extract_features(board)
    return EvaluationLabel(
        identifier=position.identifier,
        group=evaluation_group(position.identifier, board),
        fen=position.fen,
        split=position.split,
        score_cp=int(score_cp),
        baseline_cp=int(baseline_cp),
        teacher_mate=teacher_mate,
        best_move=best_move.uci() if best_move is not None else None,
        phase=extracted.phase,
        features=extracted.values,
    )


def evaluation_identity(board: chess.Board) -> str:
    """Normalize an evaluation position without irrelevant move counters."""
    fields = board.fen(en_passant="legal").split()
    return " ".join((*fields[:4], "0", "1"))


def evaluation_group(identifier: str, board: chess.Board) -> str:
    """Keep sampled positions from one source game in a single data split."""
    match = re.fullmatch(r"(g\d+)-p\d+-.+", identifier)
    return match.group(1) if match else evaluation_identity(board)


def write_labels(path: Path, labels: list[EvaluationLabel]) -> str:
    """Write deterministic JSONL labels and return their content digest."""
    if not labels:
        raise ValueError("cannot write an empty label set")
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(label.as_dict(), sort_keys=True) for label in labels]
    payload = "\n".join(lines) + "\n"
    atomic_write_text(path, payload)
    return hashlib.sha256(payload.encode()).hexdigest()


def load_labels(path: Path) -> list[EvaluationLabel]:
    labels: list[EvaluationLabel] = []
    seen_fens: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                label = EvaluationLabel.from_dict(json.loads(line))
                if label.fen in seen_fens:
                    raise ValueError("duplicate normalized FEN")
                seen_fens.add(label.fen)
                labels.append(label)
            except (TypeError, ValueError, json.JSONDecodeError) as error:
                message = f"{path}:{line_number}: invalid evaluation label: {error}"
                raise ValueError(message) from error
    if not labels:
        raise ValueError(f"label file is empty: {path}")
    return labels


def _portable_path(path: Path, manifest_path: Path) -> str:
    try:
        return Path(os.path.relpath(path.resolve(), manifest_path.parent.resolve())).as_posix()
    except ValueError:  # Different Windows drives cannot be made relative.
        return str(path.resolve())


def write_manifest(
    path: Path,
    *,
    labels_path: Path,
    suite_path: Path,
    suite_digest: str,
    engine_path: Path,
    engine_digest: str,
    nodes: int,
    labels: list[EvaluationLabel],
    label_digest: str,
    baseline_fingerprint: dict[str, object],
    split_seed: str,
    selection_count: int,
) -> None:
    """Record immutable teacher, baseline and source inputs for a label set."""
    counts: dict[str, int] = {}
    for label in labels:
        counts[label.split] = counts.get(label.split, 0) + 1
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "target": "teacher_score_cp_minus_v3_static_score_cp",
        "labels": _portable_path(labels_path, path),
        "label_sha256": label_digest,
        "suite": _portable_path(suite_path, path),
        "suite_sha256": suite_digest,
        "teacher_engine": _portable_path(engine_path, path),
        "teacher_engine_sha256": engine_digest,
        "teacher_nodes": nodes,
        "teacher_hash_mb": 128,
        "teacher_threads": 1,
        "clear_hash_each_position": True,
        "baseline": baseline_fingerprint,
        "split_seed": split_seed,
        "split_unit": "source_game_for_sampled_pgns_else_normalized_position",
        "feature_names": list(FEATURE_NAMES),
        "count": len(labels),
        "teacher_mate_count": sum(label.teacher_mate for label in labels),
        "selection_count": selection_count,
        "complete": len(labels) == selection_count,
        "split_counts": dict(sorted(counts.items())),
    }
    atomic_write_text(path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
