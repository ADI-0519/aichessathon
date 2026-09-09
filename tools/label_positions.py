"""Label a position suite for residual tuning with a local Stockfish process."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import chess
import chess.engine
import numpy as np

from tools.backtest_core import (
    fingerprint_agent,
    fingerprint_file,
    load_suite_file,
    stable_split,
    suite_digest,
)
from tools.cli import nonnegative_int, positive_int
from tools.evaluation_dataset import (
    MATE_LABEL_CP,
    evaluation_group,
    load_labels,
    make_label,
    write_labels,
    write_manifest,
)
from tools.search_diagnostics import (
    REPOSITORY,
    accumulator_row_size,
    load_engine_modules,
)

BASELINE_ROOT = REPOSITORY / "current"
baseline_engine, baseline_search = load_engine_modules(BASELINE_ROOT)


def baseline_static_score(board: chess.Board) -> int:
    """Evaluate from the side-to-move perspective with the canonical champion."""
    position = baseline_engine.position_from_board(board)
    accumulators = np.empty(
        (2, accumulator_row_size(baseline_search.nnue)),
        dtype=np.int32,
    )
    baseline_search.nnue.rebuild(position.pieces, accumulators)
    return int(baseline_search.evaluate(position.pieces, position.state, accumulators))


def _validate_resume(
    manifest_path: Path,
    labels_path: Path,
    *,
    suite_sha256: str,
    teacher_sha256: str,
    baseline_sha256: str,
    nodes: int,
    split_seed: str,
    selection_count: int,
) -> None:
    if not manifest_path.is_file():
        raise ValueError("resume manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected: dict[str, object] = {
        "schema_version": 2,
        "suite_sha256": suite_sha256,
        "teacher_engine_sha256": teacher_sha256,
        "teacher_nodes": nodes,
        "split_seed": split_seed,
        "selection_count": selection_count,
    }
    for name, value in expected.items():
        if manifest.get(name) != value:
            raise ValueError(f"resume manifest has different {name}")
    baseline = manifest.get("baseline")
    if not isinstance(baseline, dict) or baseline.get("sha256") != baseline_sha256:
        raise ValueError("resume manifest has a different evaluation baseline")
    label_sha256 = str(fingerprint_file(labels_path)["sha256"])
    if manifest.get("label_sha256") != label_sha256:
        raise ValueError("resume label file does not match its manifest digest")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--nodes", type=positive_int, required=True)
    parser.add_argument("--split-seed", default="v3-openings")
    parser.add_argument(
        "--split", choices=("all", "development", "validation", "holdout"), default="all"
    )
    parser.add_argument("--offset", type=nonnegative_int, default=0)
    parser.add_argument("--limit", type=positive_int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint-every", type=positive_int, default=25)
    args = parser.parse_args()

    suite_path = args.suite.resolve()
    engine_path = args.engine.resolve()
    if not suite_path.is_file():
        parser.error(f"suite not found: {suite_path}")
    if not engine_path.is_file():
        parser.error(f"engine not found: {engine_path}")

    source_suite = load_suite_file(suite_path, split_seed=args.split_seed)
    suite = [
        replace(
            position,
            split=stable_split(
                evaluation_group(position.identifier, chess.Board(position.fen)),
                args.split_seed,
            ),
        )
        for position in source_suite
    ]
    selected = [
        position for position in suite if args.split == "all" or position.split == args.split
    ]
    selected = selected[args.offset :]
    if args.limit is not None:
        selected = selected[: args.limit]
    if not selected:
        parser.error("selection contains no positions")

    suite_sha256 = suite_digest(suite)
    teacher_sha256 = str(fingerprint_file(engine_path)["sha256"])
    baseline_fingerprint = fingerprint_agent(BASELINE_ROOT)
    baseline_fingerprint["path"] = "current"
    baseline_sha256 = str(baseline_fingerprint["sha256"])
    manifest_path = args.output.with_suffix(".manifest.json")

    labels = []
    if args.output.exists():
        if not args.resume:
            parser.error(f"output already exists; pass --resume to continue: {args.output}")
        try:
            _validate_resume(
                manifest_path,
                args.output,
                suite_sha256=suite_sha256,
                teacher_sha256=teacher_sha256,
                baseline_sha256=baseline_sha256,
                nodes=args.nodes,
                split_seed=args.split_seed,
                selection_count=len(selected),
            )
            labels = load_labels(args.output)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            parser.error(str(error))
        if len(labels) > len(selected):
            parser.error("resume file contains more labels than the current selection")
        for label, position in zip(labels, selected, strict=False):
            expected_group = evaluation_group(position.identifier, chess.Board(position.fen))
            if (
                label.identifier != position.identifier
                or label.fen != position.fen
                or label.split != position.split
                or label.group != expected_group
            ):
                parser.error("resume labels are not a prefix of the current selection")
        print(f"resuming {args.output} after {len(labels)}/{len(selected)} labels")
    elif args.resume and manifest_path.exists():
        parser.error("resume manifest exists but the label file is missing")

    def checkpoint() -> None:
        label_digest = write_labels(args.output, labels)
        write_manifest(
            manifest_path,
            labels_path=args.output,
            suite_path=suite_path,
            suite_digest=suite_sha256,
            engine_path=engine_path,
            engine_digest=teacher_sha256,
            nodes=args.nodes,
            labels=labels,
            label_digest=label_digest,
            baseline_fingerprint=baseline_fingerprint,
            split_seed=args.split_seed,
            selection_count=len(selected),
        )

    if len(labels) == len(selected):
        print(f"already complete: {args.output} ({len(labels)} labels)")
        return

    teacher = chess.engine.SimpleEngine.popen_uci(str(engine_path))
    persisted_count = len(labels)
    try:
        teacher.configure({"Threads": 1, "Hash": 128})
        for index, position in enumerate(selected[len(labels) :], start=len(labels) + 1):
            board = chess.Board(position.fen)
            if board.is_game_over(claim_draw=True):
                raise ValueError(f"cannot label terminal position {position.identifier}")
            if "Clear Hash" in teacher.options:
                teacher.configure({"Clear Hash": None})
            result = teacher.analyse(
                board,
                chess.engine.Limit(nodes=args.nodes),
                game=position.identifier,
            )
            teacher_score = result["score"].pov(board.turn)
            score = teacher_score.score(mate_score=MATE_LABEL_CP)
            if score is None:
                raise RuntimeError(f"teacher returned no score for {position.identifier}")
            pv = result.get("pv", [])
            best_move = pv[0] if pv else None
            baseline_cp = baseline_static_score(board)
            labels.append(
                make_label(
                    position,
                    score,
                    best_move,
                    baseline_cp=baseline_cp,
                    teacher_mate=teacher_score.is_mate(),
                )
            )
            print(
                f"label {index}/{len(selected)}: {position.identifier} "
                f"teacher {score:+d}, baseline {baseline_cp:+d}, "
                f"residual {score - baseline_cp:+d} cp"
            )
            if len(labels) % args.checkpoint_every == 0:
                checkpoint()
                persisted_count = len(labels)
    finally:
        try:
            if len(labels) > persisted_count:
                checkpoint()
        finally:
            teacher.quit()

    print(f"wrote {args.output} ({len(labels)} labels)")
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
