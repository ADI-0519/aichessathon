"""Pack Lichess Fishnet Parquet rows into compact sparse-evaluator records.

PyArrow is a training-only dependency and is imported lazily.  The produced
``.npy`` files contain no moves or lookup table: each row is a position, a
teacher evaluation, and sparse piece-square inputs used to train our model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import chess
import numpy as np

from tools.backtest_core import atomic_write_text
from tools.cli import nonnegative_int, positive_int
from tools.nnue_features import MAX_PIECES, padded_indices

CP_CLAMP = 2_000
MATE_CP = CP_CLAMP
PACKED_DTYPE = np.dtype(
    [
        ("indices", np.uint16, (MAX_PIECES,)),
        ("count", np.uint8),
        ("stm", np.uint8),
        ("cp", np.int16),
    ]
)


def position_ply(fen: str) -> int:
    """Return the zero-based game ply encoded by a FEN."""
    fields = fen.split()
    if len(fields) != 6:
        raise ValueError("FEN must have six fields")
    fullmove = int(fields[5])
    if fullmove < 1 or fields[1] not in {"w", "b"}:
        raise ValueError("FEN has invalid move counters")
    return 2 * (fullmove - 1) + (fields[1] == "b")


def encode_record(
    fen: str,
    cp: int | None,
    mate: int | None,
    best_move: str | None,
    *,
    min_ply: int,
) -> np.void | None:
    """Validate and encode one quiet labelled position.

    Positions in check and positions whose teacher move is a capture are left
    to quiescence search rather than teaching the static evaluator to guess a
    tactical sequence.
    """
    try:
        if position_ply(fen) < min_ply:
            return None
        board = chess.Board(fen)
    except (TypeError, ValueError):
        return None
    if board.king(chess.WHITE) is None or board.king(chess.BLACK) is None:
        return None
    if board.is_check() or board.is_game_over(claim_draw=False):
        return None

    if mate == 0:
        return None
    if mate is not None:
        score = MATE_CP if mate > 0 else -MATE_CP
    elif cp is not None:
        score = max(-CP_CLAMP, min(CP_CLAMP, int(cp)))
    else:
        return None

    if best_move:
        try:
            move = chess.Move.from_uci(best_move)
        except ValueError:
            return None
        if move not in board.legal_moves or board.is_capture(move):
            return None

    try:
        indices, count = padded_indices(board)
    except ValueError:
        return None
    record: np.void = np.zeros(1, dtype=PACKED_DTYPE)[0]
    record["indices"] = indices
    record["count"] = count
    record["stm"] = int(board.turn == chess.WHITE)
    record["cp"] = score
    return record


def _rows(table: Any) -> Iterable[tuple[str, int | None, int | None, str | None]]:
    columns = (table[name].to_pylist() for name in ("fen", "cp", "mate", "move"))
    return zip(*columns, strict=True)


def pack_groups(
    parquet: Any,
    groups: Sequence[int],
    output: Path,
    *,
    target: int,
    min_ply: int,
) -> tuple[int, dict[str, int]]:
    """Encode selected row groups into ``output`` and return counts."""
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_suffix(output.suffix + ".staging.npy")
    records = np.lib.format.open_memmap(
        staging, mode="w+", dtype=PACKED_DTYPE, shape=(target,)
    )
    scanned = accepted = rejected = 0
    try:
        for group_index in groups:
            table = parquet.read_row_group(
                group_index, columns=["fen", "cp", "mate", "move"]
            )
            for fen, cp, mate, move in _rows(table):
                scanned += 1
                record = encode_record(
                    fen, cp, mate, move, min_ply=min_ply
                )
                if record is None:
                    rejected += 1
                    continue
                records[accepted] = record
                accepted += 1
                if accepted >= target:
                    break
            print(
                f"group {group_index}: {accepted:,}/{target:,} accepted "
                f"from {scanned:,} rows",
                flush=True,
            )
            if accepted >= target:
                break
        records.flush()
        if accepted == 0:
            raise ValueError("no usable positions were found")
        with output.open("wb") as handle:
            np.lib.format.write_array(handle, records[:accepted], allow_pickle=False)
    finally:
        del records
        staging.unlink(missing_ok=True)
    return accepted, {"scanned": scanned, "accepted": accepted, "rejected": rejected}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def pack_dataset(
    source: Path,
    train_output: Path,
    validation_output: Path,
    manifest_output: Path,
    *,
    train_target: int,
    validation_target: int,
    validation_groups: int,
    min_ply: int,
) -> None:
    """Pack training and row-group-disjoint validation datasets."""
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError(
            "packing requires the training-only dependency pyarrow"
        ) from error

    parquet = pq.ParquetFile(source)
    group_count = parquet.metadata.num_row_groups
    if not 0 < validation_groups < group_count:
        raise ValueError(
            f"validation_groups must be between 1 and {group_count - 1}"
        )
    split_at = group_count - validation_groups
    train_groups = list(range(split_at))
    held_out_groups = list(range(split_at, group_count))

    print(
        f"{source.name}: {parquet.metadata.num_rows:,} rows in {group_count} groups",
        flush=True,
    )
    train_count, train_stats = pack_groups(
        parquet,
        train_groups,
        train_output,
        target=train_target,
        min_ply=min_ply,
    )
    validation_count, validation_stats = pack_groups(
        parquet,
        held_out_groups,
        validation_output,
        target=validation_target,
        min_ply=min_ply,
    )
    manifest = {
        "schema_version": 1,
        "source": source.as_posix(),
        "source_sha256": _sha256(source),
        "source_rows": parquet.metadata.num_rows,
        "source_row_groups": group_count,
        "train_groups": train_groups,
        "validation_groups": held_out_groups,
        "train_output": train_output.as_posix(),
        "validation_output": validation_output.as_posix(),
        "train_count": train_count,
        "validation_count": validation_count,
        "train_stats": train_stats,
        "validation_stats": validation_stats,
        "min_ply": min_ply,
        "cp_clamp": CP_CLAMP,
        "filters": [
            "valid_standard_fen",
            "both_kings_present",
            "non_terminal",
            "not_in_check",
            "teacher_move_legal_and_non_capture_when_present",
        ],
        "label_perspective": "white",
    }
    atomic_write_text(manifest_output, json.dumps(manifest, indent=2) + "\n")
    print(
        f"wrote {train_output} ({train_count:,}) and "
        f"{validation_output} ({validation_count:,})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--train-output", type=Path, required=True)
    parser.add_argument("--validation-output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--train-target", type=positive_int, default=4_000_000)
    parser.add_argument("--validation-target", type=positive_int, default=500_000)
    parser.add_argument("--validation-groups", type=positive_int, default=1)
    parser.add_argument("--min-ply", type=nonnegative_int, default=12)
    args = parser.parse_args()
    if not args.source.is_file():
        parser.error(f"source Parquet file not found: {args.source}")
    pack_dataset(
        args.source,
        args.train_output,
        args.validation_output,
        args.manifest,
        train_target=args.train_target,
        validation_target=args.validation_target,
        validation_groups=args.validation_groups,
        min_ply=args.min_ply,
    )


if __name__ == "__main__":
    main()
