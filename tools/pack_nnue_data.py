"""Pack Lichess Fishnet Parquet rows into compact sparse-evaluator records.

PyArrow is a training-only dependency and is imported lazily.  The produced
``.npy`` files contain no moves or lookup table: each row is a position, a
teacher evaluation, and sparse piece-square inputs used to train our model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Collection, Iterable, Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
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


@dataclass(frozen=True)
class ReportBand:
    """A configured piece-count band included in packing diagnostics."""

    name: str
    min_pieces: int
    max_pieces: int


def load_report_bands(config_path: Path) -> tuple[ReportBand, ...]:
    """Read material-band definitions from an experiment configuration."""
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    items = raw.get("piece_bands")
    if not isinstance(items, list) or not items:
        raise ValueError("report config must contain a non-empty piece_bands list")
    bands: list[ReportBand] = []
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("each piece band must be an object")
        band = ReportBand(
            name=str(item["name"]),
            min_pieces=int(item["min_pieces"]),
            max_pieces=int(item["max_pieces"]),
        )
        if not band.name:
            raise ValueError("piece-band names cannot be empty")
        if not 0 <= band.min_pieces <= band.max_pieces <= MAX_PIECES:
            raise ValueError(f"invalid piece band: {band}")
        bands.append(band)
    if len({band.name for band in bands}) != len(bands):
        raise ValueError("piece-band names must be unique")
    return tuple(bands)


def shuffled_groups(groups: Sequence[int], seed: int) -> list[int]:
    """Return a reproducibly shuffled copy of row-group indices."""
    order = np.asarray(groups, dtype=np.int64)
    np.random.default_rng(seed).shuffle(order)
    return [int(group) for group in order]


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
    next_human_move: str | None,
    *,
    min_ply: int,
) -> np.void | None:
    """Validate and encode one quiet labelled position.

    Positions in check and positions whose next played move is a capture are
    excluded as a cheap quiet-position filter. Fishnet's ``move`` column is the
    human player's move, not the engine's preferred continuation.
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

    if next_human_move:
        try:
            move = board.parse_uci(next_human_move)
        except ValueError:
            return None
        if board.is_capture(move):
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


def position_fingerprint(record: np.void) -> bytes:
    """Return a stable identity for the evaluator inputs in one packed row.

    The teacher score is deliberately excluded. Castling and en-passant state
    are absent from the sparse evaluator's input too, so positions that differ
    only in those fields must not straddle the training/validation boundary.
    A 128-bit digest keeps the reservation set compact while making accidental
    collisions negligible at the intended dataset scale.
    """
    payload = (
        record["indices"].tobytes()
        + int(record["count"]).to_bytes(1, "little")
        + int(record["stm"]).to_bytes(1, "little")
    )
    return hashlib.blake2b(
        payload,
        digest_size=16,
        person=b"kingnet-v11-pos",
    ).digest()


def load_reserved_fingerprints(paths: Sequence[Path]) -> set[bytes]:
    """Load evaluator-input identities that must not enter a training split."""
    fingerprints: set[bytes] = set()
    for path in paths:
        records = np.load(path, mmap_mode="r", allow_pickle=False)
        if records.ndim != 1 or records.dtype != PACKED_DTYPE:
            raise ValueError(f"reserved validation file has wrong schema: {path}")
        fingerprints.update(position_fingerprint(record) for record in records)
    return fingerprints


def _rows(table: Any) -> Iterable[tuple[str, int | None, int | None, str | None]]:
    columns = (table[name].to_pylist() for name in ("fen", "cp", "mate", "move"))
    return zip(*columns, strict=True)


def default_workers() -> int:
    """Leave one core for the parent so the machine stays usable while packing."""
    return max(1, (os.cpu_count() or 2) - 1)


def pack_group(source: str, group_index: int, min_ply: int) -> tuple[np.ndarray, int, int]:
    """Encode one row group in a worker process.

    Only the per-position encoding runs here. Deduplication and holdout
    exclusion compare a position against everything accepted before it, so they
    depend on the order groups are consumed in and stay with the consumer; doing
    them here would make the output depend on which worker finished first.
    """
    import pyarrow.parquet as pq

    table = pq.ParquetFile(source).read_row_group(
        group_index, columns=["fen", "cp", "mate", "move"]
    )
    packed = np.zeros(table.num_rows, dtype=PACKED_DTYPE)
    scanned = accepted = rejected = 0
    for fen, cp, mate, move in _rows(table):
        scanned += 1
        record = encode_record(fen, cp, mate, move, min_ply=min_ply)
        if record is None:
            rejected += 1
            continue
        packed[accepted] = record
        accepted += 1
    return packed[:accepted].copy(), scanned, rejected


def _encoded_groups(
    source: Path, groups: Sequence[int], *, min_ply: int, workers: int
) -> Iterator[tuple[int, np.ndarray, int, int]]:
    """Yield each group's encoded records in the order ``groups`` gives them."""
    if workers <= 1 or len(groups) <= 1:
        for group_index in groups:
            packed, scanned, rejected = pack_group(
                source.as_posix(), group_index, min_ply
            )
            yield group_index, packed, scanned, rejected
        return
    with ProcessPoolExecutor(max_workers=min(workers, len(groups))) as pool:
        futures = [
            pool.submit(pack_group, source.as_posix(), group_index, min_ply)
            for group_index in groups
        ]
        try:
            for group_index, future in zip(groups, futures, strict=True):
                packed, scanned, rejected = future.result()
                yield group_index, packed, scanned, rejected
        finally:
            for future in futures:
                future.cancel()


def pack_groups(
    source: Path,
    groups: Sequence[int],
    output: Path,
    *,
    target: int,
    min_ply: int,
    workers: int = 1,
    report_bands: Sequence[ReportBand] = (),
    excluded_fingerprints: Collection[bytes] = (),
    deduplicate: bool = False,
    collect_fingerprints: bool = False,
) -> tuple[int, dict[str, Any], set[bytes]]:
    """Encode selected row groups into ``output`` and return counts."""
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_suffix(output.suffix + ".staging.npy")
    records = np.lib.format.open_memmap(
        staging, mode="w+", dtype=PACKED_DTYPE, shape=(target,)
    )
    scanned = accepted = rejected = 0
    excluded_overlap = within_split_duplicates = 0
    accepted_by_piece_count = [0] * (MAX_PIECES + 1)
    accepted_fingerprints: set[bytes] = set()
    try:
        for group_index, packed, group_scanned, group_rejected in _encoded_groups(
            source, groups, min_ply=min_ply, workers=workers
        ):
            scanned += group_scanned
            rejected += group_rejected
            for index in range(len(packed)):
                record = packed[index]
                fingerprint = position_fingerprint(record)
                if fingerprint in excluded_fingerprints:
                    rejected += 1
                    excluded_overlap += 1
                    continue
                if deduplicate and fingerprint in accepted_fingerprints:
                    rejected += 1
                    within_split_duplicates += 1
                    continue
                records[accepted] = record
                if deduplicate or collect_fingerprints:
                    accepted_fingerprints.add(fingerprint)
                accepted_by_piece_count[int(record["count"])] += 1
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
    accepted_by_material_band = {
        band.name: sum(
            accepted_by_piece_count[band.min_pieces : band.max_pieces + 1]
        )
        for band in report_bands
    }
    return accepted, {
        "scanned": scanned,
        "accepted": accepted,
        "rejected": rejected,
        "excluded_position_overlap": excluded_overlap,
        "within_split_duplicates": within_split_duplicates,
        "accepted_by_piece_count": accepted_by_piece_count,
        "accepted_by_material_band": accepted_by_material_band,
    }, accepted_fingerprints


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
    group_order_seed: int = 0,
    report_bands: Sequence[ReportBand] = (),
    external_validation: Sequence[Path] = (),
    workers: int = 1,
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
    train_group_order = shuffled_groups(train_groups, group_order_seed)
    external_fingerprints = load_reserved_fingerprints(external_validation)

    print(
        f"{source.name}: {parquet.metadata.num_rows:,} rows in {group_count} groups",
        flush=True,
    )
    # Reserve validation inputs first. Merely using a different Parquet row
    # group is insufficient because ordinary chess positions recur in games.
    validation_count, validation_stats, validation_fingerprints = pack_groups(
        source,
        held_out_groups,
        validation_output,
        target=validation_target,
        min_ply=min_ply,
        report_bands=report_bands,
        deduplicate=True,
        collect_fingerprints=True,
        workers=workers,
    )
    reserved_fingerprints = external_fingerprints | validation_fingerprints
    train_count, train_stats, _ = pack_groups(
        source,
        train_group_order,
        train_output,
        target=train_target,
        min_ply=min_ply,
        report_bands=report_bands,
        excluded_fingerprints=reserved_fingerprints,
        workers=workers,
    )
    manifest = {
        "schema_version": 2,
        "source": source.as_posix(),
        "source_sha256": _sha256(source),
        "source_rows": parquet.metadata.num_rows,
        "source_row_groups": group_count,
        "train_groups": train_groups,
        "train_group_order": train_group_order,
        "group_order_seed": group_order_seed,
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
            "next_human_move_legal_and_non_capture_when_present",
        ],
        "label_perspective": "white",
        "move_semantics": "next_human_move",
        "split_identity": "packed evaluator inputs excluding teacher score",
        "validation_deduplicated": True,
        "training_excludes_validation_inputs": True,
        "training_deduplicated": False,
        "external_validation": [path.as_posix() for path in external_validation],
        "reserved_validation_fingerprints": len(reserved_fingerprints),
        "report_bands": [
            {
                "name": band.name,
                "min_pieces": band.min_pieces,
                "max_pieces": band.max_pieces,
            }
            for band in report_bands
        ],
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
    parser.add_argument(
        "--workers",
        type=positive_int,
        default=default_workers(),
        help="row groups to encode concurrently (1 disables the pool)",
    )
    parser.add_argument("--group-order-seed", type=nonnegative_int, default=0)
    parser.add_argument(
        "--exclude-validation",
        type=Path,
        action="append",
        default=[],
        help=(
            "packed validation .npy whose evaluator inputs must be excluded "
            "from training; repeat for multiple files"
        ),
    )
    parser.add_argument(
        "--report-config",
        type=Path,
        help="experiment JSON whose piece_bands are reported in the manifest",
    )
    args = parser.parse_args()
    if not args.source.is_file():
        parser.error(f"source Parquet file not found: {args.source}")
    if args.report_config is not None and not args.report_config.is_file():
        parser.error(f"report config not found: {args.report_config}")
    missing_validation = [
        path for path in args.exclude_validation if not path.is_file()
    ]
    if missing_validation:
        parser.error(f"reserved validation file not found: {missing_validation[0]}")
    try:
        report_bands = (
            ()
            if args.report_config is None
            else load_report_bands(args.report_config)
        )
        pack_dataset(
            args.source,
            args.train_output,
            args.validation_output,
            args.manifest,
            train_target=args.train_target,
            validation_target=args.validation_target,
            validation_groups=args.validation_groups,
            min_ply=args.min_ply,
            group_order_seed=args.group_order_seed,
            report_bands=report_bands,
            external_validation=args.exclude_validation,
        )
    except (KeyError, TypeError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
