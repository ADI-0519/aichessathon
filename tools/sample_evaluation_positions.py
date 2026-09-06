"""Build a deterministic, phase-balanced evaluation suite from chess positions."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import os
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, cast

import chess
import chess.pgn

from tools.backtest_core import atomic_write_text, fingerprint_file
from tools.cli import nonnegative_int, positive_int
from tools.evaluation_dataset import evaluation_identity
from tools.evaluation_features import game_phase

SourceFormat = Literal["pgn", "epd", "fen"]

STRATA = (
    "opening-quiet",
    "opening-tactical",
    "middlegame-quiet",
    "middlegame-tactical",
    "endgame-quiet",
    "endgame-tactical",
)


@dataclass(frozen=True, slots=True)
class SampledPosition:
    """One deterministic sampling candidate."""

    rank: int
    identifier: str
    fen: str
    stratum: str


@dataclass(slots=True)
class SourceStatistics:
    """Collection counters for one source file."""

    path: str
    format: SourceFormat
    source_url: str | None
    records_read: int = 0
    positions_examined: int = 0
    unique_positions: int = 0
    duplicates_skipped: int = 0
    terminal_positions_skipped: int = 0


@dataclass(frozen=True, slots=True)
class CollectionStatistics:
    """Auditable totals and per-source collection counters."""

    sources: tuple[SourceStatistics, ...]

    @property
    def records_read(self) -> int:
        return sum(source.records_read for source in self.sources)

    @property
    def positions_examined(self) -> int:
        return sum(source.positions_examined for source in self.sources)

    @property
    def unique_positions(self) -> int:
        return sum(source.unique_positions for source in self.sources)

    @property
    def duplicates_skipped(self) -> int:
        return sum(source.duplicates_skipped for source in self.sources)

    @property
    def terminal_positions_skipped(self) -> int:
        return sum(source.terminal_positions_skipped for source in self.sources)


def position_stratum(board: chess.Board) -> str:
    """Classify a position by material phase and immediate forcing content."""
    phase = game_phase(board)
    phase_name = "opening" if phase >= 18 else "middlegame" if phase >= 7 else "endgame"
    tactical = board.is_check() or any(
        board.is_capture(move) or move.promotion is not None for move in board.legal_moves
    )
    return f"{phase_name}-{'tactical' if tactical else 'quiet'}"


def sample_rank(seed: str, fen: str) -> int:
    """Return a stable pseudo-random rank without relying on input ordering."""
    return int.from_bytes(hashlib.sha256(f"{seed}\0{fen}".encode()).digest(), "big")


def balanced_selection(
    buckets: dict[str, list[SampledPosition]], count: int
) -> list[SampledPosition]:
    """Round-robin deterministic strata, redistributing unavailable quota."""
    ordered = {
        name: sorted(buckets.get(name, ()), key=lambda item: (item.rank, item.fen))
        for name in STRATA
    }
    offsets = {name: 0 for name in STRATA}
    selected: list[SampledPosition] = []
    while len(selected) < count:
        made_progress = False
        for name in STRATA:
            index = offsets[name]
            if index >= len(ordered[name]):
                continue
            selected.append(ordered[name][index])
            offsets[name] += 1
            made_progress = True
            if len(selected) == count:
                break
        if not made_progress:
            break
    return selected


def source_format(path: Path) -> SourceFormat:
    """Infer the supported source format from its filename."""
    suffix = path.suffix.lower()
    if suffix not in {".pgn", ".epd", ".fen"}:
        raise ValueError(f"unsupported source type {suffix!r}; expected .pgn, .epd, or .fen")
    return cast(SourceFormat, suffix[1:])


def _consider(
    heaps: dict[str, list[tuple[int, str, SampledPosition]]],
    candidate: SampledPosition,
    capacity: int,
) -> None:
    """Retain the lowest hash-ranked candidates while keeping memory bounded."""
    heap = heaps[candidate.stratum]
    item = (-candidate.rank, candidate.fen, candidate)
    if len(heap) < capacity:
        heapq.heappush(heap, item)
    elif candidate.rank < -heap[0][0]:
        heapq.heapreplace(heap, item)


def _collect_board(
    board: chess.Board,
    *,
    identifier: str,
    seed: str,
    heaps: dict[str, list[tuple[int, str, SampledPosition]]],
    seen: set[str],
    capacity: int,
    statistics: SourceStatistics,
) -> None:
    statistics.positions_examined += 1
    if board.is_game_over(claim_draw=True):
        statistics.terminal_positions_skipped += 1
        return

    fen = evaluation_identity(board)
    if fen in seen:
        statistics.duplicates_skipped += 1
        return
    seen.add(fen)
    statistics.unique_positions += 1

    stratum = position_stratum(board)
    _consider(
        heaps,
        SampledPosition(
            rank=sample_rank(seed, fen),
            identifier=f"{identifier}-{stratum}",
            fen=fen,
            stratum=stratum,
        ),
        capacity,
    )


def _validate_board(board: chess.Board, context: str) -> None:
    if not board.is_valid():
        raise ValueError(
            f"{context}: position is not a valid chess position (status={board.status()})"
        )


def _collect_pgn(
    source: Path,
    *,
    game_offset: int,
    seed: str,
    min_ply: int,
    max_ply: int,
    stride: int,
    heaps: dict[str, list[tuple[int, str, SampledPosition]]],
    seen: set[str],
    capacity: int,
    statistics: SourceStatistics,
) -> int:
    with source.open(encoding="utf-8-sig") as handle:
        while True:
            game = chess.pgn.read_game(handle)
            if game is None:
                return game_offset + statistics.records_read
            statistics.records_read += 1
            game_number = game_offset + statistics.records_read
            if game.errors:
                raise ValueError(
                    f"{source}: game {game_number} contains PGN errors: {game.errors}"
                )
            board = game.board()
            _validate_board(board, f"{source}: game {game_number} initial position")
            for ply, move in enumerate(game.mainline_moves(), start=1):
                if move not in board.legal_moves:
                    raise ValueError(f"{source}: game {game_number} has illegal move {move}")
                board.push(move)
                if ply < min_ply or ply > max_ply or (ply - min_ply) % stride:
                    continue
                _collect_board(
                    board,
                    identifier=f"g{game_number:07d}-p{ply:03d}",
                    seed=seed,
                    heaps=heaps,
                    seen=seen,
                    capacity=capacity,
                    statistics=statistics,
                )


def _collect_epd_or_fen(
    source: Path,
    *,
    source_index: int,
    seed: str,
    heaps: dict[str, list[tuple[int, str, SampledPosition]]],
    seen: set[str],
    capacity: int,
    statistics: SourceStatistics,
) -> None:
    prefix = "e" if statistics.format == "epd" else "f"
    with source.open(encoding="utf-8-sig") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            statistics.records_read += 1
            try:
                if statistics.format == "epd":
                    board = chess.Board()
                    board.set_epd(line)
                else:
                    board = chess.Board(line)
                _validate_board(board, f"{source}:{line_number}")
            except ValueError as error:
                raise ValueError(
                    f"{source}:{line_number}: invalid {statistics.format.upper()}: {error}"
                ) from error
            _collect_board(
                board,
                identifier=f"{prefix}{source_index:03d}-l{line_number:07d}",
                seed=seed,
                heaps=heaps,
                seen=seen,
                capacity=capacity,
                statistics=statistics,
            )


def collect_positions(
    sources: list[Path],
    *,
    count: int,
    seed: str,
    min_ply: int,
    max_ply: int,
    stride: int,
    source_urls: list[str] | None = None,
) -> tuple[list[SampledPosition], CollectionStatistics]:
    """Stream supported sources while retaining bounded hash-ranked samples."""
    if source_urls is not None and len(source_urls) != len(sources):
        raise ValueError("source_urls must contain exactly one URL per source")

    heaps: dict[str, list[tuple[int, str, SampledPosition]]] = {
        name: [] for name in STRATA
    }
    seen: set[str] = set()
    statistics: list[SourceStatistics] = []
    game_offset = 0
    for source_index, source in enumerate(sources, start=1):
        file_format = source_format(source)
        source_statistics = SourceStatistics(
            path=str(source),
            format=file_format,
            source_url=None if source_urls is None else source_urls[source_index - 1],
        )
        statistics.append(source_statistics)
        if file_format == "pgn":
            game_offset = _collect_pgn(
                source,
                game_offset=game_offset,
                seed=seed,
                min_ply=min_ply,
                max_ply=max_ply,
                stride=stride,
                heaps=heaps,
                seen=seen,
                capacity=count,
                statistics=source_statistics,
            )
        else:
            _collect_epd_or_fen(
                source,
                source_index=source_index,
                seed=seed,
                heaps=heaps,
                seen=seen,
                capacity=count,
                statistics=source_statistics,
            )

    buckets = {name: [item[2] for item in heap] for name, heap in heaps.items()}
    return balanced_selection(buckets, count), CollectionStatistics(tuple(statistics))


def _portable_path(path: Path, relative_to: Path) -> str:
    try:
        return Path(os.path.relpath(path.resolve(), relative_to.resolve())).as_posix()
    except ValueError:  # Different Windows drives cannot be made relative.
        return str(path.resolve())


def build_evaluation_suite(
    sources: list[Path],
    output: Path,
    *,
    count: int,
    seed: str,
    min_ply: int,
    max_ply: int,
    stride: int,
    source_urls: list[str] | None = None,
    min_per_stratum: int = 0,
    force: bool = False,
) -> tuple[list[SampledPosition], dict[str, object]]:
    """Build the suite and manifest atomically, refusing accidental replacement."""
    manifest_path = output.with_suffix(".manifest.json")
    existing = [path for path in (output, manifest_path) if path.exists()]
    if existing and not force:
        raise FileExistsError(f"refusing to overwrite {existing[0]}; pass --force to replace it")
    output.parent.mkdir(parents=True, exist_ok=True)

    selected, statistics = collect_positions(
        sources,
        count=count,
        seed=seed,
        min_ply=min_ply,
        max_ply=max_ply,
        stride=stride,
        source_urls=source_urls,
    )
    if len(selected) < count:
        raise ValueError(f"sources yielded only {len(selected)} eligible unique positions")
    stratum_counts = Counter(item.stratum for item in selected)
    undersupplied = {
        name: stratum_counts[name] for name in STRATA if stratum_counts[name] < min_per_stratum
    }
    if undersupplied:
        details = ", ".join(f"{name}={amount}" for name, amount in undersupplied.items())
        raise ValueError(
            f"sample does not meet --min-per-stratum {min_per_stratum}: {details}"
        )

    payload = "\n".join(chess.Board(item.fen).epd(id=item.identifier) for item in selected) + "\n"
    output_digest = hashlib.sha256(payload.encode()).hexdigest()
    source_records: list[dict[str, object]] = []
    for source_statistics, source in zip(statistics.sources, sources, strict=True):
        record: dict[str, object] = asdict(source_statistics)
        record.update(fingerprint_file(source))
        record["path"] = _portable_path(source, manifest_path.parent)
        source_records.append(record)

    manifest: dict[str, object] = {
        "schema_version": 2,
        "seed": seed,
        "count": len(selected),
        "sampling": {
            "method": "sha256_rank_then_stratum_round_robin",
            "identity": "normalized_fen_without_move_counters",
            "min_per_stratum": min_per_stratum,
            "strata": dict(sorted(stratum_counts.items())),
        },
        "pgn_sampling": {"min_ply": min_ply, "max_ply": max_ply, "stride": stride},
        "counts": {
            "records_read": statistics.records_read,
            "positions_examined": statistics.positions_examined,
            "unique_positions": statistics.unique_positions,
            "duplicates_skipped": statistics.duplicates_skipped,
            "terminal_positions_skipped": statistics.terminal_positions_skipped,
        },
        "sources": source_records,
        "output": _portable_path(output, manifest_path.parent),
        "output_sha256": output_digest,
    }
    atomic_write_text(output, payload)
    atomic_write_text(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return selected, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, nargs="+", required=True, help="PGN, EPD, or FEN source files"
    )
    parser.add_argument(
        "--source-url",
        action="append",
        default=[],
        help="provenance URL; repeat once per --source file in the same order",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=positive_int, required=True)
    parser.add_argument("--seed", default="v4-residual-eval-v1")
    parser.add_argument("--min-ply", type=positive_int, default=12)
    parser.add_argument("--max-ply", type=positive_int, default=160)
    parser.add_argument("--stride", type=positive_int, default=4)
    parser.add_argument(
        "--min-per-stratum",
        type=nonnegative_int,
        default=0,
        help="fail unless every phase/tactical stratum has at least this many positions",
    )
    parser.add_argument(
        "--force", action="store_true", help="replace an existing suite and manifest"
    )
    args = parser.parse_args()

    sources = [path.resolve() for path in args.source]
    missing = [path for path in sources if not path.is_file()]
    if missing:
        parser.error(f"source not found: {missing[0]}")
    if args.max_ply < args.min_ply:
        parser.error("--max-ply must be at least --min-ply")
    if args.source_url and len(args.source_url) != len(sources):
        parser.error("repeat --source-url exactly once per --source file, or omit it")
    try:
        for source in sources:
            source_format(source)
        selected, manifest = build_evaluation_suite(
            sources,
            args.output,
            count=args.count,
            seed=args.seed,
            min_ply=args.min_ply,
            max_ply=args.max_ply,
            stride=args.stride,
            source_urls=args.source_url or None,
            min_per_stratum=args.min_per_stratum,
            force=args.force,
        )
    except (FileExistsError, ValueError) as error:
        parser.error(str(error))

    counts = cast(dict[str, int], manifest["counts"])
    sampling = cast(dict[str, object], manifest["sampling"])
    print(
        f"wrote {args.output} ({len(selected)} positions from "
        f"{counts['records_read']} source records)"
    )
    print(f"strata: {sampling['strata']}")
    print(f"wrote {args.output.with_suffix('.manifest.json')}")


if __name__ == "__main__":
    main()
