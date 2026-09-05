"""Build a deterministic, phase-balanced evaluation suite from PGN games."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import chess
import chess.pgn

from tools.backtest_core import atomic_write_text, fingerprint_file
from tools.cli import positive_int
from tools.evaluation_dataset import evaluation_identity
from tools.evaluation_features import game_phase

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
    rank: int
    identifier: str
    fen: str
    stratum: str


def position_stratum(board: chess.Board) -> str:
    phase = game_phase(board)
    phase_name = "opening" if phase >= 18 else "middlegame" if phase >= 7 else "endgame"
    tactical = board.is_check() or any(
        board.is_capture(move) or move.promotion is not None for move in board.legal_moves
    )
    return f"{phase_name}-{'tactical' if tactical else 'quiet'}"


def sample_rank(seed: str, fen: str) -> int:
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


def _consider(
    heaps: dict[str, list[tuple[int, str, SampledPosition]]],
    candidate: SampledPosition,
    capacity: int,
) -> None:
    heap = heaps[candidate.stratum]
    item = (-candidate.rank, candidate.fen, candidate)
    if len(heap) < capacity:
        heapq.heappush(heap, item)
    elif candidate.rank < -heap[0][0]:
        heapq.heapreplace(heap, item)


def collect_positions(
    sources: list[Path],
    *,
    count: int,
    seed: str,
    min_ply: int,
    max_ply: int,
    stride: int,
) -> tuple[list[SampledPosition], int]:
    """Stream PGNs while retaining only the best hash-ranked candidates."""
    heaps: dict[str, list[tuple[int, str, SampledPosition]]] = {
        name: [] for name in STRATA
    }
    seen: set[str] = set()
    games = 0
    capacity = count
    for source in sources:
        with source.open(encoding="utf-8-sig") as handle:
            while True:
                game = chess.pgn.read_game(handle)
                if game is None:
                    break
                games += 1
                if game.errors:
                    raise ValueError(f"{source}: game {games} contains PGN errors: {game.errors}")
                board = game.board()
                for ply, move in enumerate(game.mainline_moves(), start=1):
                    board.push(move)
                    if ply < min_ply or ply > max_ply or (ply - min_ply) % stride:
                        continue
                    if board.is_game_over(claim_draw=True):
                        continue
                    fen = evaluation_identity(board)
                    if fen in seen:
                        continue
                    seen.add(fen)
                    stratum = position_stratum(board)
                    identifier = f"g{games:07d}-p{ply:03d}-{stratum}"
                    _consider(
                        heaps,
                        SampledPosition(sample_rank(seed, fen), identifier, fen, stratum),
                        capacity,
                    )
    buckets = {
        name: [item[2] for item in heap]
        for name, heap in heaps.items()
    }
    return balanced_selection(buckets, count), games


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=positive_int, required=True)
    parser.add_argument("--seed", default="v4-residual-eval-v1")
    parser.add_argument("--min-ply", type=positive_int, default=12)
    parser.add_argument("--max-ply", type=positive_int, default=160)
    parser.add_argument("--stride", type=positive_int, default=4)
    args = parser.parse_args()

    sources = [path.resolve() for path in args.source]
    missing = [path for path in sources if not path.is_file()]
    if missing:
        parser.error(f"PGN source not found: {missing[0]}")
    if args.max_ply < args.min_ply:
        parser.error("--max-ply must be at least --min-ply")

    selected, games = collect_positions(
        sources,
        count=args.count,
        seed=args.seed,
        min_ply=args.min_ply,
        max_ply=args.max_ply,
        stride=args.stride,
    )
    if len(selected) < args.count:
        parser.error(f"sources yielded only {len(selected)} eligible unique positions")

    lines = [chess.Board(item.fen).epd(id=item.identifier) for item in selected]
    payload = "\n".join(lines) + "\n"
    atomic_write_text(args.output, payload)
    manifest = {
        "schema_version": 1,
        "seed": args.seed,
        "count": len(selected),
        "games_read": games,
        "min_ply": args.min_ply,
        "max_ply": args.max_ply,
        "stride": args.stride,
        "strata": dict(sorted(Counter(item.stratum for item in selected).items())),
        "sources": [
            {"path": str(path), "sha256": fingerprint_file(path)["sha256"]}
            for path in sources
        ],
        "output_sha256": hashlib.sha256(payload.encode()).hexdigest(),
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    atomic_write_text(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output} ({len(selected)} positions from {games} games)")
    print(f"strata: {manifest['strata']}")
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
