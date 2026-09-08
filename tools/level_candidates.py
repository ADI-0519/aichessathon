from __future__ import annotations

import argparse
import random
from pathlib import Path

import chess
import pyarrow.parquet as pq

SOURCE = Path("scratch/nnue/sources/standard_rated_2014_09.parquet")
HELD_OUT_GROUP = 7


def main() -> None:
    parser = argparse.ArgumentParser(
        description="candidate opening positions from held-out human games"
    )
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--group", type=int, default=HELD_OUT_GROUP)
    parser.add_argument("--min-ply", type=int, default=16)
    parser.add_argument("--max-ply", type=int, default=36)
    parser.add_argument("--label-cp", type=int, default=60)
    parser.add_argument("--min-men", type=int, default=24)
    parser.add_argument("--count", type=int, default=600)
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()

    table = pq.ParquetFile(arguments.source).read_row_group(
        arguments.group, columns=["fen", "cp", "mate"]
    )
    fens = table["fen"].to_pylist()
    cps = table["cp"].to_pylist()
    mates = table["mate"].to_pylist()
    print(f"{len(fens)} rows in group {arguments.group} of {arguments.source.name}")

    kept: list[str] = []
    seen_positions: set[str] = set()
    structures: dict[str, int] = {}
    for fen, cp, mate in zip(fens, cps, mates, strict=True):
        if mate is not None or cp is None or abs(int(cp)) > arguments.label_cp:
            continue
        fields = fen.split()
        if len(fields) != 6:
            continue
        ply = 2 * (int(fields[5]) - 1) + (fields[1] == "b")
        if not arguments.min_ply <= ply <= arguments.max_ply:
            continue
        placement = fields[0]
        if placement in seen_positions:
            continue
        try:
            board = chess.Board(fen)
        except ValueError:
            continue
        if board.is_check() or chess.popcount(board.occupied) < arguments.min_men:
            continue
        # cap how many positions share a pawn skeleton to ensure varied suit
        pawns = "".join(
            "P" if board.piece_type_at(sq) == chess.PAWN else "."
            for sq in chess.SQUARES
        )
        if structures.get(pawns, 0) >= 2:
            continue
        structures[pawns] = structures.get(pawns, 0) + 1
        seen_positions.add(placement)
        kept.append(fen)

    print(f"{len(kept)} candidates survive the ply, label, material and variety filters")
    rng = random.Random(arguments.seed)
    if len(kept) > arguments.count:
        kept = rng.sample(kept, arguments.count)
    arguments.out.write_text("\n".join(kept) + "\n")
    print(f"wrote {len(kept)} candidates to {arguments.out}")


if __name__ == "__main__":
    main()
