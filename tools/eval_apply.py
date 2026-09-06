from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

import engine
import search
from tools import eval_features as features

OLD_TABLES = """

def _relative_rank(square: int, color: int) -> int:
    rank = square >> 3
    return rank if color == engine.WHITE else 7 - rank


def _piece_square(kind: int, square: int, color: int) -> tuple[int, int]:
    file_index = square & 7
    rank = _relative_rank(square, color)
    center_distance = abs(2 * file_index - 7) + abs(2 * rank - 7)
    center = 14 - center_distance
    if kind == engine.PAWN:
        central_file = 4 - abs(2 * file_index - 7)
        return rank * 7 + central_file * 2, rank * 12 + central_file
    if kind == engine.KNIGHT:
        return center * 4, center * 3
    if kind == engine.BISHOP:
        return center * 2 + rank * 2, center * 2
    if kind == engine.ROOK:
        seventh = 22 if rank == 6 else 0
        return seventh + rank, seventh + rank * 2
    if kind == engine.QUEEN:
        return center - rank * 2, center * 2
    if kind == engine.KING:
        home_safety = 28 if rank == 0 and file_index in (2, 6) else 0
        return home_safety - center * 5, center * 5
    return 0, 0


MG_PST = np.zeros((engine.PIECE_BITBOARD_COUNT, 64), dtype=np.int32)
EG_PST = np.zeros((engine.PIECE_BITBOARD_COUNT, 64), dtype=np.int32)
for _color in (engine.WHITE, engine.BLACK):
    for _kind in range(engine.PIECE_KIND_COUNT):
        for _square in range(64):
            _mg, _eg = _piece_square(_kind, _square, _color)
            _index = _color * engine.PIECE_KIND_COUNT + _kind
            MG_PST[_index, _square] = _mg
            EG_PST[_index, _square] = _eg
"""

NAMES = ("pawn", "knight", "bishop", "rook", "queen", "king")
SCALARS = (
    "F_BISHOP_PAIR",
    "F_DOUBLED",
    "F_ISOLATED",
    "F_PASSED_RANK",
    "F_PASSED_RANK_SQUARED",
    "F_ROOK_SEMI_OPEN",
    "F_ROOK_OPEN",
    "F_KING_SHIELD",
)


def _table(weights: NDArray[np.int64], values: NDArray[np.int32]) -> str:
    lines = []
    for kind in range(engine.PIECE_KIND_COUNT):
        entries = weights[kind * 64 : kind * 64 + 64] - int(values[kind])
        if kind == engine.PAWN:
            # pawn never stands on first or last rank, so nothing constrained those
            entries = entries.copy()
            entries[:8] = 0
            entries[56:] = 0
        lines.append(f"        # {NAMES[kind]}")
        lines.append("        (")
        for rank in range(8):
            row = ", ".join(f"{int(v):>4}" for v in entries[rank * 8 : rank * 8 + 8])
            lines.append(f"            {row},")
        lines.append("        ),")
    return "\n".join(lines)


def _term(name: str, weight: int, suffix: str = "") -> str:
    # keep the subtraction where fitted weight is a penalty, as source reads now
    if weight <= 0:
        return f"{name} -= sign{suffix} * {-int(weight)}"
    return f"{name} += sign{suffix} * {int(weight)}"


def _patch(base: Path, mg: NDArray[np.int64], eg: NDArray[np.int64], tempo: int) -> str:
    tables = f"""
# fitted to Stockfish labels (white orientation, a1 first, black mirrors rank)
MG_PST_TABLE = np.array(
    (
{_table(mg, search.MG_VALUE)}
    ),
    dtype=np.int32,
)
EG_PST_TABLE = np.array(
    (
{_table(eg, search.EG_VALUE)}
    ),
    dtype=np.int32,
)

MG_PST = np.zeros((engine.PIECE_BITBOARD_COUNT, 64), dtype=np.int32)
EG_PST = np.zeros((engine.PIECE_BITBOARD_COUNT, 64), dtype=np.int32)
for _color in (engine.WHITE, engine.BLACK):
    _flip = 0 if _color == engine.WHITE else 56
    for _kind in range(engine.PIECE_KIND_COUNT):
        _index = _color * engine.PIECE_KIND_COUNT + _kind
        for _square in range(64):
            MG_PST[_index, _square] = MG_PST_TABLE[_kind, _square ^ _flip]
            EG_PST[_index, _square] = EG_PST_TABLE[_kind, _square ^ _flip]
"""
    source = base.read_text()
    if OLD_TABLES not in source:
        raise SystemExit(f"{base} no longer holds the piece-square block this replaces")
    source = source.replace(OLD_TABLES, tables)

    replacements = (
        (
            "            middlegame += sign * 32\n            endgame += sign * 42",
            f"            {_term('middlegame', mg[features.F_BISHOP_PAIR])}\n"
            f"            {_term('endgame', eg[features.F_BISHOP_PAIR])}",
        ),
        (
            "                middlegame -= sign * 11\n                endgame -= sign * 14",
            f"                {_term('middlegame', mg[features.F_DOUBLED])}\n"
            f"                {_term('endgame', eg[features.F_DOUBLED])}",
        ),
        (
            "                middlegame -= sign * 10\n                endgame -= sign * 8",
            f"                {_term('middlegame', mg[features.F_ISOLATED])}\n"
            f"                {_term('endgame', eg[features.F_ISOLATED])}",
        ),
        (
            "                middlegame += sign * relative_rank * 7\n"
            "                endgame += sign * relative_rank * relative_rank * 5",
            "                "
            + _term("middlegame", mg[features.F_PASSED_RANK], " * relative_rank")
            + "\n                "
            + _term(
                "endgame",
                eg[features.F_PASSED_RANK_SQUARED],
                " * relative_rank * relative_rank",
            ),
        ),
        (
            "                middlegame += sign * 12\n                endgame += sign * 8",
            f"                {_term('middlegame', mg[features.F_ROOK_SEMI_OPEN])}\n"
            f"                {_term('endgame', eg[features.F_ROOK_SEMI_OPEN])}",
        ),
        (
            "                    middlegame += sign * 10\n                    endgame += sign * 6",
            f"                    {_term('middlegame', mg[features.F_ROOK_OPEN])}\n"
            f"                    {_term('endgame', eg[features.F_ROOK_OPEN])}",
        ),
        (
            "                        middlegame += sign * 9",
            f"                        {_term('middlegame', mg[features.F_KING_SHIELD])}",
        ),
        (
            "    score += 10 if int(state[engine.STATE_SIDE]) == engine.WHITE else -10",
            f"    score += {tempo} if int(state[engine.STATE_SIDE]) == engine.WHITE else -{tempo}",
        ),
    )
    for old, new in replacements:
        if source.count(old) != 1:
            raise SystemExit(f"expected exactly one of: {old.strip()!r}")
        source = source.replace(old, new)
    return source


def _resync_mirror(path: Path, mg: NDArray[np.int64], eg: NDArray[np.int64], tempo: int) -> None:
    text = path.read_text()
    head, _, rest = text.partition("SCALAR_WEIGHTS = {")
    _, _, tail = rest.partition("TEMPO = ")
    _, _, tail = tail.partition("\n")
    body = "\n".join(
        f"    {name}: ({int(mg[getattr(features, name)])}, {int(eg[getattr(features, name)])}),"
        for name in SCALARS
    )
    path.write_text(f"{head}SCALAR_WEIGHTS = {{\n{body}\n}}\nTEMPO = {tempo}\n{tail}")


def main() -> None:
    parser = argparse.ArgumentParser(description="install fitted weights into evaluate()")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True, help="the unfitted search.py to patch")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mirror", type=Path, default=Path("tools/eval_features.py"))
    arguments = parser.parse_args()

    with np.load(arguments.weights) as stored:
        mg, eg, tempo = stored["middlegame"], stored["endgame"], int(stored["tempo"])
    arguments.out.write_text(_patch(arguments.base, mg, eg, tempo))
    _resync_mirror(arguments.mirror, mg, eg, tempo)
    print(f"{arguments.out} from {arguments.base}, tempo {tempo}; {arguments.mirror} resynced")


if __name__ == "__main__":
    main()
