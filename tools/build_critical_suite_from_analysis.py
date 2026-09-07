"""Turn high-loss moves from PGN analysis JSON into a search regression suite."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import chess


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")


def build_positions(
    payload: dict[str, Any], min_cp_loss: int, limit: int
) -> list[dict[str, str]]:
    """Select the largest legal, non-duplicate errors made by the focused player."""
    candidates: list[tuple[int, str, int, dict[str, Any]]] = []
    for game in payload.get("games", []):
        filename = Path(str(game.get("file", "game"))).stem
        nodes = int(game.get("analysis", {}).get("nodes_per_position", 0))
        for record in game.get("moves", []):
            loss = record.get("cp_loss")
            if not record.get("selected") or not isinstance(loss, int):
                continue
            if loss < min_cp_loss or record.get("best_uci") in {None, record.get("uci")}:
                continue
            candidates.append((loss, filename, nodes, record))
    candidates.sort(key=lambda item: (-item[0], item[1], int(item[3]["ply"])))

    positions: list[dict[str, str]] = []
    seen_fens: set[str] = set()
    seen_ids: set[str] = set()
    for loss, filename, nodes, record in candidates:
        fen = str(record["fen"])
        if fen in seen_fens:
            continue
        board = chess.Board(fen)
        played = str(record["uci"])
        reference = str(record["best_uci"])
        legal = {move.uci() for move in board.legal_moves}
        if played not in legal or reference not in legal:
            continue
        identifier = f"{_slug(filename)}-ply-{int(record['ply'])}"
        suffix = 2
        base_identifier = identifier
        while identifier in seen_ids:
            identifier = f"{base_identifier}-{suffix}"
            suffix += 1
        seen_ids.add(identifier)
        seen_fens.add(fen)
        positions.append(
            {
                "id": identifier,
                "label": f"{filename}, ply {int(record['ply'])} ({record['san']})",
                "fen": fen,
                "played_move": played,
                "reference_move": reference,
                "baseline_move": played,
                "diagnosis": (
                    f"Focused player chose {record['san']} ({played}); the "
                    f"{nodes} node teacher preferred "
                    f"{reference}, with an estimated loss of {loss} cp."
                ),
            }
        )
        if len(positions) >= limit:
            break
    return positions


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--min-cp-loss", type=int, default=80)
    parser.add_argument("--limit", type=int, default=25)
    args = parser.parse_args()
    if args.min_cp_loss < 0 or args.limit <= 0:
        parser.error("--min-cp-loss must be nonnegative and --limit positive")

    payload = json.loads(args.analysis.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or not isinstance(payload.get("games"), list):
        parser.error("analysis must be schema-version 1 PGN analysis JSON")
    positions = build_positions(payload, args.min_cp_loss, args.limit)
    if not positions:
        parser.error("analysis contains no qualifying focused-player errors")
    nodes = sorted(
        {
            int(game["analysis"]["nodes_per_position"])
            for game in payload["games"]
        }
    )
    suite = {
        "schema_version": 1,
        "description": (
            "Search regressions selected from machine-readable rated-PGN analysis; "
            f"minimum loss {args.min_cp_loss} cp, teacher node limits {nodes}."
        ),
        "positions": positions,
    }
    _write_json(args.output, suite)
    print(f"wrote {len(positions)} positions to {args.output.resolve()}")


if __name__ == "__main__":
    main()
