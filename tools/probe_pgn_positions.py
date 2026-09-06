"""Probe an agent's choices at selected positions from a PGN.

This is a development-only diagnostic and is never included in submission.zip.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import sys
import time
from pathlib import Path
from types import ModuleType

import chess
import chess.pgn


def load_agent(path: Path, serial: int) -> ModuleType:
    """Load an agent as an isolated package so relative imports stay local."""
    root = path.resolve()
    module_path = root / "agent.py"
    token = hashlib.sha256(str(root).encode()).hexdigest()[:12]
    name = f"probe_agent_{serial}_{token}"
    spec = importlib.util.spec_from_file_location(
        name,
        module_path,
        submodule_search_locations=[str(root)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", action="append", required=True, type=Path)
    parser.add_argument("--color", required=True, choices=("white", "black"))
    parser.add_argument("--fullmoves", required=True, help="Comma-separated move numbers")
    parser.add_argument("--time-ms", required=True, type=int)
    parser.add_argument("pgn", type=Path)
    args = parser.parse_args()

    with args.pgn.open(encoding="utf-8") as handle:
        game = chess.pgn.read_game(handle)
    if game is None:
        raise ValueError(f"no game in {args.pgn}")

    focus = chess.WHITE if args.color == "white" else chess.BLACK
    selected = {int(value) for value in args.fullmoves.split(",")}
    positions: list[tuple[int, str, str]] = []
    board = game.board()
    for node in game.mainline():
        move_number = board.fullmove_number
        if board.turn == focus and move_number in selected:
            positions.append((move_number, board.fen(), node.move.uci()))
        board.push(node.move)

    print(f"{args.pgn.name}: {args.color}, {args.time_ms} ms per probe")
    for move_number, fen, played in positions:
        outputs: list[str] = []
        for serial, path in enumerate(args.agent):
            agent = load_agent(path, serial)
            started = time.monotonic()
            chosen = agent.get_move(fen, args.time_ms)
            elapsed = time.monotonic() - started
            outputs.append(f"{path.name or path}: {chosen} ({elapsed:.3f}s)")
        print(f"move {move_number:>2}, played {played}: " + " | ".join(outputs))


if __name__ == "__main__":
    main()
