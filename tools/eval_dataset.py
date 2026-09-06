from __future__ import annotations

import argparse
import atexit
import json
import multiprocessing as mp
import random
from dataclasses import dataclass
from pathlib import Path

import chess
import chess.engine

import engine
import search

OPENING_WEIGHTS = (0.40, 0.25, 0.20, 0.15)
LABEL_CLAMP = 2_000
DECIDED_PLIES = 6
DECIDED_SCORE = 1_000


@dataclass(frozen=True, slots=True)
class Settings:
    stockfish: Path
    warm: bool = False
    opening_nodes: int = 0
    opening_plies: int = 0
    nodes: int = 0
    opponent_nodes: int = 0
    stride: int = 0
    per_game: int = 0
    max_plies: int = 0
    label_nodes: int = 0


_SETTINGS: Settings
_STOCKFISH: chess.engine.SimpleEngine | None = None


def _stockfish() -> chess.engine.SimpleEngine:
    global _STOCKFISH
    if _STOCKFISH is None:
        _STOCKFISH = chess.engine.SimpleEngine.popen_uci(str(_SETTINGS.stockfish))
        _STOCKFISH.configure({"Threads": 1, "Hash": 16})
        atexit.register(_STOCKFISH.quit)
    return _STOCKFISH


def _worker_init(settings: Settings) -> None:
    global _SETTINGS
    _SETTINGS = settings
    if settings.warm:
        search.warmup()


def _opening(rng: random.Random) -> chess.Board:
    # random walk through Stockfish's top four keeps openings playable+varied
    board = chess.Board()
    limit = chess.engine.Limit(nodes=_SETTINGS.opening_nodes)
    for _ in range(_SETTINGS.opening_plies):
        if board.is_game_over(claim_draw=True):
            break
        lines = _stockfish().analyse(board, limit, multipv=4)
        moves = [line["pv"][0] for line in lines if line.get("pv")]
        if not moves:
            break
        board.push(rng.choices(moves, weights=OPENING_WEIGHTS[: len(moves)])[0])
    return board


def _play_game(task: tuple[int, int, int]) -> list[dict[str, object]]:
    game_id, seed, stockfish_side = task
    rng = random.Random(seed)
    board = _opening(rng)
    opening_plies = len(board.move_stack)
    memory = search.SearchMemory.create(18)
    limit = chess.engine.Limit(nodes=_SETTINGS.opponent_nodes)

    positions: list[dict[str, object]] = []
    decided = 0
    while not board.is_game_over(claim_draw=True):
        ply = len(board.move_stack)
        if ply >= _SETTINGS.max_plies:
            break
        if (
            (ply - opening_plies) % _SETTINGS.stride == 0
            and ply >= opening_plies
            and len(positions) < _SETTINGS.per_game
            and not board.is_check()
        ):
            positions.append({"fen": board.fen(), "game": game_id, "ply": ply})

        if stockfish_side >= 0 and board.turn == (stockfish_side == chess.WHITE):
            board.push(_stockfish().play(board, limit).move)
            continue

        position = engine.position_from_board(board)
        result = search.search_position(position, memory, node_limit=_SETTINGS.nodes)
        if result.move == 0:
            break
        decided = decided + 1 if abs(result.score) >= DECIDED_SCORE else 0
        board.push(chess.Move.from_uci(engine.move_to_uci(result.move)))
        if decided >= DECIDED_PLIES:
            break
    return positions


def _label(record: dict[str, object]) -> dict[str, object]:
    board = chess.Board(str(record["fen"]))
    # fresh game object clears hash, so a label doesn't depend on what came before it
    info = _stockfish().analyse(
        board, chess.engine.Limit(nodes=_SETTINGS.label_nodes), game=object()
    )
    relative = info["score"].white()
    if relative.is_mate():
        centipawns = LABEL_CLAMP if (relative.mate() or 0) > 0 else -LABEL_CLAMP
    else:
        centipawns = max(-LABEL_CLAMP, min(LABEL_CLAMP, relative.score() or 0))
    return {**record, "cp": centipawns}


def _pool(workers: int, settings: Settings) -> mp.pool.Pool:
    # fork after warmup so workers inherit the compiled code instead of rebuilding
    return mp.get_context("fork").Pool(workers, initializer=_worker_init, initargs=(settings,))


def _generate(arguments: argparse.Namespace) -> None:
    search.warmup()
    settings = Settings(
        stockfish=arguments.stockfish,
        warm=True,
        opening_nodes=arguments.opening_nodes,
        opening_plies=arguments.opening_plies,
        nodes=arguments.nodes,
        opponent_nodes=arguments.opponent_nodes,
        stride=arguments.stride,
        per_game=arguments.per_game,
        max_plies=arguments.max_plies,
    )
    rng = random.Random(arguments.seed)
    tasks = []
    for game_id in range(arguments.games):
        side = -1
        if rng.random() < arguments.stockfish_share:
            side = chess.WHITE if game_id % 2 == 0 else chess.BLACK
        tasks.append((game_id, rng.randrange(1 << 30), side))

    seen: set[str] = set()
    written = 0
    with arguments.out.open("w") as sink, _pool(arguments.workers, settings) as pool:
        for done, positions in enumerate(pool.imap_unordered(_play_game, tasks, chunksize=4), 1):
            for record in positions:
                # placement, side, castling and en passant (clocks are not part of identity)
                key = " ".join(str(record["fen"]).split(" ")[:4])
                if key in seen:
                    continue
                seen.add(key)
                sink.write(json.dumps(record) + "\n")
                written += 1
            if done % 100 == 0:
                print(f"{done}/{len(tasks)} games, {written} positions", flush=True)
    print(f"{written} unique positions from {len(tasks)} games -> {arguments.out}")


def _label_file(arguments: argparse.Namespace) -> None:
    records = [json.loads(line) for line in arguments.source.open()]
    settings = Settings(stockfish=arguments.stockfish, label_nodes=arguments.nodes)
    labelled = 0
    with arguments.out.open("w") as sink, _pool(arguments.workers, settings) as pool:
        for record in pool.imap(_label, records, chunksize=16):
            sink.write(json.dumps(record) + "\n")
            labelled += 1
            if labelled % 2_000 == 0:
                print(f"{labelled}/{len(records)} labelled", flush=True)
    print(f"{labelled} labelled positions -> {arguments.out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="generate and label evaluation training positions")
    parser.add_argument("--stockfish", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    subparsers = parser.add_subparsers(dest="command", required=True)

    games = subparsers.add_parser("games", help="play games and sample positions")
    games.add_argument("--games", type=int, default=4_000)
    games.add_argument("--nodes", type=int, default=10_000)
    games.add_argument("--opponent-nodes", type=int, default=30_000)
    games.add_argument("--stockfish-share", type=float, default=0.25)
    games.add_argument("--opening-nodes", type=int, default=20_000)
    games.add_argument("--opening-plies", type=int, default=10)
    games.add_argument("--stride", type=int, default=5)
    games.add_argument("--per-game", type=int, default=16)
    games.add_argument("--max-plies", type=int, default=200)
    games.add_argument("--seed", type=int, default=20260906)
    games.add_argument("--out", type=Path, required=True)
    games.set_defaults(run=_generate)

    label = subparsers.add_parser("label", help="score sampled positions with Stockfish")
    label.add_argument("--source", type=Path, required=True)
    label.add_argument("--nodes", type=int, default=200_000)
    label.add_argument("--out", type=Path, required=True)
    label.set_defaults(run=_label_file)

    arguments = parser.parse_args()
    arguments.run(arguments)


if __name__ == "__main__":
    main()
