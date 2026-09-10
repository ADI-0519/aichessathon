"""Replay rated games to measure realistic persistent TT/history effects."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

import chess
import numpy as np

from tools.backtest_core import atomic_write_json, fingerprint_agent, git_state
from tools.cli import positive_int
from tools.search_diagnostics import (
    DEFAULT_SUITE,
    REPOSITORY,
    CriticalPosition,
    load_critical_positions,
    load_engine_modules,
)

DEFAULT_REPLAY_SUITE = (
    REPOSITORY / "benchmarks" / "suites" / "v3_critical_game_replays.json"
)
DEFAULT_OUTPUT = REPOSITORY / "benchmarks" / "diagnostics" / "v3-persistent-replay.json"


@dataclass(frozen=True, slots=True)
class ReplayTarget:
    case_id: str
    fullmove: int


@dataclass(frozen=True, slots=True)
class ReplayGame:
    identifier: str
    initial_fen: str
    engine_color: Literal["white", "black"]
    targets: tuple[ReplayTarget, ...]
    moves: tuple[str, ...]


def load_replay_games(
    path: Path, critical_positions: list[CriticalPosition]
) -> list[ReplayGame]:
    """Validate replay histories against the independent critical FEN suite."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError(f"unsupported replay schema in {path}")
    raw_games = payload.get("games")
    if not isinstance(raw_games, list) or not raw_games:
        raise ValueError(f"replay suite is empty: {path}")

    critical = {position.identifier: position for position in critical_positions}
    games: list[ReplayGame] = []
    seen_games: set[str] = set()
    seen_targets: set[str] = set()
    for raw in raw_games:
        if not isinstance(raw, dict):
            raise ValueError("each replay game must be an object")
        identifier = str(raw.get("id", ""))
        color_name = str(raw.get("engine_color", ""))
        if not identifier or identifier in seen_games:
            raise ValueError(f"missing or duplicate replay game id: {identifier!r}")
        if color_name not in {"white", "black"}:
            raise ValueError(f"invalid engine color in {identifier}: {color_name!r}")
        seen_games.add(identifier)

        raw_targets = raw.get("targets")
        raw_moves = raw.get("moves")
        if not isinstance(raw_targets, list) or not isinstance(raw_moves, list):
            raise ValueError(f"replay game {identifier} needs target and move lists")
        targets = tuple(
            ReplayTarget(str(target["case_id"]), int(target["fullmove"]))
            for target in raw_targets
            if isinstance(target, dict)
        )
        if len(targets) != len(raw_targets) or not targets:
            raise ValueError(f"replay game {identifier} contains malformed targets")
        if len({target.case_id for target in targets}) != len(targets):
            raise ValueError(f"replay game {identifier} contains duplicate targets")

        board = chess.Board(str(raw.get("initial_fen", "")))
        engine_color = chess.WHITE if color_name == "white" else chess.BLACK
        targets_by_move = {target.fullmove: target for target in targets}
        found: set[str] = set()
        moves: list[str] = []
        for move_text in raw_moves:
            target = (
                targets_by_move.get(board.fullmove_number)
                if board.turn == engine_color
                else None
            )
            if target is not None:
                if target.case_id not in critical:
                    raise ValueError(f"unknown critical case in replay: {target.case_id}")
                expected = critical[target.case_id]
                if board.fen(en_passant="legal") != chess.Board(expected.fen).fen(
                    en_passant="legal"
                ):
                    raise ValueError(f"replay FEN does not match {target.case_id}")
                if str(move_text) != expected.played_move:
                    raise ValueError(
                        f"replay move for {target.case_id} is {move_text}, "
                        f"expected {expected.played_move}"
                    )
                found.add(target.case_id)
            move = chess.Move.from_uci(str(move_text))
            if move not in board.legal_moves:
                raise ValueError(f"illegal replay move in {identifier}: {move_text}")
            moves.append(move.uci())
            board.push(move)
        expected_targets = {target.case_id for target in targets}
        if found != expected_targets:
            missing = sorted(expected_targets - found)
            raise ValueError(f"replay game {identifier} did not reach targets: {missing}")
        overlap = seen_targets & expected_targets
        if overlap:
            raise ValueError(f"critical cases occur in multiple replay games: {sorted(overlap)}")
        seen_targets.update(expected_targets)
        games.append(
            ReplayGame(
                identifier=identifier,
                initial_fen=str(raw["initial_fen"]),
                engine_color=cast(Literal["white", "black"], color_name),
                targets=targets,
                moves=tuple(moves),
            )
        )
    return games


def _result_payload(engine_module: Any, result: Any) -> dict[str, object]:
    nodes = int(result.nodes)
    qnodes = int(result.qnodes)
    return {
        "move": engine_module.move_to_uci(result.move),
        "score": int(result.score),
        "depth": int(result.depth),
        "nodes": nodes,
        "qnodes": qnodes,
        "qshare": qnodes / nodes if nodes else 0.0,
        "tt_hits": int(result.tt_hits),
        "tt_cutoffs": int(result.tt_cutoffs),
        "lmr_reductions": int(result.lmr_reductions),
        "lmr_researches": int(result.lmr_researches),
    }


def replay_game(
    engine_module: Any,
    search_module: Any,
    game: ReplayGame,
    critical: dict[str, CriticalPosition],
    *,
    warm_nodes: int,
    target_nodes: int,
    max_depth: int,
    tt_bits: int,
) -> list[dict[str, object]]:
    """Compare fresh and game-replayed memory at every target in one game."""
    board = chess.Board(game.initial_fen)
    engine_color = chess.WHITE if game.engine_color == "white" else chess.BLACK
    targets = {target.fullmove: target for target in game.targets}
    memory = search_module.SearchMemory.create(tt_bits)
    history = [np.uint64(engine_module.position_from_board(board).key[0])]
    prior_searches = 0
    records: list[dict[str, object]] = []

    for move_text in game.moves:
        if board.turn == engine_color:
            position = engine_module.position_from_board(board)
            prior_history = np.asarray(history, dtype=np.uint64)
            target = targets.get(board.fullmove_number)
            if target is None:
                search_module.search_position(
                    position,
                    memory,
                    node_limit=warm_nodes,
                    max_depth=max_depth,
                    prior_history=prior_history,
                )
                prior_searches += 1
            else:
                fresh = search_module.search_position(
                    position,
                    search_module.SearchMemory.create(tt_bits),
                    node_limit=target_nodes,
                    max_depth=max_depth,
                    prior_history=prior_history,
                )
                replayed = search_module.search_position(
                    position,
                    memory,
                    node_limit=target_nodes,
                    max_depth=max_depth,
                    prior_history=prior_history,
                )
                expected = critical[target.case_id]
                records.append(
                    {
                        "game": game.identifier,
                        "case": target.case_id,
                        "fullmove": target.fullmove,
                        "fen": board.fen(en_passant="legal"),
                        "actual_move": move_text,
                        "reference_move": expected.reference_move,
                        "prior_searches": prior_searches,
                        "prior_history_positions": len(history),
                        "fresh": _result_payload(engine_module, fresh),
                        "replayed": _result_payload(engine_module, replayed),
                    }
                )
                prior_searches += 1

        move = chess.Move.from_uci(move_text)
        if move not in board.legal_moves:
            raise RuntimeError(f"validated replay move became illegal: {move_text}")
        board.push(move)
        history.append(np.uint64(engine_module.position_from_board(board).key[0]))
    return records


def _print_records(records: list[dict[str, object]]) -> None:
    print("case                         prior fresh    d  replay   d  ref      changed")
    for record in records:
        fresh = cast(dict[str, object], record["fresh"])
        replayed = cast(dict[str, object], record["replayed"])
        changed = "yes" if fresh["move"] != replayed["move"] else "no"
        prior_searches = cast(int, record["prior_searches"])
        fresh_depth = cast(int, fresh["depth"])
        replayed_depth = cast(int, replayed["depth"])
        print(
            f"{record['case']!s:<28} {prior_searches:>5} "
            f"{fresh['move']!s:<8} {fresh_depth:>2} "
            f"{replayed['move']!s:<8} {replayed_depth:>2} "
            f"{record['reference_move']!s:<8} {changed}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine-root", type=Path, default=REPOSITORY / "current")
    parser.add_argument("--profile", default="baseline")
    parser.add_argument("--critical-suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--replay-suite", type=Path, default=DEFAULT_REPLAY_SUITE)
    parser.add_argument("--warm-nodes", type=positive_int, default=25_000)
    parser.add_argument("--target-nodes", type=positive_int, default=300_000)
    parser.add_argument("--max-depth", type=positive_int, default=32)
    parser.add_argument("--tt-bits", type=positive_int, default=18)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if not 10 <= args.tt_bits <= 24:
        parser.error("--tt-bits must be between 10 and 24")

    critical_positions = load_critical_positions(args.critical_suite)
    critical = {position.identifier: position for position in critical_positions}
    games = load_replay_games(args.replay_suite, critical_positions)
    engine_module, search_module = load_engine_modules(args.engine_root)
    if hasattr(search_module, "configure_experiment"):
        try:
            search_module.configure_experiment(args.profile)
        except (RuntimeError, ValueError) as error:
            parser.error(str(error))
    elif args.profile != "baseline":
        parser.error("the selected engine supports only the baseline profile")
    search_module.warmup()

    records: list[dict[str, object]] = []
    for game in games:
        records.extend(
            replay_game(
                engine_module,
                search_module,
                game,
                critical,
                warm_nodes=args.warm_nodes,
                target_nodes=args.target_nodes,
                max_depth=args.max_depth,
                tt_bits=args.tt_bits,
            )
        )
    _print_records(records)
    payload = {
        "schema_version": 1,
        "engine": fingerprint_agent(args.engine_root),
        "git": git_state(REPOSITORY),
        "configuration": {
            "profile": args.profile,
            "warm_nodes": args.warm_nodes,
            "target_nodes": args.target_nodes,
            "max_depth": args.max_depth,
            "tt_bits": args.tt_bits,
            "critical_suite": str(args.critical_suite.resolve()),
            "replay_suite": str(args.replay_suite.resolve()),
        },
        "games": [asdict(game) for game in games],
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output, payload)
    print(f"wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
