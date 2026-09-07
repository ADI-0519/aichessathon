"""Run reproducible, resumable, paired chess-agent backtests.

This is development tooling only. It never enters the submission archive.
"""

from __future__ import annotations

import argparse
import io
import json
import random
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import asdict
from functools import wraps
from pathlib import Path
from typing import Literal, cast

import chess
import chess.pgn

from harness.referee import FAILED_TERMINATIONS, Outcome, play_match
from harness.rules import PLY_CAP
from harness.sandbox import Agent, local
from tools.backtest_core import (
    CandidateResult,
    GameRecord,
    SuitePosition,
    acquire_output_lock,
    append_record,
    atomic_write_json,
    atomic_write_text,
    ensure_manifest,
    fingerprint_agent,
    fingerprint_file,
    git_state,
    load_records,
    load_suite_file,
    positions_from_fens,
    release_output_lock,
    suite_digest,
    summarize,
)
from tools.cli import nonnegative_int, positive_int
from tools.paired_arena import positions as builtin_fens
from tools.stockfish_arena import StockfishAgent

AgentFactory = Callable[[], Agent]
AGENT_LOG_LIMIT = 8 * 1024


def save_agent_logs(
    output: Path,
    game_id: str,
    candidate_agent: Agent,
    opponent_agent: Agent,
) -> None:
    """Persist bounded stderr tails so a startup crash remains diagnosable."""
    for name, agent in (
        ("candidate", candidate_agent),
        ("opponent", opponent_agent),
    ):
        if not agent.stderr_tail:
            continue
        tail = agent.stderr_tail[-AGENT_LOG_LIMIT:]
        path = output / "logs" / f"game-{game_id}-{name}.stderr.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(path, tail)
        print(f"{name} stderr saved to {path}")


def load_suite(source: str, split_seed: str) -> tuple[list[SuitePosition], str]:
    """Load the built-in regression suite or a standard EPD/PGN file."""
    if source == "builtin":
        entries = [
            (f"builtin-{index:03d}", fen) for index, fen in enumerate(builtin_fens(), start=1)
        ]
        return positions_from_fens(entries, split_seed=split_seed), "builtin"
    path = Path(source).resolve()
    return load_suite_file(path, split_seed=split_seed), str(path)


def select_positions(
    suite: Sequence[SuitePosition],
    split: str,
    order_seed: int,
    offset: int,
    limit: int | None,
) -> list[SuitePosition]:
    selected = list(suite if split == "all" else (p for p in suite if p.split == split))
    random.Random(order_seed).shuffle(selected)
    selected = selected[offset:]
    if limit is not None:
        selected = selected[:limit]
    if not selected:
        raise ValueError("the split/offset/limit selection contains no positions")
    return selected


def candidate_result(outcome: Outcome, candidate_is_white: bool) -> CandidateResult:
    if outcome.result == "void":
        return "void"
    if outcome.result == "draw":
        return "draw"
    candidate_won = (outcome.result == "white") == candidate_is_white
    return "win" if candidate_won else "loss"


def annotate_pgn(
    outcome: Outcome,
    *,
    game_id: str,
    candidate_is_white: bool,
    candidate_name: str,
    opponent_name: str,
) -> tuple[str, int]:
    game = chess.pgn.read_game(io.StringIO(outcome.pgn))
    if game is None or game.errors:
        detail = game.errors if game is not None else "no game"
        raise RuntimeError(f"referee produced malformed PGN for {game_id}: {detail}")
    game.headers["White"] = candidate_name if candidate_is_white else opponent_name
    game.headers["Black"] = opponent_name if candidate_is_white else candidate_name
    game.headers["BacktestId"] = game_id
    return str(game) + "\n", game.end().ply() - game.ply()


def print_summary(summary: dict[str, object]) -> None:
    score = summary["score"]
    score_text = f"{score:.1%}" if isinstance(score, float) else "n/a"
    elo = summary["elo"]
    elo_text = f"{elo:+.1f}" if isinstance(elo, float) else "n/a"
    print(
        f"\n+{summary['wins']} ={summary['draws']} -{summary['losses']} "
        f"void {summary['voids']}, score {score_text}, Elo {elo_text}"
    )
    print(
        f"pairs {summary['complete_pairs']}, pentanomial {summary['pentanomial']}, "
        f"candidate failures {summary['candidate_failures']}"
    )
    interval = summary["confidence_95"]
    if isinstance(interval, dict):
        low = interval.get("score_low")
        high = interval.get("score_high")
        if isinstance(low, float) and isinstance(high, float):
            print(f"paired 95% score interval {low:.1%}..{high:.1%}")


def configuration_for_run(
    *,
    candidate: Path,
    opponent: Path | None,
    stockfish: Path | None,
    stockfish_nodes: int | None,
    suite_source: str,
    suite: Sequence[SuitePosition],
    selected: Sequence[SuitePosition],
    split: str,
    split_seed: str,
    order_seed: int,
    offset: int,
    limit: int | None,
    base_ms: int,
    increment_ms: int,
    ply_cap: int,
) -> dict[str, object]:
    opponent_fingerprint: dict[str, object]
    opponent_config: dict[str, object]
    if opponent is not None:
        opponent_fingerprint = fingerprint_agent(opponent)
        opponent_config = {"type": "local", "fingerprint": opponent_fingerprint}
    elif stockfish is not None and stockfish_nodes is not None:
        opponent_fingerprint = fingerprint_file(stockfish)
        opponent_config = {
            "type": "stockfish",
            "nodes": stockfish_nodes,
            "hash_mb": 128,
            "threads": 1,
            "fingerprint": opponent_fingerprint,
        }
    else:
        raise ValueError("exactly one opponent type is required")

    split_counts = Counter(position.split for position in suite)
    return {
        "candidate": fingerprint_agent(candidate),
        "opponent": opponent_config,
        "suite": {
            "source": suite_source,
            "digest": suite_digest(suite),
            "total_positions": len(suite),
            "split_counts": dict(sorted(split_counts.items())),
            "selected": [asdict(position) for position in selected],
        },
        "selection": {
            "split": split,
            "split_seed": split_seed,
            "order_seed": order_seed,
            "offset": offset,
            "limit": limit,
        },
        "clock": {"base_ms": base_ms, "increment_ms": increment_ms, "ply_cap": ply_cap},
        "git_commit": git_state(Path.cwd())["commit"],
    }


def assert_sources_unchanged(configuration: dict[str, object]) -> None:
    """Stop before a new game if an editor would create a mixed-build run."""
    candidate = cast(dict[str, object], configuration["candidate"])
    candidate_path = Path(str(candidate["path"]))
    if fingerprint_agent(candidate_path)["sha256"] != candidate["sha256"]:
        raise RuntimeError("candidate files changed during the backtest; resume with a new output")

    opponent = cast(dict[str, object], configuration["opponent"])
    expected = cast(dict[str, object], opponent["fingerprint"])
    if opponent["type"] == "local":
        current = fingerprint_agent(Path(str(expected["path"])))
    else:
        current = fingerprint_file(Path(str(expected["path"])))
    if current["sha256"] != expected["sha256"]:
        raise RuntimeError("opponent files changed during the backtest; resume with a new output")


def make_opponent_factory(
    opponent: Path | None, stockfish: Path | None, stockfish_nodes: int | None
) -> tuple[AgentFactory, str]:
    if opponent is not None:
        resolved = opponent.resolve()
        return lambda: local(resolved), resolved.name
    if stockfish is None or stockfish_nodes is None:
        raise ValueError("Stockfish path and node limit are required")
    executable = stockfish.resolve()

    def stockfish_factory() -> Agent:
        return cast(Agent, StockfishAgent(executable, stockfish_nodes))

    return stockfish_factory, f"stockfish-{stockfish_nodes}"


def validate_arguments(arguments: argparse.Namespace) -> None:
    if arguments.stockfish is not None and arguments.stockfish_nodes is None:
        raise ValueError("--stockfish requires --stockfish-nodes")
    if arguments.stockfish is None and arguments.stockfish_nodes is not None:
        raise ValueError("--stockfish-nodes requires --stockfish")
    if arguments.split in {"holdout", "all"} and not arguments.unlock_holdout:
        raise ValueError("holdout access requires --unlock-holdout")


def with_output_lock(
    function: Callable[[argparse.Namespace], int],
) -> Callable[[argparse.Namespace], int]:
    """Serialize every read and write associated with one output directory."""

    @wraps(function)
    def locked(arguments: argparse.Namespace) -> int:
        output = arguments.output.resolve()
        output.mkdir(parents=True, exist_ok=True)
        lock = acquire_output_lock(output)
        try:
            return function(arguments)
        finally:
            release_output_lock(lock)

    return locked


@with_output_lock
def run(arguments: argparse.Namespace) -> int:
    validate_arguments(arguments)
    candidate = arguments.candidate.resolve()
    opponent = arguments.opponent.resolve() if arguments.opponent is not None else None
    stockfish = arguments.stockfish.resolve() if arguments.stockfish is not None else None
    suite, suite_source = load_suite(arguments.suite, arguments.split_seed)
    selected = select_positions(
        suite,
        arguments.split,
        arguments.order_seed,
        arguments.offset,
        arguments.limit,
    )
    output = arguments.output.resolve()
    games_directory = output / "games"
    games_directory.mkdir(exist_ok=True)

    configuration = configuration_for_run(
        candidate=candidate,
        opponent=opponent,
        stockfish=stockfish,
        stockfish_nodes=arguments.stockfish_nodes,
        suite_source=suite_source,
        suite=suite,
        selected=selected,
        split=arguments.split,
        split_seed=arguments.split_seed,
        order_seed=arguments.order_seed,
        offset=arguments.offset,
        limit=arguments.limit,
        base_ms=arguments.base_ms,
        increment_ms=arguments.increment_ms,
        ply_cap=arguments.ply_cap,
    )
    ensure_manifest(output, configuration)
    journal = output / "games.jsonl"
    records = load_records(journal)
    completed = {record.game_id for record in records}

    expected_ids = {
        f"{index:05d}-{'white' if candidate_is_white else 'black'}"
        for index in range(1, len(selected) + 1)
        for candidate_is_white in (True, False)
    }
    unexpected = completed - expected_ids
    if unexpected:
        raise ValueError(f"journal contains unexpected game ids: {sorted(unexpected)}")
    for record in records:
        if not (output / record.pgn_file).is_file():
            raise ValueError(f"journal references missing PGN: {record.pgn_file}")

    opponent_factory, opponent_name = make_opponent_factory(
        opponent, stockfish, arguments.stockfish_nodes
    )
    candidate_name = candidate.name
    total_games = len(selected) * 2

    try:
        for position_index, position in enumerate(selected, start=1):
            position_id = f"{position.source_index:06d}:{position.identifier}"
            for candidate_is_white in (True, False):
                color: Literal["white", "black"] = "white" if candidate_is_white else "black"
                game_id = f"{position_index:05d}-{color}"
                if game_id in completed:
                    print(f"skip {game_id}: already complete")
                    continue
                assert_sources_unchanged(configuration)
                candidate_agent = local(candidate)
                opponent_agent = opponent_factory()
                white, black = (
                    (candidate_agent, opponent_agent)
                    if candidate_is_white
                    else (opponent_agent, candidate_agent)
                )
                started = time.monotonic()
                outcome = play_match(
                    white,
                    black,
                    arguments.base_ms,
                    arguments.increment_ms,
                    ply_cap=arguments.ply_cap,
                    start_fen=position.fen,
                )
                save_agent_logs(
                    output,
                    game_id,
                    candidate_agent,
                    opponent_agent,
                )
                elapsed = time.monotonic() - started
                result = candidate_result(outcome, candidate_is_white)
                failed = outcome.termination in FAILED_TERMINATIONS
                candidate_failure = failed and result in {"loss", "void"}
                opponent_failure = failed and result in {"win", "void"}
                pgn_name = f"game-{game_id}-{result}.pgn"
                relative_pgn = str(Path("games") / pgn_name)
                pgn, plies = annotate_pgn(
                    outcome,
                    game_id=game_id,
                    candidate_is_white=candidate_is_white,
                    candidate_name=candidate_name,
                    opponent_name=opponent_name,
                )
                atomic_write_text(output / relative_pgn, pgn)
                record = GameRecord(
                    game_id=game_id,
                    position_id=position_id,
                    position_index=position_index,
                    fen=position.fen,
                    candidate_color=color,
                    candidate_result=result,
                    board_result=outcome.result,
                    termination=outcome.termination,
                    plies=plies,
                    elapsed_s=elapsed,
                    pgn_file=relative_pgn,
                    candidate_failure=candidate_failure,
                    opponent_failure=opponent_failure,
                )
                append_record(journal, record)
                records.append(record)
                completed.add(game_id)
                atomic_write_json(output / "summary.json", summarize(records))
                print(
                    f"game {len(records)}/{total_games}, {game_id}, "
                    f"position {position.identifier}, "
                    f"candidate {color}: {result} by {outcome.termination} ({elapsed:.1f}s)"
                )
                if candidate_failure and not arguments.continue_on_failure:
                    print_summary(summarize(records))
                    print(
                        "stopped after candidate technical failure; "
                        "use --continue-on-failure to override"
                    )
                    return 2
    except KeyboardInterrupt:
        print("\ninterrupted; completed games were saved and can be resumed")
        return 130
    finally:
        atomic_write_json(output / "summary.json", summarize(records))

    summary = summarize(records)
    print_summary(summary)
    candidate_failures = summary["candidate_failures"]
    if not isinstance(candidate_failures, int):
        raise TypeError("summary candidate_failures must be an integer")
    return 2 if candidate_failures else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, default=Path("current"))
    opponent_group = parser.add_mutually_exclusive_group(required=True)
    opponent_group.add_argument("--opponent", type=Path, help="local agent directory")
    opponent_group.add_argument("--stockfish", type=Path, help="development-only UCI executable")
    parser.add_argument("--stockfish-nodes", type=positive_int)
    parser.add_argument("--suite", default="builtin", help="builtin, or an EPD/FEN/PGN path")
    parser.add_argument(
        "--split",
        choices=("development", "validation", "holdout", "all"),
        default="development",
    )
    parser.add_argument("--unlock-holdout", action="store_true")
    parser.add_argument("--split-seed", default="aichessathon-backtest-v2")
    parser.add_argument("--order-seed", type=int, default=20260904)
    parser.add_argument("--offset", type=nonnegative_int, default=0)
    parser.add_argument("--limit", type=positive_int)
    parser.add_argument("--base-ms", type=positive_int, default=10_000)
    parser.add_argument("--increment-ms", type=nonnegative_int, default=100)
    parser.add_argument("--ply-cap", type=positive_int, default=PLY_CAP)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--continue-on-failure", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    arguments = parser.parse_args()
    try:
        raise SystemExit(run(arguments))
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as error:
        parser.exit(2, f"backtest error: {error}\n")


if __name__ == "__main__":
    main()
