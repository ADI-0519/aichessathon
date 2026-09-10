"""Pure suite, fingerprint, persistence, and statistics helpers for backtests."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, cast

import chess
import chess.pgn

from harness.package import DEFAULT_INCLUDES, members

Split = Literal["development", "validation", "holdout"]
CandidateResult = Literal["win", "draw", "loss", "void"]

SCHEMA_VERSION = 1
SPLIT_THRESHOLDS = (("development", 60), ("validation", 80), ("holdout", 100))
RESULT_POINTS = {"win": 1.0, "draw": 0.5, "loss": 0.0}
PENTANOMIAL_KEYS = ("0.0", "0.5", "1.0", "1.5", "2.0")


@dataclass(frozen=True, slots=True)
class SuitePosition:
    """One unique, normalized position and its stable split assignment."""

    identifier: str
    fen: str
    source_index: int
    split: Split


@dataclass(frozen=True, slots=True)
class GameRecord:
    """One completed candidate game, suitable for append-only JSONL storage."""

    game_id: str
    position_id: str
    position_index: int
    fen: str
    candidate_color: Literal["white", "black"]
    candidate_result: CandidateResult
    board_result: str
    termination: str
    plies: int
    elapsed_s: float
    pgn_file: str
    candidate_failure: bool
    opponent_failure: bool

    @classmethod
    def from_dict(cls, value: object) -> GameRecord:
        if not isinstance(value, dict):
            raise ValueError("game record must be a JSON object")
        try:
            candidate_color = str(value["candidate_color"])
            candidate_result = str(value["candidate_result"])
            if candidate_color not in {"white", "black"}:
                raise ValueError(f"invalid candidate color: {candidate_color}")
            if candidate_result not in {"win", "draw", "loss", "void"}:
                raise ValueError(f"invalid candidate result: {candidate_result}")
            return cls(
                game_id=str(value["game_id"]),
                position_id=str(value["position_id"]),
                position_index=int(value["position_index"]),
                fen=str(value["fen"]),
                candidate_color=cast(Literal["white", "black"], candidate_color),
                candidate_result=cast(CandidateResult, candidate_result),
                board_result=str(value["board_result"]),
                termination=str(value["termination"]),
                plies=int(value["plies"]),
                elapsed_s=float(value["elapsed_s"]),
                pgn_file=str(value["pgn_file"]),
                candidate_failure=bool(value["candidate_failure"]),
                opponent_failure=bool(value["opponent_failure"]),
            )
        except KeyError as error:
            raise ValueError(f"game record is missing {error.args[0]!r}") from error


def canonical_fen(board: chess.Board) -> str:
    """Normalize irrelevant en-passant fields while retaining clock state."""
    return board.fen(en_passant="legal")


def stable_split(fen: str, seed: str) -> Split:
    """Assign a position without depending on suite order or future additions."""
    digest = hashlib.sha256(f"{seed}\0{fen}".encode()).digest()
    bucket = int.from_bytes(digest[:8], "big") % 100
    for name, upper_bound in SPLIT_THRESHOLDS:
        if bucket < upper_bound:
            return cast(Split, name)
    raise AssertionError("split thresholds do not cover every bucket")


def positions_from_fens(
    entries: Iterable[tuple[str, str]], *, split_seed: str
) -> list[SuitePosition]:
    """Validate, normalize, and de-duplicate named FEN entries."""
    positions: list[SuitePosition] = []
    seen: set[str] = set()
    for source_index, (identifier, fen) in enumerate(entries, start=1):
        try:
            board = chess.Board(fen)
        except ValueError as error:
            raise ValueError(f"position {identifier!r} has invalid FEN: {error}") from error
        normalized = canonical_fen(board)
        if normalized in seen:
            continue
        seen.add(normalized)
        positions.append(
            SuitePosition(
                identifier=identifier,
                fen=normalized,
                source_index=source_index,
                split=stable_split(normalized, split_seed),
            )
        )
    if not positions:
        raise ValueError("position suite is empty")
    return positions


def load_epd(path: Path, *, split_seed: str) -> list[SuitePosition]:
    """Load one EPD position per non-comment line."""
    entries: list[tuple[str, str]] = []
    with path.open(encoding="utf-8-sig") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            board = chess.Board()
            try:
                operations = board.set_epd(line)
            except ValueError as error:
                raise ValueError(f"{path}:{line_number}: invalid EPD: {error}") from error
            raw_identifier = operations.get("id", f"epd-{line_number:06d}")
            entries.append((str(raw_identifier), canonical_fen(board)))
    return positions_from_fens(entries, split_seed=split_seed)


def load_fen(path: Path, *, split_seed: str) -> list[SuitePosition]:
    """Load one complete FEN per non-comment line."""
    entries: list[tuple[str, str]] = []
    with path.open(encoding="utf-8-sig") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            fen = raw_line.strip()
            if not fen or fen.startswith("#"):
                continue
            entries.append((f"fen-{line_number:06d}", fen))
    return positions_from_fens(entries, split_seed=split_seed)


def load_pgn(path: Path, *, split_seed: str) -> list[SuitePosition]:
    """Load the position after each PGN main line, as used by opening books."""
    entries: list[tuple[str, str]] = []
    with path.open(encoding="utf-8-sig") as handle:
        game_number = 0
        while True:
            game = chess.pgn.read_game(handle)
            if game is None:
                break
            game_number += 1
            if game.errors:
                raise ValueError(f"{path}: game {game_number} contains PGN errors: {game.errors}")
            board = game.board()
            for move in game.mainline_moves():
                if move not in board.legal_moves:
                    raise ValueError(f"{path}: game {game_number} contains illegal move {move}")
                board.push(move)
            identifier = game.headers.get("Opening") or f"pgn-{game_number:06d}"
            entries.append((identifier, canonical_fen(board)))
    return positions_from_fens(entries, split_seed=split_seed)


def load_suite_file(path: Path, *, split_seed: str) -> list[SuitePosition]:
    """Load a supported suite file without duplicating format dispatch in CLIs."""
    resolved = path.resolve()
    if not resolved.is_file():
        raise ValueError(f"suite file not found: {resolved}")
    suffix = resolved.suffix.lower()
    if suffix == ".epd":
        return load_epd(resolved, split_seed=split_seed)
    if suffix == ".fen":
        return load_fen(resolved, split_seed=split_seed)
    if suffix == ".pgn":
        return load_pgn(resolved, split_seed=split_seed)
    raise ValueError("suite must be a .epd, .fen, or .pgn file")


def suite_digest(positions: Sequence[SuitePosition]) -> str:
    """Fingerprint ordered identifiers, FENs, and split assignments."""
    digest = hashlib.sha256()
    for position in positions:
        digest.update(position.identifier.encode())
        digest.update(b"\0")
        digest.update(position.fen.encode())
        digest.update(b"\0")
        digest.update(position.split.encode())
        digest.update(b"\n")
    return digest.hexdigest()


def fingerprint_agent(directory: Path) -> dict[str, object]:
    """Hash exactly the source/data set that the standard packager would include."""
    root = directory.resolve()
    entries = list(members(root, DEFAULT_INCLUDES))
    if not any(name == "agent.py" for _, name in entries):
        raise ValueError(f"agent entry point not found: {root / 'agent.py'}")
    digest = hashlib.sha256()
    files: list[str] = []
    total_bytes = 0
    for path, name in entries:
        data = path.read_bytes()
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
        files.append(name)
        total_bytes += len(data)
    return {
        "kind": "local",
        "path": str(root),
        "sha256": digest.hexdigest(),
        "files": files,
        "bytes": total_bytes,
    }


def fingerprint_file(path: Path) -> dict[str, object]:
    """Hash a single external benchmark executable without reading it into memory."""
    resolved = path.resolve()
    digest = hashlib.sha256()
    size = 0
    with resolved.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return {
        "kind": "file",
        "path": str(resolved),
        "sha256": digest.hexdigest(),
        "bytes": size,
    }


def git_state(root: Path) -> dict[str, object]:
    """Capture the current commit and dirty flag without requiring Git at runtime."""

    def run(*arguments: str) -> str | None:
        try:
            completed = subprocess.run(
                ["git", *arguments],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            )
        except (FileNotFoundError, subprocess.CalledProcessError):
            return None
        return completed.stdout.strip()

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    return {"commit": commit, "dirty": bool(status) if status is not None else None}


def environment_metadata() -> dict[str, object]:
    return {
        "python": sys.version.split()[0],
        "python_chess": getattr(chess, "__version__", "unknown"),
        "platform": sys.platform,
        "created_at": datetime.now(UTC).isoformat(),
    }


def atomic_write_json(path: Path, value: object) -> None:
    """Replace a JSON file atomically so interruption cannot leave a half-report."""
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_write_text(path: Path, value: str) -> None:
    """Replace a UTF-8 text file atomically."""
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(value, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def acquire_output_lock(output: Path) -> tuple[Path, int]:
    """Atomically claim an experiment directory for one writer process."""
    path = output / ".run.lock"
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise RuntimeError(
            f"another backtest owns {output}; if no run is active, remove {path.name}"
        ) from error
    owner = f"pid={os.getpid()}\nstarted_at={datetime.now(UTC).isoformat()}\n"
    try:
        os.write(descriptor, owner.encode())
        os.fsync(descriptor)
    except OSError:
        os.close(descriptor)
        path.unlink(missing_ok=True)
        raise
    return path, descriptor


def release_output_lock(lock: tuple[Path, int]) -> None:
    """Release a lock returned by :func:`acquire_output_lock`."""
    path, descriptor = lock
    os.close(descriptor)
    path.unlink(missing_ok=True)


def ensure_manifest(output: Path, configuration: dict[str, object]) -> dict[str, object]:
    """Create the immutable run manifest or verify it before resuming."""
    path = output / "manifest.json"
    if path.exists():
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict) or loaded.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"unsupported or malformed manifest: {path}")
        if loaded.get("configuration") != configuration:
            raise ValueError(
                "output directory belongs to a different experiment; choose a new --output"
            )
        return cast(dict[str, object], loaded)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "environment": environment_metadata(),
        "configuration": configuration,
    }
    atomic_write_json(path, manifest)
    return manifest


def load_records(path: Path) -> list[GameRecord]:
    """Read an append-only journal, rejecting duplicates and partial JSON."""
    if not path.exists():
        return []
    records: list[GameRecord] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = GameRecord.from_dict(json.loads(line))
            except (json.JSONDecodeError, ValueError, TypeError) as error:
                raise ValueError(f"{path}:{line_number}: invalid game journal: {error}") from error
            if record.game_id in seen:
                raise ValueError(f"{path}:{line_number}: duplicate game id {record.game_id}")
            seen.add(record.game_id)
            records.append(record)
    return records


def append_record(path: Path, record: GameRecord) -> None:
    """Durably append one complete record before the next game starts."""
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(asdict(record), sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _score_to_elo(score: float) -> float | None:
    if not 0.0 < score < 1.0:
        return None
    return 400.0 * math.log10(score / (1.0 - score))


def complete_pair_scores(records: Sequence[GameRecord]) -> list[float]:
    """Return candidate points for every complete, non-void colour pair."""
    grouped: dict[str, list[GameRecord]] = defaultdict(list)
    for record in records:
        grouped[record.position_id].append(record)

    pair_scores: list[float] = []
    for games in grouped.values():
        colors = {game.candidate_color for game in games}
        if len(games) != 2 or colors != {"white", "black"}:
            continue
        if any(
            game.candidate_result == "void"
            or game.candidate_failure
            or game.opponent_failure
            for game in games
        ):
            continue
        pair_scores.append(sum(RESULT_POINTS[game.candidate_result] for game in games))
    return pair_scores


def pentanomial_counts(records: Sequence[GameRecord]) -> tuple[int, int, int, int, int]:
    """Return LL, LD/DL, LW/DD/WL, DW/WD, and WW pair frequencies."""
    counts = [0, 0, 0, 0, 0]
    index_for_score = {0.0: 0, 0.5: 1, 1.0: 2, 1.5: 3, 2.0: 4}
    for score in complete_pair_scores(records):
        counts[index_for_score[score]] += 1
    return counts[0], counts[1], counts[2], counts[3], counts[4]


def _paired_mean_interval(pair_scores: Sequence[float]) -> tuple[float, float]:
    """Return a descriptive normal interval using pair scores as observations."""
    normalized = [score / 2.0 for score in pair_scores]
    mean = sum(normalized) / len(normalized)
    if len(normalized) == 1:
        return 0.0, 1.0
    variance = sum((value - mean) ** 2 for value in normalized) / (len(normalized) - 1)
    margin = 1.96 * math.sqrt(variance / len(normalized))
    return max(0.0, mean - margin), min(1.0, mean + margin)


def summarize(records: Sequence[GameRecord]) -> dict[str, object]:
    """Summarize games and paired outcomes without treating voids as draws."""
    result_counts = Counter(record.candidate_result for record in records)
    scored_games = result_counts["win"] + result_counts["draw"] + result_counts["loss"]
    points = sum(RESULT_POINTS.get(record.candidate_result, 0.0) for record in records)
    score = points / scored_games if scored_games else None

    by_color: dict[str, dict[str, int]] = {}
    for color in ("white", "black"):
        color_counts = Counter(
            record.candidate_result for record in records if record.candidate_color == color
        )
        by_color[color] = {
            "wins": color_counts["win"],
            "draws": color_counts["draw"],
            "losses": color_counts["loss"],
            "voids": color_counts["void"],
        }

    pair_scores = complete_pair_scores(records)
    counts = pentanomial_counts(records)
    interval: dict[str, float | None] | None = None
    if pair_scores:
        low, high = _paired_mean_interval(pair_scores)
        interval = {
            "score_low": low,
            "score_high": high,
            "elo_low": _score_to_elo(low),
            "elo_high": _score_to_elo(high),
        }

    return {
        "games": len(records),
        "scored_games": scored_games,
        "wins": result_counts["win"],
        "draws": result_counts["draw"],
        "losses": result_counts["loss"],
        "voids": result_counts["void"],
        "score": score,
        "elo": _score_to_elo(score) if score is not None else None,
        "by_color": by_color,
        "terminations": dict(sorted(Counter(r.termination for r in records).items())),
        "candidate_failures": sum(record.candidate_failure for record in records),
        "opponent_failures": sum(record.opponent_failure for record in records),
        "complete_pairs": len(pair_scores),
        "pentanomial": dict(zip(PENTANOMIAL_KEYS, counts, strict=True)),
        "confidence_95": interval,
        "game_elapsed_s_sum": sum(record.elapsed_s for record in records),
    }
