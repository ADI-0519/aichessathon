"""Analyse PGN moves with a local UCI engine.

This is a development-only diagnostic. Scores and principal variations are
never shipped with the competition agent. JSON output deliberately keeps mate
scores separate from centipawns so forced mates cannot corrupt ACPL statistics.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import chess
import chess.engine
import chess.pgn


class PlayerHeaderError(ValueError):
    """Raised when a requested player cannot identify exactly one colour."""


@dataclass(frozen=True, slots=True)
class ScoreSnapshot:
    """One engine score from the mover's point of view."""

    cp: int | None
    mate: int | None

    def as_json(self) -> dict[str, int | None]:
        return {"cp": self.cp, "mate": self.mate}


@dataclass(slots=True)
class ClockTracker:
    """Reconstruct pre-move clocks from PGN post-move clock annotations."""

    base_s: float
    increment_s: float
    _after: dict[chess.Color, float] = field(default_factory=dict, init=False)

    def observe(
        self, color: chess.Color, clock_after_s: float | None
    ) -> tuple[float | None, float | None]:
        if clock_after_s is None:
            return None, None
        clock_before_s = (
            self._after[color] + self.increment_s
            if color in self._after
            else self.base_s
        )
        self._after[color] = clock_after_s
        return clock_before_s, max(0.0, clock_before_s - clock_after_s)


def score_snapshot(score: chess.engine.PovScore, color: chess.Color) -> ScoreSnapshot:
    """Preserve CP and mate scores rather than mapping mate to fake CP."""
    relative = score.pov(color)
    mate = relative.mate()
    if mate is not None:
        return ScoreSnapshot(cp=None, mate=mate)
    cp = relative.score()
    if cp is None:
        raise RuntimeError("engine returned an unscored position")
    return ScoreSnapshot(cp=cp, mate=None)


def classify_move(
    before: ScoreSnapshot, after: ScoreSnapshot
) -> tuple[int | None, str]:
    """Return CP loss only when both endpoints are ordinary CP scores."""
    if before.cp is not None and after.cp is not None:
        return max(0, before.cp - after.cp), "centipawn"
    if before.mate is not None and before.mate >= 0:
        if after.mate is not None and after.mate >= 0:
            return None, "forced_mate_maintained"
        return None, "forced_mate_lost"
    if after.mate is not None and after.mate < 0:
        if before.mate is not None and before.mate < 0:
            return None, "forced_mate_continues"
        return None, "forced_mate_allowed"
    if before.mate is not None and before.mate < 0:
        return None, "forced_mate_escaped"
    if after.mate is not None and after.mate >= 0:
        return None, "forced_mate_found"
    return None, "mate_transition"


def _pv_text(
    board: chess.Board, pv: list[chess.Move], plies: int
) -> tuple[list[str], list[str]]:
    replay = board.copy(stack=False)
    uci: list[str] = []
    san: list[str] = []
    for move in pv[:plies]:
        if move not in replay.legal_moves:
            break
        uci.append(move.uci())
        san.append(replay.san(move))
        replay.push(move)
    return uci, san


def _selected_color(
    game: chess.pgn.Game, focus: chess.Color | None, player: str | None
) -> chess.Color | None:
    if player is None:
        return focus
    wanted = player.casefold()
    matches = [
        color
        for color, header in ((chess.WHITE, "White"), (chess.BLACK, "Black"))
        if game.headers.get(header, "").casefold() == wanted
    ]
    if len(matches) != 1:
        raise PlayerHeaderError(
            f"player {player!r} must match exactly one White/Black header; "
            f"got White={game.headers.get('White')!r}, Black={game.headers.get('Black')!r}"
        )
    return matches[0]


def analyze_game(
    engine: chess.engine.SimpleEngine,
    path: Path,
    nodes: int,
    focus: chess.Color | None,
    *,
    player: str | None = None,
    base_s: float = 120.0,
    increment_s: float = 0.5,
    pv_plies: int = 5,
) -> dict[str, Any]:
    """Analyse one PGN, reusing every after-position on the following ply."""
    with path.open(encoding="utf-8") as handle:
        game = chess.pgn.read_game(handle)
    if game is None:
        raise ValueError(f"no PGN game found in {path}")
    if game.errors:
        raise ValueError(f"PGN parser errors in {path}: {game.errors}")

    selected_color = _selected_color(game, focus, player)
    board = game.board()
    start_fen = board.fen()
    clock = ClockTracker(base_s, increment_s)
    before_info = engine.analyse(board, chess.engine.Limit(nodes=nodes))
    records: list[dict[str, Any]] = []

    for ply, node in enumerate(game.mainline(), start=1):
        move = node.move
        mover = board.turn
        fullmove = board.fullmove_number
        san = board.san(move)
        before = score_snapshot(before_info["score"], mover)
        before_pv = list(before_info.get("pv", []))
        best_move = before_pv[0] if before_pv else None
        pv_uci, pv_san = _pv_text(board, before_pv, pv_plies)
        fen = board.fen()

        board.push(move)
        after_info = engine.analyse(board, chess.engine.Limit(nodes=nodes))
        after = score_snapshot(after_info["score"], mover)
        cp_loss, classification = classify_move(before, after)
        clock_after_s = node.clock()
        clock_before_s, think_s = clock.observe(mover, clock_after_s)
        records.append(
            {
                "ply": ply,
                "fullmove": fullmove,
                "color": chess.COLOR_NAMES[mover],
                "selected": selected_color is None or mover == selected_color,
                "fen": fen,
                "san": san,
                "uci": move.uci(),
                "best_uci": best_move.uci() if best_move is not None else None,
                "before": before.as_json(),
                "after": after.as_json(),
                "cp_loss": cp_loss,
                "classification": classification,
                "clock_before_s": clock_before_s,
                "clock_after_s": clock_after_s,
                "think_s": think_s,
                "pv_uci": pv_uci,
                "pv_san": pv_san,
            }
        )
        before_info = after_info

    selected_cp = [
        int(record["cp_loss"])
        for record in records
        if record["selected"] and record["cp_loss"] is not None
    ]
    mate_events: dict[str, int] = {}
    for record in records:
        if not record["selected"] or record["classification"] == "centipawn":
            continue
        key = str(record["classification"])
        mate_events[key] = mate_events.get(key, 0) + 1

    return {
        "schema_version": 1,
        "file": str(path.resolve()),
        "headers": dict(game.headers),
        "start_fen": start_fen,
        "analysis": {
            "nodes_per_position": nodes,
            "focus": (
                player
                if player is not None
                else chess.COLOR_NAMES[selected_color]
                if selected_color is not None
                else "both"
            ),
            "base_s": base_s,
            "increment_s": increment_s,
            "pv_plies": pv_plies,
        },
        "summary": {
            "selected_moves": sum(bool(record["selected"]) for record in records),
            "cp_scored_moves": len(selected_cp),
            "average_cp_loss": (
                sum(selected_cp) / len(selected_cp) if selected_cp else None
            ),
            "mate_events": mate_events,
        },
        "moves": records,
    }


def _score_text(value: dict[str, int | None]) -> str:
    if value["mate"] is not None:
        return f"M{value['mate']}"
    return str(value["cp"])


def print_text(report: dict[str, Any]) -> None:
    """Render a compact human-readable view."""
    print(f"\n===== {Path(report['file']).name} =====")
    print(f"start: {report['start_fen']}")
    print(
        f"result: {report['headers'].get('Result')}  "
        f"focus: {report['analysis']['focus']}"
    )
    print("ply move    played  best    before   after    loss  think   pv")
    for record in report["moves"]:
        notable = record["cp_loss"] is not None and record["cp_loss"] >= 100
        mate_event = record["classification"] in {
            "forced_mate_lost",
            "forced_mate_allowed",
        }
        if not record["selected"] and not notable and not mate_event:
            continue
        loss = "-" if record["cp_loss"] is None else str(record["cp_loss"])
        best = record["best_uci"] or "-"
        think = "?" if record["think_s"] is None else f"{record['think_s']:.2f}s"
        print(
            f"{record['ply']:>3} {record['san']:<7} {record['uci']:<7} {best:<7} "
            f"{_score_text(record['before']):>7} {_score_text(record['after']):>7} "
            f"{loss:>6} {think:>7}  {' '.join(record['pv_san'])}"
        )
    summary = report["summary"]
    acpl = summary["average_cp_loss"]
    acpl_text = "n/a" if acpl is None else f"{acpl:.1f}"
    print(
        f"selected ACPL: {acpl_text} over {summary['cp_scored_moves']} CP-scored moves; "
        f"mate events: {summary['mate_events']}"
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", required=True, type=Path)
    parser.add_argument("--nodes", type=int, default=100_000)
    selector = parser.add_mutually_exclusive_group()
    selector.add_argument(
        "--focus",
        choices=("white", "black", "both"),
        default="both",
        help="Select a colour; major opponent errors are still printed in text mode.",
    )
    selector.add_argument(
        "--player",
        help="Select the colour by an exact, case-insensitive White/Black PGN header.",
    )
    parser.add_argument("--base-s", type=float, default=120.0)
    parser.add_argument("--increment-s", type=float, default=0.5)
    parser.add_argument("--pv-plies", type=int, default=5)
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument(
        "--missing-player",
        choices=("error", "skip"),
        default="error",
        help="How --player handles old PGNs whose player headers are unavailable.",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("pgn", nargs="+", type=Path)
    args = parser.parse_args()

    if args.nodes <= 0 or args.pv_plies <= 0:
        parser.error("--nodes and --pv-plies must be positive")
    if args.base_s <= 0 or args.increment_s < 0:
        parser.error("--base-s must be positive and --increment-s nonnegative")
    if args.output is not None and args.format != "json":
        parser.error("--output requires --format json")

    focus = None if args.focus == "both" else args.focus == "white"
    engine = chess.engine.SimpleEngine.popen_uci(str(args.engine))
    try:
        engine.configure({"Threads": 1, "Hash": 128})
        reports: list[dict[str, Any]] = []
        skipped: list[str] = []
        for path in args.pgn:
            try:
                report = analyze_game(
                    engine,
                    path,
                    args.nodes,
                    focus,
                    player=args.player,
                    base_s=args.base_s,
                    increment_s=args.increment_s,
                    pv_plies=args.pv_plies,
                )
            except PlayerHeaderError:
                if args.missing_player == "error":
                    raise
                skipped.append(str(path.resolve()))
                continue
            reports.append(report)
    finally:
        engine.quit()

    payload: dict[str, Any] = {
        "schema_version": 1,
        "engine": str(args.engine.resolve()),
        "games": reports,
        "skipped_missing_player": skipped,
    }
    if args.format == "text":
        for report in reports:
            print_text(report)
        if skipped:
            print(f"\nskipped {len(skipped)} PGN(s) without a unique {args.player!r} header")
    elif args.output is not None:
        _write_json(args.output, payload)
        print(f"wrote {args.output.resolve()}")
    else:
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
