"""Count the mistakes in played games, using a deep search as the referee.

You cannot minimise blunders you do not count. This scores every move of a PGN by
how much worse it is than the best move a much deeper search finds, and reports the
average centipawn loss along with inaccuracy/mistake/blunder counts.

The referee is the compiled challenger itself at a high fixed node count. That is a
real limitation: it cannot see a mistake it would not understand at any depth, so it
under-reports rather than over-reports. Fixed *node* limits rather than time make the
audit deterministic and reproducible. When a UCI engine binary is available,
``analyze_pgn_stockfish.py`` is the stronger referee and should be preferred.

    uv run python -m tools.blunder_audit game.pgn --nodes 2000000 --side white
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import chess
import chess.pgn

CHALLENGER = Path(__file__).resolve().parent.parent / "challengers" / "numba_v1"
sys.path.insert(0, str(CHALLENGER))

import engine  # noqa: E402
import search  # noqa: E402

INACCURACY = 50
MISTAKE = 100
BLUNDER = 300

# Scores are clamped before differencing. Mate scores are not centipawns, and the gap
# between "mate in 4" and "mate in 5" is not a mistake worth 1 cp, let alone 29,997.
# Clamping also stops a hopeless position generating enormous losses on every move.
SCORE_CLAMP = 1_000


@dataclass
class Tally:
    """Per-side move quality over one or more games."""

    moves: int = 0
    total_loss: int = 0
    inaccuracies: int = 0
    mistakes: int = 0
    blunders: int = 0
    worst: list[tuple[int, str, str, str, int]] = field(default_factory=list)

    def record(self, loss: int, ply: int, fen: str, played: str, best: str) -> None:
        self.moves += 1
        self.total_loss += loss
        if loss >= BLUNDER:
            self.blunders += 1
        elif loss >= MISTAKE:
            self.mistakes += 1
        elif loss >= INACCURACY:
            self.inaccuracies += 1
        if loss >= INACCURACY:
            self.worst.append((loss, fen, played, best, ply))

    def report(self, label: str) -> str:
        if self.moves == 0:
            return f"{label}: no moves scored"
        average = self.total_loss / self.moves
        return (
            f"{label}: {self.moves} moves, average loss {average:.1f} cp, "
            f"{self.inaccuracies} inaccuracies (>={INACCURACY}), "
            f"{self.mistakes} mistakes (>={MISTAKE}), "
            f"{self.blunders} blunders (>={BLUNDER})"
        )


def _clamp(score: int) -> int:
    return max(-SCORE_CLAMP, min(SCORE_CLAMP, score))


def _score(board: chess.Board, memory: search.SearchMemory, nodes: int) -> int:
    """Clamped search score from the perspective of the side to move."""
    if board.is_checkmate():
        return -SCORE_CLAMP
    if board.is_game_over(claim_draw=True):
        return 0
    if board.legal_moves.count() == 1:
        # search_position short-circuits a forced move and reports score 0, which is
        # not an evaluation. Play the move and score the position it leads to.
        board.push(next(iter(board.legal_moves)))
        try:
            return -_score(board, memory, nodes)
        finally:
            board.pop()
    position = engine.position_from_board(board)
    result = search.search_position(position, memory, node_limit=nodes)
    return _clamp(int(result.score))


def _best(board: chess.Board, memory: search.SearchMemory, nodes: int) -> tuple[int, str]:
    position = engine.position_from_board(board)
    result = search.search_position(position, memory, node_limit=nodes)
    return _clamp(int(result.score)), engine.move_to_uci(result.move)


def audit_game(
    game: chess.pgn.Game, nodes: int, sides: set[chess.Color], quiet_plies: int
) -> dict[chess.Color, Tally]:
    tallies = {chess.WHITE: Tally(), chess.BLACK: Tally()}
    board = game.board()
    memory = search.SearchMemory.create()
    for ply, node in enumerate(game.mainline()):
        played = node.move
        # A forced move cannot be a mistake, and scoring it only adds noise.
        scored = board.turn in sides and ply >= quiet_plies and board.legal_moves.count() > 1
        if scored:
            fen = board.fen()
            mover = board.turn
            best_score, best_uci = _best(board, memory, nodes)
            if played.uci() == best_uci:
                # The referee would have played it too: nothing was lost, and the
                # second search can be skipped entirely.
                loss = 0
            else:
                board.push(played)
                # The child's score is from the opponent's view; negate to compare.
                played_score = -_score(board, memory, nodes)
                board.pop()
                loss = max(0, best_score - played_score)
            tallies[mover].record(loss, ply, fen, played.uci(), best_uci)
        board.push(played)
    return tallies


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pgn", type=Path, nargs="+")
    parser.add_argument(
        "--nodes",
        type=int,
        default=4_000_000,
        help=(
            "referee node budget per position. Must be well above what the engine "
            "searched when it played the move, or the referee simply agrees with the "
            "player and every game looks clean. The compiled challenger searches about "
            "390,000 nodes on a 3.7 s rated budget, so aim for 10x that or more"
        ),
    )
    parser.add_argument("--side", choices=("white", "black", "both"), default="both")
    parser.add_argument(
        "--skip-opening",
        type=int,
        default=0,
        help="plies to skip before scoring, so book moves are not counted",
    )
    parser.add_argument("--show-worst", type=int, default=8)
    arguments = parser.parse_args()

    sides = {
        "white": {chess.WHITE},
        "black": {chess.BLACK},
        "both": {chess.WHITE, chess.BLACK},
    }[arguments.side]

    search.warmup()
    totals = {chess.WHITE: Tally(), chess.BLACK: Tally()}
    games = 0
    for path in arguments.pgn:
        with path.open(encoding="utf-8") as handle:
            while (game := chess.pgn.read_game(handle)) is not None:
                games += 1
                result = audit_game(game, arguments.nodes, sides, arguments.skip_opening)
                for color, tally in result.items():
                    totals[color].moves += tally.moves
                    totals[color].total_loss += tally.total_loss
                    totals[color].inaccuracies += tally.inaccuracies
                    totals[color].mistakes += tally.mistakes
                    totals[color].blunders += tally.blunders
                    totals[color].worst.extend(tally.worst)

    print(f"\n{games} game(s), referee at {arguments.nodes:,} nodes per position\n")
    for color, label in ((chess.WHITE, "white"), (chess.BLACK, "black")):
        if color in sides:
            print(totals[color].report(label))

    worst = sorted(
        (item for color in sides for item in totals[color].worst), reverse=True
    )[: arguments.show_worst]
    if worst:
        print("\nworst decisions:")
        for loss, fen, played, best, ply in worst:
            print(f"  -{loss:>4} cp  ply {ply:>3}  played {played:<6} best {best:<6}  {fen}")


if __name__ == "__main__":
    main()
