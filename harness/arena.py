import argparse
import math
from dataclasses import dataclass
from pathlib import Path

from harness.referee import FAILED_TERMINATIONS, Outcome, play_match
from harness.rules import OPENINGS, PLY_CAP
from harness.sandbox import local

FAST_BASE_MS = 10_000
FAST_INCREMENT_MS = 100
CONFIDENCE = 1.96


@dataclass(frozen=True)
class Game:
    index: int
    opening: str
    plays_white: bool
    outcome: Outcome

    @property
    def colour(self) -> str:
        return "white" if self.plays_white else "black"

    @property
    def won(self) -> bool:
        return (self.outcome.result == "white") == self.plays_white


def play(index: int, agent: Path, opponent: Path, arguments: argparse.Namespace) -> Game:
    opening, fen = OPENINGS[(index // 2) % len(OPENINGS)]
    plays_white = index % 2 == 0
    white, black = (agent, opponent) if plays_white else (opponent, agent)
    outcome = play_match(
        local(white, index),
        local(black, index),
        arguments.base_ms,
        arguments.increment_ms,
        ply_cap=arguments.ply_cap,
        start_fen=fen,
    )
    return Game(index, opening, plays_white, outcome)


def main() -> None:
    parser = argparse.ArgumentParser(description="Score an agent over several games.")
    parser.add_argument("--agent", type=Path, default=Path("."))
    parser.add_argument("--opponent", type=Path, default=Path("baselines/greedy"))
    parser.add_argument("--games", type=int, default=2 * len(OPENINGS))
    parser.add_argument("--base-ms", type=int, default=FAST_BASE_MS)
    parser.add_argument("--increment-ms", type=int, default=FAST_INCREMENT_MS)
    parser.add_argument("--ply-cap", type=int, default=PLY_CAP)
    parser.add_argument("--pgn-dir", type=Path)
    arguments = parser.parse_args()

    if arguments.pgn_dir:
        arguments.pgn_dir.mkdir(parents=True, exist_ok=True)

    agent = arguments.agent.resolve()
    opponent = arguments.opponent.resolve()
    played = max(2, arguments.games - arguments.games % 2)
    games: list[Game] = []
    for index in range(played):
        game = play(index, agent, opponent, arguments)
        games.append(game)
        print(
            f"Game {game.index + 1}/{played}, {game.opening} as {game.colour}, "
            f"{game.outcome.result} by {game.outcome.termination}"
        )
        if arguments.pgn_dir:
            destination = arguments.pgn_dir / f"game-{game.index + 1:03d}.pgn"
            destination.write_text(game.outcome.pgn + "\n")

    scored = [game for game in games if game.outcome.result != "void"]
    draws = sum(1 for game in scored if game.outcome.result == "draw")
    wins = sum(1 for game in scored if game.outcome.result != "draw" and game.won)
    terminations: dict[str, int] = {}
    for game in games:
        terminations[game.outcome.termination] = terminations.get(game.outcome.termination, 0) + 1

    print(f"\n{arguments.agent} vs {arguments.opponent}")
    print("Terminations " + ", ".join(f"{name} {count}" for name, count in terminations.items()))
    if scored:
        _report(wins, draws, len(scored) - wins - draws)
    broken = {name: count for name, count in terminations.items() if name in FAILED_TERMINATIONS}
    if broken:
        raise SystemExit(
            "Your agent failed to finish a game. "
            + ", ".join(f"{name} {count}" for name, count in broken.items())
        )


def _report(wins: int, draws: int, losses: int) -> None:
    played = wins + draws + losses
    score = (wins + draws / 2) / played
    if played < 2:
        print(f"+{wins} ={draws} -{losses}, score {score:.1%}")
        return
    spread = wins * (1 - score) ** 2 + draws * (0.5 - score) ** 2 + losses * score**2
    margin = CONFIDENCE * math.sqrt(spread / (played - 1) / played)
    if margin == 0.0:
        print(f"+{wins} ={draws} -{losses}, score {score:.1%}, every game had the same result")
        return
    print(f"+{wins} ={draws} -{losses}, score {score:.1%} +- {margin:.1%}")
    if score - margin > 0.0 and score + margin < 1.0:
        low, high = _elo(score - margin), _elo(score + margin)
        print(f"Elo {_elo(score):+.0f}, 95% interval {low:+.0f} to {high:+.0f}")


def _elo(score: float) -> float:
    return 400.0 * math.log10(score / (1.0 - score))


if __name__ == "__main__":
    main()
