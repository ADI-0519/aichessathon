from __future__ import annotations

import unittest

from tools.search_diagnostics import DEFAULT_SUITE, load_critical_positions
from tools.search_memory_replay import DEFAULT_REPLAY_SUITE, load_replay_games


class SearchMemoryReplayTests(unittest.TestCase):
    def test_replays_reach_every_critical_position_legally(self) -> None:
        critical = load_critical_positions(DEFAULT_SUITE)
        games = load_replay_games(DEFAULT_REPLAY_SUITE, critical)
        self.assertEqual([game.identifier for game in games], ["corundum-ai", "negamaximus"])
        replayed_cases = {
            target.case_id for game in games for target in game.targets
        }
        self.assertEqual(replayed_cases, {position.identifier for position in critical})
        self.assertEqual(games[0].engine_color, "white")
        self.assertEqual(games[1].engine_color, "black")


if __name__ == "__main__":
    unittest.main()
