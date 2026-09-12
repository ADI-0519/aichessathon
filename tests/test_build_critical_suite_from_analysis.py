from __future__ import annotations

import unittest

from tools.build_critical_suite_from_analysis import build_positions


class BuildCriticalSuiteTests(unittest.TestCase):
    def test_selects_focused_errors_by_loss_and_deduplicates_fens(self) -> None:
        fen = "4k3/8/8/8/8/8/q7/R3K3 w - - 0 1"
        payload = {
            "games": [
                {
                    "file": "/tmp/Round 1.pgn",
                    "moves": [
                        {
                            "selected": True,
                            "cp_loss": 150,
                            "best_uci": "a1a2",
                            "uci": "e1d1",
                            "fen": fen,
                            "ply": 3,
                            "san": "Ke2",
                        },
                        {
                            "selected": True,
                            "cp_loss": 200,
                            "best_uci": "a1a2",
                            "uci": "e1d1",
                            "fen": fen,
                            "ply": 4,
                            "san": "Ke2",
                        },
                    ],
                }
            ]
        }
        positions = build_positions(payload, min_cp_loss=100, limit=5)
        self.assertEqual(len(positions), 1)
        self.assertEqual(positions[0]["reference_move"], "a1a2")
        self.assertIn("ply-4", positions[0]["id"])

    def test_ignores_unselected_small_and_same_move_records(self) -> None:
        payload = {
            "games": [
                {
                    "file": "game.pgn",
                    "moves": [
                        {"selected": False, "cp_loss": 500},
                        {"selected": True, "cp_loss": 40},
                        {
                            "selected": True,
                            "cp_loss": 500,
                            "best_uci": "e2e4",
                            "uci": "e2e4",
                        },
                    ],
                }
            ]
        }
        self.assertEqual(build_positions(payload, 80, 10), [])

    def test_filters_low_clock_errors_when_requested(self) -> None:
        payload = {
            "games": [
                {
                    "file": "round.pgn",
                    "moves": [
                        {
                            "selected": True,
                            "cp_loss": 120,
                            "clock_before_s": 4.0,
                            "best_uci": "a1a2",
                            "uci": "e1d1",
                            "fen": "4k3/8/8/8/8/8/q7/R3K3 w - - 0 1",
                            "ply": 1,
                            "san": "Kd1",
                        }
                    ],
                }
            ]
        }
        self.assertEqual(
            build_positions(payload, 80, 10, min_clock_before_s=20.0),
            [],
        )


if __name__ == "__main__":
    unittest.main()
