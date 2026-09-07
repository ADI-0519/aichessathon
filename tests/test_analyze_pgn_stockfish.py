from __future__ import annotations

import unittest

import chess
import chess.engine
import chess.pgn

from tools.analyze_pgn_stockfish import (
    ClockTracker,
    PlayerHeaderError,
    ScoreSnapshot,
    _selected_color,
    classify_move,
    score_snapshot,
)


class AnalyzePgnStockfishTests(unittest.TestCase):
    def test_cp_loss_is_computed_only_for_two_cp_scores(self) -> None:
        self.assertEqual(
            classify_move(ScoreSnapshot(80, None), ScoreSnapshot(25, None)),
            (55, "centipawn"),
        )
        self.assertEqual(
            classify_move(ScoreSnapshot(10, None), ScoreSnapshot(30, None)),
            (0, "centipawn"),
        )

    def test_mate_transitions_do_not_create_fake_cp_loss(self) -> None:
        self.assertEqual(
            classify_move(ScoreSnapshot(None, 4), ScoreSnapshot(600, None)),
            (None, "forced_mate_lost"),
        )
        self.assertEqual(
            classify_move(ScoreSnapshot(-100, None), ScoreSnapshot(None, -3)),
            (None, "forced_mate_allowed"),
        )
        self.assertEqual(
            classify_move(ScoreSnapshot(None, 2), ScoreSnapshot(None, 0)),
            (None, "forced_mate_maintained"),
        )

    def test_score_snapshot_obeys_requested_point_of_view(self) -> None:
        score = chess.engine.PovScore(chess.engine.Cp(42), chess.WHITE)
        self.assertEqual(score_snapshot(score, chess.WHITE), ScoreSnapshot(42, None))
        self.assertEqual(score_snapshot(score, chess.BLACK), ScoreSnapshot(-42, None))
        mate = chess.engine.PovScore(chess.engine.Mate(3), chess.BLACK)
        self.assertEqual(score_snapshot(mate, chess.WHITE), ScoreSnapshot(None, -3))

    def test_first_move_does_not_receive_increment_early(self) -> None:
        tracker = ClockTracker(120.0, 0.5)
        self.assertEqual(tracker.observe(chess.WHITE, 118.0), (120.0, 2.0))
        self.assertEqual(tracker.observe(chess.BLACK, 119.0), (120.0, 1.0))
        self.assertEqual(tracker.observe(chess.WHITE, 116.5), (118.5, 2.0))

    def test_player_header_selects_color_case_insensitively(self) -> None:
        game = chess.pgn.Game()
        game.headers["White"] = "AIY"
        game.headers["Black"] = "Opponent"
        self.assertEqual(_selected_color(game, None, "aiy"), chess.WHITE)
        with self.assertRaisesRegex(PlayerHeaderError, "must match exactly one"):
            _selected_color(game, None, "missing")


if __name__ == "__main__":
    unittest.main()
