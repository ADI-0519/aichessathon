from __future__ import annotations

import unittest

import chess

from tools.evaluation_features import (
    FEATURE_NAMES,
    MAX_PHASE,
    RAW_FEATURE_NAMES,
    extract_features,
)


class EvaluationFeatureTests(unittest.TestCase):
    def test_schema_and_taper_are_stable(self) -> None:
        features = extract_features(chess.Board())

        self.assertEqual(features.phase, MAX_PHASE)
        self.assertEqual(len(features.raw_values), len(RAW_FEATURE_NAMES))
        self.assertEqual(len(features.values), len(FEATURE_NAMES))
        self.assertTrue(
            all(features.values[index] == 0 for index in range(1, len(features.values), 2))
        )

    def test_extraction_does_not_mutate_board(self) -> None:
        board = chess.Board("r3k2r/ppp2ppp/2n5/3pp3/3PP3/2N5/PPP2PPP/R3K2R w KQkq - 0 1")
        before = board.fen()

        extract_features(board)

        self.assertEqual(board.fen(), before)

    def test_all_features_are_side_relative_without_a_constant_tempo_column(self) -> None:
        white = chess.Board("4k3/8/8/8/8/2N5/8/4K3 w - - 0 1")
        black = white.copy(stack=False)
        black.turn = chess.BLACK

        white_features = extract_features(white)
        black_features = extract_features(black)

        self.assertEqual(black_features.raw_values, tuple(-v for v in white_features.raw_values))
        self.assertEqual(black_features.values, tuple(-v for v in white_features.values))
        self.assertTrue(any(white_features.raw_values))

    def test_endgame_uses_only_endgame_channels(self) -> None:
        features = extract_features(chess.Board("4k3/8/8/8/4P3/8/8/4K3 w - - 0 1"))

        self.assertEqual(features.phase, 0)
        self.assertTrue(
            all(features.values[index] == 0 for index in range(0, len(features.values), 2))
        )

    def test_hanging_piece_is_a_positive_opponent_liability(self) -> None:
        board = chess.Board("n3k3/8/8/8/8/8/8/R3K3 w - - 0 1")

        self.assertGreater(extract_features(board).raw_as_dict()["hanging_minor"], 0)


if __name__ == "__main__":
    unittest.main()
