from __future__ import annotations

import unittest

import chess
import numpy as np

from tools.halfkp_features import (
    FEATURE_COUNT,
    PADDING_INDEX,
    feature_index,
    padded_perspective_indices,
    perspective_indices,
)


class HalfKpFeatureTests(unittest.TestCase):
    def test_start_position_has_one_feature_per_piece(self) -> None:
        features = perspective_indices(chess.Board(), chess.WHITE)

        self.assertEqual(len(features), 32)
        self.assertEqual(len(set(features)), 32)
        self.assertTrue(all(0 <= value < FEATURE_COUNT for value in features))

    def test_black_view_matches_mirrored_white_view(self) -> None:
        board = chess.Board(
            "r3k2r/pp2qppp/2n1bn2/2pp4/3P4/2P1PN2/PPQ1BPPP/R3K2R w KQkq - 4 12"
        )

        self.assertEqual(
            sorted(perspective_indices(board, chess.BLACK)),
            sorted(perspective_indices(board.mirror(), chess.WHITE)),
        )

    def test_king_square_changes_every_feature(self) -> None:
        first = chess.Board("7k/8/8/8/8/8/4P3/4K3 w - - 0 1")
        second = chess.Board("7k/8/8/8/8/3K4/4P3/8 w - - 1 1")

        self.assertTrue(
            set(perspective_indices(first, chess.WHITE)).isdisjoint(
                perspective_indices(second, chess.WHITE)
            )
        )

    def test_padding_uses_uint16_sentinel(self) -> None:
        encoded, count = padded_perspective_indices(
            chess.Board("7k/8/8/8/8/8/4P3/4K3 w - - 0 1"), chess.WHITE
        )

        self.assertEqual(encoded.dtype, np.uint16)
        self.assertEqual(count, 3)
        self.assertTrue(np.all(encoded[count:] == PADDING_INDEX))

    def test_feature_index_rejects_invalid_values(self) -> None:
        with self.assertRaises(ValueError):
            feature_index(768, 0)
        with self.assertRaises(ValueError):
            feature_index(0, 64)


if __name__ == "__main__":
    unittest.main()
