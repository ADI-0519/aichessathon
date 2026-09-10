from __future__ import annotations

import unittest

import chess
import numpy as np

from tools.nnue_features import (
    FEATURE_COUNT,
    MAX_PIECES,
    PADDING_INDEX,
    canonical_indices,
    flip_board_colours,
    orient_index,
    oriented_indices,
    padded_indices,
)


class NnueFeatureTests(unittest.TestCase):
    def test_start_position_encoding_is_complete_and_unique(self) -> None:
        board = chess.Board()
        indices = canonical_indices(board)

        self.assertEqual(len(indices), 32)
        self.assertEqual(len(set(indices)), 32)
        self.assertTrue(all(0 <= index < FEATURE_COUNT for index in indices))

    def test_black_perspective_matches_colour_flipped_position(self) -> None:
        board = chess.Board(
            "r3k2r/pp2qppp/2n1bn2/2pp4/3P4/2P1PN2/PPQ1BPPP/R3K2R w KQkq - 4 12"
        )
        mirrored = flip_board_colours(board)

        black_view = sorted(oriented_indices(canonical_indices(board), chess.BLACK))
        mirrored_white_view = sorted(
            oriented_indices(canonical_indices(mirrored), chess.WHITE)
        )

        self.assertEqual(black_view, mirrored_white_view)

    def test_padding_uses_out_of_vocabulary_sentinel(self) -> None:
        board = chess.Board("8/8/8/3k4/8/3K4/4P3/8 w - - 0 1")
        encoded, count = padded_indices(board)

        self.assertEqual(encoded.dtype, np.uint16)
        self.assertEqual(encoded.shape, (MAX_PIECES,))
        self.assertEqual(count, 3)
        self.assertTrue(np.all(encoded[count:] == PADDING_INDEX))

    def test_orient_index_rejects_invalid_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "outside"):
            orient_index(FEATURE_COUNT, chess.WHITE)


if __name__ == "__main__":
    unittest.main()
