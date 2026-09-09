from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import chess
import numpy as np

from tools.nnue_features import PADDING_INDEX
from tools.pack_nnue_data import (
    CP_CLAMP,
    encode_record,
    load_report_bands,
    position_ply,
    shuffled_groups,
)


class PackNnueDataTests(unittest.TestCase):
    def test_group_order_is_seeded_and_preserves_membership(self) -> None:
        groups = list(range(20))
        first = shuffled_groups(groups, 17)

        self.assertEqual(first, shuffled_groups(groups, 17))
        self.assertNotEqual(first, shuffled_groups(groups, 18))
        self.assertEqual(sorted(first), groups)
        self.assertEqual(groups, list(range(20)))

    def test_report_bands_come_from_experiment_config(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "experiment.json"
            path.write_text(
                json.dumps(
                    {
                        "piece_bands": [
                            {"name": "ending", "min_pieces": 2, "max_pieces": 12},
                            {"name": "rest", "min_pieces": 13, "max_pieces": 32},
                        ]
                    }
                ),
                encoding="utf-8",
            )

            bands = load_report_bands(path)

        self.assertEqual(
            [(band.name, band.min_pieces, band.max_pieces) for band in bands],
            [("ending", 2, 12), ("rest", 13, 32)],
        )

    def test_position_ply_uses_side_and_fullmove(self) -> None:
        self.assertEqual(position_ply(chess.STARTING_FEN), 0)
        self.assertEqual(position_ply("8/8/8/8/8/8/4K3/7k b - - 0 9"), 17)

    def test_encode_record_preserves_white_perspective_label(self) -> None:
        fen = "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/2N2N2/PPPP1PPP/R1BQKB1R w KQkq - 2 3"
        record = encode_record(fen, 37, None, "f1b5", min_ply=0)

        self.assertIsNotNone(record)
        assert record is not None
        self.assertEqual(int(record["cp"]), 37)
        self.assertEqual(int(record["stm"]), 1)
        self.assertEqual(int(record["count"]), 32)
        self.assertTrue(np.all(record["indices"] < PADDING_INDEX))

    def test_encode_record_filters_tactical_or_unlabelled_rows(self) -> None:
        capture_fen = "rnbqkbnr/pppp1ppp/8/4p3/3PP3/8/PPP2PPP/RNBQKBNR b KQkq - 0 2"
        self.assertIsNone(
            encode_record(capture_fen, 10, None, "e5d4", min_ply=0)
        )
        self.assertIsNone(
            encode_record(chess.STARTING_FEN, None, None, "e2e4", min_ply=0)
        )
        self.assertIsNone(
            encode_record(chess.STARTING_FEN, None, 0, "e2e4", min_ply=0)
        )

    def test_mate_and_cp_scores_are_bounded(self) -> None:
        fen = "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/2N2N2/PPPP1PPP/R1BQKB1R b KQkq - 2 3"
        mate = encode_record(fen, None, -4, "g8f6", min_ply=0)
        cp = encode_record(fen, 99_999, None, "g8f6", min_ply=0)

        self.assertIsNotNone(mate)
        self.assertIsNotNone(cp)
        assert mate is not None and cp is not None
        self.assertEqual(int(mate["cp"]), -CP_CLAMP)
        self.assertEqual(int(cp["cp"]), CP_CLAMP)


if __name__ == "__main__":
    unittest.main()
