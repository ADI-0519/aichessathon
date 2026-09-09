from __future__ import annotations

import json
import pathlib
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
    load_reserved_fingerprints,
    pack_groups,
    position_fingerprint,
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

    def test_position_fingerprint_uses_inputs_but_not_teacher_score(self) -> None:
        fen = "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/2N2N2/PPPP1PPP/R1BQKB1R w KQkq - 2 3"
        first = encode_record(fen, 37, None, "f1b5", min_ply=0)
        second = encode_record(fen, -91, None, "f1b5", min_ply=0)

        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        assert first is not None and second is not None
        self.assertEqual(position_fingerprint(first), position_fingerprint(second))

    def test_load_reserved_fingerprints_validates_schema(self) -> None:
        fen = "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/2N2N2/PPPP1PPP/R1BQKB1R w KQkq - 2 3"
        record = encode_record(fen, 37, None, "f1b5", min_ply=0)
        self.assertIsNotNone(record)
        assert record is not None

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            valid = root / "valid.npy"
            invalid = root / "invalid.npy"
            np.save(valid, np.asarray([record], dtype=record.dtype))
            np.save(invalid, np.asarray([1, 2, 3], dtype=np.int64))

            self.assertEqual(
                load_reserved_fingerprints([valid]),
                {position_fingerprint(record)},
            )
            with self.assertRaisesRegex(ValueError, "wrong schema"):
                load_reserved_fingerprints([invalid])

    def test_pack_groups_deduplicates_and_honours_reserved_inputs(self) -> None:
        quiet = "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/2N2N2/PPPP1PPP/R1BQKB1R w KQkq - 2 3"
        other = "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/2N2N2/PPPP1PPP/R1BQKB1R b KQkq - 2 3"

        class Column:
            def __init__(self, values: list[object]) -> None:
                self.values = values

            def to_pylist(self) -> list[object]:
                return self.values

        class Parquet:
            def read_row_group(
                self, group: int, *, columns: list[str]
            ) -> dict[str, Column]:
                del group, columns
                rows = [
                    (quiet, 10, None, "f1b5"),
                    (quiet, 20, None, "f1b5"),
                    (other, 30, None, "g8f6"),
                ]
                return {
                    name: Column([row[index] for row in rows])
                    for index, name in enumerate(("fen", "cp", "mate", "move"))
                }

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            count, stats, fingerprints = pack_groups(
                Parquet(),
                [0],
                root / "validation.npy",
                target=3,
                min_ply=0,
                deduplicate=True,
                collect_fingerprints=True,
            )
            reserved = {position_fingerprint(np.load(
                root / "validation.npy", allow_pickle=False
            )[0])}
            train_count, train_stats, _ = pack_groups(
                Parquet(),
                [0],
                root / "train.npy",
                target=3,
                min_ply=0,
                excluded_fingerprints=reserved,
            )

        self.assertEqual(count, 2)
        self.assertEqual(len(fingerprints), 2)
        self.assertEqual(stats["within_split_duplicates"], 1)
        self.assertEqual(train_count, 1)
        self.assertEqual(train_stats["excluded_position_overlap"], 2)

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

    def test_encode_record_accepts_fishnet_chess960_castling_uci(self) -> None:
        fen = "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"

        record = encode_record(fen, 25, None, "e1h1", min_ply=0)

        self.assertIsNotNone(record)

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
def _sample_rows(count: int) -> dict[str, list[object]]:
    """Build Parquet-shaped rows, some of which the encoder must reject."""
    board = chess.Board()
    fens: list[str] = []
    moves: list[str | None] = []
    while len(fens) < count:
        legal = list(board.legal_moves)
        if not legal or board.is_game_over(claim_draw=False):
            board = chess.Board()
            continue
        move = legal[len(fens) % len(legal)]
        fens.append(board.fen())
        moves.append(move.uci())
        board.push(move)
    return {
        "fen": [*fens],
        "cp": [(index % 401) - 200 for index in range(count)],
        "mate": [None] * count,
        "move": [*moves],
    }


try:  # pyarrow is a training-only dependency, so this suite is conditional
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover - exercised only without the training env
    pa = pq = None


@unittest.skipIf(pq is None, "pyarrow is only installed in the training environment")
class ParallelPackingTests(unittest.TestCase):
    """The worker pool must not change which positions land in the dataset."""

    def test_parallel_packing_matches_a_serial_scan(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            source = root / "rows.parquet"
            pq.write_table(pa.table(_sample_rows(900)), source, row_group_size=100)
            groups = list(range(pq.ParquetFile(source).metadata.num_row_groups))
            self.assertGreater(len(groups), 1)

            serial, serial_stats, _ = pack_groups(
                source, groups, root / "serial.npy", target=200, min_ply=0, workers=1
            )
            parallel, parallel_stats, _ = pack_groups(
                source, groups, root / "parallel.npy", target=200, min_ply=0, workers=4
            )

            self.assertEqual(serial, parallel)
            self.assertEqual(serial_stats, parallel_stats)
            self.assertEqual(serial, 200, "the fixture must exercise the target cut-off")
            np.testing.assert_array_equal(
                np.load(root / "serial.npy"), np.load(root / "parallel.npy")
            )
