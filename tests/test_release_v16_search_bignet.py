from __future__ import annotations

import hashlib
import json
import random
import subprocess
import sys
import unittest
from pathlib import Path
from typing import Any, ClassVar

import chess
import numpy as np

from tools.search_diagnostics import load_engine_modules

REPOSITORY = Path(__file__).resolve().parents[1]
CHALLENGER = REPOSITORY / "challengers" / "exp_release_v16_search_bignet"
MODEL_SHA256 = "78931e8692e0ad3b90a6fa4614643aa9c2b9d8bba8731419c26eae2108b93647"


class ReleaseV16SearchBigNetTests(unittest.TestCase):
    engine: ClassVar[Any]
    nnue: ClassVar[Any]

    @classmethod
    def setUpClass(cls) -> None:
        cls.engine, search = load_engine_modules(CHALLENGER)
        cls.nnue = search.nnue

    def _packed_move(self, position: Any, move: chess.Move) -> int:
        for candidate in self.engine.legal_moves(position):
            packed = int(candidate)
            if self.engine.move_to_uci(packed) == move.uci():
                return packed
        self.fail(f"engine did not generate legal move {move.uci()}")

    def _assert_post_move_update(
        self,
        board: chess.Board,
        move: chess.Move,
        *,
        force_stale: bool = False,
    ) -> None:
        position = self.engine.position_from_board(board)
        packed = self._packed_move(position, move)
        parent = np.empty((2, self.nnue.ACCUMULATOR_ROW), dtype=np.int32)
        child = np.empty_like(parent)
        rebuilt = np.empty_like(parent)
        self.nnue.rebuild(position.pieces, parent)

        if force_stale:
            parent[:, self.nnue.BUCKET_SLOT] = self.nnue.STALE

        working = position.copy()
        undo = np.empty(self.engine.UNDO_SIZE, dtype=np.int64)
        undo_key = np.empty(1, dtype=np.uint64)
        self.assertTrue(
            self.engine.make_move(
                working.pieces,
                working.state,
                working.key,
                packed,
                undo,
                undo_key,
            )
        )
        self.nnue.update_after_move(packed, undo, parent, child)

        if force_stale:
            np.testing.assert_array_equal(
                child[:, self.nnue.BUCKET_SLOT],
                np.full(2, self.nnue.STALE, dtype=np.int32),
            )
            np.testing.assert_array_equal(
                child[:, self.nnue.COUNT_SLOT], np.zeros(2, dtype=np.int32)
            )

        self.nnue.refresh(working.pieces, child)
        self.nnue.rebuild(working.pieces, rebuilt)
        np.testing.assert_array_equal(child, rebuilt)

    def test_model_matches_the_teammate_export(self) -> None:
        model = CHALLENGER / "weights" / "model.npz"
        self.assertEqual(hashlib.sha256(model.read_bytes()).hexdigest(), MODEL_SHA256)
        with np.load(model, allow_pickle=False) as archive:
            self.assertEqual(int(archive["format_version"]), 3)
            self.assertEqual(archive["feature_weights"].shape, (12_288, 256))
            self.assertEqual(archive["hidden_weights"].shape, (8, 32, 256))

    def test_incremental_updates_and_fixed_point_inference(self) -> None:
        completed = subprocess.run(
            (
                sys.executable,
                "-m",
                "tools.verify_kingnet",
                "--candidate",
                str(CHALLENGER),
                "--random-plies",
                "80",
                "--benchmark-iterations",
                "1000",
            ),
            cwd=REPOSITORY,
            check=True,
            capture_output=True,
            text=True,
            timeout=90.0,
        )
        report = json.loads(completed.stdout)
        self.assertEqual(report["positions_checked"], 86)
        self.assertEqual(report["maximum_incremental_error"], 0.0)
        self.assertLess(report["maximum_inference_error_cp"], 8.0)
        self.assertGreaterEqual(report["stale_bucket_transitions"], 1)

    def test_post_move_updates_cover_special_moves_and_bucket_crossing(self) -> None:
        cases = (
            (chess.STARTING_FEN, "e2e4", "quiet move"),
            ("4k3/8/8/8/8/8/3p4/3RK3 w - - 0 1", "d1d2", "capture"),
            ("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", "e5d6", "en passant"),
            ("4k3/P7/8/8/8/8/8/4K3 w - - 0 1", "a7a8q", "promotion"),
            (
                "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
                "e1g1",
                "castling",
            ),
            ("7k/8/8/8/8/8/1K6/8 w - - 0 1", "b2c3", "king bucket"),
        )
        for fen, uci, label in cases:
            with self.subTest(label=label):
                self._assert_post_move_update(
                    chess.Board(fen), chess.Move.from_uci(uci)
                )

    def test_post_move_updates_match_random_legal_sequences(self) -> None:
        generator = random.Random(0xA1C4E55A)
        checked = 0
        for _game in range(4):
            board = chess.Board()
            for _ply in range(32):
                legal_moves = list(board.legal_moves)
                if not legal_moves:
                    break
                move = generator.choice(legal_moves)
                self._assert_post_move_update(board, move)
                board.push(move)
                checked += 1
        self.assertGreaterEqual(checked, 100)

    def test_post_move_update_propagates_and_repairs_a_stale_parent(self) -> None:
        self._assert_post_move_update(
            chess.Board("7k/8/8/8/8/8/1K6/R7 w - - 0 1"),
            chess.Move.from_uci("a1a2"),
            force_stale=True,
        )


if __name__ == "__main__":
    unittest.main()
