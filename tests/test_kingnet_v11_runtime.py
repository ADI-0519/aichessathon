from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Any

import chess
import numpy as np

from tools.search_diagnostics import load_engine_modules
from tools.train_kingnet_v11 import ModelConfig, V11BigEvaluator, export_model

REPOSITORY = Path(__file__).resolve().parents[1]
RUNTIME_ROOT = REPOSITORY / "challengers" / "exp_kingnet_v11_big"


class KingNetV11RuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name) / "engine"
        shutil.copytree(RUNTIME_ROOT, cls.root)

        head_map = tuple(0 if count <= 16 else 1 for count in range(33))
        config = ModelConfig(
            accumulator=8,
            hidden=4,
            pairwise_width=4,
            cp_scale=400.0,
            piece_head_map=head_map,
        )
        model = V11BigEvaluator(config)
        export_model(model, cls.root / "weights" / "model.npz")
        cls.engine, cls.search = load_engine_modules(cls.root)
        cls.nnue = cls.search.nnue

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def _packed_move(self, position: Any, uci: str) -> int:
        engine: Any = self.engine
        for candidate in engine.legal_moves(position):
            packed = int(candidate)
            if engine.move_to_uci(packed) == uci:
                return packed
        self.fail(f"engine did not generate legal move {uci}")

    def test_quantized_evaluation_tracks_exported_float_model(self) -> None:
        engine: Any = self.engine
        nnue: Any = self.nnue
        positions = (
            chess.STARTING_FEN,
            "r3k2r/ppp2ppp/2n1bn2/3qp3/3P4/2P1BN2/PPQ2PPP/R3K2R w KQkq - 2 10",
            "8/8/4k3/8/3P4/3K4/8/8 b - - 0 40",
        )
        for fen in positions:
            with self.subTest(fen=fen):
                position = engine.position_from_fen(fen)
                accumulator = np.empty((2, nnue.ACCUMULATOR_ROW), dtype=np.int32)
                nnue.rebuild(position.pieces, accumulator)
                side = int(position.state[engine.STATE_SIDE])
                quantized = nnue.evaluate(position.pieces, accumulator, side)
                reference = nnue.evaluate_reference(position.pieces, side)
                self.assertLessEqual(abs(quantized - reference), 2.0)

    def test_incremental_update_refreshes_a_stale_parent(self) -> None:
        engine: Any = self.engine
        nnue: Any = self.nnue
        position = engine.position_from_fen("7k/8/8/8/8/8/1K6/R7 w - - 0 1")
        parent = np.empty((2, nnue.ACCUMULATOR_ROW), dtype=np.int32)
        first_child = np.empty_like(parent)
        second_child = np.empty_like(parent)
        rebuilt = np.empty_like(parent)
        nnue.rebuild(position.pieces, parent)

        first = self._packed_move(position, "b2c3")
        nnue.update_for_move(position.pieces, position.state, first, parent, first_child)
        self.assertEqual(first_child[0, nnue.BUCKET_SLOT], nnue.STALE)
        undo = np.empty(engine.UNDO_SIZE, dtype=np.int64)
        undo_key = np.empty(1, dtype=np.uint64)
        self.assertTrue(
            engine.make_move(
                position.pieces, position.state, position.key, first, undo, undo_key
            )
        )

        second = self._packed_move(position, "h8g8")
        nnue.update_for_move(
            position.pieces, position.state, second, first_child, second_child
        )
        self.assertNotEqual(first_child[0, nnue.BUCKET_SLOT], nnue.STALE)
        self.assertTrue(
            engine.make_move(
                position.pieces, position.state, position.key, second, undo, undo_key
            )
        )
        nnue.rebuild(position.pieces, rebuilt)
        np.testing.assert_array_equal(second_child, rebuilt)


if __name__ == "__main__":
    unittest.main()
