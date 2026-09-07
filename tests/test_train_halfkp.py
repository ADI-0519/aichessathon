from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import chess
import numpy as np
import torch

from tools.halfkp_features import FEATURE_COUNT, PADDING_INDEX
from tools.nnue_features import MAX_PIECES, canonical_indices
from tools.nnue_features import PADDING_INDEX as BASE_PADDING_INDEX
from tools.train_halfkp import (
    FORMAT_VERSION,
    KingConditionedEvaluator,
    ModelConfig,
    export_model,
    initialise_from_v5,
    king_conditioned_indices,
)
from tools.train_nnue import black_perspective


def _tensor(board: chess.Board) -> torch.Tensor:
    values = list(canonical_indices(board))
    values.extend([BASE_PADDING_INDEX] * (MAX_PIECES - len(values)))
    return torch.tensor([values], dtype=torch.long)


class TrainHalfKpTests(unittest.TestCase):
    def test_conditioned_indices_preserve_padding_and_range(self) -> None:
        canonical = _tensor(chess.Board())
        canonical[0, -1] = BASE_PADDING_INDEX

        for black in (False, True):
            encoded = king_conditioned_indices(canonical, black=black)
            self.assertEqual(int(encoded[0, -1]), PADDING_INDEX)
            self.assertTrue(bool(torch.all(encoded[0, :-1] < FEATURE_COUNT)))

    def test_model_is_colour_symmetric(self) -> None:
        torch.manual_seed(11)
        model = KingConditionedEvaluator(ModelConfig(accumulator=8, hidden=4)).eval()
        board = chess.Board(
            "r3k2r/pp2qppp/2n1bn2/2pp4/3P4/2P1PN2/PPQ1BPPP/R3K2R w KQkq - 4 12"
        )

        original = model(_tensor(board), torch.tensor([board.turn]))
        mirrored = board.mirror()
        reflected = model(_tensor(mirrored), torch.tensor([mirrored.turn]))

        self.assertTrue(torch.allclose(original, reflected, atol=1e-6))

    def test_v5_lift_preserves_float_output(self) -> None:
        rng = np.random.default_rng(19)
        old_accumulator = 4
        hidden = 3
        feature_weights = rng.normal(0.0, 0.01, (768, old_accumulator)).astype(
            np.float32
        )
        accumulator_bias = rng.normal(0.0, 0.01, old_accumulator).astype(np.float32)
        hidden_weights = rng.normal(
            0.0, 0.05, (hidden, 2 * old_accumulator)
        ).astype(np.float32)
        hidden_bias = rng.normal(0.0, 0.01, hidden).astype(np.float32)
        output_weights = rng.normal(0.0, 0.05, (1, hidden)).astype(np.float32)
        output_bias = rng.normal(0.0, 0.01, 1).astype(np.float32)

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "v5.npz"
            np.savez(
                path,
                format_version=np.asarray(1, dtype=np.int32),
                cp_scale=np.asarray(400.0, dtype=np.float32),
                feature_weights=feature_weights,
                accumulator_bias=accumulator_bias,
                hidden_weights=hidden_weights,
                hidden_bias=hidden_bias,
                output_weights=output_weights,
                output_bias=output_bias,
            )
            model = KingConditionedEvaluator(
                ModelConfig(accumulator=8, hidden=hidden)
            ).eval()
            initialise_from_v5(model, path)

        canonical = _tensor(chess.Board())
        white_indices = canonical[0, :32]
        black_indices = black_perspective(canonical)[0, :32]
        old_white = torch.from_numpy(feature_weights)[white_indices].sum(0)
        old_white += torch.from_numpy(accumulator_bias)
        old_black = torch.from_numpy(feature_weights)[black_indices].sum(0)
        old_black += torch.from_numpy(accumulator_bias)
        old_inputs = torch.cat((old_white, old_black)).clamp(0.0, 1.0)
        old_hidden = (
            torch.from_numpy(hidden_weights) @ old_inputs
            + torch.from_numpy(hidden_bias)
        ).clamp(0.0, 1.0)
        expected = (
            torch.from_numpy(output_weights) @ old_hidden
            + torch.from_numpy(output_bias)
        )[0]

        actual = model(canonical, torch.tensor([True]))[0]
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))

        extra_features = model.embedding.weight[:FEATURE_COUNT, old_accumulator:]
        own_extra = model.hidden.weight[:, old_accumulator : 2 * old_accumulator]
        opponent_extra = model.hidden.weight[:, 3 * old_accumulator :]
        self.assertGreater(float(extra_features.detach().abs().max()), 0.0)
        self.assertEqual(float(own_extra.detach().abs().max()), 0.0)
        self.assertEqual(float(opponent_extra.detach().abs().max()), 0.0)

    def test_export_contains_only_bounded_integer_runtime_arrays(self) -> None:
        model = KingConditionedEvaluator(ModelConfig(accumulator=8, hidden=4))
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "model.npz"
            export_model(model, output)
            with np.load(output, allow_pickle=False) as archive:
                self.assertEqual(int(archive["format_version"]), FORMAT_VERSION)
                self.assertEqual(archive["feature_weights_q"].dtype, np.int16)
                self.assertEqual(archive["accumulator_bias_q"].dtype, np.int32)
                self.assertEqual(archive["hidden_weights_q"].dtype, np.int16)
                self.assertEqual(archive["hidden_bias_q"].dtype, np.int32)
                self.assertEqual(archive["output_weights_q"].dtype, np.int16)
                self.assertEqual(archive["output_bias_q"].dtype, np.int64)


if __name__ == "__main__":
    unittest.main()
