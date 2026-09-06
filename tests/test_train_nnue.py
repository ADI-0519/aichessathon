from __future__ import annotations

import unittest

import chess
import torch

from tools.nnue_features import PADDING_INDEX, canonical_indices
from tools.train_nnue import (
    ModelConfig,
    SparseEvaluator,
    black_perspective,
    probability_loss,
)


def _tensor(board: chess.Board) -> torch.Tensor:
    values = list(canonical_indices(board))
    values.extend([PADDING_INDEX] * (32 - len(values)))
    return torch.tensor([values], dtype=torch.long)


class TrainNnueTests(unittest.TestCase):
    def test_black_perspective_preserves_padding_and_range(self) -> None:
        encoded = _tensor(chess.Board())
        encoded[0, -1] = PADDING_INDEX
        oriented = black_perspective(encoded)

        self.assertEqual(int(oriented[0, -1]), PADDING_INDEX)
        self.assertTrue(
            bool(
                torch.all(
                    (oriented[:, :-1] >= 0)
                    & (oriented[:, :-1] < PADDING_INDEX)
                )
            )
        )

    def test_model_is_colour_symmetric(self) -> None:
        torch.manual_seed(7)
        model = SparseEvaluator(ModelConfig(accumulator=16, hidden=8)).eval()
        board = chess.Board(
            "r3k2r/pp2qppp/2n1bn2/2pp4/3P4/2P1PN2/PPQ1BPPP/R3K2R w KQkq - 4 12"
        )
        mirrored = board.mirror()

        original = model(_tensor(board), torch.tensor([board.turn]))
        reflected = model(_tensor(mirrored), torch.tensor([mirrored.turn]))

        self.assertTrue(torch.allclose(original, reflected, atol=1e-6))

    def test_probability_loss_is_zero_for_matching_scaled_logit(self) -> None:
        cp = torch.tensor([-400.0, 0.0, 400.0])
        prediction = cp / 400.0

        self.assertEqual(float(probability_loss(prediction, cp)), 0.0)


if __name__ == "__main__":
    unittest.main()
