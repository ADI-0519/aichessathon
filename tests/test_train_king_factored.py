from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import chess
import numpy as np
import torch

from tools.king_features import (
    BASE_FEATURE_COUNT,
    FEATURE_COUNT,
    KING_BUCKET_COUNT,
    PADDING_INDEX,
    king_bucket,
)
from tools.nnue_features import MAX_PIECES, canonical_indices
from tools.nnue_features import PADDING_INDEX as BASE_PADDING_INDEX
from tools.train_king_factored import (
    ModelConfig,
    SparseEvaluator,
    apply_king_bucket,
    black_perspective,
    export_model,
)


def _tensor(board: chess.Board) -> torch.Tensor:
    values = list(canonical_indices(board))
    values.extend([BASE_PADDING_INDEX] * (MAX_PIECES - len(values)))
    return torch.tensor([values], dtype=torch.long)


class TrainKingFactoredTests(unittest.TestCase):
    def test_bucket_map_covers_the_board(self) -> None:
        buckets = {king_bucket(square) for square in chess.SQUARES}
        self.assertEqual(buckets, set(range(KING_BUCKET_COUNT)))
        self.assertEqual(king_bucket(chess.A1), 0)
        self.assertEqual(king_bucket(chess.H8), KING_BUCKET_COUNT - 1)

    def test_conditioned_indices_preserve_padding_and_range(self) -> None:
        canonical = _tensor(chess.Board())
        canonical[0, -1] = BASE_PADDING_INDEX

        for oriented in (canonical, black_perspective(canonical)):
            encoded = apply_king_bucket(oriented)
            self.assertEqual(int(encoded[0, -1]), PADDING_INDEX)
            self.assertTrue(bool(torch.all(encoded[0, :-1] < FEATURE_COUNT)))

    def test_model_is_colour_symmetric(self) -> None:
        torch.manual_seed(23)
        model = SparseEvaluator(ModelConfig(accumulator=8, hidden=4)).eval()
        board = chess.Board(
            "r3k2r/pp2qppp/2n1bn2/2pp4/3P4/2P1PN2/PPQ1BPPP/R3K2R w KQkq - 4 12"
        )
        mirrored = board.mirror()

        original = model(_tensor(board), torch.tensor([board.turn]))
        reflected = model(_tensor(mirrored), torch.tensor([mirrored.turn]))
        self.assertTrue(torch.allclose(original, reflected, atol=1e-6))

    def test_export_folds_shared_and_bucket_weights(self) -> None:
        model = SparseEvaluator(ModelConfig(accumulator=8, hidden=4))
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "model.npz"
            export_model(model, output)
            with np.load(output, allow_pickle=False) as archive:
                self.assertEqual(int(archive["format_version"]), 2)
                self.assertEqual(archive["feature_weights"].shape, (FEATURE_COUNT, 8))
                self.assertEqual(archive["hidden_weights"].shape, (4, 16))
                expected = (
                    model.embedding.weight[0].detach().numpy()
                    + model.factor.weight[0].detach().numpy()
                )
                np.testing.assert_allclose(archive["feature_weights"][0], expected)
                self.assertEqual(BASE_FEATURE_COUNT, 768)


if __name__ == "__main__":
    unittest.main()
