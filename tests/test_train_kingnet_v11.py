from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.nnue_features import PADDING_INDEX as BASE_PADDING_INDEX
from tools.pack_nnue_data import PACKED_DTYPE
from tools.train_kingnet_v11 import _horizontal_mirror, load_config


class TrainKingNetV11Tests(unittest.TestCase):
    def test_horizontal_mirror_preserves_piece_slots_and_padding(self) -> None:
        indices = np.array(
            [
                [0, 7, 64 + 8, 11 * 64 + 63, BASE_PADDING_INDEX],
                [5, 10, 130, 700, BASE_PADDING_INDEX],
            ],
            dtype=np.int64,
        )
        original_slots = indices // 64
        rows = np.array([True, False], dtype=np.bool_)

        _horizontal_mirror(indices, rows)

        np.testing.assert_array_equal(indices[0, :4], np.array([7, 0, 64 + 15, 11 * 64 + 56]))
        self.assertEqual(int(indices[0, 4]), BASE_PADDING_INDEX)
        np.testing.assert_array_equal(indices[1], np.array([5, 10, 130, 700, BASE_PADDING_INDEX]))
        np.testing.assert_array_equal(indices[:, :4] // 64, original_slots[:, :4])

    def test_config_paths_are_relative_to_config_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            data.mkdir()
            records = np.zeros(2, dtype=PACKED_DTYPE)
            records["count"] = 8
            np.save(data / "train.npy", records, allow_pickle=False)
            np.save(data / "validation.npy", records, allow_pickle=False)

            config = {
                "model": {"accumulator": 128, "hidden": 32},
                "training": {
                    "epochs": 1,
                    "samples_per_epoch": 2,
                    "batch_size": 1,
                    "learning_rate": 0.001,
                    "device": "cpu",
                },
                "piece_bands": [
                    {"name": "all", "min_pieces": 2, "max_pieces": 32, "weight": 1.0}
                ],
                "train_shards": [
                    {"name": "train", "path": "data/train.npy", "weight": 1.0}
                ],
                "validation_sets": [
                    {
                        "name": "validation",
                        "path": "data/validation.npy",
                        "weight": 1.0,
                    }
                ],
            }
            path = root / "experiment.json"
            path.write_text(json.dumps(config), encoding="utf-8")

            loaded = load_config(path)

            self.assertEqual(loaded.train_shards[0].path, (data / "train.npy").resolve())
            self.assertEqual(
                loaded.validation_sets[0].path,
                (data / "validation.npy").resolve(),
            )

    def test_piece_band_overlap_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records = np.zeros(2, dtype=PACKED_DTYPE)
            records["count"] = 8
            np.save(root / "train.npy", records, allow_pickle=False)
            np.save(root / "validation.npy", records, allow_pickle=False)
            config = {
                "training": {
                    "epochs": 1,
                    "samples_per_epoch": 2,
                    "batch_size": 1,
                    "learning_rate": 0.001,
                },
                "piece_bands": [
                    {"name": "a", "min_pieces": 2, "max_pieces": 12, "weight": 1.0},
                    {"name": "b", "min_pieces": 12, "max_pieces": 32, "weight": 1.0},
                ],
                "train_shards": [{"name": "t", "path": "train.npy"}],
                "validation_sets": [{"name": "v", "path": "validation.npy"}],
            }
            path = root / "experiment.json"
            path.write_text(json.dumps(config), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "overlaps"):
                load_config(path)


if __name__ == "__main__":
    unittest.main()
