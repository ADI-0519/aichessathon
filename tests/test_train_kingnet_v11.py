from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import chess
import numpy as np
import torch

from tools.nnue_features import PADDING_INDEX as BASE_PADDING_INDEX
from tools.nnue_features import padded_indices
from tools.pack_nnue_data import PACKED_DTYPE
from tools.train_kingnet_v11 import (
    ModelConfig,
    SelectionObjective,
    V11BigEvaluator,
    _horizontal_mirror,
    _selection_score,
    export_model,
    load_config,
    train,
)

ONE_HEAD_MAP = [0] * 33


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
                "model": {
                    "accumulator": 128,
                    "hidden": 32,
                    "pairwise_width": 64,
                    "cp_scale": 400.0,
                    "piece_head_map": ONE_HEAD_MAP,
                },
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
                "selection_objective": {"overall": 1.0},
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
                "model": {
                    "accumulator": 128,
                    "hidden": 32,
                    "pairwise_width": 64,
                    "cp_scale": 400.0,
                    "piece_head_map": ONE_HEAD_MAP,
                },
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
                "selection_objective": {"overall": 1.0},
            }
            path = root / "experiment.json"
            path.write_text(json.dumps(config), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "overlaps"):
                load_config(path)

    def test_same_dataset_under_two_names_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records = np.zeros(2, dtype=PACKED_DTYPE)
            records["count"] = 8
            np.save(root / "shared.npy", records, allow_pickle=False)
            config = {
                "model": {
                    "accumulator": 128,
                    "hidden": 32,
                    "pairwise_width": 64,
                    "cp_scale": 400.0,
                    "piece_head_map": ONE_HEAD_MAP,
                },
                "training": {
                    "epochs": 1,
                    "samples_per_epoch": 2,
                    "batch_size": 1,
                    "learning_rate": 0.001,
                },
                "piece_bands": [
                    {"name": "all", "min_pieces": 2, "max_pieces": 32, "weight": 1.0}
                ],
                "train_shards": [{"name": "train", "path": "shared.npy"}],
                "validation_sets": [{"name": "validation", "path": "shared.npy"}],
                "selection_objective": {"overall": 1.0},
            }
            path = root / "experiment.json"
            path.write_text(json.dumps(config), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "dataset leakage"):
                load_config(path)

    def test_material_aware_selection_uses_configured_weights(self) -> None:
        metrics = {
            "overall": {"probability_mse": 0.10},
            "by_piece_band": {
                "endgame": {"probability_mse": 0.30},
                "opening": {"probability_mse": 0.05},
            },
        }
        selection = SelectionObjective(
            overall=1.0,
            by_piece_band=(("endgame", 2.0), ("opening", 0.0)),
        )
        self.assertAlmostEqual(_selection_score(metrics, selection), 0.7 / 3.0)

    def test_export_contains_configured_piece_head_map(self) -> None:
        mapping = tuple([0] * 9 + [1] * 24)
        model = V11BigEvaluator(
            ModelConfig(
                accumulator=8,
                hidden=4,
                pairwise_width=4,
                cp_scale=400.0,
                piece_head_map=mapping,
            )
        )
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "model.npz"
            export_model(model, output)
            with np.load(output, allow_pickle=False) as archive:
                self.assertEqual(int(archive["format_version"]), 3)
                np.testing.assert_array_equal(
                    archive["piece_head_map"], np.asarray(mapping, dtype=np.int32)
                )
                self.assertEqual(archive["hidden_weights"].shape, (2, 4, 8))

    def test_float16_feature_export_has_bounded_runtime_quantization_drift(self) -> None:
        model = V11BigEvaluator(
            ModelConfig(
                accumulator=8,
                hidden=4,
                pairwise_width=4,
                cp_scale=400.0,
                piece_head_map=tuple(0 for _ in range(33)),
            )
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            full = root / "full.npz"
            compact = root / "compact.npz"
            export_model(model, full, feature_storage="float32")
            export_model(model, compact, feature_storage="float16")
            with np.load(full) as full_archive, np.load(compact) as compact_archive:
                full_q = np.rint(full_archive["feature_weights"] * 2048).astype(np.int16)
                compact_q = np.rint(
                    compact_archive["feature_weights"].astype(np.float32) * 2048
                ).astype(np.int16)
                maximum_drift = int(
                    np.max(np.abs(full_q.astype(np.int32) - compact_q))
                )
                self.assertLessEqual(maximum_drift, 1)
                self.assertEqual(compact_archive["feature_weights"].dtype, np.float16)
                self.assertEqual(
                    int(compact_archive["feature_quantization_max_delta"]),
                    maximum_drift,
                )
                self.assertEqual(int(compact_archive["runtime_input_scale"]), 2048)
            self.assertLess(compact.stat().st_size, full.stat().st_size * 0.55)

    def test_tiny_training_writes_resumable_checkpoint_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train_records = np.zeros(4, dtype=PACKED_DTYPE)
            validation_records = np.zeros(3, dtype=PACKED_DTYPE)
            for records, cp in ((train_records, 40), (validation_records, -25)):
                indices, count = padded_indices(chess.Board())
                records["indices"] = indices
                records["count"] = count
                records["stm"] = 1
                records["cp"] = cp
            np.save(root / "train.npy", train_records, allow_pickle=False)
            np.save(root / "validation.npy", validation_records, allow_pickle=False)
            config = {
                "model": {
                    "accumulator": 8,
                    "hidden": 4,
                    "pairwise_width": 4,
                    "cp_scale": 400.0,
                    "piece_head_map": ONE_HEAD_MAP,
                },
                "training": {
                    "epochs": 1,
                    "samples_per_epoch": 4,
                    "batch_size": 2,
                    "learning_rate": 0.001,
                    "device": "cpu",
                    "seed": 7,
                },
                "piece_bands": [
                    {"name": "all", "min_pieces": 2, "max_pieces": 32, "weight": 1.0}
                ],
                "selection_objective": {
                    "overall": 0.5,
                    "by_piece_band": {"all": 0.5},
                },
                "train_shards": [{"name": "train", "path": "train.npy"}],
                "validation_sets": [
                    {"name": "validation", "path": "validation.npy"}
                ],
            }
            config_path = root / "experiment.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            output = root / "model.npz"
            manifest = root / "manifest.json"
            checkpoint = root / "recovery.pt"

            train(config_path, output, manifest, checkpoint)

            metadata = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(metadata["schema_version"], 3)
            slope = metadata["history"][0]["validation"]["validation"]["overall"][
                "calibration_slope"
            ]
            self.assertIsInstance(slope, float)
            self.assertTrue(output.is_file())
            self.assertTrue(checkpoint.is_file())
            recovery = torch.load(checkpoint, map_location="cpu", weights_only=False)
            self.assertEqual(recovery["completed_epoch"], 1)
            self.assertEqual(recovery["global_step"], 2)
            self.assertIn("model_state", recovery)
            self.assertIn("best_model_state", recovery)
            self.assertIn("optimizer_state", recovery)
            self.assertIn("numpy_rng_state", recovery)
            self.assertIn("torch_rng_state", recovery)


if __name__ == "__main__":
    unittest.main()
