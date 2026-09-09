from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.kingnet_width_benchmark import (
    DEFAULT_SOURCE,
    materialize_width,
    positive_widths,
)
from tools.train_kingnet_v11 import load_architecture_config


class KingNetWidthBenchmarkTests(unittest.TestCase):
    def test_width_list_validation(self) -> None:
        self.assertEqual(positive_widths("128,256,1024"), (128, 256, 1024))
        for invalid in ("", "0", "127", "128,128", "bad"):
            with self.subTest(value=invalid), self.assertRaises(
                argparse.ArgumentTypeError
            ):
                positive_widths(invalid)

    def test_materialized_width_uses_configured_heads_and_compact_storage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = root / "experiment.json"
            config_path.write_text(
                json.dumps(
                    {
                        "model": {
                            "accumulator": 128,
                            "hidden": 4,
                            "pairwise_width": 64,
                            "cp_scale": 360.0,
                            "piece_head_map": [0] * 17 + [1] * 16,
                        },
                        "export": {"feature_storage": "float16"},
                    }
                ),
                encoding="utf-8",
            )
            architecture, export = load_architecture_config(config_path)
            destination = root / "candidate"

            materialize_width(
                DEFAULT_SOURCE,
                destination,
                architecture,
                width=16,
                feature_storage=export.feature_storage,
            )

            with np.load(destination / "weights" / "model.npz") as archive:
                self.assertEqual(archive["feature_weights"].dtype, np.float16)
                self.assertEqual(archive["feature_weights"].shape, (12_288, 16))
                self.assertEqual(archive["hidden_weights"].shape, (2, 4, 16))
                self.assertEqual(float(archive["cp_scale"]), 360.0)


if __name__ == "__main__":
    unittest.main()
