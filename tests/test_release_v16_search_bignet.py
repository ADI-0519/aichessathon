from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

REPOSITORY = Path(__file__).resolve().parents[1]
CHALLENGER = REPOSITORY / "challengers" / "exp_release_v16_search_bignet"
MODEL_SHA256 = "78931e8692e0ad3b90a6fa4614643aa9c2b9d8bba8731419c26eae2108b93647"


class ReleaseV16SearchBigNetTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
