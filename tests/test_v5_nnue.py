from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT / "challengers" / "v5_nnue"
MODEL_SHA256 = "76336cbb0e2b270ab5e626e2f88201afef43972d0fd1a9bd056db8486b2614ef"


class V5NnueTests(unittest.TestCase):
    def test_model_artifact_is_exact_training_export(self) -> None:
        model = CANDIDATE / "weights" / "model.npz"
        digest = hashlib.sha256(model.read_bytes()).hexdigest()
        self.assertEqual(digest, MODEL_SHA256)

    def test_incremental_updates_and_inference_parity(self) -> None:
        completed = subprocess.run(
            (
                sys.executable,
                "-m",
                "tools.verify_v5_nnue",
                "--candidate",
                str(CANDIDATE),
                "--random-plies",
                "80",
                "--benchmark-iterations",
                "1000",
            ),
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=60.0,
        )
        report = json.loads(completed.stdout)
        self.assertEqual(report["positions_checked"], 86)
        self.assertEqual(report["maximum_accumulator_error"], 0.0)
        self.assertLess(report["maximum_quantization_error_cp"], 8.0)
        self.assertGreater(report["evaluations_per_second"], 10_000)


if __name__ == "__main__":
    unittest.main()
