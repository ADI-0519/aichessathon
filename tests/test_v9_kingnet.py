from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT / "challengers" / "v9_kingnet"
MODEL_SHA256 = "9348d4e0ca5e2ee10c11003e7363953316e721df16db3bdafc272451e7550087"


class V9KingnetTests(unittest.TestCase):
    def test_model_artifact_matches_teammate_export(self) -> None:
        model = CANDIDATE / "weights" / "model.npz"
        self.assertEqual(hashlib.sha256(model.read_bytes()).hexdigest(), MODEL_SHA256)

    def test_incremental_updates_and_inference_parity(self) -> None:
        completed = subprocess.run(
            (
                sys.executable,
                "-m",
                "tools.verify_kingnet",
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
            timeout=90.0,
        )
        report = json.loads(completed.stdout)
        self.assertEqual(report["positions_checked"], 86)
        self.assertEqual(report["maximum_incremental_error"], 0.0)
        self.assertLess(report["maximum_inference_error_cp"], 8.0)
        self.assertGreaterEqual(report["stale_bucket_transitions"], 1)


if __name__ == "__main__":
    unittest.main()
