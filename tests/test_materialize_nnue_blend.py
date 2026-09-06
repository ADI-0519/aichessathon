from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tools.materialize_nnue_blend import materialize

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "challengers" / "v5_nnue"


class MaterializeNnueBlendTests(unittest.TestCase):
    def test_materializes_complete_candidate_with_requested_blend(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "candidate"
            materialize(SOURCE, output, 50)

            search = (output / "search.py").read_text(encoding="utf-8")
            self.assertIn("NNUE_BLEND = 50\n", search)
            self.assertTrue((output / "weights" / "model.npz").is_file())
            self.assertEqual(
                (output / "weights" / "model.npz").read_bytes(),
                (SOURCE / "weights" / "model.npz").read_bytes(),
            )

    def test_rejects_invalid_or_existing_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "candidate"
            with self.assertRaises(ValueError):
                materialize(SOURCE, output, 101)
            output.mkdir()
            with self.assertRaises(FileExistsError):
                materialize(SOURCE, output, 25)


if __name__ == "__main__":
    unittest.main()
