from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from harness.package import DEFAULT_INCLUDES, build

REPOSITORY = Path(__file__).resolve().parents[1]
CURRENT = REPOSITORY / "current"


class PackageTests(unittest.TestCase):
    def test_current_archive_has_platform_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            destination = Path(temporary_directory) / "submission.zip"
            written = build(CURRENT, destination, DEFAULT_INCLUDES)

            with zipfile.ZipFile(destination) as archive:
                names = archive.namelist()

            self.assertEqual(names, written)
            self.assertIn("agent.py", names)
            self.assertIn("weights/model.npz", names)
            self.assertTrue(all(not name.startswith("current/") for name in names))
            self.assertTrue(all("__pycache__" not in name for name in names))

    def test_missing_agent_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            with self.assertRaisesRegex(SystemExit, "agent.py.*does not exist"):
                build(root, root / "submission.zip", DEFAULT_INCLUDES)

    def test_oversize_archive_is_rejected_before_writing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            destination = Path(temporary_directory) / "submission.zip"
            with (
                patch("harness.package.MAX_UNZIPPED_BYTES", 1),
                self.assertRaisesRegex(SystemExit, "over the 0 MB limit"),
            ):
                build(CURRENT, destination, DEFAULT_INCLUDES)
            self.assertFalse(destination.exists())


if __name__ == "__main__":
    unittest.main()
