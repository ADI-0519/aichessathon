from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tools.probe_pgn_positions import load_agent


class ProbePgnPositionTests(unittest.TestCase):
    def test_load_agent_resolves_relative_imports_inside_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "helper.py").write_text("MOVE = 'a2a3'\n", encoding="utf-8")
            (root / "agent.py").write_text(
                "from .helper import MOVE\n"
                "def get_move(fen: str, time_left_ms: int) -> str:\n"
                "    return MOVE\n",
                encoding="utf-8",
            )

            loaded = load_agent(root, 0)

        self.assertEqual(loaded.get_move("unused", 1), "a2a3")


if __name__ == "__main__":
    unittest.main()
