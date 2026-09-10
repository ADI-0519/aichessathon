from __future__ import annotations

import unittest
from pathlib import Path
from typing import Any, ClassVar

from tools.search_diagnostics import load_engine_modules

REPOSITORY = Path(__file__).resolve().parents[1]
CHALLENGER = REPOSITORY / "challengers" / "exp_release_v15_nuclear_phase"


class NuclearPhaseTests(unittest.TestCase):
    search: ClassVar[Any]

    @classmethod
    def setUpClass(cls) -> None:
        _, cls.search = load_engine_modules(CHALLENGER)

    def test_rfp_policy_is_bounded_and_monotonic(self) -> None:
        parameters = self.search._phase_rfp_parameters.py_func
        values = [parameters(piece_count) for piece_count in range(2, 33)]

        self.assertEqual(
            values[10],
            (
                self.search.V10_RFP_MAX_DEPTH,
                self.search.V10_RFP_MARGIN_BASE,
                self.search.V10_RFP_MARGIN_PER_DEPTH,
            ),
        )
        self.assertEqual(
            values[26],
            (
                self.search.RFP_RICH_MAX_DEPTH,
                self.search.RFP_RICH_MARGIN_BASE,
                self.search.RFP_RICH_MARGIN_PER_DEPTH,
            ),
        )
        self.assertEqual(values[26:], [values[26]] * len(values[26:]))

        depths = [value[0] for value in values]
        bases = [value[1] for value in values]
        per_depth = [value[2] for value in values]
        self.assertEqual(depths, sorted(depths))
        self.assertEqual(bases, sorted(bases, reverse=True))
        self.assertEqual(per_depth, sorted(per_depth, reverse=True))


if __name__ == "__main__":
    unittest.main()
