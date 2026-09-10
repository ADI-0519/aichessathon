from __future__ import annotations

import unittest
from typing import Any, ClassVar

from tools.search_diagnostics import REPOSITORY, load_engine_modules


class SearchV10ProfileTests(unittest.TestCase):
    search: ClassVar[Any]

    @classmethod
    def setUpClass(cls) -> None:
        _, cls.search = load_engine_modules(
            REPOSITORY / "challengers" / "exp_search_v10"
        )

    def test_profiles_cover_every_v10_mechanism(self) -> None:
        expected = {
            "baseline",
            "current",
            "no-dynamic-nmp",
            "no-reverse-futility",
            "no-late-move-pruning",
            "no-quiet-futility",
            "no-see-pruning",
            "no-contextual-lmr",
        }
        self.assertEqual(set(self.search.available_profiles()), expected)

    def test_current_profile_disables_only_v10_mechanisms(self) -> None:
        self.search.configure_experiment("current")
        self.assertFalse(self.search.ENABLE_V10_DYNAMIC_NMP)
        self.assertFalse(self.search.ENABLE_V10_REVERSE_FUTILITY)
        self.assertFalse(self.search.ENABLE_V10_LATE_MOVE_PRUNING)
        self.assertFalse(self.search.ENABLE_V10_QUIET_FUTILITY)
        self.assertFalse(self.search.ENABLE_V10_SEE_PRUNING)
        self.assertFalse(self.search.ENABLE_V10_CONTEXTUAL_LMR)
        self.search.configure_experiment("baseline")

    def test_unknown_profile_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown search profile"):
            self.search.configure_experiment("not-a-profile")


if __name__ == "__main__":
    unittest.main()
