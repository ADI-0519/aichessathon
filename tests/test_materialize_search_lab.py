from __future__ import annotations

import unittest

from tools.materialize_search_lab import instrument_search


class MaterializeSearchLabTests(unittest.TestCase):
    def test_instrumentation_adds_each_guard_once(self) -> None:
        source = """NNUE_BLEND = 50

def search():
    if (
        allow_null
        and depth >= 3
    ):
        pass
        reduced = (
            depth >= 3
        )

def warmup() -> None:
    pass
"""
        result = instrument_search(source)
        self.assertEqual(result.count("ENABLE_NULL_MOVE\n        and allow_null"), 1)
        self.assertEqual(result.count("ENABLE_LMR\n            and depth"), 1)
        self.assertEqual(result.count("def configure_experiment"), 1)
        self.assertIn('"hce-only": (0, True, True)', result)
        self.assertIn('"nnue-only": (100, True, True)', result)

    def test_instrumentation_rejects_changed_source_shape(self) -> None:
        with self.assertRaisesRegex(ValueError, "NNUE blend"):
            instrument_search("def warmup() -> None:\n    pass\n")


if __name__ == "__main__":
    unittest.main()
