from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

from tools.search_ablations import _node_list, _trial_list, _write_report, summarize_trial


class SearchAblationTests(unittest.TestCase):
    def test_trial_and_node_lists_are_validated(self) -> None:
        self.assertEqual(
            [trial.name for trial in _trial_list("baseline,check-extension")],
            ["baseline", "check-extension"],
        )
        self.assertEqual(_node_list("100,200,300"), [100, 200, 300])
        with self.assertRaises(argparse.ArgumentTypeError):
            _trial_list("baseline,baseline")
        with self.assertRaises(argparse.ArgumentTypeError):
            _trial_list("unknown")
        with self.assertRaises(argparse.ArgumentTypeError):
            _node_list("200,100")

    def test_summary_uses_highest_node_probe_and_finds_reference_rank(self) -> None:
        report = {
            "positions": [
                {
                    "case": {"identifier": "critical", "reference_move": "a1a2"},
                    "node_probes": [
                        {"move": "e1e2", "depth": 2, "score": 10, "nodes": 100, "qnodes": 50},
                        {
                            "move": "a1a2",
                            "depth": 4,
                            "score": 90,
                            "nodes": 1_000,
                            "qnodes": 700,
                        },
                    ],
                    "root_lines": [
                        {"move": "e1e2", "score": 100},
                        {"move": "a1a2", "score": 90},
                    ],
                }
            ]
        }
        row = summarize_trial(report)[0]
        self.assertEqual(row["move"], "a1a2")
        self.assertTrue(row["matches_reference"])
        self.assertEqual(row["reference_root_rank"], 2)
        self.assertEqual(row["reference_root_score"], 90)
        self.assertEqual(row["qshare"], 0.7)

    def test_report_write_is_complete_json(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "nested" / "report.json"
            _write_report(path, {"schema_version": 1, "reports": []})
            self.assertEqual(json.loads(path.read_text()), {"schema_version": 1, "reports": []})
            self.assertFalse(path.with_suffix(".json.tmp").exists())


if __name__ == "__main__":
    unittest.main()
