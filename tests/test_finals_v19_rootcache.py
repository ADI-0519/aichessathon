from __future__ import annotations

import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT / "challengers" / "exp_finals_v19_rootcache_init30"


def _constants(path: Path) -> dict[str, object]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    values: dict[str, object] = {}
    for statement in tree.body:
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
            continue
        target = statement.targets[0]
        if isinstance(target, ast.Name):
            try:
                values[target.id] = ast.literal_eval(statement.value)
            except (TypeError, ValueError):
                continue
    return values


class FinalsV19RootCacheTests(unittest.TestCase):
    def test_candidate_keeps_proven_searchmax_profile(self) -> None:
        constants = _constants(CANDIDATE / "search.py")
        self.assertEqual(constants["NNUE_BLEND"], 75)
        self.assertEqual(constants["LMR_REDUCTION_DIVISOR"], 1.75)
        self.assertEqual(constants["V10_RFP_MAX_DEPTH"], 6)
        self.assertEqual(constants["V10_QF_MAX_DEPTH"], 4)
        self.assertEqual(constants["V10_LMP_MAX_DEPTH"], 4)

    def test_root_and_cache_experiment_is_explicit(self) -> None:
        constants = _constants(CANDIDATE / "search.py")
        self.assertEqual(constants["Q_EVAL_BITS"], 18)
        self.assertEqual(constants["ASPIRATION_INITIAL_WINDOW"], 18)
        self.assertEqual(constants["ASPIRATION_GROWTH_NUMERATOR"], 3)
        self.assertEqual(constants["ASPIRATION_GROWTH_DENOMINATOR"], 2)
        self.assertEqual(constants["UNSTABLE_SCORE_DELTA"], 50)

    def test_thirty_second_adapter_is_retained(self) -> None:
        constants = _constants(CANDIDATE / "agent.py")
        self.assertEqual(constants["INIT_READY_TARGET_S"], 20.0)


if __name__ == "__main__":
    unittest.main()
