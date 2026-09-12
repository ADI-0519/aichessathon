import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "challengers"


def _constant(candidate: str, name: str) -> str:
    source = (ROOT / candidate / "search.py").read_text(encoding="utf-8")
    match = re.search(rf"^{re.escape(name)}\s*=\s*([^#\r\n]+)", source, re.MULTILINE)
    if match is None:
        raise AssertionError(f"{candidate} does not define {name}")
    return match.group(1).strip()


class FinalsV18MatrixTests(unittest.TestCase):
    def test_core_preserves_v16_selectivity(self) -> None:
        core = "exp_finals_v18_core"
        self.assertEqual(_constant(core, "NNUE_BLEND"), "75")
        self.assertEqual(_constant(core, "LMR_REDUCTION_DIVISOR"), "2.25")
        self.assertEqual(_constant(core, "V10_RFP_MAX_DEPTH"), "4")
        self.assertEqual(_constant(core, "V10_QF_MAX_DEPTH"), "3")
        self.assertEqual(_constant(core, "V10_LMP_MAX_DEPTH"), "3")

    def test_eval50_changes_only_the_intended_blend(self) -> None:
        candidate = "exp_finals_v18_eval50"
        self.assertEqual(_constant(candidate, "NNUE_BLEND"), "50")
        self.assertEqual(_constant(candidate, "LMR_REDUCTION_DIVISOR"), "2.25")
        self.assertEqual(_constant(candidate, "V10_RFP_MAX_DEPTH"), "4")

    def test_searchmax_retains_the_proven_blend(self) -> None:
        candidate = "exp_finals_v18_searchmax"
        self.assertEqual(_constant(candidate, "NNUE_BLEND"), "75")
        self.assertEqual(_constant(candidate, "LMR_REDUCTION_DIVISOR"), "1.75")
        self.assertEqual(_constant(candidate, "V10_RFP_MAX_DEPTH"), "6")
        self.assertEqual(_constant(candidate, "V10_QF_MAX_DEPTH"), "4")
        self.assertEqual(_constant(candidate, "V10_LMP_MAX_DEPTH"), "4")

    def test_max_combines_eval_and_search_lanes(self) -> None:
        candidate = "exp_finals_v18_max"
        self.assertEqual(_constant(candidate, "NNUE_BLEND"), "50")
        self.assertEqual(_constant(candidate, "LMR_REDUCTION_DIVISOR"), "1.75")
        self.assertEqual(_constant(candidate, "V10_RFP_MAX_DEPTH"), "6")
