from __future__ import annotations

import argparse
import unittest

from tools.numba_search_scaling import Probe, assert_deterministic, positive_int_list


def probe(*, move: str = "e2e4", score: int = 12) -> Probe:
    return Probe(
        node_limit=10_000,
        repeat=1,
        move=move,
        score=score,
        depth=5,
        nodes=10_001,
        qnodes=7_500,
        elapsed_s=0.1,
        nps=100_010.0,
        tt_hits=100,
        tt_cutoffs=50,
        lmr_reductions=20,
        lmr_researches=2,
    )


class NumbaSearchScalingTests(unittest.TestCase):
    def test_positive_node_list(self) -> None:
        self.assertEqual(positive_int_list("100,2000"), [100, 2000])
        for invalid in ("", "0", "100,-2", "bad"):
            with self.subTest(invalid=invalid), self.assertRaises(
                argparse.ArgumentTypeError
            ):
                positive_int_list(invalid)

    def test_repeated_chess_results_must_be_deterministic(self) -> None:
        assert_deterministic([probe(), probe()])
        with self.assertRaisesRegex(RuntimeError, "not deterministic"):
            assert_deterministic([probe(), probe(move="d2d4")])
        with self.assertRaisesRegex(RuntimeError, "not deterministic"):
            assert_deterministic([probe(), probe(score=13)])


if __name__ == "__main__":
    unittest.main()
