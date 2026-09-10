from __future__ import annotations

import argparse
import unittest

from tools.numba_search_scaling import Probe, assert_deterministic, positive_int_list


def probe(*, move: str = "e2e4", score: int = 12) -> Probe:
    return Probe(
        position_id="test",
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
        beta_cutoffs=500,
        lmr_reductions=20,
        lmr_researches=2,
        q_eval_probes=0,
        q_eval_hits=0,
        q_moves_considered=0,
        q_accumulator_updates=0,
        q_pruned_after_update=0,
        q_check_saves=0,
        q_children_searched=0,
        qtt_probes=0,
        qtt_hits=0,
        qtt_cutoffs=0,
        qtt_stores=0,
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
