from __future__ import annotations

import unittest

from harness.referee import play_match
from harness.sandbox import Agent, AgentFailure


class InstantAgent(Agent):
    def __init__(self, name: str) -> None:
        super().__init__([], name)

    def start(self, init_budget_s: float) -> None:
        del init_budget_s

    def suspend(self) -> None:
        pass

    def resume(self) -> None:
        pass

    def move(
        self,
        fen: str,
        time_left_ms: int,
    ) -> str:
        del fen, time_left_ms
        return "a1a2"

    def stop(self) -> None:
        pass


class FlaggingAgent(InstantAgent):
    def move(self, fen: str, time_left_ms: int) -> str:
        del fen, time_left_ms
        raise AgentFailure("flag")


class RefereeTests(unittest.TestCase):
    def test_flag_draw_still_records_failed_side(self) -> None:
        white = InstantAgent("white")
        black = InstantAgent("black")

        # White has mating material; Black has king only.
        #
        # If White flags, the result is a draw because Black
        # cannot possibly checkmate.
        fen = "7k/8/8/8/8/8/8/R6K w - - 0 1"

        outcome = play_match(
            white,
            black,
            base_ms=-1,
            increment_ms=0,
            start_fen=fen,
        )

        self.assertEqual(outcome.result, "draw")
        self.assertEqual(outcome.termination, "flag")
        self.assertEqual(outcome.failed_side, "white")
        self.assertIn('[FailureSide "white"]', outcome.pgn)

    def test_watchdog_flag_draw_still_records_failed_side(self) -> None:
        white = FlaggingAgent("white")
        black = InstantAgent("black")

        # Black has only a king and therefore cannot possibly mate.
        fen = "7k/8/8/8/8/8/8/R6K w - - 0 1"

        outcome = play_match(
            white,
            black,
            base_ms=1_000,
            increment_ms=0,
            start_fen=fen,
        )

        self.assertEqual(outcome.result, "draw")
        self.assertEqual(outcome.termination, "flag")
        self.assertEqual(outcome.failed_side, "white")


if __name__ == "__main__":
    unittest.main()