from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import chess

CURRENT = Path(__file__).resolve().parents[1] / "current"


def _module_from_path(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_current_agent() -> ModuleType:
    """Load submission-style sibling imports without polluting other engine tests."""
    aliases = ("engine", "nnue", "search", "time_manager")
    previous = {name: sys.modules.get(name) for name in aliases}
    try:
        engine = _module_from_path("_current_test_engine", CURRENT / "engine.py")
        sys.modules["engine"] = engine
        nnue = _module_from_path("_current_test_nnue", CURRENT / "nnue.py")
        sys.modules["nnue"] = nnue
        search = _module_from_path("_current_test_search", CURRENT / "search.py")
        sys.modules["search"] = search
        time_manager = _module_from_path(
            "_current_test_time_manager", CURRENT / "time_manager.py"
        )
        sys.modules["time_manager"] = time_manager
        return _module_from_path("_current_test_agent", CURRENT / "agent.py")
    finally:
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


try:
    agent = _load_current_agent()
except Exception:
    sys.modules.pop("_current_test_agent", None)
    raise


class AgentTests(unittest.TestCase):
    def setUp(self) -> None:
        agent._game_board = None
        agent._position_history.clear()
        agent._memory.clear()

    def test_import_warmup_fits_initialization_allowance(self) -> None:
        self.assertLess(agent._warmup_elapsed_s, 75.0)

    def test_low_clock_returns_legal_move(self) -> None:
        board = chess.Board()
        move = chess.Move.from_uci(agent.get_move(board.fen(), 1))
        self.assertIn(move, board.legal_moves)

    def test_normal_search_returns_legal_move(self) -> None:
        board = chess.Board()
        move = chess.Move.from_uci(agent.get_move(board.fen(), 1_000))
        self.assertIn(move, board.legal_moves)

    def test_persistent_state_tracks_our_and_opponent_moves(self) -> None:
        board = chess.Board()
        our_move = chess.Move.from_uci(agent.get_move(board.fen(), 150))
        board.push(our_move)
        opponent_move = next(iter(board.legal_moves))
        board.push(opponent_move)

        reply = chess.Move.from_uci(agent.get_move(board.fen(), 150))
        self.assertIn(reply, board.legal_moves)
        self.assertEqual(len(agent._position_history), 4)

    def test_internal_failure_keeps_legal_fallback(self) -> None:
        board = chess.Board()
        with patch.object(agent, "_choose_move", side_effect=RuntimeError("boom")):
            move = chess.Move.from_uci(agent.get_move(board.fen(), 1_000))
        self.assertIn(move, board.legal_moves)

    def test_clock_schedule_always_preserves_a_reserve(self) -> None:
        for remaining in (1, 10, 100, 1_000, 10_000, 60_000, 120_000):
            with self.subTest(remaining=remaining):
                budget = agent._move_budget_ms(remaining, 30)
                self.assertGreaterEqual(budget, 0)
                self.assertLess(budget, remaining)


if __name__ == "__main__":
    unittest.main()
