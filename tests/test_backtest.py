from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

import chess

from harness.referee import Outcome
from harness.sandbox import Agent
from tools import backtest
from tools.backtest_core import (
    GameRecord,
    acquire_output_lock,
    append_record,
    ensure_manifest,
    fingerprint_agent,
    load_epd,
    load_fen,
    load_pgn,
    load_records,
    positions_from_fens,
    release_output_lock,
    stable_split,
    suite_digest,
    summarize,
)


class BacktestCoreTests(unittest.TestCase):
    def test_agent_stderr_is_bounded_and_persisted(self) -> None:
        candidate = Agent([])
        opponent = Agent([])
        candidate.stderr_tail = "prefix" + "x" * backtest.AGENT_LOG_LIMIT
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            backtest.save_agent_logs(output, "00001-white", candidate, opponent)
            saved = (
                output / "logs" / "game-00001-white-candidate.stderr.log"
            ).read_text(encoding="utf-8")
            self.assertEqual(saved, "x" * backtest.AGENT_LOG_LIMIT)
            self.assertFalse(
                (output / "logs" / "game-00001-white-opponent.stderr.log").exists()
            )

    def test_normalization_deduplicates_phantom_en_passant(self) -> None:
        plain = "8/8/8/8/8/8/4K3/7k w - - 0 1"
        phantom = "8/8/8/8/8/8/4K3/7k w - e3 0 1"
        positions = positions_from_fens((("plain", plain), ("phantom", phantom)), split_seed="test")
        self.assertEqual(len(positions), 1)
        self.assertEqual(positions[0].fen, plain)

    def test_split_and_digest_are_deterministic_and_order_sensitive(self) -> None:
        fens = (
            ("start", chess.STARTING_FEN),
            ("endgame", "8/5pk1/6p1/3p4/3P4/5KP1/5P2/8 w - - 0 1"),
        )
        first = positions_from_fens(fens, split_seed="stable")
        second = positions_from_fens(reversed(fens), split_seed="stable")
        split_by_fen = {position.fen: position.split for position in first}
        self.assertEqual(
            split_by_fen,
            {position.fen: position.split for position in second},
        )
        self.assertNotEqual(suite_digest(first), suite_digest(second))
        self.assertEqual(stable_split(first[0].fen, "stable"), first[0].split)

    def test_loads_epd_and_pgn_opening_positions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            epd = root / "suite.epd"
            epd.write_text(
                "# comment\n" + chess.Board().epd(id="initial") + "\n",
                encoding="utf-8",
            )
            epd_positions = load_epd(epd, split_seed="test")
            self.assertEqual(epd_positions[0].identifier, "initial")
            self.assertEqual(epd_positions[0].fen, chess.STARTING_FEN)

            pgn = root / "openings.pgn"
            pgn.write_text(
                '[Event "Line one"]\n[Result "*"]\n\n1. e4 e5 2. Nf3 *\n\n'
                '[Event "Line two"]\n[Result "*"]\n\n1. d4 d5 2. c4 *\n',
                encoding="utf-8",
            )
            pgn_positions = load_pgn(pgn, split_seed="test")
            self.assertEqual(len(pgn_positions), 2)
            first_board = chess.Board()
            for move in ("e2e4", "e7e5", "g1f3"):
                first_board.push_uci(move)
            self.assertEqual(pgn_positions[0].fen, first_board.fen())

    def test_loads_complete_fen_lines_without_losing_clock_state(self) -> None:
        fen = "8/5pk1/6p1/3p4/3P4/5KP1/5P2/8 w - - 17 42"
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "suite.fen"
            path.write_text(f"# comment\n{fen}\n", encoding="utf-8")
            positions = load_fen(path, split_seed="test")
        self.assertEqual(positions[0].fen, fen)

    def test_agent_fingerprint_matches_packaged_inputs_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "agent.py").write_text("def get_move(fen, time): return 'e2e4'\n")
            (root / "ignored.txt").write_text("first")
            weights = root / "weights"
            weights.mkdir()
            (weights / "model.bin").write_bytes(b"weights")
            first = fingerprint_agent(root)
            (root / "ignored.txt").write_text("second")
            self.assertEqual(first["sha256"], fingerprint_agent(root)["sha256"])
            (weights / "model.bin").write_bytes(b"changed")
            self.assertNotEqual(first["sha256"], fingerprint_agent(root)["sha256"])

    def test_manifest_resume_requires_identical_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            created = ensure_manifest(output, {"candidate": "one"})
            resumed = ensure_manifest(output, {"candidate": "one"})
            self.assertEqual(created, resumed)
            with self.assertRaisesRegex(ValueError, "different experiment"):
                ensure_manifest(output, {"candidate": "two"})

    def test_output_lock_rejects_a_second_writer_and_can_be_reacquired(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            first = acquire_output_lock(output)
            with self.assertRaisesRegex(RuntimeError, "another backtest owns"):
                acquire_output_lock(output)
            release_output_lock(first)
            second = acquire_output_lock(output)
            release_output_lock(second)
            self.assertFalse((output / ".run.lock").exists())

    def test_run_claims_output_before_creating_a_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            existing_lock = acquire_output_lock(output)
            try:
                with self.assertRaisesRegex(RuntimeError, "another backtest owns"):
                    backtest.run(argparse.Namespace(output=output))
                self.assertFalse((output / "manifest.json").exists())
            finally:
                release_output_lock(existing_lock)

    def test_journal_round_trip_and_duplicate_rejection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            journal = Path(temporary) / "games.jsonl"
            record = make_record("00001-white", "p1", "white", "win")
            append_record(journal, record)
            self.assertEqual(load_records(journal), [record])
            append_record(journal, record)
            with self.assertRaisesRegex(ValueError, "duplicate game id"):
                load_records(journal)

    def test_summary_uses_complete_pairs_and_excludes_voids(self) -> None:
        records = [
            make_record("00001-white", "p1", "white", "win"),
            make_record("00001-black", "p1", "black", "win"),
            make_record("00002-white", "p2", "white", "loss"),
            make_record("00002-black", "p2", "black", "draw"),
            make_record(
                "00003-white",
                "p3",
                "white",
                "void",
                termination="both_failed",
                candidate_failure=True,
                opponent_failure=True,
            ),
        ]
        summary = summarize(records)
        self.assertEqual(summary["scored_games"], 4)
        self.assertEqual(summary["score"], 0.625)
        self.assertEqual(summary["complete_pairs"], 2)
        self.assertEqual(
            summary["pentanomial"],
            {"0.0": 0, "0.5": 1, "1.0": 0, "1.5": 0, "2.0": 1},
        )
        self.assertEqual(summary["candidate_failures"], 1)
        self.assertEqual(summary["opponent_failures"], 1)

    def test_perfect_small_sample_keeps_an_uncertain_interval(self) -> None:
        records = [
            make_record("00001-white", "p1", "white", "win"),
            make_record("00001-black", "p1", "black", "win"),
            make_record("00002-white", "p2", "white", "win"),
            make_record("00002-black", "p2", "black", "win"),
        ]
        interval = summarize(records)["confidence_95"]
        self.assertIsInstance(interval, dict)
        assert isinstance(interval, dict)
        self.assertLess(interval["score_low"], 1.0)
        self.assertEqual(interval["score_high"], 1.0)

    def test_result_mapping_handles_colors_draws_and_voids(self) -> None:
        white_win = Outcome("white", "checkmate", "")
        self.assertEqual(backtest.candidate_result(white_win, True), "win")
        self.assertEqual(backtest.candidate_result(white_win, False), "loss")
        self.assertEqual(backtest.candidate_result(Outcome("draw", "stalemate", ""), True), "draw")
        self.assertEqual(
            backtest.candidate_result(Outcome("void", "both_failed", ""), True), "void"
        )

    def test_source_guard_detects_a_mixed_build(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate = root / "candidate"
            opponent = root / "opponent"
            candidate.mkdir()
            opponent.mkdir()
            (candidate / "agent.py").write_text("VERSION = 1\n")
            (opponent / "agent.py").write_text("VERSION = 1\n")
            configuration: dict[str, object] = {
                "candidate": fingerprint_agent(candidate),
                "opponent": {
                    "type": "local",
                    "fingerprint": fingerprint_agent(opponent),
                },
            }
            backtest.assert_sources_unchanged(configuration)
            (candidate / "agent.py").write_text("VERSION = 2\n")
            with self.assertRaisesRegex(RuntimeError, "candidate files changed"):
                backtest.assert_sources_unchanged(configuration)

    def test_holdout_requires_explicit_unlock(self) -> None:
        arguments = argparse.Namespace(
            stockfish=None,
            stockfish_nodes=None,
            split="holdout",
            unlock_holdout=False,
        )
        with self.assertRaisesRegex(ValueError, "unlock-holdout"):
            backtest.validate_arguments(arguments)


class BacktestIntegrationTests(unittest.TestCase):
    def test_tiny_run_resumes_without_replaying_games(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        random_agent = repository / "baselines" / "random"
        with tempfile.TemporaryDirectory() as temporary:
            arguments = argparse.Namespace(
                candidate=random_agent,
                opponent=random_agent,
                stockfish=None,
                stockfish_nodes=None,
                suite="builtin",
                split="development",
                unlock_holdout=False,
                split_seed="integration-test",
                order_seed=7,
                offset=0,
                limit=1,
                base_ms=1_000,
                increment_ms=0,
                ply_cap=4,
                output=Path(temporary),
                continue_on_failure=False,
            )
            self.assertEqual(backtest.run(arguments), 0)
            first = load_records(Path(temporary) / "games.jsonl")
            self.assertEqual(len(first), 2)
            self.assertEqual(backtest.run(arguments), 0)
            second = load_records(Path(temporary) / "games.jsonl")
            self.assertEqual(second, first)
            summary = json.loads((Path(temporary) / "summary.json").read_text())
            self.assertEqual(summary["games"], 2)
            self.assertEqual(summary["complete_pairs"], 1)


def make_record(
    game_id: str,
    position_id: str,
    color: str,
    result: str,
    *,
    termination: str = "checkmate",
    candidate_failure: bool = False,
    opponent_failure: bool = False,
) -> GameRecord:
    if color not in {"white", "black"}:
        raise ValueError(color)
    if result not in {"win", "draw", "loss", "void"}:
        raise ValueError(result)
    return GameRecord.from_dict(
        {
            "game_id": game_id,
            "position_id": position_id,
            "position_index": 1,
            "fen": chess.STARTING_FEN,
            "candidate_color": color,
            "candidate_result": result,
            "board_result": "draw" if result == "draw" else "white",
            "termination": termination,
            "plies": 10,
            "elapsed_s": 1.0,
            "pgn_file": "games/example.pgn",
            "candidate_failure": candidate_failure,
            "opponent_failure": opponent_failure,
        }
    )


if __name__ == "__main__":
    unittest.main()
