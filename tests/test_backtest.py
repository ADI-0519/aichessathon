from __future__ import annotations

import argparse
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import chess

from harness.referee import Outcome
from harness.sandbox import Agent
from tools import backtest
from tools.backtest_core import (
    GameRecord,
    acquire_output_lock,
    append_record,
    complete_pair_scores,
    ensure_manifest,
    fingerprint_agent,
    load_epd,
    load_fen,
    load_pgn,
    load_records,
    pentanomial_counts,
    positions_from_fens,
    release_output_lock,
    stable_split,
    suite_digest,
    summarize,
)


class BacktestCoreTests(unittest.TestCase):
    def test_workers_argument_defaults_to_one_and_requires_positive_values(self) -> None:
        parser = backtest.build_parser()
        default = parser.parse_args(["--opponent", "opponent", "--output", "output"])
        parallel = parser.parse_args(
            ["--opponent", "opponent", "--output", "output", "--workers", "4"]
        )
        self.assertEqual(default.workers, 1)
        self.assertEqual(parallel.workers, 4)
        with patch("sys.stderr"), self.assertRaises(SystemExit):
            parser.parse_args(
                ["--opponent", "opponent", "--output", "output", "--workers", "0"]
            )

    def test_pair_worker_runs_candidate_colours_sequentially(self) -> None:
        position = backtest.SuitePosition("position", chess.STARTING_FEN, 7, "development")
        task = backtest.PairTask(3, position, (True, False))
        calls: list[bool] = []

        def fake_play_game(*, candidate_is_white: bool, **_: object) -> backtest.GamePayload:
            calls.append(candidate_is_white)
            color = "white" if candidate_is_white else "black"
            record = make_record(f"00003-{color}", "000007:position", color, "draw")
            return backtest.GamePayload(record, f"{color} pgn", (("candidate", color),))

        with patch("tools.backtest.play_game", side_effect=fake_play_game):
            result = backtest.play_pair(
                task,
                candidate=Path("candidate"),
                opponent_factory=lambda: Agent([], "opponent"),
                candidate_name="candidate",
                opponent_name="opponent",
                configuration={},
                base_ms=1,
                increment_ms=0,
                ply_cap=1,
            )

        self.assertEqual(calls, [True, False])
        self.assertEqual(
            [game.record.game_id for game in result.games],
            ["00003-white", "00003-black"],
        )
        self.assertEqual([game.pgn for game in result.games], ["white pgn", "black pgn"])
        
    def test_failure_attribution_does_not_depend_on_game_result(self) -> None:
    # White flags, but Black has insufficient mating material.
        outcome = Outcome(
            "draw",
            "flag",
            "",
            failed_side="white",
        )

        self.assertEqual(
            backtest.candidate_result(
                outcome,
                candidate_is_white=True,
            ),
            "draw",
        )

        self.assertEqual(
            backtest.failure_roles(
                outcome,
                candidate_is_white=True,
            ),
            (True, False),
        )

        # Same referee outcome, but candidate is Black:
        # now the opponent was the side that failed.
        self.assertEqual(
            backtest.failure_roles(
                outcome,
                candidate_is_white=False,
            ),
            (False, True),
        )
        
    def test_both_failed_attributes_failure_to_both_agents(self) -> None:
        outcome = Outcome(
            "void",
            "both_failed",
            "",
            failed_side="both",
        )

        self.assertEqual(
            backtest.failure_roles(
                outcome,
                candidate_is_white=True,
            ),
            (True, True),
        )

        self.assertEqual(
            backtest.failure_roles(
                outcome,
                candidate_is_white=False,
            ),
            (True, True),
        )

    def test_pair_worker_does_not_start_second_colour_after_stop(self) -> None:
        position = backtest.SuitePosition("position", chess.STARTING_FEN, 7, "development")
        task = backtest.PairTask(1, position, (True, False))
        stop_event = threading.Event()
        calls: list[bool] = []

        def fake_play_game(*, candidate_is_white: bool, **_: object) -> backtest.GamePayload:
            calls.append(candidate_is_white)
            stop_event.set()
            color = "white" if candidate_is_white else "black"
            record = make_record(f"00001-{color}", "p1", color, "draw")
            return backtest.GamePayload(record, "mock pgn", ())

        with patch("tools.backtest.play_game", side_effect=fake_play_game):
            result = backtest.play_pair(
                task,
                candidate=Path("candidate"),
                opponent_factory=lambda: Agent([], "opponent"),
                candidate_name="candidate",
                opponent_name="opponent",
                configuration={},
                base_ms=1,
                increment_ms=0,
                ply_cap=1,
                stop_event=stop_event,
            )

        self.assertEqual(calls, [True])
        self.assertEqual(len(result.games), 1)
        self.assertEqual(result.games[0].record.game_id, "00001-white")

    def test_agent_stderr_is_bounded_and_persisted(self) -> None:
        candidate = Agent([], "candidate")
        opponent = Agent([], "opponent")
        candidate.stderr_log = "prefix" + "x" * backtest.AGENT_LOG_LIMIT
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
        positions = positions_from_fens(
            (("plain", plain), ("phantom", phantom)), split_seed="test"
        )
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
            (root / "agent.py").write_text(
                "def get_move(fen, time): return 'e2e4'\n", encoding="utf-8"
            )
            (root / "ignored.txt").write_text("first", encoding="utf-8")
            weights = root / "weights"
            weights.mkdir()
            (weights / "model.bin").write_bytes(b"weights")
            first = fingerprint_agent(root)
            (root / "ignored.txt").write_text("second", encoding="utf-8")
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

    def test_execution_policy_is_part_of_the_immutable_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate = root / "candidate"
            opponent = root / "opponent"
            output = root / "output"
            candidate.mkdir()
            opponent.mkdir()
            output.mkdir()
            (candidate / "agent.py").write_text("VERSION = 1\n", encoding="utf-8")
            (opponent / "agent.py").write_text("VERSION = 1\n", encoding="utf-8")
            position = backtest.SuitePosition(
                "position", chess.STARTING_FEN, 1, "development"
            )
            common: dict[str, object] = {
                "candidate": candidate,
                "opponent": opponent,
                "stockfish": None,
                "stockfish_nodes": None,
                "suite_source": "test",
                "suite": [position],
                "selected": [position],
                "split": "development",
                "split_seed": "test",
                "order_seed": 1,
                "offset": 0,
                "limit": 1,
                "base_ms": 1_000,
                "increment_ms": 0,
                "ply_cap": 2,
                "sprt": None,
            }
            config_workers_1 = backtest.configuration_for_run(
                **common, workers=1, continue_on_failure=False
            )
            config_workers_6 = backtest.configuration_for_run(
                **common, workers=6, continue_on_failure=False
            )
            config_continue_on_failure = backtest.configuration_for_run(
                **common, workers=1, continue_on_failure=True
            )

            self.assertEqual(
                config_workers_1["execution"],
                {"workers": 1, "continue_on_failure": False},
            )
            self.assertNotEqual(config_workers_1, config_workers_6)
            self.assertNotEqual(config_workers_1, config_continue_on_failure)

            ensure_manifest(output, config_workers_1)
            with self.assertRaisesRegex(ValueError, "different experiment"):
                ensure_manifest(output, config_workers_6)
            with self.assertRaisesRegex(ValueError, "different experiment"):
                ensure_manifest(output, config_continue_on_failure)

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

    def test_summary_uses_only_clean_complete_pairs(self) -> None:
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
        self.assertEqual(complete_pair_scores(records), [2.0, 0.5])
        self.assertEqual(pentanomial_counts(records), (0, 1, 0, 0, 1))

    def test_strength_statistics_exclude_technical_failure_pairs(self) -> None:
        records = [
            make_record(
                "00001-white",
                "p1",
                "white",
                "win",
                termination="init",
                opponent_failure=True,
            ),
            make_record("00001-black", "p1", "black", "draw"),
        ]

        self.assertEqual(complete_pair_scores(records), [])
        self.assertEqual(pentanomial_counts(records), (0, 0, 0, 0, 0))
        self.assertEqual(summarize(records)["complete_pairs"], 0)

        arguments = argparse.Namespace(
            sprt=True,
            sprt_elo0=0.0,
            sprt_elo1=20.0,
            sprt_alpha=0.05,
            sprt_beta=0.05,
            sprt_min_pairs=0,
        )
        verdict = backtest.sprt_verdict(records, arguments)
        self.assertIsNotNone(verdict)
        assert verdict is not None
        self.assertEqual(verdict.pairs, 0)

    def test_paired_interval_uses_pair_variance(self) -> None:
        records = [
            make_record("00001-white", "p1", "white", "win"),
            make_record("00001-black", "p1", "black", "win"),
            make_record("00002-white", "p2", "white", "win"),
            make_record("00002-black", "p2", "black", "win"),
        ]
        interval = summarize(records)["confidence_95"]
        self.assertIsInstance(interval, dict)
        assert isinstance(interval, dict)
        self.assertEqual(interval["score_low"], 1.0)
        self.assertEqual(interval["score_high"], 1.0)

    def test_sprt_snapshot_uses_only_complete_pairs(self) -> None:
        arguments = argparse.Namespace(
            sprt=True,
            sprt_elo0=0.0,
            sprt_elo1=20.0,
            sprt_alpha=0.05,
            sprt_beta=0.05,
            sprt_min_pairs=0,
        )
        incomplete = [make_record("00001-white", "p1", "white", "win")]
        incomplete_verdict = backtest.sprt_verdict(incomplete, arguments)
        self.assertIsNotNone(incomplete_verdict)
        assert incomplete_verdict is not None
        self.assertEqual(incomplete_verdict.pairs, 0)

        complete = [
            *incomplete,
            make_record("00001-black", "p1", "black", "win"),
        ]
        complete_verdict = backtest.sprt_verdict(complete, arguments)
        self.assertIsNotNone(complete_verdict)
        assert complete_verdict is not None
        self.assertEqual(complete_verdict.pairs, 1)

    def test_result_mapping_handles_colors_draws_and_voids(self) -> None:
        white_win = Outcome("white", "checkmate", "")
        self.assertEqual(backtest.candidate_result(white_win, True), "win")
        self.assertEqual(backtest.candidate_result(white_win, False), "loss")
        self.assertEqual(
            backtest.candidate_result(Outcome("draw", "stalemate", ""), True),
            "draw",
        )
        self.assertEqual(
            backtest.candidate_result(Outcome("void", "both_failed", ""), True),
            "void",
        )

    def test_source_guard_detects_a_mixed_build(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate = root / "candidate"
            opponent = root / "opponent"
            candidate.mkdir()
            opponent.mkdir()
            (candidate / "agent.py").write_text("VERSION = 1\n", encoding="utf-8")
            (opponent / "agent.py").write_text("VERSION = 1\n", encoding="utf-8")
            configuration: dict[str, object] = {
                "candidate": fingerprint_agent(candidate),
                "opponent": {
                    "type": "local",
                    "fingerprint": fingerprint_agent(opponent),
                },
            }
            backtest.assert_sources_unchanged(configuration)
            (candidate / "agent.py").write_text("VERSION = 2\n", encoding="utf-8")
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
    def test_parallel_fixed_run_journals_complete_pairs_in_suite_order(self) -> None:
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
                split_seed="parallel-integration-test",
                order_seed=11,
                offset=0,
                limit=2,
                base_ms=1_000,
                increment_ms=0,
                ply_cap=2,
                output=Path(temporary),
                continue_on_failure=False,
                workers=2,
                sprt=False,
            )
            self.assertEqual(backtest.run(arguments), 0)
            records = load_records(Path(temporary) / "games.jsonl")
            self.assertEqual(
                [record.game_id for record in records],
                ["00001-white", "00001-black", "00002-white", "00002-black"],
            )
            self.assertTrue(
                all((Path(temporary) / record.pgn_file).is_file() for record in records)
            )
            summary = json.loads(
                (Path(temporary) / "summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["complete_pairs"], 2)

            arguments.workers = 6
            with self.assertRaisesRegex(ValueError, "different experiment"):
                backtest.run(arguments)

    def test_parallel_out_of_order_completion_is_committed_in_suite_order(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        random_agent = repository / "baselines" / "random"
        pair_1_started = threading.Event()
        pair_2_completed_first = threading.Event()

        def fake_payload(
            task: backtest.PairTask,
            candidate_is_white: bool,
        ) -> backtest.GamePayload:
            color = "white" if candidate_is_white else "black"
            game_id = f"{task.position_index:05d}-{color}"
            record = GameRecord.from_dict(
                {
                    "game_id": game_id,
                    "position_id": task.position.identifier,
                    "position_index": task.position_index,
                    "fen": task.position.fen,
                    "candidate_color": color,
                    "candidate_result": "draw",
                    "board_result": "draw",
                    "termination": "stalemate",
                    "plies": 10,
                    "elapsed_s": 1.0,
                    "pgn_file": f"games/{game_id}.pgn",
                    "candidate_failure": False,
                    "opponent_failure": False,
                }
            )
            return backtest.GamePayload(record=record, pgn=f"mock {game_id}\n", logs=())

        def fake_play_pair(
            task: backtest.PairTask,
            **_: object,
        ) -> backtest.PairResult:
            if task.position_index == 1:
                pair_1_started.set()
                if not pair_2_completed_first.wait(timeout=5):
                    raise AssertionError(
                        "pair 2 did not finish while pair 1 was blocked"
                    )
            elif task.position_index == 2:
                if not pair_1_started.wait(timeout=5):
                    raise AssertionError("pair 1 did not start before pair 2")
                pair_2_completed_first.set()

            return backtest.PairResult(
                tuple(
                    fake_payload(task, candidate_is_white)
                    for candidate_is_white in task.candidate_colours
                )
            )

        with tempfile.TemporaryDirectory() as temporary:
            arguments = argparse.Namespace(
                candidate=random_agent,
                opponent=random_agent,
                stockfish=None,
                stockfish_nodes=None,
                suite="builtin",
                split="development",
                unlock_holdout=False,
                split_seed="out-of-order-test",
                order_seed=17,
                offset=0,
                limit=2,
                base_ms=1_000,
                increment_ms=0,
                ply_cap=2,
                output=Path(temporary),
                continue_on_failure=False,
                workers=2,
                sprt=False,
            )
            with patch("tools.backtest.play_pair", side_effect=fake_play_pair):
                self.assertEqual(backtest.run(arguments), 0)

            records = load_records(Path(temporary) / "games.jsonl")
            self.assertEqual(
                [record.game_id for record in records],
                ["00001-white", "00001-black", "00002-white", "00002-black"],
            )
            self.assertTrue(pair_2_completed_first.is_set())

    def test_resume_repairs_half_completed_pair_before_later_pairs(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        random_agent = repository / "baselines" / "random"

        def fake_payload(
            task: backtest.PairTask,
            candidate_is_white: bool,
        ) -> backtest.GamePayload:
            color = "white" if candidate_is_white else "black"
            game_id = f"{task.position_index:05d}-{color}"
            record = GameRecord.from_dict(
                {
                    "game_id": game_id,
                    "position_id": task.position.identifier,
                    "position_index": task.position_index,
                    "fen": task.position.fen,
                    "candidate_color": color,
                    "candidate_result": "draw",
                    "board_result": "draw",
                    "termination": "stalemate",
                    "plies": 10,
                    "elapsed_s": 1.0,
                    "pgn_file": f"games/{game_id}.pgn",
                    "candidate_failure": False,
                    "opponent_failure": False,
                }
            )
            return backtest.GamePayload(record=record, pgn=f"mock {game_id}\n", logs=())

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            arguments = argparse.Namespace(
                candidate=random_agent,
                opponent=random_agent,
                stockfish=None,
                stockfish_nodes=None,
                suite="builtin",
                split="development",
                unlock_holdout=False,
                split_seed="half-pair-resume-test",
                order_seed=23,
                offset=0,
                limit=2,
                base_ms=1_000,
                increment_ms=0,
                ply_cap=2,
                output=output,
                continue_on_failure=False,
                workers=1,
                sprt=False,
            )

            def first_play_pair(
                task: backtest.PairTask,
                **_: object,
            ) -> backtest.PairResult:
                if task.position_index == 1:
                    return backtest.PairResult((fake_payload(task, True),))
                return backtest.PairResult(())

            with patch("tools.backtest.play_pair", side_effect=first_play_pair):
                self.assertEqual(backtest.run(arguments), 0)

            first_records = load_records(output / "games.jsonl")
            self.assertEqual(
                [record.game_id for record in first_records],
                ["00001-white"],
            )

            resumed_tasks: list[tuple[int, tuple[bool, ...]]] = []

            def resumed_play_pair(
                task: backtest.PairTask,
                **_: object,
            ) -> backtest.PairResult:
                resumed_tasks.append((task.position_index, task.candidate_colours))
                return backtest.PairResult(
                    tuple(
                        fake_payload(task, candidate_is_white)
                        for candidate_is_white in task.candidate_colours
                    )
                )

            with patch("tools.backtest.play_pair", side_effect=resumed_play_pair):
                self.assertEqual(backtest.run(arguments), 0)

            records = load_records(output / "games.jsonl")
            self.assertEqual(
                [record.game_id for record in records],
                ["00001-white", "00001-black", "00002-white", "00002-black"],
            )
            self.assertEqual(resumed_tasks[0], (1, (False,)))
            self.assertEqual(resumed_tasks[1], (2, (True, False)))

            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["complete_pairs"], 2)

    def test_resume_does_not_continue_past_recorded_technical_failure(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        random_agent = repository / "baselines" / "random"
        calls = 0

        def failed_pair(
            task: backtest.PairTask,
            **_: object,
        ) -> backtest.PairResult:
            nonlocal calls
            calls += 1
            game_id = f"{task.position_index:05d}-white"
            record = make_record(
                game_id,
                task.position.identifier,
                "white",
                "win",
                termination="init",
                opponent_failure=True,
            )
            return backtest.PairResult(
                (backtest.GamePayload(record, "mock pgn\n", ()),)
            )

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            arguments = argparse.Namespace(
                candidate=random_agent,
                opponent=random_agent,
                stockfish=None,
                stockfish_nodes=None,
                suite="builtin",
                split="development",
                unlock_holdout=False,
                split_seed="technical-failure-resume-test",
                order_seed=31,
                offset=0,
                limit=2,
                base_ms=1_000,
                increment_ms=0,
                ply_cap=2,
                output=output,
                continue_on_failure=False,
                workers=1,
                sprt=False,
            )

            with patch("tools.backtest.play_pair", side_effect=failed_pair):
                self.assertEqual(backtest.run(arguments), 2)

            first_call_count = calls
            records = load_records(output / "games.jsonl")
            self.assertEqual(len(records), 1)
            self.assertTrue(records[0].opponent_failure)

            with patch(
                "tools.backtest.play_pair",
                side_effect=AssertionError("resume must not start another pair"),
            ):
                self.assertEqual(backtest.run(arguments), 2)

            self.assertEqual(calls, first_call_count)
            resumed_records = load_records(output / "games.jsonl")
            self.assertEqual(resumed_records, records)

    def test_sprt_stops_after_a_pair_and_resume_replays_nothing(self) -> None:
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
                limit=2,
                base_ms=1_000,
                increment_ms=0,
                ply_cap=4,
                output=Path(temporary),
                continue_on_failure=False,
                workers=2,
                sprt=True,
                sprt_elo0=0.0,
                sprt_elo1=1000.0,
                sprt_alpha=0.05,
                sprt_beta=0.05,
                sprt_min_pairs=0,
            )
            self.assertEqual(backtest.run(arguments), 0)
            first = load_records(Path(temporary) / "games.jsonl")
            self.assertEqual(len(first), 2)
            self.assertEqual(
                [record.game_id for record in first],
                ["00001-white", "00001-black"],
            )
            self.assertEqual(backtest.run(arguments), 0)
            second = load_records(Path(temporary) / "games.jsonl")
            self.assertEqual(second, first)
            summary = json.loads(
                (Path(temporary) / "summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["games"], 2)
            self.assertEqual(summary["complete_pairs"], 1)
            self.assertEqual(summary["sprt"]["decision"], "accept_h0")


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
