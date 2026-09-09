"""Run named search ablations in isolated processes on critical positions."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from tools.backtest_core import fingerprint_agent, git_state
from tools.search_diagnostics import DEFAULT_SUITE, REPOSITORY, load_critical_positions

DEFAULT_ENGINE_ROOT = REPOSITORY / "challengers" / "v4_lab"
DEFAULT_OUTPUT = REPOSITORY / "benchmarks" / "diagnostics" / "v4-search-ablations.json"


@dataclass(frozen=True, slots=True)
class Trial:
    """One compile-time search profile and memory-lifetime policy."""

    name: str
    profile: str
    memory_mode: str = "fresh"


TRIALS = {
    trial.name: trial
    for trial in (
        Trial("baseline", "baseline"),
        Trial("no-lmr", "no-lmr"),
        Trial("no-q-pruning", "no-q-pruning"),
        Trial("check-extension", "check-extension"),
        Trial("persistent", "baseline", "persistent"),
        Trial("hce-only", "hce-only"),
        Trial("nnue-only", "nnue-only"),
        Trial("no-null", "no-null"),
        Trial("no-lmr-no-null", "no-lmr-no-null"),
        Trial("v10-current", "current"),
        Trial("v10-no-dynamic-nmp", "no-dynamic-nmp"),
        Trial("v10-no-rfp", "no-reverse-futility"),
        Trial("v10-no-lmp", "no-late-move-pruning"),
        Trial("v10-no-quiet-futility", "no-quiet-futility"),
        Trial("v10-no-see", "no-see-pruning"),
        Trial("v10-no-contextual-lmr", "no-contextual-lmr"),
    )
}


def _trial_list(value: str) -> list[Trial]:
    names = [name.strip() for name in value.split(",") if name.strip()]
    unknown = [name for name in names if name not in TRIALS]
    if not names or unknown:
        choices = ", ".join(TRIALS)
        detail = f"unknown trials: {', '.join(unknown)}; " if unknown else ""
        raise argparse.ArgumentTypeError(f"{detail}choose from {choices}")
    if len(set(names)) != len(names):
        raise argparse.ArgumentTypeError("trials must not contain duplicates")
    return [TRIALS[name] for name in names]


def _node_list(value: str) -> list[int]:
    try:
        nodes = [int(item.strip()) for item in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("nodes must be comma-separated integers") from error
    if not nodes or any(node <= 0 for node in nodes):
        raise argparse.ArgumentTypeError("nodes must be comma-separated positive integers")
    if nodes != sorted(set(nodes)):
        raise argparse.ArgumentTypeError("nodes must be unique and increasing")
    return nodes


def run_trial(
    trial: Trial,
    *,
    engine_root: Path,
    suite: Path,
    nodes: list[int],
    root_depth: int,
    tt_bits: int,
    pv_plies: int,
    timeout_s: int = 3_600,
) -> dict[str, Any]:
    """Execute one profile in a clean interpreter and return its JSON report."""
    command = [
        sys.executable,
        "-m",
        "tools.search_diagnostics",
        "--engine-root",
        str(engine_root),
        "--profile",
        trial.profile,
        "--memory-mode",
        trial.memory_mode,
        "--suite",
        str(suite),
        "--all-cases",
        "--nodes",
        ",".join(str(node) for node in nodes),
        "--root-depth",
        str(root_depth),
        "--tt-bits",
        str(tt_bits),
        "--pv-plies",
        str(pv_plies),
        "--format",
        "json",
    ]
    completed = subprocess.run(
        command,
        cwd=REPOSITORY,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"trial {trial.name} failed with code {completed.returncode}: {detail}")
    try:
        report = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"trial {trial.name} returned malformed JSON") from error
    if not isinstance(report, dict):
        raise RuntimeError(f"trial {trial.name} returned a non-object report")
    report["trial"] = {
        "name": trial.name,
        "profile": trial.profile,
        "memory_mode": trial.memory_mode,
    }
    return report


def summarize_trial(report: dict[str, Any]) -> list[dict[str, object]]:
    """Extract the highest-node choice and reference rank for each position."""
    rows: list[dict[str, object]] = []
    for position in report["positions"]:
        case = position["case"]
        final_probe = position["node_probes"][-1]
        reference = case["reference_move"]
        root_lines = position["root_lines"]
        reference_rank = next(
            (rank for rank, line in enumerate(root_lines, start=1) if line["move"] == reference),
            None,
        )
        reference_line = next(
            (line for line in root_lines if line["move"] == reference),
            None,
        )
        nodes = int(final_probe["nodes"])
        qnodes = int(final_probe["qnodes"])
        rows.append(
            {
                "case": case["identifier"],
                "move": final_probe["move"],
                "reference": reference,
                "matches_reference": final_probe["move"] == reference,
                "depth": int(final_probe["depth"]),
                "score": int(final_probe["score"]),
                "nodes": nodes,
                "qshare": qnodes / nodes if nodes else 0.0,
                "reference_root_rank": reference_rank,
                "reference_root_score": (
                    reference_line["score"] if reference_line is not None else None
                ),
            }
        )
    return rows


def _print_summary(trial: Trial, report: dict[str, Any]) -> None:
    print(
        f"\n{trial.name}: profile={trial.profile}, memory={trial.memory_mode}, "
        f"warmup={float(report['warmup_s']):.3f}s"
    )
    print("case                         move     ref      hit depth  score    q% ref-rank")
    for row in summarize_trial(report):
        marker = "yes" if row["matches_reference"] else "no"
        rank = row["reference_root_rank"] if row["reference_root_rank"] is not None else "-"
        depth = cast(int, row["depth"])
        score = cast(int, row["score"])
        qshare = cast(float, row["qshare"])
        print(
            f"{row['case']!s:<28} {row['move']!s:<8} {row['reference']!s:<8} "
            f"{marker:<3} {depth:>5} {score:>6} {qshare:>5.1%} {rank!s:>8}"
        )


def _write_report(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine-root", type=Path, default=DEFAULT_ENGINE_ROOT)
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument(
        "--trials",
        type=_trial_list,
        default=_trial_list("baseline,no-lmr,no-q-pruning,check-extension,persistent"),
    )
    parser.add_argument(
        "--nodes", type=_node_list, default=_node_list("25000,100000,300000")
    )
    parser.add_argument("--root-depth", type=int, default=5)
    parser.add_argument("--tt-bits", type=int, default=18)
    parser.add_argument("--pv-plies", type=int, default=12)
    parser.add_argument(
        "--trial-timeout-s",
        type=int,
        default=3_600,
        help="Maximum wall time for each isolated profile process.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if args.root_depth <= 0 or args.pv_plies <= 0 or args.trial_timeout_s <= 0:
        parser.error("--root-depth, --pv-plies, and --trial-timeout-s must be positive")
    if not 10 <= args.tt_bits <= 24:
        parser.error("--tt-bits must be between 10 and 24")
    load_critical_positions(args.suite)

    reports: list[dict[str, Any]] = []
    for trial in args.trials:
        print(f"running {trial.name}...", flush=True)
        report = run_trial(
            trial,
            engine_root=args.engine_root,
            suite=args.suite,
            nodes=args.nodes,
            root_depth=args.root_depth,
            tt_bits=args.tt_bits,
            pv_plies=args.pv_plies,
            timeout_s=args.trial_timeout_s,
        )
        reports.append(report)
        _print_summary(trial, report)

    payload: dict[str, object] = {
        "schema_version": 1,
        "engine": fingerprint_agent(args.engine_root),
        "git": git_state(REPOSITORY),
        "configuration": {
            "suite": str(args.suite.resolve()),
            "nodes": args.nodes,
            "root_depth": args.root_depth,
            "tt_bits": args.tt_bits,
            "pv_plies": args.pv_plies,
            "trial_timeout_s": args.trial_timeout_s,
        },
        "reports": reports,
    }
    _write_report(args.output, payload)
    print(f"\nwrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
