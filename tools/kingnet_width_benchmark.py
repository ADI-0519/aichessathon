"""Measure V11-BIG width costs with equal, neutral-output search semantics.

Each width is materialized as a separate source tree and benchmarked in a fresh
process. Nonzero hidden work is arranged in exactly cancelling unit pairs, which
keeps the search tree identical without letting a compiler erase the evaluator.
Differences therefore measure implementation cost rather than random evaluator
quality. This tool does not estimate Elo and does not train a network.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, cast

for _variable in (
    "MKL_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
):
    os.environ[_variable] = "1"

import torch  # noqa: E402

from harness.rules import MAX_UNZIPPED_BYTES  # noqa: E402
from tools.backtest_core import atomic_write_json  # noqa: E402
from tools.cli import positive_int  # noqa: E402
from tools.train_kingnet_v11 import (  # noqa: E402
    ModelConfig,
    V11BigEvaluator,
    export_model,
    load_architecture_config,
)

REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = REPOSITORY / "challengers" / "exp_kingnet_v11_big"
SOURCE_FILES = ("__init__.py", "agent.py", "engine.py", "nnue.py", "search.py", "time_manager.py")
DEFAULT_FEN = "r3k2r/ppp2ppp/2n1bn2/3qp3/3P4/2P1BN2/PPQ2PPP/R3K2R w KQkq - 2 10"


def positive_widths(text: str) -> tuple[int, ...]:
    try:
        widths = tuple(int(value) for value in text.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("widths must be comma-separated integers") from error
    if not widths or any(width <= 0 or width % 2 for width in widths):
        raise argparse.ArgumentTypeError("widths must be positive even integers")
    if len(set(widths)) != len(widths):
        raise argparse.ArgumentTypeError("widths must be unique")
    return widths


def _neutral_workload_model(model: V11BigEvaluator) -> None:
    """Fill every expensive path while making paired output contributions cancel."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(0xA1C4E55)
    with torch.no_grad():
        model.embedding.weight.uniform_(-0.01, 0.01, generator=generator)
        model.factor.weight.uniform_(-0.01, 0.01, generator=generator)
        model.embedding.weight[-1].zero_()
        model.factor.weight[-1].zero_()
        model.accumulator_bias.fill_(0.5)
        model.hidden_weight.uniform_(-0.05, 0.05, generator=generator)
        model.hidden_bias.uniform_(-0.05, 0.05, generator=generator)
        model.output_relu_weight.zero_()
        model.output_clipped_square_weight.zero_()
        model.output_bias.zero_()
        for unit in range(0, model.config.hidden - 1, 2):
            model.hidden_weight[:, unit + 1].copy_(model.hidden_weight[:, unit])
            model.hidden_bias[:, unit + 1].copy_(model.hidden_bias[:, unit])
            model.output_relu_weight[:, unit] = 0.125
            model.output_relu_weight[:, unit + 1] = -0.125
            model.output_clipped_square_weight[:, unit] = 0.125
            model.output_clipped_square_weight[:, unit + 1] = -0.125


def materialize_width(
    source: Path,
    destination: Path,
    architecture: ModelConfig,
    width: int,
    feature_storage: str,
) -> None:
    """Create one immutable synthetic candidate for a cost-only benchmark."""
    if destination.exists():
        raise FileExistsError(f"benchmark candidate already exists: {destination}")
    missing = [name for name in SOURCE_FILES if not (source / name).is_file()]
    if missing:
        raise ValueError(f"source runtime is incomplete: {', '.join(missing)}")

    destination.mkdir(parents=True)
    for name in SOURCE_FILES:
        shutil.copy2(source / name, destination / name)
    (destination / "weights").mkdir()
    model = V11BigEvaluator(
        ModelConfig(
            accumulator=width,
            hidden=architecture.hidden,
            pairwise_width=width // 2,
            cp_scale=architecture.cp_scale,
            piece_head_map=architecture.piece_head_map,
        )
    )
    _neutral_workload_model(model)
    export_model(
        model,
        destination / "weights" / "model.npz",
        feature_storage=feature_storage,
    )


def _candidate_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def run_benchmark(arguments: argparse.Namespace) -> None:
    source = arguments.source.resolve()
    config_path = arguments.config.resolve()
    output = arguments.output.resolve()
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    architecture, export = load_architecture_config(config_path)
    output.mkdir(parents=True)

    reports: list[dict[str, Any]] = []
    reference_signature: tuple[object, ...] | None = None
    for width in arguments.widths:
        print(f"materializing width {width}...", flush=True)
        candidate = output / "candidates" / f"acc{width}"
        materialize_width(source, candidate, architecture, width, export.feature_storage)
        command = [
            sys.executable,
            "-m",
            "tools.kingnet_width_worker",
            "--engine-root",
            str(candidate),
            "--fen",
            arguments.fen,
            "--nodes",
            str(arguments.nodes),
            "--evaluation-iterations",
            str(arguments.evaluation_iterations),
            "--update-iterations",
            str(arguments.update_iterations),
            "--wall-time-s",
            str(arguments.wall_time_s),
        ]
        started = time.perf_counter()
        completed = subprocess.run(
            command,
            cwd=REPOSITORY,
            check=False,
            capture_output=True,
            text=True,
        )
        process_s = time.perf_counter() - started
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(f"width {width} worker failed: {detail}")
        report = cast(dict[str, Any], json.loads(completed.stdout))
        report["process_s"] = process_s
        report["candidate_bytes"] = _candidate_bytes(candidate)
        report["within_submission_limit"] = report["candidate_bytes"] <= MAX_UNZIPPED_BYTES
        report["within_configured_local_init_ceiling"] = (
            report["total_init_s"] <= arguments.init_budget_s
        )
        fixed = report["fixed_search"]
        signature = (
            fixed["move"],
            fixed["score"],
            fixed["depth"],
            fixed["nodes"],
            fixed["qnodes"],
        )
        if reference_signature is None:
            reference_signature = signature
        elif signature != reference_signature:
            raise RuntimeError(
                f"width {width} changed the fixed-node search tree: {signature} "
                f"!= {reference_signature}"
            )
        reports.append(report)
        atomic_write_json(
            output / "summary.partial.json",
            {
                "schema_version": 1,
                "purpose": "incomplete cost-only neutral-output width comparison",
                "reports": reports,
            },
        )
        timed_depth = report["timed_search"]["completed_depth"]
        print(
            f"width {width}: init={report['total_init_s']:.2f}s, "
            f"eval/s={report['evaluations_per_s']:,.0f}, "
            f"update/s={report['updates_per_s']:,.0f}, "
            f"search={fixed['nps']:,.0f} nps, timed depth={timed_depth}",
            flush=True,
        )

    baseline = reports[0]
    for report in reports:
        report["relative_to_first_width"] = {
            "evaluation_throughput": report["evaluations_per_s"] / baseline["evaluations_per_s"],
            "update_throughput": report["updates_per_s"] / baseline["updates_per_s"],
            "fixed_search_nps": report["fixed_search"]["nps"] / baseline["fixed_search"]["nps"],
        }
    result = {
        "schema_version": 1,
        "purpose": "cost-only neutral-output width comparison; not an Elo estimate",
        "source": source.as_posix(),
        "config": config_path.as_posix(),
        "feature_storage": export.feature_storage,
        "fen": arguments.fen,
        "node_limit": arguments.nodes,
        "wall_time_s": arguments.wall_time_s,
        "configured_local_init_ceiling_s": arguments.init_budget_s,
        "init_measurement": (
            "fresh local import of candidate agent.py; not a prediction of official hardware"
        ),
        "widths": list(arguments.widths),
        "reports": reports,
    }
    destination = output / "summary.json"
    atomic_write_json(destination, result)
    print(f"report={destination}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--widths", type=positive_widths, default=(128, 256, 512, 1024))
    parser.add_argument("--fen", default=DEFAULT_FEN)
    parser.add_argument("--nodes", type=positive_int, default=100_000)
    parser.add_argument("--evaluation-iterations", type=positive_int, default=100_000)
    parser.add_argument("--update-iterations", type=positive_int, default=100_000)
    parser.add_argument("--wall-time-s", type=float, default=1.0)
    parser.add_argument("--init-budget-s", type=float, default=90.0)
    return parser


def main() -> None:
    parser = build_parser()
    arguments = parser.parse_args()
    if arguments.wall_time_s <= 0.0 or arguments.init_budget_s <= 0.0:
        parser.error("time limits must be positive")
    if arguments.config is None or arguments.output is None:
        parser.error("--config and --output are required")
    try:
        run_benchmark(arguments)
    except (FileExistsError, RuntimeError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
