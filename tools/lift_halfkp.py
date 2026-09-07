"""Create an untrained king-conditioned runtime model equivalent to V5."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from tools.backtest_core import atomic_write_text
from tools.cli import positive_int
from tools.train_halfkp import (
    KingConditionedEvaluator,
    ModelConfig,
    export_model,
    initialise_from_v5,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def lift(
    source: Path,
    output: Path,
    manifest: Path,
    *,
    accumulator: int,
    hidden: int,
) -> None:
    """Expand V5 weights into every king bucket without changing its graph."""
    torch.manual_seed(20260906)
    model = KingConditionedEvaluator(ModelConfig(accumulator, hidden)).eval()
    initialise_from_v5(model, source)
    export_model(model, output)
    metadata = {
        "schema_version": 1,
        "architecture": "friendly_king_conditioned_12_piece_accumulator",
        "trained": False,
        "purpose": "runtime parity scaffold; do not promote as a learned V6 model",
        "source": source.as_posix(),
        "source_sha256": _sha256(source),
        "accumulator": accumulator,
        "hidden": hidden,
        "output": output.as_posix(),
        "output_sha256": _sha256(output),
        "output_bytes": output.stat().st_size,
    }
    atomic_write_text(manifest, json.dumps(metadata, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("challengers/v5_nnue/weights/model.npz"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--accumulator", type=positive_int, default=256)
    parser.add_argument("--hidden", type=positive_int, default=32)
    args = parser.parse_args()
    if not args.source.is_file():
        parser.error(f"source model not found: {args.source}")
    if args.output.exists() or args.manifest.exists():
        parser.error("output and manifest paths must not already exist")
    lift(
        args.source,
        args.output,
        args.manifest,
        accumulator=args.accumulator,
        hidden=args.hidden,
    )
    print(f"exported lifted model to {args.output}")
    print(f"wrote {args.manifest}")


if __name__ == "__main__":
    main()
