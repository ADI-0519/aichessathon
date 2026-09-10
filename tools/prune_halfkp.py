"""Remove provably inactive accumulator channels from a HalfKP export."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from tools.backtest_core import atomic_write_text
from tools.cli import positive_int
from tools.train_halfkp import FORMAT_VERSION


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _array(archive: Any, name: str) -> NDArray[Any]:
    if name not in archive:
        raise ValueError(f"model is missing {name}")
    # Preserve zero-dimensional metadata arrays. ``ascontiguousarray`` turns
    # a scalar into shape ``(1,)``, which violates the runtime archive format.
    return np.asarray(archive[name])


def prune_model(source: Path, output: Path, accumulator: int) -> dict[str, object]:
    """Keep the first ``accumulator`` channels when all removed outputs are zero.

    A channel can contain arbitrary feature values and still be inactive if its
    outgoing weights are exactly zero in both perspective halves.  Refusing any
    non-zero removed connection makes this transformation graph-preserving rather
    than an approximate magnitude-pruning operation.
    """
    with np.load(source, allow_pickle=False) as archive:
        version = int(archive["format_version"])
        if version != FORMAT_VERSION:
            raise ValueError(f"unsupported HalfKP model format: {version}")
        cp_scale = _array(archive, "cp_scale")
        input_scale = _array(archive, "input_scale")
        weight_scale = _array(archive, "weight_scale")
        features = _array(archive, "feature_weights_q")
        accumulator_bias = _array(archive, "accumulator_bias_q")
        hidden_weights = _array(archive, "hidden_weights_q")
        hidden_bias = _array(archive, "hidden_bias_q")
        output_weights = _array(archive, "output_weights_q")
        output_bias = _array(archive, "output_bias_q")

    if features.ndim != 2:
        raise ValueError("feature_weights_q must be two-dimensional")
    source_accumulator = features.shape[1]
    if not 0 < accumulator < source_accumulator:
        raise ValueError(
            f"target accumulator must be between 1 and {source_accumulator - 1}"
        )
    if accumulator_bias.shape != (source_accumulator,):
        raise ValueError("accumulator_bias_q has an incompatible shape")
    if hidden_weights.ndim != 2 or hidden_weights.shape[1] != 2 * source_accumulator:
        raise ValueError("hidden_weights_q has an incompatible shape")

    removed_own = hidden_weights[:, accumulator:source_accumulator]
    removed_opponent = hidden_weights[:, source_accumulator + accumulator :]
    removed_connections = int(np.count_nonzero(removed_own)) + int(
        np.count_nonzero(removed_opponent)
    )
    if removed_connections:
        raise ValueError(
            f"refusing to remove {removed_connections} non-zero outgoing connections"
        )

    retained_hidden = np.concatenate(
        (
            hidden_weights[:, :accumulator],
            hidden_weights[:, source_accumulator : source_accumulator + accumulator],
        ),
        axis=1,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        format_version=np.asarray(FORMAT_VERSION, dtype=np.int32),
        cp_scale=cp_scale,
        input_scale=input_scale,
        weight_scale=weight_scale,
        feature_weights_q=np.ascontiguousarray(features[:, :accumulator]),
        accumulator_bias_q=np.ascontiguousarray(accumulator_bias[:accumulator]),
        hidden_weights_q=np.ascontiguousarray(retained_hidden),
        hidden_bias_q=hidden_bias,
        output_weights_q=output_weights,
        output_bias_q=output_bias,
    )
    return {
        "source_accumulator": source_accumulator,
        "target_accumulator": accumulator,
        "removed_connections": removed_connections,
        "exact_graph_preserving_transform": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--accumulator", type=positive_int, default=128)
    args = parser.parse_args()
    if not args.source.is_file():
        parser.error(f"source model not found: {args.source}")
    if args.output.exists() or args.manifest.exists():
        parser.error("output and manifest paths must not already exist")
    try:
        details = prune_model(args.source, args.output, args.accumulator)
    except ValueError as error:
        parser.error(str(error))

    metadata = {
        "schema_version": 1,
        "source": args.source.as_posix(),
        "source_sha256": _sha256(args.source),
        **details,
        "output": args.output.as_posix(),
        "output_sha256": _sha256(args.output),
        "output_bytes": args.output.stat().st_size,
    }
    atomic_write_text(args.manifest, json.dumps(metadata, indent=2) + "\n")
    print(f"exported exact pruned model to {args.output}")
    print(f"wrote {args.manifest}")


if __name__ == "__main__":
    main()
