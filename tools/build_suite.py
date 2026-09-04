"""Build a deterministic, provenance-tracked EPD benchmark suite.

The generated suite is development data. It is not part of an agent submission.
"""

from __future__ import annotations

import argparse
import hashlib
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import chess

from tools.backtest_core import (
    SuitePosition,
    atomic_write_json,
    atomic_write_text,
    fingerprint_file,
    load_suite_file,
    suite_digest,
)
from tools.cli import positive_int

SCHEMA_VERSION = 1
SELECTION_METHOD = "lowest-sha256-v1"


def selection_key(position: SuitePosition, seed: str) -> tuple[bytes, str]:
    """Rank a position independently of its location in the source file."""
    digest = hashlib.sha256(f"{seed}\0{position.fen}".encode()).digest()
    return digest, position.fen


def select_positions(
    positions: Sequence[SuitePosition], *, count: int, seed: str
) -> list[SuitePosition]:
    """Select a stable hash-ranked subset without replacement."""
    if count <= 0:
        raise ValueError("position count must be positive")
    if count > len(positions):
        raise ValueError(f"requested {count} positions from a source with only {len(positions)}")
    return sorted(positions, key=lambda position: selection_key(position, seed))[:count]


def render_epd(positions: Sequence[SuitePosition], *, source_name: str) -> str:
    """Serialize positions while retaining halfmove and fullmove counters."""
    lines = [
        "# Deterministic engine-test suite; never package with the competition agent.",
        f"# Source: {source_name}",
    ]
    for position in positions:
        board = chess.Board(position.fen)
        lines.append(
            board.epd(
                id=position.identifier,
                hmvc=board.halfmove_clock,
                fmvn=board.fullmove_number,
            )
        )
    return "\n".join(lines) + "\n"


def build_suite(
    source: Path,
    output: Path,
    *,
    count: int,
    selection_seed: str,
    split_seed: str,
    source_url: str | None,
    force: bool,
) -> dict[str, object]:
    """Create an EPD suite and adjacent manifest, returning the manifest."""
    source = source.resolve()
    output = output.resolve()
    manifest_path = output.with_suffix(".manifest.json")
    if output.suffix.lower() != ".epd":
        raise ValueError("output must use the .epd suffix")
    if output == source:
        raise ValueError("output must differ from the source file")
    existing = [path for path in (output, manifest_path) if path.exists()]
    if existing and not force:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(f"refusing to overwrite {names}; pass --force to rebuild")

    positions = load_suite_file(source, split_seed=split_seed)
    selected = select_positions(positions, count=count, seed=selection_seed)
    epd = render_epd(selected, source_name=source.name)
    epd_bytes = epd.encode("utf-8")
    source_fingerprint = fingerprint_file(source)
    split_counts = Counter(position.split for position in selected)
    source_record: dict[str, object] = {
        "name": source.name,
        "sha256": source_fingerprint["sha256"],
        "bytes": source_fingerprint["bytes"],
    }
    if source_url is not None:
        source_record["url"] = source_url
    manifest: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "source": source_record,
        "selection": {
            "method": SELECTION_METHOD,
            "seed": selection_seed,
            "requested_positions": count,
            "unique_source_positions": len(positions),
        },
        "splits": {
            "seed": split_seed,
            "counts": {
                name: split_counts[name] for name in ("development", "validation", "holdout")
            },
        },
        "output": {
            "name": output.name,
            "sha256": hashlib.sha256(epd_bytes).hexdigest(),
            "bytes": len(epd_bytes),
            "suite_digest": suite_digest(selected),
        },
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(output, epd)
    atomic_write_json(manifest_path, manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=positive_int, default=500)
    parser.add_argument("--selection-seed", default="aichessathon-openings-v1")
    parser.add_argument("--split-seed", default="aichessathon-backtest-v2")
    parser.add_argument("--source-url")
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    arguments = parser.parse_args()
    try:
        manifest = build_suite(
            arguments.source,
            arguments.output,
            count=arguments.count,
            selection_seed=arguments.selection_seed,
            split_seed=arguments.split_seed,
            source_url=arguments.source_url,
            force=arguments.force,
        )
    except (OSError, ValueError) as error:
        parser.exit(2, f"suite build error: {error}\n")
    output = manifest["output"]
    splits = manifest["splits"]
    print(f"wrote {arguments.count} positions to {arguments.output}")
    print(f"output: {output}")
    print(f"splits: {splits}")


if __name__ == "__main__":
    main()
