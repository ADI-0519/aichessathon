"""Create an immutable local V5 candidate with one explicit evaluation blend.

Optionally swaps in a freshly trained model.  The accumulator width is read
from the model itself rather than passed in, so a candidate can never be built
with weights the inference code would misread.
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

import numpy as np

SOURCE_FILES = ("agent.py", "engine.py", "nnue.py", "search.py")
ACCUMULATOR_PATTERN = re.compile("^ACCUMULATOR_SIZE = [0-9]+$", re.MULTILINE)
HIDDEN_PATTERN = re.compile("^HIDDEN_SIZE = [0-9]+$", re.MULTILINE)
BLEND_PATTERN = re.compile(r"^NNUE_BLEND = \d+$", re.MULTILINE)


def model_geometry(model: Path) -> tuple[int, int]:
    """Return ``(accumulator, hidden)`` as stored in a trained ``model.npz``."""
    with np.load(model) as archive:
        feature_weights = archive["feature_weights"]
        hidden_weights = archive["hidden_weights"]
    accumulator = int(feature_weights.shape[1])
    hidden = int(hidden_weights.shape[0])
    if hidden_weights.shape[1] != 2 * accumulator:
        raise ValueError(
            f"{model}: the hidden layer takes {hidden_weights.shape[1]} inputs, "
            f"but two accumulators supply {2 * accumulator}"
        )
    return accumulator, hidden


def materialize(
    source: Path, output: Path, blend: int, model_override: Path | None = None
) -> None:
    """Copy the runnable files and freeze ``blend`` into the generated search."""
    if not 0 <= blend <= 100:
        raise ValueError("blend must be between 0 and 100")
    source = source.resolve()
    output = output.resolve()
    if not source.is_dir():
        raise ValueError(f"source candidate does not exist: {source}")
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    if output == source or source in output.parents:
        raise ValueError("output must not be inside the source candidate")

    missing = [name for name in SOURCE_FILES if not (source / name).is_file()]
    model = source / "weights" / "model.npz"
    if not model.is_file():
        missing.append("weights/model.npz")
    if missing:
        raise ValueError(f"source candidate is incomplete: {', '.join(missing)}")

    search_source = (source / "search.py").read_text(encoding="utf-8")
    rewritten, replacements = BLEND_PATTERN.subn(
        f"NNUE_BLEND = {blend}", search_source
    )
    if replacements != 1:
        raise ValueError("search.py must contain exactly one NNUE_BLEND assignment")

    if model_override is not None:
        if not model_override.is_file():
            raise ValueError(f"model not found: {model_override}")
        model = model_override
    accumulator, hidden = model_geometry(model)

    nnue_source = (source / "nnue.py").read_text(encoding="utf-8")
    nnue_source, accumulator_hits = ACCUMULATOR_PATTERN.subn(
        f"ACCUMULATOR_SIZE = {accumulator}", nnue_source
    )
    nnue_source, hidden_hits = HIDDEN_PATTERN.subn(
        f"HIDDEN_SIZE = {hidden}", nnue_source
    )
    if accumulator_hits != 1 or hidden_hits != 1:
        raise ValueError(
            "nnue.py must hold exactly one ACCUMULATOR_SIZE and one HIDDEN_SIZE"
        )

    output.mkdir(parents=True)
    for name in SOURCE_FILES:
        if name not in {"search.py", "nnue.py"}:
            shutil.copy2(source / name, output / name)
    (output / "search.py").write_text(rewritten, encoding="utf-8", newline="\n")
    (output / "nnue.py").write_text(nnue_source, encoding="utf-8", newline=chr(10))
    (output / "weights").mkdir()
    shutil.copy2(model, output / "weights" / "model.npz")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("challengers/v5_nnue"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--blend", type=int, required=True)
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="trained model.npz to ship instead of the source candidate's",
    )
    arguments = parser.parse_args()
    try:
        materialize(
            arguments.source, arguments.output, arguments.blend, arguments.model
        )
        geometry = model_geometry(
            arguments.model or arguments.source / "weights" / "model.npz"
        )
    except (FileExistsError, ValueError) as error:
        parser.error(str(error))
    print(
        f"created {arguments.output} with NNUE_BLEND={arguments.blend}, "
        f"{geometry[0]}x{geometry[1]}"
    )


if __name__ == "__main__":
    main()
