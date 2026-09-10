"""Create an immutable local candidate with one explicit evaluation blend."""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

SOURCE_FILES = ("agent.py", "engine.py", "nnue.py", "search.py")
BLEND_PATTERN = re.compile(r"^NNUE_BLEND = \d+$", re.MULTILINE)


def materialize(source: Path, output: Path, blend: int) -> None:
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

    output.mkdir(parents=True)
    for name in SOURCE_FILES:
        if name != "search.py":
            shutil.copy2(source / name, output / name)
    (output / "search.py").write_text(rewritten, encoding="utf-8", newline="\n")
    (output / "weights").mkdir()
    shutil.copy2(model, output / "weights" / "model.npz")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("challengers/v5_nnue"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--blend", type=int, required=True)
    arguments = parser.parse_args()
    try:
        materialize(arguments.source, arguments.output, arguments.blend)
    except (FileExistsError, ValueError) as error:
        parser.error(str(error))
    print(f"created {arguments.output} with NNUE_BLEND={arguments.blend}")


if __name__ == "__main__":
    main()
