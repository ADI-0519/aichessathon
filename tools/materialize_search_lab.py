"""Create an isolated engine build with compile-time search ablation profiles."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import uuid
from pathlib import Path

from tools.backtest_core import fingerprint_agent, git_state
from tools.search_diagnostics import REPOSITORY

DEFAULT_SOURCE = REPOSITORY / "current"
DEFAULT_OUTPUT = (
    REPOSITORY / "benchmarks" / "runs" / "candidates" / "current_search_lab"
)
BLEND_ASSIGNMENT = re.compile(r"^NNUE_BLEND\s*=\s*(\d+)\s*$", re.MULTILINE)


def _profile_source(baseline_blend: int) -> str:
    """Build profiles around the source engine's configured evaluation blend."""
    return f'''

# Development-lab switches. Numba treats these globals as compile-time
# constants, so configure_experiment() must run before warmup.
ENABLE_LMR = True
ENABLE_NULL_MOVE = True
ACTIVE_PROFILE = "baseline"
_EXPERIMENT_PROFILES = {{
    "baseline": ({baseline_blend}, True, True),
    "hce-only": (0, True, True),
    "nnue-only": (100, True, True),
    "no-lmr": ({baseline_blend}, False, True),
    "no-null": ({baseline_blend}, True, False),
    "no-lmr-no-null": ({baseline_blend}, False, False),
}}
'''

CONFIGURE_SOURCE = '''

def available_profiles() -> tuple[str, ...]:
    """Return the isolated evaluator and pruning profiles in this lab."""
    return tuple(_EXPERIMENT_PROFILES)


def configure_experiment(name: str) -> None:
    """Select one profile before Numba compiles the recursive search."""
    global ACTIVE_PROFILE, ENABLE_LMR, ENABLE_NULL_MOVE, NNUE_BLEND
    if name not in _EXPERIMENT_PROFILES:
        choices = ", ".join(available_profiles())
        raise ValueError(f"unknown search profile {name!r}; choose from {choices}")
    if _negamax.signatures or _quiescence.signatures:
        raise RuntimeError("search profile must be selected before warmup")
    NNUE_BLEND, ENABLE_LMR, ENABLE_NULL_MOVE = _EXPERIMENT_PROFILES[name]
    ACTIVE_PROFILE = name
'''


def _replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise ValueError(f"expected one {label} insertion point, found {count}")
    return source.replace(old, new, 1)


def _baseline_blend(source: str) -> tuple[int, re.Match[str]]:
    """Read and validate the source engine's single blend assignment."""
    assignments = list(BLEND_ASSIGNMENT.finditer(source))
    if len(assignments) != 1:
        raise ValueError(
            "expected one integer NNUE blend assignment, "
            f"found {len(assignments)}"
        )
    assignment = assignments[0]
    baseline_blend = int(assignment.group(1))
    if not 0 <= baseline_blend <= 100:
        raise ValueError("NNUE blend must be between 0 and 100")
    return baseline_blend, assignment


def instrument_search(source: str) -> str:
    """Add profile constants without changing baseline search semantics."""
    baseline_blend, assignment = _baseline_blend(source)
    source = (
        source[: assignment.end()]
        + _profile_source(baseline_blend)
        + source[assignment.end() :]
    )
    source = _replace_once(
        source,
        "    if (\n        allow_null\n",
        "    if (\n        ENABLE_NULL_MOVE\n        and allow_null\n",
        "null-move guard",
    )
    source = _replace_once(
        source,
        "        reduced = (\n            depth >= 3\n",
        "        reduced = (\n            ENABLE_LMR\n            and depth >= 3\n",
        "LMR guard",
    )
    return _replace_once(
        source,
        "\ndef warmup() -> None:\n",
        CONFIGURE_SOURCE + "\n\ndef warmup() -> None:\n",
        "configuration function",
    )


def materialize(source: Path, output: Path) -> None:
    """Copy a frozen challenger and instrument only its search module."""
    source = source.resolve()
    output = output.resolve()
    if not (source / "agent.py").is_file() or not (source / "search.py").is_file():
        raise ValueError(f"source is not a packaged agent: {source}")
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    # tempfile.mkdtemp() deliberately creates a private directory. On Windows
    # that restrictive ACL survives the final rename and can make the lab
    # unreadable from the user's normal terminal. A unique directory created
    # normally inherits the repository ACL while still giving us atomic rename.
    temporary = output.parent / f".{output.name}-{uuid.uuid4().hex}.tmp"
    temporary.mkdir()
    try:
        for path in source.iterdir():
            if path.is_file() and path.name != "README.md":
                shutil.copy2(path, temporary / path.name)
            elif path.is_dir() and path.name not in {
                "__pycache__",
                ".mypy_cache",
                ".ruff_cache",
            }:
                shutil.copytree(
                    path,
                    temporary / path.name,
                    ignore=shutil.ignore_patterns(
                        "__pycache__", ".mypy_cache", ".ruff_cache", "*.pyc"
                    ),
                )
        search_path = temporary / "search.py"
        original = search_path.read_text(encoding="utf-8")
        baseline_blend, _ = _baseline_blend(original)
        search_path.write_text(instrument_search(original), encoding="utf-8")
        manifest = {
            "schema_version": 1,
            "purpose": "development-only search/evaluator ablation lab",
            "source": fingerprint_agent(source),
            "baseline_nnue_blend": baseline_blend,
            "profiles": [
                "baseline",
                "hce-only",
                "nnue-only",
                "no-lmr",
                "no-null",
                "no-lmr-no-null",
            ],
            "git": git_state(REPOSITORY),
        }
        (temporary / "LAB_MANIFEST.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        temporary.replace(output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    try:
        materialize(args.source, args.output)
    except (FileExistsError, ValueError) as error:
        parser.error(str(error))
    print(f"materialized {args.output.resolve()}")


if __name__ == "__main__":
    main()
