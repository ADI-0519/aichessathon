import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "scratch/nnue"
PACK_PY = ROOT / ".venv-pack/bin/python"
PIPELINE = ROOT / "pipeline"

# size target to the shard as packer allocates 68 bytes per slot up front
SHARDS = {"2015_05": 14_000_000, "2015_06": 17_000_000, "2016_01": 28_000_000}
CONCURRENCY = 2


def command(shard: str, target: int) -> list[str]:
    return [
        str(PACK_PY), "-m", "tools.pack_nnue_data",
        "--source", str(ROOT / f"sources/standard_rated_{shard}.parquet"),
        "--train-output", str(ROOT / f"run/train-{shard}.npy"),
        "--validation-output", str(ROOT / f"run/discard-{shard}.npy"),
        "--manifest", str(ROOT / f"run/manifest-{shard}.json"),
        "--train-target", str(target),
        "--validation-target", "1000",
        "--validation-groups", "1",
        "--min-ply", "12",
    ]


def arena_running() -> bool:
    found = subprocess.run(["pgrep", "-f", "paired_arena"], capture_output=True, text=True)
    return found.returncode == 0


def main() -> None:
    if arena_running():
        raise SystemExit("an arena is running; packing beside it corrupts both")

    pending = [(s, t) for s, t in SHARDS.items() if not (ROOT / f"run/train-{s}.npy").exists()]
    for shard in SHARDS:
        if (ROOT / f"run/train-{shard}.npy").exists():
            print(f"reusing train-{shard}.npy")

    for batch in [pending[i : i + CONCURRENCY] for i in range(0, len(pending), CONCURRENCY)]:
        running = []
        for shard, target in batch:
            log = (ROOT / f"run/pack-{shard}.log").open("w")
            print(f"packing {shard} (target {target:,}) ...", flush=True)
            running.append((shard, log, subprocess.Popen(
                command(shard, target), cwd=PIPELINE, stdout=log, stderr=subprocess.STDOUT
            )))
        for shard, log, process in running:
            code = process.wait()
            log.close()
            tail = (ROOT / f"run/pack-{shard}.log").read_text().strip().splitlines()
            print(f"{shard} exited {code}: {tail[-1] if tail else '(no output)'}", flush=True)
            if code != 0:
                sys.exit(f"packing {shard} failed; see run/pack-{shard}.log")


if __name__ == "__main__":
    main()
