import argparse
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1] / "scratch/nnue/run"


def fingerprint(records: np.ndarray) -> np.ndarray:
    indices = np.ascontiguousarray(records["indices"]).view(np.uint8)
    digest = np.zeros(len(records), dtype=np.uint64)
    prime = np.uint64(1099511628211)
    for column in range(indices.shape[1]):
        digest = (digest ^ indices[:, column].astype(np.uint64)) * prime
    stm = np.asarray(records["stm"], dtype=np.uint64)
    return digest ^ (stm * np.uint64(0x9E3779B97F4A7C15))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation", type=Path, default=ROOT / "validation-500k.npy")
    parser.add_argument("--add", nargs="+", required=True)
    args = parser.parse_args()

    held_out = np.unique(fingerprint(np.load(args.validation, allow_pickle=False)))
    base = np.load(args.base, mmap_mode="r", allow_pickle=False)
    print(f"{args.base.name}: {len(base):,} (already deduplicated)", flush=True)

    # two passes so output memmap is allocated once at its final length
    kept = {}
    total = len(base)
    for shard in args.add:
        path = ROOT / f"train-{shard}.npy"
        part = np.load(path, mmap_mode="r", allow_pickle=False)
        collides = np.isin(fingerprint(part[:]), held_out)
        kept[shard] = ~collides
        total += int((~collides).sum())
        print(
            f"{path.name}: {len(part):,}, dropped {collides.sum():,} "
            f"colliding ({collides.mean():.4%})",
            flush=True,
        )

    combined = np.lib.format.open_memmap(
        args.output, mode="w+", dtype=base.dtype, shape=(total,)
    )
    # trainer reshuffles every epoch, on-disk order never reaches the model
    at = len(base)
    combined[:at] = base
    for shard in args.add:
        part = np.load(ROOT / f"train-{shard}.npy", mmap_mode="r", allow_pickle=False)
        keep = part[kept[shard]]
        combined[at : at + len(keep)] = keep
        at += len(keep)
    combined.flush()
    size = args.output.stat().st_size / 1e9
    print()
    print(f"wrote {args.output} with {total:,} positions ({size:.2f} GB)")


if __name__ == "__main__":
    main()
