#!/usr/bin/env bash
# Gate a freshly trained KingNet V11-BIG model before it goes anywhere near an upload.
#
# The cluster run trains a format-3 net: 256-wide accumulator, 128 pairwise, 32
# hidden, eight material heads. challengers/exp_kingnet_v11_big is byte-identical
# to the deployed build except for nnue.py, which reads format 3 as well as
# format 2 -- so swapping the weights in is a net-only A/B against what we ship.
#
# Nothing here has ever run a format-3 net: the v11_big weights/ directory still
# holds the old 128-wide format-2 model. The v3 branch of nnue.py is therefore
# untested code, and a silent fixed-point bug there would look exactly like a
# weak net. Hence stages 2 and 3 before any game is played.
#
# Usage: scripts/test_new_net.sh /path/to/model.npz [name]
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MODEL="${1:?usage: scripts/test_new_net.sh <model.npz> [name]}"
NAME="${2:-bignet256}"
DEST="benchmarks/opponents/$NAME"

[ -f "$MODEL" ] || { echo "no such model: $MODEL" >&2; exit 1; }

echo "== 1/5  archive contents =="
uv run python - "$MODEL" <<'PY'
import sys
import numpy as np
path = sys.argv[1]
with np.load(path, allow_pickle=False) as archive:
    names = sorted(archive.files)
    params = 0
    for name in names:
        value = archive[name]
        tail = f"  = {value}" if value.ndim == 0 else ""
        if value.ndim:
            params += value.size
        print(f"  {name:24} {str(value.dtype):9} {str(value.shape):18}{tail}")
    version = int(np.asarray(archive["format_version"]).item())
    print(f"\n  format_version {version}   trainable parameters {params:,}")
    if version != 3:
        raise SystemExit(f"expected format 3 from train_kingnet_v11, got {version}")
    accumulator = archive["feature_weights"].shape[1]
    pairwise = int(np.asarray(archive["pairwise_width"]).item())
    heads = len(set(int(v) for v in archive["piece_head_map"][2:]))
    print(f"  accumulator {accumulator}  pairwise {pairwise}  heads {heads}")
    # The v3 runtime rejects a net whose halves do not fit the accumulator.
    if 2 * pairwise > accumulator:
        raise SystemExit("pairwise_width does not fit the accumulator")
PY

echo
echo "== 2/5  build a net-only challenger =="
rm -rf "$DEST"
mkdir -p "$DEST/weights"
cp challengers/exp_kingnet_v11_big/*.py "$DEST/"
cp "$MODEL" "$DEST/weights/model.npz"
printf "  %s  (%s)\n" "$DEST" "$(du -sh "$DEST" | cut -f1)"
# The package cap is 50 MB unzipped; float32 feature weights are the bulk of it.
printf "  weights %s\n" "$(du -h "$DEST/weights/model.npz" | cut -f1)"

echo
echo "== 3/5  correctness: king-bucket updates and fixed-point inference =="
uv run python -m tools.verify_kingnet --candidate "$DEST"

echo
echo "== 4/5  import + warmup wall clock (platform budget 90 s, we are at 88.5) =="
uv run python - "$DEST" <<'PY'
import subprocess
import sys
import time
# A fresh interpreter, because numba compiles at import and this process is warm.
code = (
    "import sys, time;"
    f"sys.path.insert(0, {sys.argv[1]!r});"
    "start = time.perf_counter();"
    "import agent;"
    "print(f'{time.perf_counter() - start:.1f}')"
)
start = time.perf_counter()
result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
if result.returncode:
    print(result.stderr[-2000:])
    raise SystemExit("import failed")
inner = float(result.stdout.strip().splitlines()[-1])
print(f"  local cold import {inner:.1f} s")
# Rated rounds 97-100 measured 70.1/77.0/85.7/88.5 s against a local idle 46 s:
# the platform runs this compile 1.5x to 1.9x slower than this machine does.
print(f"  platform estimate {inner * 1.5:.0f}-{inner * 1.9:.0f} s of the 90 s budget")
if inner * 1.9 > 90:
    print("  WARNING: the pessimistic estimate misses the budget")
PY

echo
echo "== 5/5  ready to play =="
cat <<NEXT
  net-only A/B against what we ship (the strongest signal, same search both sides):
    uv run python -m tools.paired_arena_shards \
      --candidate $DEST --opponent challengers/exp_kingnet_v11_big \
      --base-ms 120000 --increment-ms 500 --positions 24 --shards 4 \
      --log-dir benchmarks/runs/arena-$NAME-vs-current

  against the external benchmark (comparable with the 21.9-26.1% we have scored):
    uv run python -m tools.paired_arena_shards \
      --candidate $DEST --opponent benchmarks/opponents/external_a \
      --base-ms 120000 --increment-ms 500 --positions 24 --shards 4 \
      --log-dir benchmarks/runs/arena-$NAME-vs-external

  Run one at a time. Two heavy arenas at once exhausted memory earlier this week
  and lost 203 of 240 games. The adi_v14 arena is using the machine right now.
NEXT
