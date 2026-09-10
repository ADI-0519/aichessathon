#!/usr/bin/env bash
# Material-conditional contempt against the external public engine.
#
# LOCAL ONLY: opponent lives under benchmarks/opponents/, which is gitignored.
# A measuring stick, as Stockfish is; no code from it reaches a submission.
#
# Real tournament control: that opponent defers Numba compilation past import
# (cold: import 72.7s, first move 32.4s) and simply flags at a fast control.
#
# This is the right opponent for this feature. Contempt-when-behind only fires
# in positions we are losing, which is most of them against something 200 Elo
# stronger, so if it is worth anything it should show here first.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONUNBUFFERED=1
log_dir=benchmarks/runs/arena-v13-vs-external
mkdir -p "$log_dir"
uv run python -m tools.paired_arena_shards \
  --candidate benchmarks/opponents/v13_contempt \
  --opponent benchmarks/opponents/external_a \
  --base-ms 120000 --increment-ms 500 \
  --positions 24 --shards 2 \
  --log-dir "$log_dir" > "$log_dir/arena.log" 2> "$log_dir/arena.err"
echo "v13 contempt vs external complete $(date -Iseconds)"
