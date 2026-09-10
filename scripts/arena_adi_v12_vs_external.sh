#!/usr/bin/env bash
# Adi's release v12 against the external public engine.
#
# LOCAL ONLY. The opponent lives under benchmarks/opponents/, which is
# gitignored. It is a measuring stick, as Stockfish is; no code from it reaches
# current/ or any packaged challenger.
#
# Real tournament control. That opponent defers Numba compilation past import --
# measured cold: import 72.7s, first move 32.4s, second move 0.0s -- so at a
# fast control it flags on move one and every result is a timeout, not a game.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONUNBUFFERED=1
log_dir=benchmarks/runs/arena-adiv12-vs-external
mkdir -p "$log_dir"
uv run python -m tools.paired_arena_shards \
  --candidate benchmarks/opponents/adi_v12 \
  --opponent benchmarks/opponents/external_a \
  --base-ms 120000 --increment-ms 500 \
  --positions 24 --shards 2 \
  --log-dir "$log_dir" > "$log_dir/arena.log" 2> "$log_dir/arena.err"
echo "adi v12 vs external complete $(date -Iseconds)"
