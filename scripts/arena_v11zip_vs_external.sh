#!/usr/bin/env bash
# The uploaded submission_v11.zip against the external public engine.
#
# LOCAL ONLY: the opponent lives under benchmarks/opponents/, which is
# gitignored, and this runs on a branch that is never pushed. It is a measuring
# stick, as Stockfish is; no code from it may reach current/ or any packaged
# challenger.
#
# Real tournament control, not the fast one. That opponent defers its Numba
# compilation past import -- cold it measures import 72.7s, first move 32.4s,
# second move 0.0s -- so at 4000ms it simply flags on move one and every result
# is a timeout rather than a game. Two shards so it has a core to compile on.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONUNBUFFERED=1
log_dir=benchmarks/runs/arena-v11zip-vs-external
mkdir -p "$log_dir"
uv run python -m tools.paired_arena_shards \
  --candidate benchmarks/opponents/sub_v11 \
  --opponent benchmarks/opponents/external_a \
  --base-ms 120000 --increment-ms 500 \
  --positions 24 --shards 2 \
  --log-dir "$log_dir" > "$log_dir/arena.log" 2> "$log_dir/arena.err"
echo "v11zip vs external complete $(date -Iseconds)"
