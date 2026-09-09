#!/usr/bin/env bash
# Price V8's three most aggressive prunings, one at a time.
#
# Each candidate is full V8 with a single flag off, played against full V8.
# A score above 50% means that technique is costing Elo and should come out.
#
# Two arenas at a time: six shards alongside another job exhausted memory once
# already and Windows killed the run, so concurrency is capped rather than
# trusted.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONUNBUFFERED=1

run_arena() {
  local name="$1" log_dir="benchmarks/runs/arena-$1"
  mkdir -p "$log_dir"
  uv run python -m tools.paired_arena_shards \
    --candidate "challengers/$name" \
    --opponent challengers/v8_search \
    --base-ms 4000 --increment-ms 100 \
    --positions 120 --shards 4 \
    --log-dir "$log_dir" > "$log_dir/arena.log" 2> "$log_dir/arena.err"
  echo "finished: $name $(date -Iseconds)"
}

run_arena v8_no_lmp &
run_arena v8_no_futility &
wait
run_arena v8_no_razor
echo "all ablations complete $(date -Iseconds)"
