#!/usr/bin/env bash
# Does the king-bucketed evaluation add on top of the V8 search, and how much
# do the two together beat the champion?
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONUNBUFFERED=1

run_arena() {
  local opponent="$1" log_dir="benchmarks/runs/$2"
  mkdir -p "$log_dir"
  uv run python -m tools.paired_arena_shards \
    --candidate challengers/v10_kingnet_search \
    --opponent "$opponent" \
    --base-ms 4000 --increment-ms 100 \
    --positions 120 --shards 4 \
    --log-dir "$log_dir" > "$log_dir/arena.log" 2> "$log_dir/arena.err"
  echo "finished: v10 vs $opponent $(date -Iseconds)"
}

run_arena challengers/v8_search arena-v10-vs-v8 &
run_arena current arena-v10-vs-current &
wait
echo "v10 arenas complete $(date -Iseconds)"
