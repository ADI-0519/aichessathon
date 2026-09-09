#!/usr/bin/env bash
# Complete the net-by-blend grid, one factor per arena.
#
#   blend  : blend100_v5net  vs v5_nnue          -- same net, 50 -> 100
#   net    : blend100_a128e24 vs blend100_v5net  -- same blend 100, net swapped
#
# The net was already measured at blend 50 (nnue_a128_e24, 50.2%). If a better
# net buys nothing at blend 100 either, the net is not the lever.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONUNBUFFERED=1

run_arena() {
  local candidate="$1" opponent="$2" log_dir="$3"
  mkdir -p "$log_dir"
  uv run python -m tools.paired_arena_shards \
    --candidate "$candidate" --opponent "$opponent" \
    --base-ms 4000 --increment-ms 100 \
    --positions 120 --shards 6 \
    --log-dir "$log_dir" > "$log_dir/arena.log" 2> "$log_dir/arena.err"
  echo "finished: $candidate vs $opponent"
}

run_arena challengers/blend100_v5net challengers/v5_nnue \
  benchmarks/runs/arena-blend100-v5net &
run_arena challengers/blend100_a128e24 challengers/blend100_v5net \
  benchmarks/runs/arena-blend100-a128e24 &
wait
echo "blend arenas complete $(date -Iseconds)"
