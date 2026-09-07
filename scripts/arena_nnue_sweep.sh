#!/usr/bin/env bash
# Both sweep nets against the shipped V5 net, 240 paired games each.
#
# Each candidate is challengers/v5_nnue with exactly one file changed --
# weights/model.npz (and the width constant the model implies) -- so a result
# is attributable to the net alone.  The two arenas run together: a paired game
# only ever has one side thinking, so 2 x 6 shards is 12 busy cores, and any
# contention that remains slows both sides of every game equally.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONUNBUFFERED=1

run_arena() {
  local candidate="$1" log_dir="$2"
  mkdir -p "$log_dir"
  uv run python -m tools.paired_arena_shards \
    --candidate "$candidate" \
    --opponent challengers/v5_nnue \
    --base-ms 4000 --increment-ms 100 \
    --positions 120 --shards 6 \
    --log-dir "$log_dir" > "$log_dir/arena.log" 2> "$log_dir/arena.err"
  echo "finished: $candidate"
}

run_arena challengers/nnue_a128_e24 benchmarks/runs/arena-a128-e24 &
run_arena challengers/nnue_a256_e24 benchmarks/runs/arena-a256-e24 &
wait
echo "both arenas complete $(date -Iseconds)"
