#!/usr/bin/env bash
# What are the seven selectivity techniques worth on top of the evaluation and
# cache the team already has? V11 is Adi's current plus those techniques.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONUNBUFFERED=1
log_dir=benchmarks/runs/arena-v11-vs-adi
mkdir -p "$log_dir"
uv run python -m tools.paired_arena_shards \
  --candidate challengers/v11_full \
  --opponent challengers/adi_current \
  --base-ms 4000 --increment-ms 100 \
  --positions 120 --shards 6 \
  --log-dir "$log_dir" > "$log_dir/arena.log" 2> "$log_dir/arena.err"
echo "v11 arena complete $(date -Iseconds)"
