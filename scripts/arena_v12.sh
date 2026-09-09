#!/usr/bin/env bash
# V12 (all three improvements plus the repaired clock) against Adi's build.
# Waits for any running arena first: two at once exhausted memory once already.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONUNBUFFERED=1

# Git Bash has no pgrep, so wait on the other arena's completion marker
# instead of on its processes. Running two at once has exhausted memory before.
while [ -f benchmarks/runs/arena-v11-vs-adi/arena.log ]    && ! grep -q "paired standard error" benchmarks/runs/arena-v11-vs-adi/arena.log; do
  sleep 30
done

log_dir=benchmarks/runs/arena-v12-vs-adi
mkdir -p "$log_dir"
uv run python -m tools.paired_arena_shards \
  --candidate challengers/v12_clock \
  --opponent challengers/adi_current \
  --base-ms 4000 --increment-ms 100 \
  --positions 120 --shards 5 \
  --log-dir "$log_dir" > "$log_dir/arena.log" 2> "$log_dir/arena.err"
echo "v12 arena complete $(date -Iseconds)"
