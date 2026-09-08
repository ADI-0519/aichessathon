#!/usr/bin/env bash
# V8 search selectivity against the V7 champion, 240 paired games.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONUNBUFFERED=1
mkdir -p benchmarks/runs/arena-v8-search
uv run python -m tools.paired_arena_shards \
  --candidate challengers/v8_search \
  --opponent current \
  --base-ms 4000 --increment-ms 100 \
  --positions 120 --shards 4 \
  --log-dir benchmarks/runs/arena-v8-search \
  > benchmarks/runs/arena-v8-search/arena.log 2> benchmarks/runs/arena-v8-search/arena.err
echo "v8 arena complete $(date -Iseconds)"
