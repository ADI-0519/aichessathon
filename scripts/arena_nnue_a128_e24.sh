#!/usr/bin/env bash
# 240 paired games: the 24-epoch cosine net against the shipped V5 net.
# The two candidates differ only in weights/model.npz.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONUNBUFFERED=1
uv run python -m tools.paired_arena_shards \
  --candidate challengers/nnue_a128_e24 \
  --opponent challengers/v5_nnue \
  --base-ms 4000 --increment-ms 100 \
  --positions 120 --shards 6 \
  --log-dir benchmarks/runs/arena-a128-e24
