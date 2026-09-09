#!/usr/bin/env bash
# Reproducible KingNet V11 training entry point.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONFIG="${1:-configs/kingnet_v11.json}"
RUN_DIR="${2:-benchmarks/runs/kingnet-v11}"
TRAIN_ENV="${NNUE_TRAIN_ENV:-.venv-training}"
PY="${NNUE_TRAIN_PY:-$TRAIN_ENV/Scripts/python.exe}"

if [[ ! -f "$CONFIG" ]]; then
  echo "error: config not found: $CONFIG" >&2
  echo "copy configs/kingnet_v11.example.json and point it at packed shards" >&2
  exit 2
fi
if [[ ! -x "$PY" ]]; then
  echo "error: training Python not found: $PY" >&2
  exit 2
fi

mkdir -p "$RUN_DIR"
export PYTHONUNBUFFERED=1

"$PY" -m tools.train_kingnet_v11 \
  --config "$CONFIG" \
  --output "$RUN_DIR/model.npz" \
  --manifest "$RUN_DIR/manifest.json" \
  --checkpoint "$RUN_DIR/best.pt"
