#!/usr/bin/env bash
# Reproducible KingNet V11 training entry point.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <experiment-config.json> [run-directory]" >&2
  echo "the configuration schema and example are in docs/KINGNET_v11.md" >&2
  exit 2
fi

CONFIG="$1"
RUN_DIR="${2:-benchmarks/runs/kingnet-v11}"
TRAIN_ENV="${NNUE_TRAIN_ENV:-.venv-training}"
PY="${NNUE_TRAIN_PY:-$TRAIN_ENV/Scripts/python.exe}"

if [[ ! -f "$CONFIG" ]]; then
  echo "error: config not found: $CONFIG" >&2
  echo "create a run config under benchmarks/runs; see docs/KINGNET_v11.md" >&2
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
