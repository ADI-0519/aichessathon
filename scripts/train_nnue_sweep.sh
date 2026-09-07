#!/usr/bin/env bash
# Train evaluator variants against the dataset scripts/train_nnue_v1.sh packed.
#
# Each variant changes one thing from the nnue-v1 baseline (128x32, 8 epochs,
# constant learning rate) so a paired-game result can be attributed to it.
# Rerunnable: a variant whose model.npz already exists is skipped.  FORCE=1
# retrains everything.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TRAIN_PY="${NNUE_TRAIN_ENV:-.venv-training}/Scripts/python.exe"
DATA_DIR="${NNUE_DATA_DIR:-benchmarks/runs/nnue-v1}"
TRAIN_DATA="$DATA_DIR/train-4m.npy"
VALIDATION_DATA="$DATA_DIR/validation-500k.npy"
SWEEP_DIR="${NNUE_SWEEP_DIR:-benchmarks/runs/nnue-sweep}"
FORCE="${FORCE:-0}"

# name:accumulator:hidden:epochs:schedule
VARIANTS=(
  "a128-e24-cosine:128:32:24:cosine"
  "a256-e24-cosine:256:32:24:cosine"
)

for path in "$TRAIN_PY" "$TRAIN_DATA" "$VALIDATION_DATA"; do
  [[ -e "$path" ]] || { echo "error: missing $path -- run scripts/train_nnue_v1.sh first" >&2; exit 1; }
done

mkdir -p "$SWEEP_DIR"
export PYTHONUNBUFFERED=1
LOG="$SWEEP_DIR/sweep.log"
exec > >(tee -a "$LOG") 2>&1

printf '\nNNUE sweep started %s\n' "$(date -Iseconds)"

for variant in "${VARIANTS[@]}"; do
  IFS=: read -r name accumulator hidden epochs schedule <<< "$variant"
  out_dir="$SWEEP_DIR/$name"
  mkdir -p "$out_dir"
  if [[ "$FORCE" != "1" && -f "$out_dir/model.npz" && -f "$out_dir/model-manifest.json" ]]; then
    echo "== $name: reusing existing model"
    continue
  fi
  printf '\n== %s: %sx%s, %s epochs, %s schedule (%s)\n' \
    "$name" "$accumulator" "$hidden" "$epochs" "$schedule" "$(date -Iseconds)"
  "$TRAIN_PY" -m tools.train_nnue \
    --train "$TRAIN_DATA" \
    --validation "$VALIDATION_DATA" \
    --output "$out_dir/model.npz" \
    --manifest "$out_dir/model-manifest.json" \
    --accumulator "$accumulator" \
    --hidden "$hidden" \
    --epochs "$epochs" \
    --batch-size 8192 \
    --lr-schedule "$schedule" \
    --device cuda
done

printf '\nsweep complete %s\n' "$(date -Iseconds)"
