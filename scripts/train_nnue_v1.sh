#!/usr/bin/env bash
# Build the first learned-evaluator artifact from public labels.
#
# This script is safe to rerun: completed data and model artifacts are reused.
# Set FORCE=1 to rebuild them, or override the NNUE_* variables for an experiment.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TRAIN_ENV="${NNUE_TRAIN_ENV:-.venv-training}"
TRAIN_PY="$TRAIN_ENV/Scripts/python.exe"
SOURCE="benchmarks/suites/sources/standard_rated_2014_09.parquet"
SOURCE_URL="https://huggingface.co/datasets/Lichess/fishnet-evals/resolve/main/standard_rated_2014_09.parquet?download=true"
SOURCE_SHA256="b2d0d3cc3ea2f2795e6fffa4d33f5c1ff6b26d8a0c1cbb5a74768f3228b9a5ef"
RUN_DIR="${NNUE_RUN_DIR:-benchmarks/runs/nnue-v1}"
TRAIN_TARGET="${NNUE_TRAIN_TARGET:-4000000}"
VALIDATION_TARGET="${NNUE_VALIDATION_TARGET:-500000}"
EPOCHS="${NNUE_EPOCHS:-8}"
BATCH_SIZE="${NNUE_BATCH_SIZE:-8192}"
FORCE="${FORCE:-0}"

mkdir -p benchmarks/suites/sources benchmarks/runs/uv-cache "$RUN_DIR"
export UV_CACHE_DIR="$ROOT/benchmarks/runs/uv-cache"
export PYTHONUNBUFFERED=1

LOG="$RUN_DIR/pipeline.log"
exec > >(tee -a "$LOG") 2>&1

printf '\nNNUE V1 pipeline started %s\n' "$(date -Iseconds)"

if ! command -v uv >/dev/null 2>&1; then
  echo "error: uv is not available on PATH" >&2
  exit 1
fi

# uv renamed --system-certs to --native-tls; support whichever this uv has so the
# script runs on every machine on the team.
UV_TLS=""
if uv venv --help 2>&1 | grep -q -- "--native-tls"; then
  UV_TLS="--native-tls"
elif uv venv --help 2>&1 | grep -q -- "--system-certs"; then
  UV_TLS="--system-certs"
fi

if [[ ! -x "$TRAIN_PY" ]]; then
  uv ${UV_TLS} venv "$TRAIN_ENV" --python 3.12
fi

uv ${UV_TLS} pip install \
  --python "$TRAIN_PY" \
  --torch-backend cu128 \
  "chess==1.11.2" \
  "numpy==2.5.2" \
  pyarrow \
  torch

"$TRAIN_PY" -c \
  "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); raise SystemExit(0 if torch.cuda.is_available() else 'CUDA is unavailable in the training environment')"

if [[ ! -f "$SOURCE" ]]; then
  curl.exe -fL "$SOURCE_URL" -o "$SOURCE.download"
  printf '%s  %s\n' "$SOURCE_SHA256" "$SOURCE.download" | sha256sum --check
  mv "$SOURCE.download" "$SOURCE"
fi
printf '%s  %s\n' "$SOURCE_SHA256" "$SOURCE" | sha256sum --check

TRAIN_DATA="$RUN_DIR/train-4m.npy"
VALIDATION_DATA="$RUN_DIR/validation-500k.npy"
DATA_MANIFEST="$RUN_DIR/data-manifest.json"
MODEL="$RUN_DIR/model.npz"
MODEL_MANIFEST="$RUN_DIR/model-manifest.json"

if [[ "$FORCE" == "1" || ! -f "$TRAIN_DATA" || ! -f "$VALIDATION_DATA" || ! -f "$DATA_MANIFEST" ]]; then
  "$TRAIN_PY" -m tools.pack_nnue_data \
    --source "$SOURCE" \
    --train-output "$TRAIN_DATA" \
    --validation-output "$VALIDATION_DATA" \
    --manifest "$DATA_MANIFEST" \
    --train-target "$TRAIN_TARGET" \
    --validation-target "$VALIDATION_TARGET" \
    --validation-groups 1 \
    --min-ply 12
else
  echo "reusing packed data in $RUN_DIR"
fi

if [[ "$FORCE" == "1" || ! -f "$MODEL" || ! -f "$MODEL_MANIFEST" ]]; then
  "$TRAIN_PY" -m tools.train_nnue \
    --train "$TRAIN_DATA" \
    --validation "$VALIDATION_DATA" \
    --output "$MODEL" \
    --manifest "$MODEL_MANIFEST" \
    --accumulator 128 \
    --hidden 32 \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --device cuda
else
  echo "reusing trained model in $RUN_DIR"
fi

echo "pipeline complete: $MODEL"
echo "log: $LOG"
