#!/usr/bin/env bash
# One command to run the big training experiment on a shared GPU cluster.
#
#   bash scripts/cluster_train.sh
#
# It is written to be a good neighbour first and fast second, because the box is
# shared: it claims exactly one idle GPU, caps its own CPU and thread use well
# below what is available, runs at low priority, writes only under its own
# directory, refuses to start a second copy of itself, and cleans up on exit.
#
# Every stage is resumable. Re-running skips work that already finished, so an
# interrupted run costs only the stage it was in.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# ---------------------------------------------------------------- configuration
MONTHS="${MONTHS:-12}"                  # Parquet months to download; the last is held out
ACCUMULATOR="${ACCUMULATOR:-256}"
HIDDEN="${HIDDEN:-32}"
EPOCHS="${EPOCHS:-12}"
BATCH_SIZE="${BATCH_SIZE:-16384}"
TRAIN_TARGET="${TRAIN_TARGET:-60000000}"
VALIDATION_TARGET="${VALIDATION_TARGET:-1000000}"
LR_SCHEDULE="${LR_SCHEDULE:-cosine}"

# Politeness. 64 cores exist; other people are using them.
# nproc reports the CPUs this process may actually use, which on a scheduled
# node is often far fewer than the machine has. Oversubscribing here would
# thrash our own packing and everyone else's jobs with it.
VISIBLE_CPUS="$(nproc)"
WORKERS="${WORKERS:-$(( VISIBLE_CPUS > 4 ? VISIBLE_CPUS - 2 : 2 ))}"
THREADS="${THREADS:-$(( VISIBLE_CPUS > 8 ? 4 : 1 ))}"
NICE="${NICE:-15}"
GPU_FREE_MB="${GPU_FREE_MB:-40000}"     # a GPU must have at least this free
GPU_MAX_UTIL="${GPU_MAX_UTIL:-10}"      # ...and be no busier than this percent

# Where large files go. /data is the big volume; keep everything under our own
# directory there so nothing of ours lands in someone else's space.
WORK="${WORK:-/data/$USER/aichessathon}"
RUN_DIR="$WORK/runs/$(date +%Y%m%d-%H%M%S)"
SOURCE_DIR="$WORK/sources"
VENV="$WORK/venv"
LOCK="$WORK/.cluster_train.lock"

mkdir -p "$WORK" "$SOURCE_DIR" "$RUN_DIR"

# ------------------------------------------------------------------- single copy
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "another cluster_train.sh is already running (lock: $LOCK). Refusing to start." >&2
  exit 1
fi

LOG="$RUN_DIR/run.log"
STATUS="$RUN_DIR/status.txt"
exec > >(tee -a "$LOG") 2>&1

say() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
stage() { printf '%s | %s\n' "$(date -Iseconds)" "$*" >> "$STATUS"; say "== $*"; }

cleanup() {
  local code=$?
  stage "exited with code $code"
  [[ -n "${TRAIN_PID:-}" ]] && kill "$TRAIN_PID" 2>/dev/null || true
  exit "$code"
}
trap cleanup EXIT INT TERM

say "run directory: $RUN_DIR"
say "log:           $LOG"
say "status:        $STATUS"

# ------------------------------------------------------------------- memory
stage "checking memory"
AVAILABLE_KB="$(awk '/MemAvailable/ {print $2}' /proc/meminfo)"
AVAILABLE_GB=$((AVAILABLE_KB / 1024 / 1024))
# 68 bytes a position, plus roughly a gigabyte per packing worker for the
# Parquet row group it decodes.
NEED_GB=$(( (TRAIN_TARGET * 68 / 1024 / 1024 / 1024) + WORKERS + 4 ))
say "available: ${AVAILABLE_GB} GB, this run needs about ${NEED_GB} GB"
if (( AVAILABLE_GB < NEED_GB + 16 )); then
  echo "Not enough memory: ${AVAILABLE_GB} GB available, ${NEED_GB} GB needed plus headroom." >&2
  echo "Others are using this machine; lower TRAIN_TARGET or WORKERS, or wait." >&2
  exit 1
fi

# ----------------------------------------------------------------- pick one GPU
stage "selecting a GPU"
# GPU=6 picks a card explicitly; leaving it unset finds an idle one.
GPU_ID="${GPU:-${GPU_ID:-}}"
if [[ -n "$GPU_ID" ]]; then
  if ! nvidia-smi --query-gpu=index --format=csv,noheader,nounits        | grep -qx "$GPU_ID"; then
    echo "GPU $GPU_ID does not exist on this machine." >&2
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv >&2
    exit 1
  fi
  read -r used_mb util < <(nvidia-smi --id="$GPU_ID"     --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits     | tr -d ',')
  say "using GPU $GPU_ID as asked: ${used_mb} MB already in use, ${util}% busy"
  if (( used_mb > 2000 )); then
    say "WARNING: someone else appears to be on GPU $GPU_ID. Continuing because"
    say "         you named it, but check gpustat if that was not intended."
  fi
fi
if [[ -z "$GPU_ID" ]]; then
  GPU_ID="$(nvidia-smi --query-gpu=index,memory.free,utilization.gpu \
            --format=csv,noheader,nounits \
    | awk -F', ' -v m="$GPU_FREE_MB" -v u="$GPU_MAX_UTIL" \
        '$2 >= m && $3 <= u { print $1; exit }')"
fi
if [[ -z "$GPU_ID" ]]; then
  echo "No GPU is free enough (need >= ${GPU_FREE_MB} MB free and <= ${GPU_MAX_UTIL}% busy)." >&2
  echo "Someone else is using them all. Current state:" >&2
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv >&2
  exit 1
fi
export CUDA_VISIBLE_DEVICES="$GPU_ID"
say "claimed GPU $GPU_ID only; the others stay invisible to this process"

# --------------------------------------------------------------- thread limits
for pool in OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS \
            NUMEXPR_NUM_THREADS VECLIB_MAXIMUM_THREADS; do
  export "$pool=$THREADS"
done
export PYTHONUNBUFFERED=1
say "visible CPUs: $VISIBLE_CPUS; packing workers: $WORKERS; threads each: $THREADS"
if (( WORKERS * THREADS > VISIBLE_CPUS * 2 )); then
  say "WARNING: workers x threads exceeds twice the visible CPUs; lower WORKERS"
fi

# ------------------------------------------------------------------ environment
stage "preparing the python environment"
if [[ ! -x "$VENV/bin/python" ]]; then
  python3 -m venv "$VENV"
fi
PY="$VENV/bin/python"
"$PY" -m pip install --quiet --upgrade pip
"$PY" -m pip install --quiet "numpy>=2" pyarrow "chess==1.11.2"

# pip's default torch is built against the newest CUDA, which a slightly older
# driver cannot load: the first run here installed cu130 against a 12.8 driver
# and reported no CUDA at all. Ask the driver what it supports and take the
# matching wheel index.
DRIVER_CUDA="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader   | head -1 | awk -F. '{print $1}')"
CUDA_TAG="${CUDA_TAG:-}"
if [[ -z "$CUDA_TAG" ]]; then
  RUNTIME="$(nvidia-smi | awk -F'CUDA Version: ' '/CUDA Version/ {print $2}'     | awk '{print $1}' | head -1)"
  case "$RUNTIME" in
    13.*) CUDA_TAG=cu130 ;;
    12.8|12.9) CUDA_TAG=cu128 ;;
    12.6|12.7) CUDA_TAG=cu126 ;;
    12.[0-5]) CUDA_TAG=cu121 ;;
    11.*) CUDA_TAG=cu118 ;;
    *) CUDA_TAG=cu128 ;;
  esac
  say "driver $DRIVER_CUDA reports CUDA $RUNTIME, using torch wheels for $CUDA_TAG"
fi
if ! "$PY" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"      2>/dev/null; then
  say "installing torch for $CUDA_TAG (this takes a few minutes)"
  "$PY" -m pip install --quiet --force-reinstall     --index-url "https://download.pytorch.org/whl/$CUDA_TAG" torch
fi
"$PY" - <<'PYCHECK'
import torch
print(f"torch {torch.__version__}, cuda available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"visible device: {torch.cuda.get_device_name(0)}")
else:
    raise SystemExit("CUDA is not available in this environment")
PYCHECK

# ----------------------------------------------------------------------- data
stage "fetching $MONTHS Parquet months into $SOURCE_DIR"
BASE="https://huggingface.co/datasets/Lichess/fishnet-evals/resolve/main"
SOURCES=()
fetched=0
for year in 2014 2015 2016; do
  for month in 01 02 03 04 05 06 07 08 09 10 11 12; do
    (( fetched >= MONTHS )) && break 2
    name="standard_rated_${year}_${month}.parquet"
    target="$SOURCE_DIR/$name"
    if [[ ! -s "$target" ]]; then
      if ! curl -fsSL --retry 3 "$BASE/$name?download=true" -o "$target.part"; then
        rm -f "$target.part"
        say "  $name unavailable, skipping"
        continue
      fi
      mv "$target.part" "$target"
      say "  downloaded $name ($(du -h "$target" | cut -f1))"
    else
      say "  reusing $name"
    fi
    SOURCES+=(--source "$target")
    fetched=$((fetched + 1))
  done
done
if (( fetched < 2 )); then
  echo "need at least two months; only $fetched available" >&2
  exit 1
fi
say "using $fetched months, holding the last out for validation"

# --------------------------------------------------------------------- packing
TRAIN_NPY="$WORK/packed/train-${TRAIN_TARGET}.npy"
VAL_NPY="$WORK/packed/validation-${VALIDATION_TARGET}.npy"
DATA_MANIFEST="$WORK/packed/data-manifest.json"
mkdir -p "$WORK/packed"

if [[ -s "$TRAIN_NPY" && -s "$VAL_NPY" && -s "$DATA_MANIFEST" ]]; then
  stage "reusing packed data ($(du -h "$TRAIN_NPY" | cut -f1))"
else
  stage "packing positions (this is the long CPU stage)"
  nice -n "$NICE" "$PY" -m tools.pack_nnue_data \
    "${SOURCES[@]}" \
    --train-output "$TRAIN_NPY" \
    --validation-output "$VAL_NPY" \
    --manifest "$DATA_MANIFEST" \
    --train-target "$TRAIN_TARGET" \
    --validation-target "$VALIDATION_TARGET" \
    --workers "$WORKERS" \
    --min-ply 12
fi
say "packed training set: $(du -h "$TRAIN_NPY" | cut -f1)"

# -------------------------------------------------------------------- training
MODEL="$RUN_DIR/model.npz"
MODEL_MANIFEST="$RUN_DIR/model-manifest.json"
stage "training ${ACCUMULATOR}x${HIDDEN} for $EPOCHS epochs on GPU $GPU_ID"
nice -n "$NICE" "$PY" -m tools.train_nnue \
  --train "$TRAIN_NPY" \
  --validation "$VAL_NPY" \
  --output "$MODEL" \
  --manifest "$MODEL_MANIFEST" \
  --accumulator "$ACCUMULATOR" \
  --hidden "$HIDDEN" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --lr-schedule "$LR_SCHEDULE" \
  --device cuda &
TRAIN_PID=$!
wait "$TRAIN_PID"
TRAIN_PID=""

stage "complete"
say "model:    $MODEL"
say "manifest: $MODEL_MANIFEST"
say "log:      $LOG"
