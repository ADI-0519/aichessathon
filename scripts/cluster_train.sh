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
# Months to fetch; the last is held out for validation. Recent months are huge,
# so three is already far more data than every net we have shipped.
MONTHS="${MONTHS:-3}"
ACCUMULATOR="${ACCUMULATOR:-256}"
HIDDEN="${HIDDEN:-32}"
EPOCHS="${EPOCHS:-24}"
# Samples drawn per epoch. The default of twenty million was written for
# four-million-position datasets; against 200M it means each position is seen
# about once across the whole run. Defaulting to the training target makes an
# epoch a full pass.
SAMPLES_PER_EPOCH="${SAMPLES_PER_EPOCH:-$TRAIN_TARGET}"
BATCH_SIZE="${BATCH_SIZE:-16384}"
TRAIN_TARGET="${TRAIN_TARGET:-200000000}"
VALIDATION_TARGET="${VALIDATION_TARGET:-1000000}"
LR_SCHEDULE="${LR_SCHEDULE:-cosine}"

# Politeness. 64 cores exist; other people are using them.
# nproc reports the CPUs this process may actually use, which on a scheduled
# node is often far fewer than the machine has. Oversubscribing here would
# thrash our own packing and everyone else's jobs with it.
VISIBLE_CPUS="$(nproc)"
# Half the visible cores, capped at 16. Taking nearly all of them starves the
# other jobs on this node, which need CPU to feed their own GPUs.
WORKERS="${WORKERS:-$(( VISIBLE_CPUS > 4 ? (VISIBLE_CPUS / 2 > 16 ? 16 : VISIBLE_CPUS / 2) : 2 ))}"
THREADS="${THREADS:-1}"   # packing is pure Python; native threads only contend
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
set +o pipefail   # detection below reads tools whose pipelines may close early
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
# `head` closes the pipe as soon as it has its line, which hands SIGPIPE to
# whatever is upstream; under `set -o pipefail` that is exit 141 and `set -e`
# then kills the run. awk does the same job inside one process.
DRIVER_CUDA="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader   | awk 'NR==1 {split($0, v, "."); print v[1]}')"
CUDA_TAG="${CUDA_TAG:-}"
if [[ -z "$CUDA_TAG" ]]; then
  RUNTIME="$(nvidia-smi     | awk -F'CUDA Version: ' '/CUDA Version/ {split($2, f, " "); print f[1]; exit}')"
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

set -o pipefail

# ----------------------------------------------------------------------- data
stage "fetching $MONTHS Parquet months into $SOURCE_DIR"
BASE="https://huggingface.co/datasets/Lichess/fishnet-evals/resolve/main"
SOURCES=()
fetched=0
# Newest first. The corpus runs 2013 to 2025 and grows enormously: a 2014 month
# is 25-200 MB where a 2024 month is 6-7 GB, so one recent month carries more
# positions than two years of early ones. YEARS overrides the order.
for year in ${YEARS:-2024 2023 2022 2021 2020 2019 2018 2017 2016 2015 2014}; do
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
      size=$(stat -c%s "$target")
      if (( size < 65536 )); then
        say "  $name is empty ($size bytes), that month has no data; skipping"
        rm -f "$target"
        continue
      fi
      say "  downloaded $name ($(du -h "$target" | cut -f1))"
    elif (( $(stat -c%s "$target") < 65536 )); then
      say "  $name is empty, skipping"
      continue
    else
      say "  reusing $name ($(du -h "$target" | cut -f1))"
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
# The engine loads a format-2 king-bucketed network with 16 * 768 features.
# tools/train_nnue.py builds the retired 768-feature format-1 evaluator, whose
# output current/nnue.py refuses to load, so training goes through the KingNet
# trainer and its JSON contract.
MODEL="$RUN_DIR/model.npz"
MODEL_MANIFEST="$RUN_DIR/model-manifest.json"
CONFIG="$RUN_DIR/config.json"

"$PY" - "$CONFIG" "$TRAIN_NPY" "$VAL_NPY" "$ACCUMULATOR" "$HIDDEN"       "$EPOCHS" "$BATCH_SIZE" "$SAMPLES_PER_EPOCH" <<'PYCONF'
import json, sys
config_path, train_npy, val_npy, acc, hidden, epochs, batch, spe = sys.argv[1:9]
# Sampling and selection lean towards the endings, where our static evaluation
# is worst: measured against Stockfish it errs by 94 cp with 26 or more pieces
# and 346 cp with seven or fewer, and both long rated losses were endgames.
config = {
    "model": {
        "accumulator": int(acc),
        "hidden": int(hidden),
        "pairwise_width": int(acc) // 2,
        "cp_scale": 400.0,
        "piece_head_map": [0]*9 + [1]*4 + [2]*4 + [3]*4 + [4]*4 + [5]*3 + [6]*3 + [7]*2,
    },
    "export": {"feature_storage": "float16"},
    "training": {
        "epochs": int(epochs),
        "samples_per_epoch": int(spe),
        "batch_size": int(batch),
        "learning_rate": 0.0003,
        "min_learning_rate": 0.00001,
        "warmup_fraction": 0.05,
        "weight_decay": 0.00001,
        "mirror_probability": 0.5,
        "gradient_clip_norm": 1.0,
        "seed": 20260910,
        "device": "cuda",
        "init_model": None,
        "resume_checkpoint": None,
    },
    "piece_bands": [
        {"name": "2_8",   "min_pieces": 2,  "max_pieces": 8,  "weight": 0.10},
        {"name": "9_12",  "min_pieces": 9,  "max_pieces": 12, "weight": 0.225},
        {"name": "13_16", "min_pieces": 13, "max_pieces": 16, "weight": 0.20},
        {"name": "17_24", "min_pieces": 17, "max_pieces": 24, "weight": 0.25},
        {"name": "25_32", "min_pieces": 25, "max_pieces": 32, "weight": 0.225},
    ],
    "selection_objective": {
        "overall": 0.35,
        "by_piece_band": {"2_8": 0.05, "9_12": 0.20, "13_16": 0.20,
                          "17_24": 0.10, "25_32": 0.10},
    },
    "train_shards": [
        {"name": "human_train", "path": train_npy, "weight": 1.0,
         "kind": "human_fishnet"}
    ],
    "validation_sets": [
        {"name": "human_validation", "path": val_npy, "kind": "human_fishnet"}
    ],
}
with open(config_path, "w", encoding="utf-8") as handle:
    json.dump(config, handle, indent=2)
print(f"wrote {config_path}")
PYCONF

say "epoch = $SAMPLES_PER_EPOCH samples over $TRAIN_TARGET positions; $EPOCHS epochs"
stage "training KingNet ${ACCUMULATOR}/pair$((ACCUMULATOR / 2))/${HIDDEN} for $EPOCHS epochs on GPU $GPU_ID"
nice -n "$NICE" "$PY" -m tools.train_kingnet_v11   --config "$CONFIG"   --output "$MODEL"   --manifest "$MODEL_MANIFEST" &
TRAIN_PID=$!
wait "$TRAIN_PID"
TRAIN_PID=""

stage "complete"
say "model:    $MODEL"
say "manifest: $MODEL_MANIFEST"
say "config:   $CONFIG"
say "log:      $LOG"
