# KingNet V11-BIG

KingNet V11-BIG is the next evaluator experiment. It keeps the proven 128-wide
king-conditioned sparse accumulator, but replaces the old single dense head with:

- pairwise interactions between the accumulator's two halves;
- independently selected material heads;
- ReLU and clipped-square activation branches;
- a material-specific output combining both branches.

The documented configuration starts at width 128 as a safe control; it is not a
preselected winner. A 16-bucket 1024-wide float32 feature table consumes roughly
the entire submission allowance before source files and other weights. Compact
storage can make it fit on disk, but every incremental update still scales with
the in-memory accumulator width and the platform still enforces its init budget.

The exporter can store the folded sparse feature table as float16 while retaining
float32 training and fixed-point runtime evaluation. This reduces its on-disk size
by roughly half. Compact storage is a packaging optimization, not a claim that a
wider network is stronger or fast enough for the competition CPU.

## Configuration is the experiment contract

Create a run-specific JSON file under ignored `benchmarks/runs/`. Dataset paths,
sampling weights, material bands, training parameters, selection weights, and
the complete piece-count-to-head mapping live in that file rather than Python.

`model.piece_head_map` contains exactly 33 entries. Entry `n` selects the head
for a position containing `n` pieces. The exporter stores this array in the model;
the runtime must load and validate it instead of embedding material thresholds.

The example uses eight heads, but neither the trainer nor export format assumes
that number. Head identifiers used for legal piece counts must be contiguous from
zero.

## Data

Use multiple source-disjoint human/Fishnet shards and keep at least one entire
source validation-only. The first serious run should target 20–50 million diverse
positions rather than repeatedly sampling one old four-million-position shard.
Engine-labelled positions can be included as a minority distribution, not as a
replacement for human-game positions.

The configured material sampler should deliberately cover the weak 9–16-piece
range. Horizontal reflection is training-only and costs nothing during search.

Prepare each source with the audited packer:

```bash
PY="./.venv-training/Scripts/python.exe"

"$PY" -m tools.pack_nnue_data \
  --source benchmarks/suites/sources/<source>.parquet \
  --train-output benchmarks/runs/kingnet-v11-data/<source>_train.npy \
  --validation-output benchmarks/runs/kingnet-v11-data/<source>_validation.npy \
  --manifest benchmarks/runs/kingnet-v11-data/<source>_pack_manifest.json \
  --train-target 4000000 \
  --validation-target 500000 \
  --validation-groups 1 \
  --min-ply 12
```

The trainer rejects reused paths, hard links, and byte-identical train/validation
files. Every input is hashed before optimization so a run cannot silently change
its data halfway through its provenance record.

## Training

Create `benchmarks/runs/kingnet-v11-mixed20m.json` using the example below and
replace its dataset paths with the packed, source-disjoint shards for the run.
Do not commit this machine-specific configuration.

```bash
./scripts/train_kingnet_v11.sh \
  benchmarks/runs/kingnet-v11-mixed20m.json \
  benchmarks/runs/kingnet-v11-mixed20m
```

The required structure is:

```json
{
  "model": {
    "accumulator": 128,
    "hidden": 32,
    "pairwise_width": 64,
    "cp_scale": 400.0,
    "piece_head_map": [
      0, 0, 0, 0, 0, 0, 0, 0, 0,
      1, 1, 1, 1,
      2, 2, 2, 2,
      3, 3, 3, 3,
      4, 4, 4, 4,
      5, 5, 5,
      6, 6, 6,
      7, 7
    ]
  },
  "export": {
    "feature_storage": "float16"
  },
  "training": {
    "epochs": 8,
    "samples_per_epoch": 20000000,
    "batch_size": 8192,
    "learning_rate": 0.0003,
    "min_learning_rate": 0.00001,
    "warmup_fraction": 0.05,
    "weight_decay": 0.00001,
    "mirror_probability": 0.5,
    "gradient_clip_norm": 1.0,
    "seed": 20260909,
    "device": "cuda",
    "init_model": null,
    "resume_checkpoint": null
  },
  "piece_bands": [
    {"name": "2_8", "min_pieces": 2, "max_pieces": 8, "weight": 0.10},
    {"name": "9_12", "min_pieces": 9, "max_pieces": 12, "weight": 0.225},
    {"name": "13_16", "min_pieces": 13, "max_pieces": 16, "weight": 0.20},
    {"name": "17_24", "min_pieces": 17, "max_pieces": 24, "weight": 0.25},
    {"name": "25_32", "min_pieces": 25, "max_pieces": 32, "weight": 0.225}
  ],
  "selection_objective": {
    "overall": 0.35,
    "by_piece_band": {
      "2_8": 0.05,
      "9_12": 0.20,
      "13_16": 0.20,
      "17_24": 0.10,
      "25_32": 0.10
    }
  },
  "train_shards": [
    {
      "name": "human_train",
      "path": "kingnet-v11-data/human_train.npy",
      "weight": 1.0,
      "kind": "human_fishnet"
    },
    {
      "name": "engine_train",
      "path": "kingnet-v11-data/engine_train.npy",
      "weight": 0.35,
      "kind": "strong_engine_labelled"
    }
  ],
  "validation_sets": [
    {
      "name": "human_holdout",
      "path": "kingnet-v11-data/human_validation.npy",
      "weight": 1.0,
      "kind": "human_fishnet_holdout"
    },
    {
      "name": "engine_holdout",
      "path": "kingnet-v11-data/engine_validation.npy",
      "weight": 0.5,
      "kind": "strong_engine_labelled_holdout"
    }
  ]
}
```

Paths are resolved relative to the run configuration, so the example assumes
the JSON file and `kingnet-v11-data/` are both under `benchmarks/runs/`.

## Width-cost benchmark

Before the expensive training run, benchmark 128, 256, 512, and 1024 widths in
fresh processes. The tool creates zero-valued synthetic models, ensuring every
width has identical evaluation semantics and a matching fixed-node tree. It
measures model/candidate size, model loading, JIT warmup, evaluation and update
throughput, fixed-node NPS, and completed depth under a wall-time limit.

The benchmark reads only the `model` and `export` sections of the run JSON, so
the referenced datasets do not need to exist yet:

```bash
PY="./.venv/Scripts/python.exe"

"$PY" -m tools.kingnet_width_benchmark \
  --config benchmarks/runs/kingnet-v11-mixed20m.json \
  --widths 128,256,512,1024 \
  --nodes 100000 \
  --wall-time-s 1.0 \
  --output benchmarks/runs/kingnet-v11-width-cost
```

Run it alone on an idle machine. Each width pays a fresh Numba compilation and
the complete run can take several minutes. `summary.json` explicitly flags the
50 MB package and 90-second initialization gates. Width selection should use the
largest candidate that retains acceptable full-search NPS, timed depth, and init
headroom; synthetic weights cannot predict Elo.

The recovery checkpoint contains the current and best model states, optimizer,
manual scheduler position, Python/NumPy/Torch RNG states, validation history, and
dataset fingerprints. Set `training.resume_checkpoint` in a copied configuration
to resume an interrupted run.

`training.init_model` accepts our format-v2 KingNet as an optional accumulator
warm start. The V11 pairwise material heads are new and remain freshly initialized.
Use warm-start and from-scratch runs as separate experiments.

## Model selection

Every validation set reports overall and per-material-band:

- probability MSE;
- centipawn MAE and RMSE;
- prediction-to-teacher calibration slope.

`selection_objective` controls checkpoint selection. It combines overall and
per-band probability MSE using explicit JSON weights, then combines independent
validation sets using their own weights. This prevents a large opening population
from concealing a regression in the 9–16-piece bands.

The architecture name in the manifest is derived from the configured accumulator,
pairwise width, hidden width, and head count. It is not a fixed label.

## Promotion gates

An offline loss improvement is necessary but insufficient:

1. Validate the exported format, array shapes, head map, finite values, and
   fixed-point/runtime parity.
2. Compare calibration against the deployed model overall and by material band.
3. Run the rated critical-position suite as a regression veto.
4. Measure single-process evaluation throughput and full-search NPS.
5. Run a short paired screen against the search champion, then an untouched split.
6. Run an official-clock safety pair and packaging inspection before promotion.

Training artifacts remain under ignored `benchmarks/runs/`. Preserve the manifest
and run-specific configuration beside every candidate model.
