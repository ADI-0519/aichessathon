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
wider network is stronger or fast enough for the competition CPU. The exporter
rejects float16 when it would change a runtime feature weight by more than the
configured quantization tolerance. Use float32 for a long training run when the
uncompressed model still fits comfortably; compact a completed model only after
the exporter accepts it and the normal runtime gates pass.

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
  --min-ply 12 \
  --group-order-seed 20260909 \
  --exclude-validation benchmarks/runs/nnue-v1/validation-500k.npy \
  --report-config benchmarks/runs/<experiment-config>.json
```

Training row groups are visited in a deterministic seeded order, while held-out
validation groups remain separate. Validation is packed first and deduplicated;
its evaluator-input fingerprints are then excluded from training. This matters
because common chess positions can recur even across different games and row
groups. Training duplicates remain allowed so the source distribution retains
its natural frequency without a large global deduplication index. When
`--report-config` is supplied, the manifest records overlap rejections and
accepted counts for the exact material bands used by training.
Fishnet's `move` column is the next human move, not the teacher's best move; the
capture filter therefore acts only as a cheap quiet-position heuristic.

Repeat `--exclude-validation` to reserve validation inputs from other source
months. Pack the designated modern holdout month first; for later training
months, exclude that modern holdout as well as the older independent validation
set. A per-month holdout that was not excluded from every other training shard
must not be used to select checkpoints.

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
fresh processes. The tool creates nonzero synthetic models whose paired hidden
units cancel exactly, ensuring every width has identical evaluation semantics
and a matching fixed-node tree while the neural loops still execute. It
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
50 MB package limit and the configured local initialization ceiling, while
labelling initialization as a local measurement rather than an official-hardware
prediction. Width selection should require comfortable init headroom and retain
acceptable full-search NPS and timed depth; synthetic weights cannot predict Elo.
Completed widths are also checkpointed in `summary.partial.json`.

The recovery checkpoint contains the current and best model states, optimizer,
manual scheduler position, Python/NumPy/Torch RNG states, validation history, and
dataset fingerprints. Set `training.resume_checkpoint` in a copied configuration
to resume an interrupted run.

Whenever validation improves, training also atomically writes
`model.best.partial.npz` beside the requested final model. This is a complete
format-3 runtime export of the best epoch so far. It can be copied into a V11
challenger and tested while later epochs continue; the `.pt` file remains the
artifact used to resume optimization.

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

The deployed 75% learned / 25% handcrafted blend was selected for the older
network and is not assumed optimal for V11. After one trained V11 file passes
the correctness and throughput gates, materialize separate 75% and 100%
candidates from that exact file. Screen 75% against the current champion first,
then screen 100% against the winning V11 candidate. No retraining is required;
the pure learned candidate also avoids the handcrafted evaluation call.

Use the V11-compatible runtime as the source while supplying the trained export
explicitly. The materializer copies every top-level Python module required by
the source candidate, including its time manager:

```bash
PY="./.venv/Scripts/python.exe"
MODEL="benchmarks/runs/kingnet-v11-mixed80m/model.npz"

"$PY" -m tools.materialize_nnue_blend \
  --source challengers/exp_kingnet_v11_big \
  --model "$MODEL" \
  --blend 75 \
  --output benchmarks/runs/candidates/kingnet-v11-256-75

"$PY" -m tools.materialize_nnue_blend \
  --source challengers/exp_kingnet_v11_big \
  --model "$MODEL" \
  --blend 100 \
  --output benchmarks/runs/candidates/kingnet-v11-256-100
```

Training artifacts remain under ignored `benchmarks/runs/`. Preserve the manifest
and run-specific configuration beside every candidate model.

## Mixed80m result

The completed 256-wide run selected epoch 16 with validation objective
`0.0075940354`. Better offline validation did not translate into stronger timed
search against the deployed Search V10 champion:

- V11 at 75% learned blend: 13 wins, 7 draws, 20 losses (41.25%);
- V11 at 100% learned blend: 11 wins, 12 draws, 17 losses (42.5%).

Both comparisons used the same 20 paired development openings at 10 seconds
plus 0.1 seconds and had zero technical failures. The half-point recovery from
removing HCE is too small and noisy to rescue the model. Both runtime candidates
are rejected for the final-day release; the data, configuration, checkpoints and
model remain useful evidence for a later retraining cycle.
