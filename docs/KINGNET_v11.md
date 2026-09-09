KingNet V11 training lane

This directory keeps the deployed KingNet architecture unchanged and improves the
training pipeline instead.  The goal is to get a submission-compatible model with
complete provenance and much broader coverage before experimenting with a larger
runtime architecture.

What the current KingNet evidence actually proves

The shipped V9/current model has SHA-256

9348d4e0ca5e2ee10c11003e7363953316e721df16db3bdafc272451e7550087.

The repository retained the associated factored trainer, but not the manifest
that links that exact export to exact dataset hashes, hyperparameters, and best
validation epoch.  The retained V9 README points at the old NNUE V1 packed corpus
(train-4m.npy / validation-500k.npy), so that is the intended/reconstructed
training route, not auditable proof for the exact bytes above.

The NNUE V1 corpus is produced from Lichess fishnet-evals human-game positions.
The checked-in V1 script used standard_rated_2014_09.parquet, packed 4,000,000
training rows and 500,000 row-group-disjoint validation rows.  Packing removes
early positions before ply 12, illegal/terminal/check positions, and positions
whose provided teacher move is a capture; centipawn labels are clamped to ±2000.

Why V11 is a data/training experiment first

The current runtime is already cheap enough to search.  Increasing the accumulator
from 128 to 1024 would make the exported 16-bucket float32 feature table about
50 MiB by itself, before the rest of the agent, so directly imitating a much wider
network is incompatible with the competition's 50 MiB unpacked budget without a
separate compression/runtime project.

V11 therefore keeps:

16 king buckets;

128-wide factored accumulator;

32-wide dense hidden layer;

the exact format-v2 export consumed by current/nnue.py.

It changes the training distribution and training quality:

any number of memory-mapped packed shards;

explicit per-shard weights;

explicit piece-count mixture;

optional horizontal mirror augmentation;

cosine LR decay with warmup;

named independent validation sets;

per-material-band validation metrics;

atomic best checkpoints;

complete SHA-256 provenance for every input and output.

No month, file path, material band, or sampling ratio is embedded in Python code.
Those choices live in a JSON experiment config.

Recommended first serious run

Prefer several different human-game/Fishnet months rather than taking 20M adjacent
rows from one month.  Keep at least one separate month completely validation-only.
A sensible first target is 20M unique packed training rows across ~5 source shards,
then sample 20M examples per epoch for 8-10 epochs.  The example config deliberately
upweights the 9-12 and 13-16 piece bands because the platform post-mortems identified
those as weak bands; treat those weights as an experiment config, not engine logic.

Horizontal mirroring is training-only.  The packed feature representation contains
piece-square features but no castling-right/en-passant feature, so file reflection
is a symmetry of what this evaluator can observe.  It costs nothing at inference.

Do not make a pure self-play/engine-position corpus the only source.  If a
self-play-labelled shard is available, add it as another explicitly weighted source
and keep human-position validation separate.

Preparing shards

Use the existing audited packer once per source Parquet.  Example:

PY="./.venv-training/Scripts/python.exe"

"$PY" -m tools.pack_nnue_data \
  --source benchmarks/suites/sources/<month>.parquet \
  --train-output benchmarks/runs/kingnet-v11-data/<month>_train.npy \
  --validation-output benchmarks/runs/kingnet-v11-data/<month>_validation.npy \
  --manifest benchmarks/runs/kingnet-v11-data/<month>_pack_manifest.json \
  --train-target 4000000 \
  --validation-target 500000 \
  --validation-groups 1 \
  --min-ply 12

For training months, leave their validation output unused or use it only for
diagnostics.  The final model-selection validation sets should be source-disjoint
from the training months.

Copy configs/kingnet_v11.example.json to a run-specific config and point it at
the packed files that actually exist.

Training

PY="./.venv-training/Scripts/python.exe"
RUN="benchmarks/runs/kingnet-v11-mixed20m"

mkdir -p "$RUN"

"$PY" -m tools.train_kingnet_v11 \
  --config configs/kingnet_v11.json \
  --output "$RUN/model.npz" \
  --manifest "$RUN/manifest.json" \
  --checkpoint "$RUN/best.pt"

For a clean-provenance candidate, leave training.init_model as null.  The
trainer also supports a format-v2 KingNet warm start; it exactly decomposes the
exported bucket table into a shared factor plus residuals.  Use that only as a
separate experiment because it inherits the provenance of the starting model.

Promotion gates

A better validation loss is not enough.  Before any game test:

tools.verify_kingnet must pass against a challenger containing the new model.

Report validation MSE/MAE separately for:

the untouched human holdout;

9-12 pieces;

13-16 pieces;

17-24 pieces;

25-32 pieces.

Compare evaluator calibration/slope against the current model.

Run the rated critical-position suite as a regression veto, not as a tuning set.

Only then run a short paired game screen against the current search champion.

The manifest written by tools.train_kingnet_v11 is the provenance record that
the current V9 model is missing.  Preserve it beside every candidate.