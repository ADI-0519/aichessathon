# V6 king-conditioned learned-evaluation challenger

This isolated challenger inherits V5's search, clock management, legal fallback,
and 50% handcrafted/neural blend. It changes only the learned evaluator's sparse
representation.

Each feature is:

```text
(friendly king square, oriented piece, oriented piece square)
```

Both kings are retained as pieces. This extends conventional HalfKP slightly,
preserves king-to-king geometry, and allows V5's 768-input model to be lifted
exactly into every king context before training. The production configuration
uses a 256-wide accumulator and a 32-wide dense layer.

`weights/model.npz` is currently a generated, untrained V5-equivalent scaffold
and is intentionally ignored by Git. It must be replaced by a team-trained
export before this challenger can be considered for promotion.

## Local correctness gate

Generate the V5-equivalent scaffold:

```bash
./.venv/Scripts/python.exe -m tools.lift_halfkp \
  --output challengers/v6_halfkp/weights/model.npz \
  --manifest benchmarks/runs/v6-halfkp-lift.manifest.json \
  --accumulator 256 \
  --hidden 32
```

Then verify feature and incremental-update parity:

```bash
./.venv/Scripts/python.exe -m tools.verify_halfkp \
  --candidate challengers/v6_halfkp
```

The verifier covers ordinary moves, captures, en passant, promotions, castling,
king moves, king captures, and a deterministic random game. Both maximum feature
and incremental errors must be exactly zero.

## Training

The existing V5 packed arrays are reused without repacking:

```bash
python -m tools.train_halfkp \
  --train benchmarks/runs/nnue-v1/train-4m.npy \
  --validation benchmarks/runs/nnue-v1/validation-500k.npy \
  --initial-model challengers/v5_nnue/weights/model.npz \
  --output benchmarks/runs/v6-halfkp-full/model.npz \
  --manifest benchmarks/runs/v6-halfkp-full/manifest.json \
  --accumulator 256 \
  --hidden 32 \
  --epochs 6 \
  --batch-size 8192 \
  --learning-rate 0.0001 \
  --freeze-dense-epochs 2 \
  --device cuda
```

The initial V5 lift is treated as epoch zero and remains the selected checkpoint
unless training improves validation probability MSE. The embedding uses sparse
gradients so the GPU updates only features present in each batch. The first
two epochs leave V5's proven dense head frozen while its lifted king buckets
begin to specialise. The extra feature channels have small random initial values and
zero outgoing weights, preserving V5's output without creating an untrainable
zero-to-zero path. A 100,000-position CPU pilot showed that `5e-4` was too aggressive;
`1e-4` improved validation after its first epoch and the protected checkpoint
correctly rejected a weaker second epoch. The export contains integer arrays
only and is approximately 25 MB.

The trainer writes to an ignored research directory. Inspect `best_epoch` in
the manifest before copying `model.npz` into this candidate's `weights`
directory.

Offline loss does not promote this challenger. It must retain acceptable full
search throughput, pass fixed-position diagnostics, beat exact V5 in timed
development pairs, and then survive untouched validation and packaging checks.
