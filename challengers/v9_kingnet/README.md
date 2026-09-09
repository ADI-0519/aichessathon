# V9 king-conditioned evaluator

This immutable challenger was imported from teammate branch `dev` at commit
`576348b` (`raise nnue blend to 75 in v9 challenger`). It is an experiment,
not the deployable champion.

V9 retains the frozen V7 search and continuous time manager, while replacing
the 768-feature learned evaluator with a king-conditioned network:

- 16 king buckets per perspective;
- 12,288 sparse input features (`16 * 12 * 64`);
- a 128-wide incremental accumulator;
- a `256 -> 32 -> 1` clipped dense head;
- lazy rebuilding of only the perspective whose king crosses a bucket;
- a 75% learned / 25% handcrafted evaluation blend.

The bundled model is 6,327,088 bytes and has SHA-256 digest
`9348d4e0ca5e2ee10c11003e7363953316e721df16db3bdafc272451e7550087`.
The `dev` branch does not contain a manifest that connects this exact export
to training-data hashes and validation metrics. Preserve that evidence before
considering the model for a competition submission.

The associated factored training pipeline is retained in
`tools/king_features.py` and `tools/train_king_factored.py`. It accepts the
same packed datasets as the existing NNUE trainer and writes both a model and
a training manifest. Always train to a new experiment directory; never
overwrite this frozen challenger in place.

```bash
uv run python -m tools.train_king_factored \
  --train benchmarks/runs/nnue-v1/train-4m.npy \
  --validation benchmarks/runs/nnue-v1/validation-500k.npy \
  --output benchmarks/runs/kingnet/model.npz \
  --manifest benchmarks/runs/kingnet/manifest.json
```

Verify incremental updates and fixed-point inference before a match:

```bash
uv run python -m tools.verify_kingnet \
  --candidate challengers/v9_kingnet
```

Compare it first with frozen V7. That isolates evaluator quality because V9
does not contain the V8 qsearch evaluation cache in `current/`:

```bash
uv run python -m tools.backtest \
  --candidate challengers/v9_kingnet \
  --opponent challengers/v7_continuous_time \
  --suite benchmarks/suites/openings_8moves_v3_500.epd \
  --split development \
  --limit 20 \
  --base-ms 10000 \
  --increment-ms 100 \
  --workers 2 \
  --output benchmarks/runs/v9-kingnet-vs-v7-development-20
```

If V9 passes that gate, port only its evaluator into a fresh copy of
`current/` and compare the combined candidate with `current/`. Do not promote
this V7-based directory directly.
