# The integration branch, 10 September 2026

Three branches were developing in parallel against baselines that no longer
existed, and two of them independently built the same seven search techniques.
This branch is the union.

## What came from where

| from | what |
| --- | --- |
| `adi_branch` | `current/` as the deployable engine: king-bucketed evaluation, quiescence evaluation cache, and the V10 selectivity bundle. KingNet v11 with pairwise heads, the width benchmark, the training pipeline, the packer's deduplication and holdout protection, the rated-loss diagnostics through round 90. |
| `yaseen` | The phase-aware clock, the Stockfish referee tooling, the evaluation calibration study, the rated-mistake suite, parallel Parquet packing, streaming arena shards, the audit referee fix, and the V8 to V12 challengers. |

## Conflicts and how they were called

Twenty-eight files. The engine in `current/`, the actively developed diagnostic
tools, and the project's own state documents came from `adi_branch`, which is
further along on all of them. Four tools came from `yaseen`, where the work was
strictly additive: `materialize_nnue_blend` gained `--model` and reads the
accumulator width out of the weights, `paired_arena` gained the paired-interval
reporting that `paired_arena_shards` depends on, `train_nnue` gained the cosine
schedule, and `train_nnue_v1.sh` handles uv's rename of `--system-certs`.

Three files needed real merging rather than a choice:

- **`tools/pack_nnue_data.py`.** Both sides had extended it and the extensions
  were complementary. `adi_branch` added deduplication, holdout-fingerprint
  exclusion, group shuffling and material-band reporting -- all data quality.
  `yaseen` added a process pool. Deduplication and exclusion compare a position
  against everything accepted before it, so they depend on the order groups are
  consumed in; the merged version encodes groups in workers and keeps all
  order-dependent filtering in the consumer, which preserves the semantics of a
  serial scan while parallelising the expensive part.
- **`tests/test_pack_nnue_data.py`.** Both suites kept.
- **`.gitignore`.** Both sets of rules kept.

## What this invalidates

`v11_full` at +33 Elo and `v12_clock` at +45 were measured against
`challengers/adi_current`, a snapshot of `adi_branch` taken before the V10
bundle landed. That baseline no longer describes anything, since the current
engine now contains the selectivity layer those numbers were measuring. Both
comparisons need re-running against the merged `current/`, and the margins
should be expected to shrink.

## What is still only on this branch

The phase-aware clock lives in `challengers/v12_clock`, which is built on the
pre-merge engine. Porting it onto the merged `current/` is a small change --
`time_manager.py` plus the banking check in the deepening loop -- and it has not
been done here, because it should be measured on its own rather than folded in
silently.
