# NNUE training results, 2026-09-07

Retrained the learned evaluator on the same public labels V5 was built from, and measured
the results in paired games rather than by validation loss. **Nothing was promoted.**
`challengers/v5_nnue` remains the submission.

## What was run

`scripts/train_nnue_v1.sh` packs 4,000,000 positions from `standard_rated_2014_09.parquet`
(Lichess `fishnet-evals`, SHA-pinned) and trains a 128x32 net. `scripts/train_nnue_sweep.sh`
then trains variants against that same packed data, one factor at a time.

| run | epochs | schedule | accumulator | val MSE | val MAE | params |
| --- | --- | --- | --- | --- | --- | --- |
| nnue-v1 baseline | 8 | constant | 128 | 0.008205 | 175.3 cp | 106,689 |
| a128-e24-cosine | 24 | cosine | 128 | 0.007926 | 161.4 cp | 106,689 |
| a256-e24-cosine | 24 | cosine | 256 | 0.007896 | 160.0 cp | 213,313 |

The baseline was **underfitting**: validation MSE fell on every one of its 8 epochs and never
turned up, so the epoch budget -- not capacity -- was binding. Training 24 epochs with cosine
decay took 3.4% off the MSE at zero inference cost. Doubling the accumulator on top of that
took a further 0.4%, while the train/validation gap widened from 0.0013 to 0.0023: the extra
capacity went into memorising 4M positions, not into generalising.

## What the games said

240 paired games per candidate against `challengers/v5_nnue` at 4000ms+100ms, from 120
positions played from both colours. Each candidate is `v5_nnue` with exactly one file changed
(`weights/model.npz`, plus the width constant the model implies), so the result is
attributable to the net alone.

| candidate | score | 95% interval | Elo |
| --- | --- | --- | --- |
| nnue_a128_e24 | 50.2% (+102 =37 -101) | 46.6% to 53.8% | ~ +1 |
| nnue_a256_e24 | 47.7% (+101 =27 -112) | 44.4% to 51.0% | ~ -16 |

**A 3.4% better validation loss and a 13.9 cp better MAE bought no measurable Elo.** The
interval is tight enough (+/- 3.6%) to rule out a gain above roughly 25 Elo, so this is a
result, not an unresolved measurement. The wider net is, if anything, slightly worse -- it
pays about 7% of the node rate for 0.4% of validation loss.

## Why, most likely

`search.NNUE_BLEND = 50`: the learned evaluator supplies only half of the static evaluation
and the hand-tuned evaluator supplies the other half. An improvement to the net is therefore
halved before the search ever sees it. Two consequences worth testing before training anything
further:

1. **Raise the blend.** This is a source constant, so `tools/materialize_nnue_blend.py` can
   build candidates at 75 and 100 with no retraining. If a better net still buys nothing at
   blend 100, the net is not the lever and the search is.
2. **More data, not more parameters.** The seven training row groups hold only ~4.2M usable
   positions and we packed 4.0M of them, so this Parquet is exhausted. Another month of
   `fishnet-evals` is a ~94 MB download and now costs ~3 minutes to repack.

## Measurement notes

Four games ended `- by init`: the agent failed to import inside `harness.rules.INIT_BUDGET_S`
(90s, the platform's real budget). All four were the candidate, and all four were black. Both
agents in a local game start and JIT-compile at the same moment, so under a 12-shard run they
contend for cores and the loser of that race can exceed 90s. The real platform runs one agent
per machine, so this is an artifact of parallel local testing, not a defect in the nets --
but it is scored as a candidate loss, and excluding those games moves the scores to 50.4% and
48.3%, which changes no conclusion.

Import time remains the thing to watch on its own account. Measured today on a loaded machine:
47.7s for the 128-wide build, 50.0s for the 256-wide one, against the 90s budget. The width
costs ~2.3s; the run-to-run variance recorded in CLAUDE.md (27s to 67s) is far larger and is
the real risk.
