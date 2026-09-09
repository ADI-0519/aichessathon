# V8 search selectivity

`current/` (V7) with seven standard search techniques added, each behind a
compile-time flag so `tools/search_ablations.py` can price them one at a time.
Nothing else differs: with every flag set to False this build returns the same
move and score as V7 on fixed-node searches.

| flag | technique |
| --- | --- |
| `ENABLE_LMR_SCALING` | log-log late-move reductions, replacing a flat one ply |
| `ENABLE_REVERSE_FUTILITY` | return early when static eval clears beta by a depth-scaled margin |
| `ENABLE_RAZORING` | drop straight to quiescence when far below alpha at depth <= 2 |
| `ENABLE_FUTILITY` | skip quiets that cannot reach alpha at shallow depth |
| `ENABLE_LATE_MOVE_PRUNING` | skip late quiets at shallow depth by move count |
| `ENABLE_HISTORY_LMR` | reduce less for quiets with cutoff history |
| `ENABLE_SEE_PRUNING` | skip losing captures at shallow depth, before make_move |

## Measured so far

Time to reach depth 9 over four positions: 20.39s for V7, 2.58s here, -87%.

That number is **not** a strength claim. Depth is not comparable across pruning
regimes -- a pruned depth 9 sees less than an unpruned depth 9 -- and this repo
has already recorded a change that bought three plies and zero Elo. Only paired
games decide.

## Not implemented

- Singular extensions: needs an exclusion re-search through the transposition
  table and is the largest remaining item.
- Continuation history: needs a previous-move-indexed table and the plumbing to
  carry the previous move down the tree.
- Transposition buckets with aging, and a quiescence evaluation cache.
- Hard history pruning: this history table only grows, so a low score means
  "never caused a cutoff" rather than "known bad". Maluses would give it meaning
  but were measured at 40.0% and rejected.
