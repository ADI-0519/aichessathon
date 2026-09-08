# V8 search selectivity, 8 September 2026

Seven standard search techniques added to the V7 champion, each behind a
compile-time flag: log-log late-move reductions replacing a flat one-ply cut,
reverse futility, razoring, futility and late-move pruning of quiets,
history-modulated reductions, and SEE pruning of losing captures.

## Correctness

With every flag set to False the build returns identical moves and scores to
V7 on fixed-node searches over six positions, so the flags are the only
behavioural difference and the shared static-evaluation refactor is inert.

## Result

240 paired games against `current` at 4000ms+100ms, from 120 positions played
from both colours:

| | |
| --- | --- |
| score | 54.0% (+116 =27 -97) |
| paired 95% interval | 49.8% to 58.1% |
| Elo estimate | about +28 |
| failed terminations | none |

**Not resolved.** The lower bound touches 50%, so this does not yet clear the
promotion gate. It is, however, the first change measured in this repository
that trends clearly positive rather than landing on 50%, and a second 240-game
run at a different seed is running to resolve it.

Reaching depth 9 over four positions costs 2.58s against V7's 20.39s. That is a
selectivity measurement and not a strength claim: pruned depth is not
comparable to unpruned depth, and this repository already records a change that
bought three plies and no Elo.

## Import time

Measured alternately on the same machine, against the platform's 90s budget:

| build | run 1 | run 2 |
| --- | --- | --- |
| current | 60.5s | 66.1s |
| v8_search | 63.1s | 57.9s |

No measurable regression: V8 falls inside V7's own spread. An earlier 70.8s
reading for V8 was machine noise, not the added branches. Both builds sit at
58-66s under load, which is a pre-existing property worth watching rather than
something V8 introduced.

## Still to do

- Ablate the seven flags individually with `tools/search_ablations.py`. They are
  currently measured only as a bundle, and some may be neutral or harmful.
- Singular extensions, continuation history, transposition buckets with aging,
  and a quiescence evaluation cache remain unimplemented.
