# Combining three branches' work, 9 September 2026

Three improvements to the same engine existed in three branches. All of them
use byte-identical king-bucketed weights, so the evaluation is settled and only
the search differs.

| part | author | in |
| --- | --- | --- |
| king-bucketed evaluation, blend 75 | Immanuel | `v9_kingnet`, Adi's `current`, `v10`, `v11` |
| quiescence evaluation cache | Adi | Adi's `current`, `v11` |
| seven search selectivity techniques | this branch | `v8_search`, `v10`, `v11` |

## Measured

240 paired games each at 4000ms+100ms unless noted.

| comparison | score | 95% interval | Elo | verdict |
| --- | --- | --- | --- | --- |
| `v8_search` vs V7 | 53.6% over 480 | 50.8 to 56.5 | +25 | resolved |
| `v10` vs V7 | 54.8% | 50.6 to 59.0 | +33 | resolved |
| `v10` vs `v8_search` | 53.3% | 49.4 to 57.2 | +23 | not resolved |

Correcting for init failures, which fall almost entirely on the side carrying
the larger weights, `v10` vs V7 becomes 55.7% and `v10` vs `v8_search` 53.8%.

## The gains are partly additive, not fully

The search is worth about +25 over V7 on its own. The evaluation adds about
+23 to +27 on top of it. Those should compose to roughly +50, and the direct
measurement of `v10` against V7 is +33 to +40. So most of each gain survives
being combined, but not all of it: both improvements ultimately buy better
moves per unit time, and they compete for the same headroom.

Anyone planning on +25 and +53 adding to +78 should not.

## Init failures are now a real risk

Across the two `v10` arenas, seven games were lost to an agent failing to
import inside the harness's 90 second budget, and all but one of them were the
side carrying the 6.3 MB king-bucketed weights rather than the 0.43 MB ones.
Idle imports measure 41 to 50 seconds, but CLAUDE.md records identical code
varying between 27 and 67 seconds, and an init failure on the platform is a
lost game rather than a slow one.

This applies to every build carrying these weights, including Adi's `current`.
Measure the worst of several imports on a quiet machine before uploading, not
the mean.
