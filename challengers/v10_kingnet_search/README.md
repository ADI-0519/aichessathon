# V10: king-bucketed evaluation on the V8 search

Two independently measured improvements against the same V7 ancestor, combined
because they touch different code paths.

| part | source | measured |
| --- | --- | --- |
| seven search selectivity techniques | `challengers/v8_search` (this branch) | 53.6% over 480 paired games against V7, 95% interval 50.8 to 56.5, about +25 Elo |
| king-bucketed network at blend 75 | `challengers/v9_kingnet` (origin/dev, Immanuel) | 75W 53L 16D against Adi's V8, 57.6%, about +53 Elo, reported not committed |

The evaluation half is 16 king buckets over the usual 12x64 board, so
`FEATURE_COUNT` is 12,288 rather than 768 and the weights are 6.3 MB rather
than 0.43 MB. Inference cost per position is unchanged: the accumulator is
still 128 wide and still updated incrementally, with the bucket carried in an
extra slot so a king move can invalidate the row.

Adapting the V8 search to it took four edits: the blend constant, the extra
`pieces` argument `nnue.evaluate` now needs, and `ACCUMULATOR_ROW` in place of
`ACCUMULATOR_SIZE` where the stack is sized and copied.

## Why this might break the evaluator plateau

Three attempts to improve the old network bought nothing: a 3.4% better
validation loss scored 50.2%, a 256-wide accumulator 47.7%, and pure NNUE
41.2%. All three reshuffled a 768-input encoding that cannot represent where
the king is, so it cannot represent that a placement is bad *because* the king
is on e8. That is the shape of both rounds 66 and 67. King buckets change the
representation rather than the fit.

## Status

Unmeasured. It compiles, warms up, plays legal moves, and plays Stockfish's
choice on both positions that lost rounds 66 and 67. Import measured 62.0s with
the machine running ablations, so it needs an idle reading against the 90s
budget before anything is uploaded.

Adi's quiescence evaluation cache, now in his `current/`, is a third orthogonal
gain not included here.
