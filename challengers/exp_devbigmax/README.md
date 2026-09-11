# exp_devbigmax

The high-ceiling dev lane: `exp_devbigmax_defer` (deferred move generation, tree-identical to
`exp_v14_bignet`) plus dev's two selectivity changes. Only `search.py` differs from `exp_v14_bignet`.

| | exp_v14_bignet | this | dev |
|---|---|---|---|
| LMR table | `0.75 + ln d * ln i / 2.25` | `/ 1.75` | `/ 1.75` |
| reverse futility max depth | 4 | 6 | 6 |
| quiet futility max depth | 3 | 4 | 4 |
| late-move pruning max depth | 3 | 4 | 4 |
| TT fifty-move matching | exact | exact | relaxed below 70 -- not ported |

Margins are V14's (`V10_*_MARGIN_*`), now applied at the deeper limits; LMP's limit is the
formula `2 + d^2 + d`, so depth 4 needs no table. dev's null-move, capture pruning, history, TT
layout, agent, engine, nnue and weights were not imported; V14 already has more mature versions.

Smoke, fixed 150,000 nodes vs `exp_v14_bignet` on 8 balanced openings: trees differ as intended,
one ply deeper in 4 of 8, 1.13x faster wall-clock. Strength is untested until its arena reports.
