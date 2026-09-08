# V11: all three improvements together

Three people improved different parts of the same engine. None of the builds
carries more than two of them.

| part | author | present in |
| --- | --- | --- |
| king-bucketed evaluation, blend 75 | Immanuel | his `v9_kingnet`, Adi's `current`, our `v10`, here |
| quiescence evaluation cache | Adi | his `current`, here |
| seven search selectivity techniques | this branch | our `v8_search`, our `v10`, here |

Built from Adi's `current` (evaluation plus cache) with the seven flags grafted
on. All three weight files are byte-identical, so the evaluation is the same
network everywhere and only the search differs.

Grafting needed one repair beyond the mechanical copy: razoring calls
`_quiescence` directly, and Adi's signature takes three more arguments than the
one this branch's razoring was written against, so the cache arrays had to be
threaded through. Numba caught it at warm-up rather than silently.

## Status

Unmeasured. Imports, warms up, plays legal moves, and plays Stockfish's choice
on both positions that lost rounds 66 and 67.

The comparison that matters is against Adi's `current`, which isolates what the
seven selectivity techniques are worth on top of the evaluation and cache that
are already there.
