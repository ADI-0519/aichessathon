# Python engine, search-tuned

The `champions/python_v2` engine with the search work from commits `2e45646` and `147e600`:
precomputed piece-square tables, bitboard evaluation, capture-only quiescence generation,
partial iteration results, soft/hard time limits, a bucketed transposition key, check
extensions, null-move pruning and static exchange evaluation.

It scored **64.7% against `python_v2`** over 58 paired games at 4 s + 0.1 s, Elo +105
(95% CI +29 to +192).

It is not the submission: the compiled engine at the repository root beat it 89% (+18 =5 -0)
over 23 paired games at 10 s + 0.3 s. It is kept as the strongest pure-Python opponent for
`tools/paired_arena.py`, and as a fallback if the compiled engine ever fails validation.
