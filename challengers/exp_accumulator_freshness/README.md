# Accumulator freshness experiment

This challenger is an exact copy of the current champion with one correctness
change: every incremental NNUE update refreshes a stale parent accumulator
before applying move deltas.

A parent can be stale after its king crosses a bucket. Search does not promise
to evaluate every such node before visiting a child because TT returns and
qsearch-evaluation-cache hits can skip evaluation. Applying deltas to the stale
bucket silently corrupts the accumulator instead of crashing.

No search, evaluation, time-management, or model parameter is otherwise
changed. Promotion requires accumulator/rebuild parity, deterministic search
regression checks, timed paired games, and the normal technical gates.
