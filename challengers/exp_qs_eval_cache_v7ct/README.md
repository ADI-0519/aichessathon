# Experimental V7 quiescence evaluation cache

This challenger is an exact-search-throughput experiment derived from `current/`. The only engine
behavioural change is in `search.py`: a 65,536-entry direct-mapped cache stores the blended static
evaluation used by quiescence search.

The cache stores only a complete Zobrist key and an `int32` evaluation. It does not store moves,
alpha-beta bounds, draw decisions, or terminal results. A separate validity array handles the
otherwise ambiguous all-zero entry, and `SearchMemory.clear()` invalidates every entry between
games. Probe and hit counters are development diagnostics only.

Initial isolated evidence:

- exact move, score, depth, node, qnode, TT, beta-cutoff, and LMR parity across the six V5 priority
  positions at 100,000 nodes;
- exact parity across six repeated starting-position probes at 100,000 and 300,000 nodes;
- median starting-position NPS gains of 7.19% and 7.29%, with 20.6% and 22.9% cache hit rates.

This is not the deployable champion until broader throughput measurements and technical games pass.
Do not package it as `current/` merely from this microbenchmark.
