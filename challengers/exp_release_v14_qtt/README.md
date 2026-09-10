# V14 qsearch TT

Forked from `exp_release_v14_runtime` and adds qsearch bounds to the existing
transposition table. It is a selective-search experiment, not an exact-speed
transformation.

Safety rules:

- probes require the full Zobrist key and matching capped halfmove clock;
- qsearch entries use depth zero and never replace a same-position main-search
  entry;
- current-generation main-search entries also survive collision traffic;
- beta cutoffs may store a lower bound;
- nodes that skipped moves through SEE or delta pruning do not store exact or
  upper bounds;
- stored mate scores retain the existing ply normalization.

`SearchResult` exposes qTT probes, hits, cutoffs, and stores. Gate this build
first for legal/technical correctness, then compare it against V14 Runtime as
an isolated strength experiment.
