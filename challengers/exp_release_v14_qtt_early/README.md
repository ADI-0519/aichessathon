# V14 early-probe qsearch TT

Forked from `exp_release_v14_qtt`. It moves the qTT probe ahead of check
detection and move generation, allowing a qTT cutoff to avoid those costs.
The stored bounds and replacement policy are otherwise unchanged.

Repetition remains ahead of the probe because it is path-dependent. Positions
at the fifty-move threshold skip the early probe and retain terminal-before-draw
ordering, so checkmate still takes precedence over a claimable draw.

Safety rules:

- probes require the full Zobrist key and matching capped halfmove clock;
- qsearch entries use depth zero and never replace a same-position main-search
  entry;
- current-generation main-search entries also survive collision traffic;
- beta cutoffs may store a lower bound;
- nodes that skipped moves through SEE or delta pruning do not store exact or
  upper bounds;
- stored mate scores retain the existing ply normalization.

This should preserve the qTT search tree exactly. Gate it against ordinary qTT
at fixed nodes before measuring NPS; every chess result and counter except
elapsed time must match.
