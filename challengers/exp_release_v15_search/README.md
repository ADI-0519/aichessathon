# V15 search challenger

This experimental engine starts from the frozen V14 Runtime champion and tests
one coherent depth-and-ordering generation:

- a 2^22-entry main transposition table arranged as two-slot buckets;
- depth-, age- and bound-aware replacement within each bucket;
- bounded capture history inside the existing good/bad SEE bands;
- a countermove hint below killers and above losing captures;
- conservative single-ply singular extensions from trustworthy TT bounds.

Singular verification disables null move, reverse futility, late-move pruning
and LMR, does not update ordering histories, and cannot write a TT result for
an excluded-move node. Its TT score and bound are trusted only when the stored
halfmove-clock signature matches the node. Double extensions and singular
multicut are intentionally absent.

This is not deployable until it beats `current/` in paired games and passes the
platform-clock release gate. No third-party engine source, weights, opening
positions or opponent-specific moves are included.
