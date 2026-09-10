# V15 capture-history challenger

This experimental engine starts from the V15 TT2 arm and isolates capture
ordering feedback:

- a 2^22-entry main transposition table arranged as two-slot buckets;
- depth-, age- and bound-aware replacement within each bucket;
- bounded capture history inside the existing good/bad SEE bands;
- no countermoves or singular extensions.

This is not deployable until it beats `current/` in paired games and passes the
platform-clock release gate. No third-party engine source, weights, opening
positions or opponent-specific moves are included.
