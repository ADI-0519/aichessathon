# V15 TT2 challenger

This experimental engine starts from the frozen V14 Runtime champion and tests
one bundled main transposition-table upgrade:

- a 2^22-entry main transposition table arranged as two-slot buckets;
- depth-, age- and bound-aware replacement within each bucket;
- no capture-history, countermove or singular-extension changes.

Relative to V14 this changes both associativity and total capacity (2^20 to
2^22 entries). It is a strength candidate, not an associativity-only ablation.

This is not deployable until it beats `current/` in paired games and passes the
platform-clock release gate. No third-party engine source, weights, opening
positions or opponent-specific moves are included.
