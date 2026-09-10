# V15 TT2 challenger

This experimental engine starts from the frozen V14 Runtime champion and
isolates the main transposition-table change:

- a 2^22-entry main transposition table arranged as two-slot buckets;
- depth-, age- and bound-aware replacement within each bucket;
- no capture-history, countermove or singular-extension changes.

This is not deployable until it beats `current/` in paired games and passes the
platform-clock release gate. No third-party engine source, weights, opening
positions or opponent-specific moves are included.
