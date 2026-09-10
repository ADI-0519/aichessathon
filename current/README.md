# V14 Runtime champion

This is the canonical promoted copy of `challengers/exp_release_v14_runtime`.
It keeps the KingNet75 model, qsearch evaluation cache and V12 safety work,
then adds the V13 core/search changes:

- material-aware time horizon, a larger direct-mapped TT, and safe partial-root recovery;
- log-log interior LMR with contextual corrections;
- guarded root LMR with full-depth verification;
- post-pruning main-search NNUE updates and removal of redundant perft warm-up.

No opening positions, opponent moves, third-party engine code or third-party
weights are embedded. V14 Runtime scored 64.17% over 30 paired development
openings against frozen V12, with zero technical failures.
