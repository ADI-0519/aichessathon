# V14 runtime

Forked from `exp_release_v13_search`. This is an exact-speed experiment: it
must not intentionally change the searched tree.

Changes:

- delays main-search NNUE feature updates until a move survives pruning;
- refreshes the parent accumulator before making the move, preserving the
  stale-bucket invariant;
- removes the redundant depth-one perft call from import warmup because the
  legal-move and legal-capture entry points are already warmed directly.

Acceptance requires deterministic fixed-node parity with V13 Search, followed
by a repeatable single-worker NPS gain. Match strength alone is not sufficient.
