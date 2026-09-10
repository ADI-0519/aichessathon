# V12 last-day release candidate

This candidate keeps the current KingNet75 model, qsearch evaluation cache and
validated Search V10 base. It combines four independently developed changes:

- signed bounded history with maluses and conservative quiet SEE pruning;
- adaptive soft, normal and hard iterative-deepening deadlines;
- mandatory refresh of stale king-bucket accumulators before incremental use;
- post-pruning qsearch accumulator updates, avoiding feature work for rejected
  children.

No opening positions, opponent moves, third-party engine code or third-party
weights are embedded. This directory is an experiment until correctness,
fixed-node and timed-match gates pass.
