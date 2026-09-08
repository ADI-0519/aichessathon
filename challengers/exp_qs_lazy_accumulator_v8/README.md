# Lazy qsearch accumulator candidate

This isolated challenger is derived from the cached V8 champion in `current/`. It changes only
when quiescence search constructs a child NNUE accumulator.

The champion eagerly updates the accumulator before making each qsearch move, even when SEE or
delta pruning immediately rejects that move. The profiling challenger measured wasted-update rates
of roughly 49–50% across six critical positions and three fixed-node budgets.

This candidate makes the move first, retains the existing gives-check safeguard, and constructs
the child accumulator only when the move will be searched. `nnue.update_after_move()` obtains the
moving and captured piece metadata from the engine's populated undo record. Pre-move and post-move
updates share one feature-delta implementation.

The candidate must satisfy these gates before promotion:

1. Exact accumulator parity with a complete rebuild for quiet moves, captures, en passant,
   promotions, capture-promotions, castling, and randomized legal sequences.
2. Exact fixed-node parity with `current/` across the critical suite.
3. A repeatable single-process NPS improvement.
4. A timed paired reliability and gross-regression screen with zero technical failures.

Do not package this directory until every gate passes and the candidate has been explicitly
promoted into `current/`.
