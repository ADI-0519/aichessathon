# V6 stable timeout

This challenger is an exact copy of `challengers/v5_nnue` except for one
search policy: when a root iteration is interrupted, it returns the move from
the last fully completed depth. V5 can instead promote a move from the partial
iteration even though that move has not been compared with every legal root
alternative.

The experiment is intentionally isolated. Its evaluator, weights, time
allocation, transposition table, history heuristic, pruning, and move ordering
are unchanged from V5.
