# Finals V19 RootCache

This challenger starts from the positive Finals V18 SearchMax candidate and
retains its learned evaluator, selectivity, long-horizon clock, and deferred
move generation. It adds:

- a larger exact q-evaluation cache (`2^18` entries);
- an 18-centipawn aspiration window with progressive widening; and
- earlier recognition of volatile iteration scores when allocating time.

The 75% BigNet blend and SearchMax pruning constants are deliberately unchanged.
The 50% blend already lost its paired screen, while a 90% blend and special-case
passed-pawn extension have no supporting match evidence. This remains an
experimental challenger and must beat V18 SearchMax at the official clock before
promotion.

## Finals 30-second initialization adapter

This variant starts the unchanged engine warm-up on one daemon thread and gives
the runner control after at most 20 seconds of total import time. The first move
joins any unfinished compilation before touching engine state and subtracts the
join time from the clock supplied to the time manager. This preserves one-core
execution and search semantics while preventing the previous long cold compile
from losing during the new 30-second initialization phase.
