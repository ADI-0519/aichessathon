# Finals V18 SearchMax

This challenger combines the Finals V18 long-horizon clock with the previously
tested V17 SearchMax selectivity:

- deferred move generation;
- stronger contextual LMR; and
- deeper reverse futility, quiet futility, and late-move pruning.

It is intentionally a high-upside challenger, not the safe default. SearchMax
was modestly positive against V16 but changes the explored tree, so it must beat
the core candidate and an external opponent before promotion.

## Finals 30-second initialization adapter

This variant starts the unchanged engine warm-up on one daemon thread and gives
the runner control after at most 20 seconds of total import time. The first move
joins any unfinished compilation before touching engine state and subtracts the
join time from the clock supplied to the time manager. This preserves one-core
execution and search semantics while preventing the previous long cold compile
from losing during the new 30-second initialization phase.
