# Finals V18 core

This challenger starts from the accepted V16 Search + BigNet build and combines
two independent changes:

- deferred legal-move generation at nodes where rule-draw and static cutoffs may
  return first; and
- a continuous long-game clock horizon that does not assume sparse positions end
  quickly.

The deferred-generation implementation comes from the team's V17 work, but its
aggressive LMR and deeper RFP, quiet-futility, and LMP settings are reset to the
frozen V16 values. It is intended to preserve the V16 fixed-node tree.

The time manager retains V16's soft/normal/hard deadline structure, reserve,
increment credit, maximum move budgets, completed-iteration stability checks,
and legal emergency fallback. Only the estimate of remaining decisions changes.

This directory is a local challenger. It is not the canonical engine and must not
be packaged for the final until exact-tree, clock, agent-contract, and paired-game
gates pass.

## Finals 30-second initialization adapter

This variant starts the unchanged engine warm-up on one daemon thread and gives
the runner control after at most 20 seconds of total import time. The first move
joins any unfinished compilation before touching engine state and subtracts the
join time from the clock supplied to the time manager. This preserves one-core
execution and search semantics while preventing the previous long cold compile
from losing during the new 30-second initialization phase.
