# V17 SearchMax challenger

This final-stage experiment preserves the complete V16 SearchBig engine and
tests one additional search-efficiency layer:

- defer full move generation until after node-level cutoffs when a legal king
  step proves that the position cannot be mate or stalemate;
- apply the same deferral to tactical qsearch generation after stand-pat;
- increase late-move reductions by changing the schedule divisor from 2.25 to
  1.75; and
- extend reverse futility, quiet futility, and late-move pruning eligibility
  from depths 4/3/3 to 6/4/4 respectively.

The deferred-generation design was independently developed on the team-combined
branch. Its V16 integration explicitly preserves singular verification's
shared move-buffer regeneration. Terminal positions retain eager generation,
and a defensive late terminal check protects the helper invariant.

Everything else remains V16: BigNet and its weights, the 75% blend, two-slot
TT, exact halfmove matching, History V2, capture history, countermoves, quiet
SEE, dynamic null move, singular extensions, corrected persistent-state agent
wrapper, and time manager.

This is deliberately an experimental challenger. Its aggressive selectivity
changes the searched tree and requires a direct official-clock match against
the frozen V16 candidate before promotion.
