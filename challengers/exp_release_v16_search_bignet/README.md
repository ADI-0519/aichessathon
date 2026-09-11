# V16 Search + BigNet challenger

This release candidate combines two independently measured team improvements:

- the frozen V15 Search challenger, which scored 62.5% against the V14 champion
  over 12 paired development openings at 20+0.5; and
- the team's format-3 BigNet evaluator, trained on 200 million positions, which
  scored 65.6% against the previous evaluator in an otherwise identical engine
  over 24 official-clock pairs.

The engine retains the V15 two-slot main transposition table, capture history,
countermoves, and conservative singular extensions. It replaces only the NNUE
runtime and model with the team's 256-wide, eight-output-head BigNet. The
existing 75% NNUE blend is unchanged.

Initialization also pins the `is_square_attacked` Numba signature and passes
typed constants into the qsearch ordering and root negamax entry points. These
changes reduce redundant JIT specializations without changing chess semantics.

The agent includes the persistent-state emergency fallback and the V15
excluded-move safeguards. No third-party engine code, network, opening book,
tablebase, or opponent-specific move is included.

This directory is an experimental release candidate. Do not replace `current/`
until it passes differential NNUE checks, initialization and packaging gates,
and paired games against both V14 BigNet and the current champion.
