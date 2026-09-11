# exp_devbigmax_defer

`exp_v14_bignet` plus deferred move generation, ported from the dev branch (`2a94ab1`, with the
stalemate guard from `aa32bf8`). Nothing else changes; only `search.py` differs.

`_negamax` used to generate every legal move on entry, before the TT probe, rule-draw check,
reverse futility and null move -- none of which read the list. Generating first only matters for
scoring a node with no legal moves, so it now happens eagerly only when that cannot be ruled out
cheaply (in check, king and pawns only, or a king with no legal step, via `_king_can_move`), and
otherwise just before move ordering. Quiescence does the same for its capture list around stand-pat.

`_king_can_move` is exact only when not in check, which is the only place it is called: an
unattacked king is on no enemy slider's line, so stepping off cannot uncover an attack on the
destination.

Gate: `tools.fixed_node_equivalence` against `exp_v14_bignet` on 60 positions -- the 36 in
`benchmarks/suites/stalemate_mate_traps_v1.fens` (stalemate traps, mate in 1, cornered kings) and 24
balanced openings -- at 150,000 nodes: 0 mismatches in move, score, depth, nodes, qnodes, TT hits
or beta cutoffs. Search time 35.95s vs 33.05s (1.09x), measured with an arena running; re-measure on
an idle machine before quoting it.

Exact fifty-move TT matching is kept; dev's HALFMOVE_SENSITIVE relaxation was deliberately not ported.
