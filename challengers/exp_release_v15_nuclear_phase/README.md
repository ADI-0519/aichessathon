# V15 phase-protected Nuclear

Forked from `exp_release_v14_runtime`. This challenger tests a materially
stronger reverse-futility policy without applying middlegame assumptions
unchanged to sparse endings.

The existing conservative RFP parameters remain active at twelve pieces and
below. Between twelve and twenty-eight pieces, maximum depth and margins are
linearly interpolated toward the previously measured Nuclear parameters. Rich
positions therefore receive the full selective-search policy while endgames
retain the incumbent safeguards.

This is a behavior-changing search experiment. It requires critical-position
screening and direct paired games against V14 Runtime before any promotion.
