# V7 continuous time

This challenger copies `challengers/v6_stable_timeout` exactly and changes
only clock allocation. It removes the 60-second and 10-second discontinuities,
uses the FEN fullmove number for a bounded moves-to-go estimate, and models only
part of the fixed increment. A hard reserve and low-clock taper remain in place.

The schedule is intended to spend more of the clock in late middlegames without
changing evaluation, search, persistent state, or the stable-timeout policy.
