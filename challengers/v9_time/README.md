# V9 adaptive time control

V7 with a position-aware clock. `time_manager.hard_budget_ms` adds a ceiling
above the flat target, and the deepening loop moves between them:

- **Bank.** When the root move has not changed for `TIME_SETTLED_DEPTHS`
  completed depths and the score is steady, stop early and leave the rest of the
  budget for a later move.
- **Extend.** When a completed depth returns a score `TIME_INSTABILITY_DROP`
  below the previous one, re-arm the abort timer at the hard budget, once per
  move, and only past `TIME_EXTEND_MIN_DEPTH` with at least
  `TIME_EXTEND_MIN_CLOCK_MS` on the clock.

All of it is behind `ENABLE_ADAPTIVE_TIME`.

## Safety

The first tuning would have lost on time. Simulated against a decreasing clock
it spent 106.8s over 20 moves and had 23.2s left, against V7's 44.1s over 15
moves. Extending on any root-move change is far too generous: this search
changes its root move constantly at depth 7, which is iterative deepening
working rather than the position being hard.

Restricted to a real score drop, the same simulation spends **39.8s and ends
with 87.7s**, against V7's **44.1s and 83.4s** -- slightly *cheaper* than V7,
because banking fires and extension does not. Maximum single move 4.04s, equal
to V7.

## The motivating case is refuted

This was built to fix two rated losses where we spent the 4.5s cap on the
losing move. On both positions it changes nothing:

| position | V7 | V9 |
| --- | --- | --- |
| R66 before 11...Ne5 (-464 cp) | g4e5 in 3.85s | g4e5 in 3.84s |
| R67 before 17...Nd4 (-315 cp) | c6d4 in 4.11s | c6d4 in 4.12s |

The extension never fires, because the search never sees a score drop: it plays
both moves confidently. Instability time control can only buy time when the
engine notices it is in trouble, and here it does not. Being confidently wrong
is invisible to a clock.

So this build is unproven and its original justification is gone. What remains
is the banking half, which is real but small, and an extension that would help
only on positions where the search does detect a drop. It should be measured on
that basis, not on the two games that prompted it.
