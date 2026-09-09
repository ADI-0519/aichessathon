# Playing without blunders

The goal is not to find brilliant moves. At this level, games are decided by the worst move
either side plays, and the round-4 loss is the proof: it was not one missed best move but six
consecutive large errors, each compounding the last. An engine that never drops below "reasonable"
beats an engine that alternates between excellent and terrible.

This document is the strategy for getting there, in the order the work is worth doing.

## 0. Measure first

You cannot minimise what you do not count.

```bash
# play games and keep the game files
uv run python -m tools.paired_arena --candidate . --opponent champions/python_tuned \
    --base-ms 10000 --increment-ms 300 --extra-positions 14 --pgn-dir games/

# score every move against a much deeper search
uv run python -m tools.blunder_audit games/*.pgn --nodes 1000000 --skip-opening 8
```

`blunder_audit` reports average centipawn loss and counts of inaccuracies (≥50 cp), mistakes
(≥100 cp) and blunders (≥300 cp), plus the worst individual decisions with their FENs. Those FENs
are the regression suite: a fix is real when the position that produced a blunder stops producing
one.

Two things to understand about the referee:

- **It is the challenger judging itself** at a high fixed node count. It cannot see a mistake it
  would not understand at any depth, so it *under*-reports. It is still a far better judge than
  the shallow search that chose the move under a clock.
- **Node limits, not time limits.** The audit is deterministic and reproducible; two runs on the
  same PGN give the same numbers. Time-limited audits do not.

Two traps the first version of the tool fell into, both worth knowing if you extend it:

- **Mate scores are not centipawns.** Differencing them directly produced reported losses of
  29,997 cp on moves that *were* the best move. Both scores are clamped to +/-1000 cp before
  subtracting, which also stops a lost position generating a huge "loss" on every subsequent move.
- **`search_position` returns score 0 for a forced move.** That is a short-circuit, not an
  evaluation. Positions with one legal move are skipped entirely -- a forced move cannot be a
  mistake -- and where a forced move appears inside scoring, it is played through instead.

There is also a shortcut worth keeping: when the deep search's best move *is* the move that was
played, the loss is zero by definition and the second search is skipped. That is typically half the
positions in a game, so the audit runs about twice as fast as the naive version.

When a UCI engine binary is available, `tools/analyze_pgn_stockfish.py` is the stronger referee and
should replace this one. Nothing from it ships in the submission; it is a development diagnostic,
which the rules permit.

## 0b. The baseline, measured over all eleven rated games

Our side in each game was identified by reproduction -- give `champions/python_v2` the same
position and the same time the player actually spent, and see whose moves it reproduces. The
method validates 2/2 against the games whose colour is known independently (rounds 4 and 11), with
margins like 100% vs 30% in round 6. It also independently reproduces the team's own description
of the standings after round 5: one win, four losses.

**The record is 5 wins, 5 losses, 1 draw**, and it improved sharply after the trial-recovery
upload: rounds 1-5 were 1-4, rounds 6-11 were 4 wins, 1 loss and a draw.

Audited at 500,000 referee nodes, our moves only, first four plies skipped:

| | Moves | Average loss | Inaccuracies | Mistakes | Blunders |
|---|---:|---:|---:|---:|---:|
| **All our moves** | 465 | **15.3 cp** | 19 | 21 | **2** |
| In games we won | 221 | **10.3 cp** | | | 1 |
| In games we lost | 214 | **21.2 cp** | | | 1 |

### What this changes

**We are not a blundering engine.** Two blunders in 465 moves. The premise this document opened
with -- that games are decided by the worst move -- is not what the data says about *our* games.
Whatever else is worth doing, "stop blundering" is already close to solved.

**We lose to a factor of two in average move quality**, not to catastrophes: 10.3 cp per move in
wins against 21.2 cp in losses, with 21 mistakes in the 100-300 cp band against 2 blunders above
it. That is steady leakage, and it is what fitting the evaluation weights addresses. It is not what
a targeted king-safety or mobility term addresses.

**There is no phase to target.** Of the twelve worst decisions, eight are middlegame positions with
queens on, one is an opening, three are endgames -- roughly proportional to where moves get played.
No localised weakness to bolt a term onto.

**Round 4 was an outlier, not a representative sample.** At 67.7 cp average over its scored moves
it is between three and six times our normal standard, in won and lost games alike. The roadmap in
[HYPERCOMPETITIVE_ROADMAP.md](HYPERCOMPETITIVE_ROADMAP.md) and
[ROUND4_RECOVERY_PLAN.md](ROUND4_RECOVERY_PLAN.md) was built on that one game.

**Two attributions in [TRIAL_RECOVERY_REPORT.md](TRIAL_RECOVERY_REPORT.md) are wrong.** It assigns
rounds 1 and 3 by clock timing; reproduction puts round 3 at 87% white against 32% black, so its
round-3 forensics analysed the opponent's moves as ours. The round-4 analysis stands.

### The honest limit on these numbers

The referee is our own compiled engine. It cannot see a mistake it would not understand at any
depth, so the absolute figures are optimistic. The comparison between won and lost games is the
part to trust, because the same referee judges both.

## 1. The round-11 draw: blind to a draw one move away

Round 11 was drawn by threefold repetition from a position worth **+1.08 to us** at 3,000,000
nodes, with material dead level. `Re8-g8` was available at +1.18 at three separate points. We
shuffled a bishop instead.

The cause is a single condition in `champions/python_v2`, the build that played it:

```python
if ply >= 4 and board.halfmove_clock >= 8 and board.is_repetition(3):
    return 0
```

`ply >= 4` hides any repetition within four plies of the root. The engine statically evaluated
that position at **+1.38 for us**, played `Be7`, and the referee claimed the threefold
immediately. It was not indifferent to being better and it was not blind to the advantage: it
could not see that its own move ended the game.

Two hypotheses were wrong before the measurement settled it, and both would have produced bad
fixes:

- **Missing contempt.** Plausible -- a repetition scores a flat 0 and there is no contempt term
  anywhere. But the engine scored itself +1.38, so contempt would have changed nothing here. Worse,
  in positions the engine misjudges, contempt pushes it away from draws it should take.
- **Evaluation blindness.** Also wrong. It saw the advantage clearly.

**The compiled engine at the root does not have this bug.** `engine.is_repetition_draw` separates
game history (three occurrences required) from repeats inside the current search line via
`root_history_count`, with no ply floor. Replayed through the agent boundary with the real history,
it deviates at every repetition point in that game -- `Kc8` at move 35, `Rg8` at 36 and 38 -- and
`Rg8` is the move the deep search independently rates best. Round 11 would have been played on.

This is the strongest single argument for getting the compiled engine uploaded: it fixes a bug that
demonstrably cost half a point in the most recent rated game.

If the `ply >= N` shortcut appears anywhere else in a draw-detection path, it deserves the same
scrutiny. It is a natural-looking optimisation and it is wrong in exactly the case that matters.

## 2. Time thrown away — largest, cheapest, already measured

The challenger inherits the clock bug that was just removed from the Python agent. Measured on
three positions:

| Budget | Time used | Wasted | Move chosen on the round-4 position |
|---:|---:|---:|---|
| 1 s | 1.01 s | — | `b7b5` |
| 2 s | 1.04 s | **48%** | **`c8c3`** — the losing sacrifice |
| 4 s | 3.07 s | 23% | `f7f5` |
| 8 s | 7.63 s | 5% | `f7f5` |

The wasted half-second at 2 s is the difference between a sound move and the blunder that lost
round 4. Two independent causes, in `search.search_position`:

**1a. Completed work is thrown away.** `_search_root` correctly tracks its running best move and
returns it even when aborted. The driver then ignores it:

```python
if aborted:
    stopped = True
    break          # `move` is discarded; the previous depth's move is kept
```

Root ordering searches the previous best first, so any move that replaced it did so on a deeper
search. Keeping it is strictly better. This is a three-line change.

**1b. Iterations that would have finished are never started.**

```python
if remaining <= 0 or (previous_iteration_s > 0 and previous_iteration_s * 1.8 >= remaining):
```

This predictor declines to start an iteration it guesses will not finish, then sits idle. Replace
it with a soft deadline that stops *new* iterations and a hard deadline that cuts off a running
one — safe precisely because 1a keeps the partial result. This pattern is already implemented and
game-tested in the root `agent.py`; port it.

## 3. Search selectivity — where a pruning rule bets wrong

Every pruning rule is a bet that a move cannot matter. Audit them in this order.

**2a. No check extension.** In `_negamax`, `depth <= 0` drops straight into quiescence even when
the side to move is in check. Quiescence does handle evasions correctly, so this is milder than it
sounds, but a check at the horizon deserves the extra full-width ply.

**2b. No panic time.** When the aspiration search fails low, the engine has just discovered the
position is worse than it thought — the single most valuable moment to spend extra clock. Instead
it re-searches and moves on. Extending the soft deadline on a fail-low, or when the root move
changes, targets blunders directly rather than average strength.

**2c. Stopping the search when being mated.** `if abs(score) >= MATE_BOUND: break` ends iterative
deepening on any mate score, including one against us. When you are being mated you want to keep
searching for the longest defence, because opponents at this level do not always find the mate.

**2d. What is already right, and should stay that way.** Quiescence preserves every check evasion
and every checking move, exempting them from SEE and delta pruning. LMR re-searches at full depth
whenever a reduced search raises alpha. These are the conservative choices that keep the engine out
of trouble; do not trade them away for node counts.

A warning drawn from the Python work: **nominal depth is not comparable across pruning regimes.**
Adding null-move and futility pruning to `agent.py` raised reported depth from 5 to 8 and produced
no measurable strength gain, because a heavily pruned depth 8 searches less than a full depth 8.
Judge every pruning change by games and by the blunder audit, never by the depth number.

## 4. Reliability — a crash is the worst blunder available

An illegal move, a crash, an out-of-memory, a flag or an init failure loses the whole game
immediately. No amount of playing strength compensates.

**4a. The import budget, measured.** Timed on this 16-core machine, same code, same session:

| Run | Import | Headroom vs 90 s |
|---|---:|---:|
| challenger, early | 27.2 s | 62.8 s |
| root, all cores | 51.5 s | 38.5 s |
| root, pinned to ONE core | 48.4 s | 41.6 s |
| challenger, later | 66.5 s | 23.5 s |
| root, later | 67.2 s | 22.8 s |

**27 s to 67 s for the same code.** Core count is not the threat: pinning to a single core cost
nothing, because numba's compilation is serial, so the platform giving one core is fine. Machine
state is the threat, and the worst case leaves 23 s of margin on fast hardware.

Where it goes, from instrumenting all 1,816 compilations:

| Function | Compile time |
|---|---:|
| `search._search_root` | 39.1 s |
| `search._negamax` | 30.5 s (nested inside the above) |
| `engine.generate_legal_moves` | 18.5 s |
| `search._quiescence` | 14.4 s |

`_search_root` largely duplicates `_negamax`, and 22 functions are `inline="always"`, so LLVM is
optimising enormous inlined bodies. Folding the root into negamax as a flagged special case is the
one change that would meaningfully cut this, and it is a refactor of the search that currently
wins, so it needs its own careful pass.

**`cache=True` is not an option; it is actively dangerous.** Tested: the cold import writes 41
cache entries correctly, and the *warm* import then dies with
`LLVM ERROR: Symbol not found: .numba.unresolved$_ZN6search11_quiescence...`. Numba's cache cannot
round-trip the mutually recursive search (`_negamax` and `_quiescence` call each other). On the
platform that is an init failure and an automatic loss on every game after the first. The existing
`cache=False` on all 43 functions should stay, and may well be deliberate.

**What to do about it.** Upload anyway. `AGENTS.md` says the latest submission *that passed
validation* is the one that plays, so a validation failure costs an upload slot and not a single
game. That makes the platform's own validation the cheapest and most accurate test of this budget
available. The residual risk is validation passing on a fast container while a rated game later
lands on a slow one.

**3b. Keep the python-chess safety net.** Root move validation against python-chess plus a
deterministic legal fallback is what turns a compiled-engine bug into a bad move instead of a
forfeit. It is already there. It stays.

**3c. Differential fuzzing.** `tools/fuzz_numba_core.py` exists; the highest-value extension is
comparing move generation and make/unmake against python-chess over many random positions,
including the paths that are rare in play and therefore untested: en passant that would expose the
king, underpromotion, double check, castling through attacked squares, and the fifty-move and
repetition boundaries.

## 5. Evaluation — positional blunders

Neither engine's evaluation weights have ever been fitted. Material values, piece-square tables,
pawn structure and king shield are all hand-guessed constants, and there is no mobility term and no
king-attack term at all. Round 4 was lost to a mating attack the evaluation had no way to anticipate.

The fix is to fit all the weights jointly against labelled positions rather than guessing more of
them — the [Lichess open database](https://database.lichess.org/) provides CC0 games with engine
evaluations, and engine-labelled training data is explicitly permitted. A previously hand-guessed
mobility term scored 50.0%, which is evidence that guessing weights fails, not that mobility is
worthless.

This is genuinely valuable, and it is genuinely fourth. Sections 1 and 2 are hours of work against
measured defects; this is days of work against a modelled one.

## 6. Draws are a scoreboard question too

Across 162 games of local testing, every non-decisive game ended in threefold repetition: 120
checkmates, 42 repetitions, no stalemates, fifty-move draws or adjudications. 26% of games are
repetition draws. Some of that is inevitable between near-identical engines, but if a share of them
are positions where we were better, a small contempt term — decline the repetition while ahead —
converts half-points into points. With PGNs now written by `--pgn-dir`, checking the material
balance at the point of repetition settles it.

Related: the 300-ply cap adjudicates on material. When ahead, simplify and avoid shuffling; when
behind, avoid the adjudication.

## The order of work

| Priority | Change | Status | Basis |
|---|---|---|---|
| 1 | Keep partial iteration results (1a) | **tried, reverted** | see below |
| 2 | Soft/hard deadlines, drop the 1.8x predictor (1b) | **tried, reverted** | see below |
| 3 | Panic time on fail-low (2b) | **tried, reverted** | see below |
| 4 | Check extension at the horizon (2a) | **tried, reverted** | see below |
| 5 | Verify import time on constrained CPU (3a) | open | automatic loss if wrong |
| 6 | Differential fuzzing against python-chess (3c) | open, half a day | reliability |
| 7 | Fit the evaluation weights (4) | open, days | now the leading candidate |
| 8 | Separate "fuller tree" from "better evaluation" | open | decides how much of 7 to do |

## Items 1-4 were implemented, measured, and reverted

All four were built and tested against an unmodified snapshot of the same engine, 34 paired games
at 10 s + 0.3 s.

**First attempt: 38.0% (+6 =7 -12).** A clear regression. The cause was a reasoning error, not a
code defect. The measured problem was that the search left 17-50% of its *allotted* budget unspent;
the fix should have been to spend that budget. Instead the hard limit was also raised to 2.5x the
budget, a ratio carried over from the Python agent without re-justification. Simulated over 60
moves at 10 s + 0.3 s, the modified engine finished with 2.7 s on the clock where the baseline had
9.9 s -- it played the back half of every game in time trouble. The Python agent tolerated that
ratio because it is slow enough that extra seconds buy real depth. This engine is about four times
faster, so the same seconds buy far less and cost exactly as much later.

**Second attempt: 47.0% (+13 =5 -15), Elo -21, 95% CI -139 to +92.** The hard limit was capped at
exactly the original schedule's budget at every clock value, so the engine could only use time it
was already allowed. End-to-end budget use went from 55-80% to ~100%. That recovered most of the
regression and landed at parity. Not resolved at this sample size, and by the promotion gate in
[PROMOTION_TESTING.md](PROMOTION_TESTING.md) that means it does not ship. Reverted.

### What to take from this

- **Converting wasted clock into search bought nothing here.** The engine already reached depth 7
  within its budget; roughly 40% more time is a fraction of a ply. The same diminishing return that
  made the Python agent's Tier 2 pruning worthless applies to giving this engine more time.
- **Nothing transfers between engines unmeasured.** The soft/hard split was worth +105 Elo in
  `agent.py` and roughly -85 here, from the same code shape.
- **Three consecutive search changes have now produced no measurable gain** -- Tier 2 on the Python
  agent, and both versions of this. The one change that produced a large measured effect all day
  was replacing the Python engine with the compiled one. That is the case for item 7 being the real
  ceiling, and for item 8 being the cheap experiment that confirms it before days are spent.
- If items 1-4 are revisited, bisect rather than stack them. The check extension is the one that
  changes the shape of the tree rather than the clock, and is the most likely to be actively
  costing something.

Nothing is promoted without clearing the gate in [PROMOTION_TESTING.md](PROMOTION_TESTING.md), and
from now on that gate includes the blunder audit: average centipawn loss must not get worse.
