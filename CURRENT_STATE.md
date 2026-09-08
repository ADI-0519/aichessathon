# Current engine state

Updated: 8 September 2026

## Deployable champion

`current/` is the only canonical deployable source directory. The default play, arena, backtest,
diagnostic, type-checking, and packaging commands all target it. Packaging copies the contents of
that directory to the archive root, where the platform imports `agent.py`.

- Engine generation: V7 continuous time plus exact qsearch evaluation caching
- Frozen source ancestor: `challengers/v7_continuous_time/`
- Rollback artifact: `submission_v7.zip`
- Fresh local artifact: `submission.zip` (gitignored and rebuilt with `make zip`)
- Lineage: V5 NNUE -> V6 stable completed-depth timeout -> V7 continuous allocation -> qsearch evaluation cache

The current source adds the independently implemented, fixed-node-equivalent qsearch evaluation
cache to the frozen V7 engine. The model weights are byte-identical. Engine files no longer live
at the repository root, and the repository root must not be packaged as an agent.

## What V7 contains

- Team-trained, incrementally updated 768-input NNUE blended 50:50 with the handcrafted evaluator.
- Numba-compiled board and alpha-beta search with iterative deepening and principal-variation
  search.
- A fixed-size array transposition table, aspiration windows, quiescence search, SEE move scoring,
  killer moves, quiet history, late-move reductions, and guarded null-move pruning.
- A 65,536-entry direct-mapped cache for exact blended static evaluations reached in quiescence.
- Persistent per-game search memory and repetition history.
- Last-completed-iteration timeout safety and continuous, move-aware clock allocation.

It does not contain pondering, an opening book, Syzygy tablebases, a lazy NNUE accumulator, or
code/model assets copied from another engine.

## Strength evidence

| Test | Result | Interpretation |
|---|---:|---|
| V5-50 vs exact V4, development, 20 pairs | 73.75% | Established the 50% learned blend as a strong V5 candidate. |
| V5-50 vs exact V4, validation, 20 pairs | 81.25% | Positive result with no technical failures; the positions are no longer untouched. |
| V6 stable timeout vs V5, development, 20 pairs | 53.75% | Non-regression plus an exact repair of the round-47 incomplete-iteration failure. |
| V7 vs V6 stable timeout, development, 20 pairs | 65.0% | Strong directional development result with no failures. |
| V7 vs V6 stable timeout, validation, 20 pairs | 48.75% | No independent evidence of a general Elo gain. |
| V7 vs submitted V5, validation, 10 pairs | 50.0% | No measured regression or superiority. |
| Qsearch evaluation cache, clean fixed-node suite | Exact parity over 90 probes; +7.50%, +4.49%, and +5.49% median NPS at 100k, 300k, and 1M nodes | Repeatable semantic-preserving throughput gain. |
| Qsearch evaluation cache vs prior V7, development, 20 pairs | 51.25%, zero technical failures | Passed the timed gross-regression and reliability screen; too few games to claim Elo. |

V7 is therefore a reliability champion, not a proven large Elo improvement. Ladder losses from V5
remain useful diagnostics because V7 changes timeout handling and allocation rather than the core
evaluation/search blind spots.

## Completed experiments

The full evidence ledger is in `docs/EXPERIMENT_LEDGER.md`. The most important decisions are:

- V5's 50% NNUE blend was promoted; pure NNUE was rejected.
- Countermove ordering plus quiet-history maluses scored 40.0% and was rejected.
- The 256-wide HalfKP model scored 35.0% and was rejected; exact pruning recovered speed but its
  ten-pair 50.0% screen did not establish a gain.
- The linear V8 residual greatly improved offline error metrics but failed critical probes and its
  smoke match, so it was rejected.
- Global no-LMR and no-null configurations did not repair the two persistent critical misses in
  the completed six-profile V7 mechanism matrix.
- Exact qsearch evaluation caching was promoted after deterministic parity, clean throughput, and
  timed reliability gates.

## What the mechanism matrix established

The checked-in `benchmarks/diagnostics/v7-priority-mechanisms.json` compares baseline, HCE-only,
NNUE-only, no-LMR, no-null, and no-LMR/no-null searches on six rated-error positions.

- Four references are reachable by unchanged V7 with sufficient fresh-search nodes: round 47
  `...Nd4`, round 48 `...Kh7`, and round 50 `...b4` and `...Qd6+`.
- Round 46 `Be2` remains missed at one million nodes by every profile: a shared evaluation/search
  blind spot, subject to deeper teacher confirmation.
- Round 50 `...Qd7` appears much earlier with HCE-only than with the 50% blend; the NNUE delays the
  correction, although unchanged V7 eventually finds it at five million nodes.
- Disabling LMR or null-move pruning does not systematically repair either persistent miss and
  usually buys less depth. A global pruning rollback is not justified.

## Next development decision

Do not start another broad neural-network run or combine fashionable search features. Qsearch
evaluation caching completed the first exact-throughput experiment. The next candidate should:

1. The six-position profile retained exact parity across all 18 probes and found wasted qsearch
   accumulator-update rates of 49.33%, 50.28%, and 49.47% at 100k, 300k, and 1M nodes.
2. `challengers/exp_qs_lazy_accumulator_v8` now implements the isolated post-move update. Exact
   rebuild tests pass for both colours, captures, en passant, promotions, and castling; fixed-node
   parity and throughput gates remain before any promotion.
3. Require exact fixed-node equivalence, a repeatable speed gain, critical-position checks, then
   paired development and untouched validation games against `current/`.

Behaviour-changing candidates now use the integrated five-bin logistic-Elo GSPRT. Exact semantic
optimisations use equivalence plus throughput gates instead of an inappropriate Elo hypothesis.

The reviewed top-20 Toby Coad repository supports these as hypotheses, especially lazy accumulator
updates and quiescence evaluation caching. Its ranking is not causal evidence, and its code or
network must not be copied into the submission.

## Competition contract

The canonical source is <https://aichessathon.com/docs>. At this update: 120 s + 0.5 s, a 90 s
initialization budget, one CPU core, 2 GB RAM, no GPU or network, a draw at 600 plies, 50 MB
uncompressed, and ten uploads per day. The checked-in harness is synced with the current official starter semantics: 600-ply draw, suspended pondering, platform-style scratch paths/log truncation seeded opening-pair arenas, and extracted-ZIP smoke testing.
