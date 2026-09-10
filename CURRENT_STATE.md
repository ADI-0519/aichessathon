# Current engine state

Updated: 10 September 2026

## Deployable champion

`current/` is the only canonical deployable source directory. The default play, arena, backtest,
diagnostic, type-checking, and packaging commands all target it. Packaging copies the contents of
that directory to the archive root, where the platform imports `agent.py`.

- Engine generation: V14 Runtime with KingNet75, exact qsearch caching and V13 search
- Exact promoted snapshot: `challengers/exp_release_v14_runtime/`
- Frozen rollback champion: `challengers/exp_release_v12/`
- Versioned release artifact: `submission_v14.zip`
- Fresh canonical artifact: `submission.zip` (gitignored and rebuilt with `make zip`)
- Lineage: V7 -> qcache -> KingNet75 -> Search V10 -> V12 -> V13 Search -> V14 Runtime

The current executable files and model are byte-identical to the tested V14 Runtime snapshot;
only descriptive package metadata differs. Engine files no longer live at the repository root,
and the repository root must not be packaged as an agent.

## Promoted evaluator

`challengers/v9_kingnet/` is an exact import of the executable V9 files from teammate branch
`dev` at commit `576348b`, with an added local README and verifier. It uses a 16-bucket,
king-conditioned 128-wide learned accumulator at a 75% blend. Its incremental and fixed-point
checks pass, including king moves across bucket boundaries, and a one-move smoke test completed
with a 44.5-second Numba warmup and a legal move.

Raw V9 scored 62.5% over 20 development pairs against frozen V7. The combined KingNet75/qcache
candidate then scored 54.0% over 100 pairs against raw V9, with zero failures. Most importantly,
it scored 72.54% over 61 pairs against the prior canonical V7/qcache champion and crossed the
integrated 0-versus-20 Elo pentanomial SPRT upper boundary. It is therefore promoted over raw V9.

The teammate branch lacks a retained training manifest for the exact bundled model hash. Recover
the dataset, trainer, command/configuration, checkpoint-selection, and model-hash provenance before
the weights are considered submission-ready.

## What the current champion contains

- Team-trained, incrementally updated 16-bucket king-conditioned NNUE with a 128-wide accumulator
  and 32-wide hidden layer, blended 75:25 with the handcrafted evaluator.
- Numba-compiled board and alpha-beta search with iterative deepening and principal-variation
  search.
- A fixed-size array transposition table, aspiration windows, quiescence search, SEE move scoring,
  killer moves, quiet history, late-move reductions, and guarded null-move pruning.
- Search V10's dynamic null-move reduction, reverse futility, late-move pruning, quiet futility,
  SEE pruning, and contextual LMR.
- V12's signed history maluses, conservative quiet SEE, stale-accumulator repair, and adaptive
  completed-iteration timing.
- V13's material-aware horizon, larger transposition table, safe partial-root recovery, log-log
  interior LMR, and guarded root LMR with verification.
- V14's exact post-pruning main-search accumulator updates and reduced import warm-up.
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
| Raw V9 KingNet vs frozen V7, development, 20 pairs | 62.5%, about +89 Elo, zero failures | Strong directional evidence for the king-conditioned evaluator. |
| KingNet75/qcache vs raw V9, development, 100 pairs | 54.0%, about +28 Elo, zero failures | Qcache was non-regressive and directionally positive with KingNet. |
| KingNet75/qcache vs prior `current/`, development SPRT | 71W 35D 16L, 72.54%, about +169 Elo over 61 pairs | Accepted H1 in the 0-versus-20 Elo pentanomial SPRT; promote. |
| Search V10 vs KingNet75/qcache, development, 25 pairs | 24W 10D 16L, 58.0%, about +56 Elo, zero failures | Positive development result. |
| Search V10 vs KingNet75/qcache, independent validation, 25 pairs | 24W 11D 15L, 59.0%, about +63 Elo, zero failures | Confirmed the direction on the held-out split; promote. |
| V14 Runtime vs frozen V12, development, 30 pairs | 27W 23D 10L, 64.17%, about +101 Elo; paired 95% score interval 56.49% to 71.85%; zero failures | Direct promotion evidence; V14 Runtime becomes canonical champion. |
| V14 qTT vs V14 Runtime, 10 pairs | 4W 8D 8L, 40.0%, about -70 Elo; zero failures | Reject qTT and its exact-tree early-probe optimisation. |

V14 Runtime is now the strength champion. Historical ladder losses remain useful as
position-distribution diagnostics but are not direct measurements of this engine.

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
- The team-trained KingNet75 evaluator and qcache combination was promoted after its direct SPRT
  against the previous champion accepted the positive hypothesis with zero failures.

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

Keep `current/` frozen while the next lanes are measured independently:

1. Run faithful V12/V14 replay over rounds 97 onward and retain only genuine decision regressions.
2. Investigate V14's one-million-node `Rxb2` miss without globally disabling pruning.
3. Measure initialization on the platform validation log; do not trade playing NPS for local-Colab
   warm-up improvements.
4. Keep qTT rejected and test any further search mechanism in an isolated challenger.

Behaviour-changing candidates now use the integrated five-bin logistic-Elo GSPRT. Exact semantic
optimisations use equivalence plus throughput gates instead of an inappropriate Elo hypothesis.

The reviewed top-20 Toby Coad repository supports these as hypotheses, especially lazy accumulator
updates and quiescence evaluation caching. Its ranking is not causal evidence, and its code or
network must not be copied into the submission.

## Competition contract

The canonical source is <https://aichessathon.com/docs>. At this update: 120 s + 0.5 s, a 90 s
initialization budget, one CPU core, 2 GB RAM, no GPU or network, a draw at 600 plies, 50 MB
uncompressed, and ten uploads per day. The checked-in harness is synced with the current official starter semantics: 600-ply draw, suspended pondering, platform-style scratch paths/log truncation seeded opening-pair arenas, and extracted-ZIP smoke testing.
