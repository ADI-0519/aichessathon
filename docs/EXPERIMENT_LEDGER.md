# Experiment ledger

Updated: 9 September 2026

This is the compact decision record for engine experiments. Detailed analysis remains in the
dated research documents and machine-readable benchmark files. A development score is evidence
for choosing what to validate next, not proof of Elo; technical smokes establish reliability only.

## Engine lineage and decisions

| Experiment | Evidence | Decision |
|---|---|---|
| Reliable Python champion | 61.7% against the previously accepted build; 82.5% against the starter baseline | Superseded by the compiled engine. |
| V3 compiled Numba engine | Legal move-generation and protocol gates; packaged and smoke-tested as both colours | Released, then superseded. |
| V4 search ablations | No-LMR and no-qsearch-pruning fixed none of four critical cases; capped check extension fixed one case but cost depth | No global ablation promoted. |
| V4 lazy ordering | Implemented as an isolated, fixed-node-equivalent experiment | Not promoted; retained as historical research code. |
| V5 NNUE blend sweep | Against V4 over 20 development pairs: 25%=75.0%, 50%=73.75%, 100%=65.0%; 50% beat 25% directly by 62.5% | Promote 50% blend; reject pure NNUE. |
| V5 independent validation | 30W 5D 5L against V4, 81.25%, zero technical failures | Confirmed V5; inspected positions ceased to be untouched validation. |
| Endgame phase taper | Initial Colab run failed initialization before chess; redundant development warm-up was removed | Inconclusive, not promoted, and not counted as a chess loss. |
| Countermoves and history maluses | 12W 8D 20L against V5, 40.0%, zero technical failures | Reject this implementation. |
| HalfKP-256 | Offline probability MSE improved 3.38%; 7W 14D 19L against V5, 35.0% | Reject. |
| Exact-pruned HalfKP-128 | Same model output with about 22.8% more probe throughput than HalfKP-256; 5W 10D 5L over ten pairs | No promotion evidence; stop expanding this rejected parent. |
| V6 stable completed-depth timeout | Repaired round 47 exactly; 14W 15D 11L against V5, 53.75%, zero failures | Promote as reliability baseline. |
| V7 continuous time | 20W 12D 8L against V6 in development (65.0%); 48.75% in independent validation; 50.0% directly against V5 | Promoted for causal reliability fixes; later superseded by KingNet75/qcache. |
| V8 linear strategic residual | Validation residual RMSE 267.6 -> 204.0 cp with exact runtime parity; failed all three round-50 probes and lost the first three smoke games | Reject; offline fit did not translate through search. |
| V7 six-profile mechanism matrix | Compared baseline, HCE-only, NNUE-only, no-LMR, no-null, and combined no-LMR/no-null on six rated errors | Complete; rejects a global LMR/null rollback and points to throughput plus evaluator-disagreement work. |
| V7 qsearch evaluation cache | Exact parity over 90 clean probes; median NPS gains of 7.50%, 4.49%, and 5.49% at 100k, 300k, and 1M nodes; 51.25% over 20 timed pairs with zero failures | Promote as an exact throughput improvement; the timed sample establishes non-regression, not Elo. |
| Qsearch lazy-accumulator profile | Exact parity across all 18 six-position probes; wasted-update rates were 49.33%, 50.28%, and 49.47% at 100k, 300k, and 1M nodes | Build an isolated post-move lazy-update candidate; do not change the champion yet. |
| Lazy qsearch accumulator V8 | Post-move update shares the eager feature-delta implementation and matches full rebuilds across special moves plus 200+ deterministic random positions; two-colour smoke passed | Candidate implemented; fixed-node parity and clean throughput remain mandatory before promotion. |
| Raw V9 KingNet | 18W 14D 8L against frozen V7 over 20 development pairs: 62.5%, about +89 Elo, zero failures | Strong evaluator candidate; retained as the no-qcache control. |
| KingNet75 plus qcache | 54.0% over 100 pairs against raw V9; then 71W 35D 16L against prior `current/` over 61 pairs (72.54%, about +169 Elo), accepting the 0-vs-20 Elo pentanomial SPRT with zero failures | Promote as canonical champion; recover the exact training manifest before upload. |
| Dedicated qsearch tactical generator | Exact move, score, depth, node, qnode, and qcache-stat parity across 12 critical probes; ordered tactical sets matched the champion and python-chess across special cases and 400 deterministic positions. First clean 300k-node comparison produced a mixed six-position geometric-mean NPS gain of about 1.9%. | Retain as a correct experiment, but do not promote below the predeclared 5% throughput threshold. |
| Adaptive iterative-deepening time | Separate challenger with reserve-safe soft/normal/hard deadlines, root move/score stability, aspiration-failure response, and iteration-cost gating. Deployment-style fresh replay chose the stable teacher moves `17...Qe5` and `18...Nf6` in round 78; two-colour smoke passed. Against uploaded KingNet75/qcache it scored 83W 43D 74L over 100 development pairs (52.25%, about +15.6 Elo), with zero failures. The 0-vs-20 Elo pentanomial SPRT exhausted its cap without a decision (LLR 0.259; bounds +/-2.944; 95% Elo interval -25.5 to +57.2). | Directionally positive but inconclusive; do not promote from reused development positions. Advance to independent validation and long-clock safety gates. |
| Round 79 five-profile diagnosis | At 100k/300k/1M nodes, KingNet75, HCE-only, and no-LMR selected `f3` throughout. Pure KingNet selected `f3`, `f3`, then `Bxg4`; no-null selected `Na4` only at 100k before reverting to `f3`. A separate 5M-node Stockfish MultiPV check retained `Na4` and scored `f3` about 67 cp lower. | The reference is stable, but the ablation does not isolate evaluator blend, LMR, or null move as the cause. Keep `current/` frozen and use the position as a regression, not a tuning target. |
| Search V10 selective bundle | Independently implemented dynamic NMP with verification from depth 8, reverse futility, late-move and quiet-futility pruning, shallow capture SEE pruning, and contextual LMR. A safety review made PV classification use the entry window, protected killers and high-history quiets from shallow pruning, allowed protected late moves to remain unreduced, and made quiet futility request its own static evaluation. Its all-off `current` profile exactly reproduces current move, score, depth, nodes, qnodes, TT and LMR counts on Round 79 at 100k/300k/1M. The reviewed bundle reaches depth 10 at 1M versus current depth 9, but still selects the inferior `f3` at that cutoff. On the six-position priority suite it finds four of six references at 300k versus three for both current and the original aggressive V10: it retains the `Nd4` repair and restores current's `Qd7` solution. | Reviewed candidate is safer and stronger on deterministic regressions, but is not promoted. Run a technical smoke and a predeclared paired screen before adding TT buckets, singular extensions, or richer histories. |

## Reliability and infrastructure completed

- Reproducible paired backtests with source fingerprints, resumable journals, colour-swapped
  openings, split protection, PGNs, failure attribution, paired intervals, and optional five-bin
  logistic-Elo GSPRT stopping after complete pairs.
- Engine-selectable repeated fixed-node scaling with deterministic-result enforcement and durable
  JSON reports.
- Fixed-node search diagnostics with principal variations, root alternatives, qsearch share, TT
  statistics, and LMR statistics.
- Faithful persistent-memory replay for reproducing rated decisions.
- Stockfish-only development analysis and label generation; no Stockfish asset enters a submission.
- Differential board fuzzing, NNUE feature/update parity, packaging tests, strict mypy, Ruff, and
  wire-protocol smoke games.
- Canonical `current/` packaging with the model under `weights/model.npz`.

## Considered but not yet tested as candidates

- Lazy NNUE accumulator updates and quiescence/static-evaluation caching.
- Pawn/HCE caching after separating pawn-only terms from king-dependent shelter terms.
- Opening books, pending measured coverage of curated starts.
- Small Syzygy tablebases as endgame polish rather than a solution to middlegame errors.
- Pondering is currently unavailable in practice because the process is suspended while the
  opponent thinks.

## Evidence locations

- Learned-evaluator and match narrative: `docs/LEARNED_EVALUATOR.md`
- Search/evaluation strategy reset: `docs/ENGINE_STRATEGY_RESET.md`
- V4 mechanism results: `docs/V4_SEARCH_ABLATIONS.md`
- Ordering result: `benchmarks/diagnostics/v6-ordering-vs-v5-development-20.json`
- V7 mechanism matrix: `benchmarks/diagnostics/v7-priority-mechanisms.json`
- Rated regression positions: `benchmarks/suites/v5_priority_losses.json` and the other
  `v5_round*.json` suites

Raw `benchmarks/runs/` directories are local, ignored working evidence. Promote durable conclusions
into this ledger or a small checked-in diagnostic summary rather than committing entire run trees.
