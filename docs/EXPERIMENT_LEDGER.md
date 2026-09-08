# Experiment ledger

Updated: 8 September 2026

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
| V7 continuous time | 20W 12D 8L against V6 in development (65.0%); 48.75% in independent validation; 50.0% directly against V5 | Canonical champion for its causal fixes, not proven Elo superiority. |
| V8 linear strategic residual | Validation residual RMSE 267.6 -> 204.0 cp with exact runtime parity; failed all three round-50 probes and lost the first three smoke games | Reject; offline fit did not translate through search. |
| V7 six-profile mechanism matrix | Compared baseline, HCE-only, NNUE-only, no-LMR, no-null, and combined no-LMR/no-null on six rated errors | Complete; rejects a global LMR/null rollback and points to throughput plus evaluator-disagreement work. |

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
