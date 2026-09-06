# V4 search ablation results

Run on 5 September 2026 with the four rated-game critical positions, fresh processes, 25,000,
100,000, and 300,000-node probes, and independently searched depth-5 root lines. The complete
machine-readable report is `benchmarks/diagnostics/v4-search-ablations.json`.

## Highest-node choices

| Trial | Bxf4 case | Be1 case | Kg1 case | Rxb2 case | Decision |
|---|---|---|---|---|---|
| V3-equivalent baseline | Bg4 | Qc2 | Qc7+ | g6 | Control reproduced |
| No LMR | Bg4 | Qc2 | Qc7+ | g6 | Reject |
| No qsearch pruning | Bg4 | Qc2 | Qc7+ | g6 | Reject |
| One check extension per line | Bg4 | Qc2 | **Kg1** | g6 | Advance to paired games |
| Persistent same-position ladder | Bg4 | Qc2 | **Kg1** | g6 | Diagnostic only |

Disabling LMR fixed nothing and generally lost one completed ply. Disabling SEE/delta pruning in
qsearch fixed nothing, increased qsearch share, and reduced the false positive score on Qc7+ without
changing the move at 300,000 nodes.

The capped check extension changed Qc7+ to the reference defence Kg1 at 300,000 nodes and changed
the evaluation from a false `+0.78` to `-0.20`. It did not improve the other three positions and
usually cost one completed nominal ply. This is promising enough for games, not enough for
promotion.

The persistent trial reused the 25,000 and 100,000-node work before its 300,000-node probe. It
therefore consumed 425,000 cumulative nodes and is not an equal-compute comparison. It establishes
TT/history sensitivity only. A faithful game-memory test should replay the preceding game positions
instead of repeatedly searching the same FEN.

## Runtime checks

- The V4 lab baseline exactly matches frozen V3 at a deterministic 20,000-node probe.
- Reconfiguring a profile after Numba warm-up is rejected.
- Every compile-time profile runs in a separate interpreter.
- At the Corundum clock value of 58,062 ms, the check-extension agent returns `Kg1` through the
  normal `get_move` boundary.
- A 5,000+100 ms harness smoke game finished normally. At that deliberately tiny clock the agent
  had only about 170 ms for the critical move and still selected Qc7+, so this is a reliability
  smoke test rather than tactical evidence.

## Next gate

Keep V3 as production. Compare `challengers/v4_lab` (currently selecting `check-extension`) against
the frozen root with paired openings. If it does not beat V3, reject the extension despite solving
one hand-selected position. SF500 should only be run after the direct V3 screen.
