# Search diagnostics

`tools.search_diagnostics` explains *why* a source-compatible engine selects a move. It is a
development tool, not part of `submission.zip`, and it does not alter the engine being inspected.

The checked-in critical suite contains four positions from the Corundum AI and Negamaximus rated
games. Each position records the played V3 move, the reproducible V3 baseline move, and a
Stockfish-checked reference move. Reference moves are investigation targets rather than unit-test
oracles: a future engine can choose a different strong move and still be correct.

## Run it

From Git Bash, inspect all four positions with the standard fixed-node ladder and exact-depth root
breakdown:

```bash
./.venv/Scripts/python.exe -m tools.search_diagnostics \
  --all-cases \
  --nodes 25000,100000,300000 \
  --root-depth 5 \
  --top 8
```

Inspect one position more deeply:

```bash
./.venv/Scripts/python.exe -m tools.search_diagnostics \
  --case corundum-43-kg1 \
  --nodes 100000,300000,1000000 \
  --root-depth 7 \
  --top 12
```

Point the same tool at an experimental challenger without copying or promoting it:

```bash
./.venv/Scripts/python.exe -m tools.search_diagnostics \
  --engine-root challengers/v4-check-extension \
  --all-cases
```

The V4 lab exposes named compile-time profiles. Run the full isolated comparison with:

```bash
./.venv/Scripts/python.exe -m tools.search_ablations \
  --nodes 25000,100000,300000 \
  --root-depth 5 \
  --output benchmarks/diagnostics/v4-search-ablations.json
```

This launches a clean interpreter for the baseline, no-LMR, no-qsearch-pruning, capped-check-
extension, and persistent-memory trials. A clean process is required because Numba freezes the
profile flags when it compiles the recursive search.

Generate an isolated lab from the canonical engine rather than modifying `current/`:

```bash
./.venv/Scripts/python.exe -m tools.materialize_search_lab \
  --source current \
  --output benchmarks/runs/candidates/current_search_lab

./.venv/Scripts/python.exe -m tools.search_ablations \
  --engine-root benchmarks/runs/candidates/current_search_lab \
  --suite benchmarks/suites/v5_priority_losses.json \
  --trials baseline,hce-only,nnue-only,no-lmr,no-null,no-lmr-no-null \
  --nodes 25000,100000,300000,1000000 \
  --root-depth 5 \
  --output benchmarks/diagnostics/v7-priority-mechanisms.json
```

The generator reads the source engine's `NNUE_BLEND` assignment. Its baseline, no-LMR and no-null
profiles retain that exact blend; HCE-only and NNUE-only explicitly select 0 and 100. The generated
baseline therefore has the same evaluator and search path as its source rather than assuming a
particular historical blend. The other profiles change exactly one axis except
`no-lmr-no-null`, which is an explicit interaction check. These positions diagnose mechanisms;
they do not estimate Elo.

### Completed V7 result

The checked-in matrix is complete. Unchanged V7 reaches the round-47 `...Nd4`, round-48 `...Kh7`,
and round-50 `...b4` and `...Qd6+` references as node limits rise. No LMR/null configuration finds
round-46 `Be2`, and none provides a systematic fix for round-50 `...Qd7`; globally disabling those
pruners also reduces completed depth. HCE-only finds `...Qd7` materially earlier than the blended
evaluator, identifying evaluator disagreement and throughput—not a blanket pruning rollback—as
the next investigation. See `CURRENT_STATE.md` and `docs/EXPERIMENT_LEDGER.md` for the current
decision.

For a realistic memory comparison, replay the actual rated-game histories. The tool searches each
earlier position on V3's turns using one persistent TT/history object, follows the historical moves,
and compares fresh and replayed memory at every critical position:

```bash
./.venv/Scripts/python.exe -m tools.search_memory_replay \
  --engine-root current \
  --warm-nodes 25000 \
  --target-nodes 300000 \
  --output benchmarks/diagnostics/v3-persistent-replay.json
```

Do not run this while an arena is active because both processes would compete for timed CPU work.

Use `--format json` when another script needs stable machine-readable output. An arbitrary position
can be supplied with `--fen` and, optionally, `--reference-move`.

## Read the output

The first table runs normal iterative deepening with fresh memory at every node limit. It reports
the last completed depth, chosen move, score, quiescence share, TT cutoffs, and LMR
reductions/re-searches.

The second table forces every legal root move and searches it independently to the requested
nominal depth. `static` is the evaluation immediately after the move; `score` is the backed-up
search value; and `pv` is the legal continuation reconstructed from that move's transposition
table. Fresh memory makes root candidates comparable and exposes cases where selective search
reverses the static ordering. Because normal iterative search shares history, killers, and TT data
between root moves, its selected move need not equal the first independently searched line. That
difference is itself useful evidence.

An incomplete root line means its per-move `--root-node-limit` was exhausted. The default is zero,
which allows every exact-depth line to finish.

## Development workflow

Before an exact throughput experiment, capture repeated fresh-memory fixed-node measurements from
the canonical engine. The tool rejects a node limit if the move, score, depth, node counts, or
search statistics change between repeats:

```bash
./.venv/Scripts/python.exe -m tools.numba_search_scaling \
  --engine-root current \
  --suite benchmarks/suites/v5_priority_losses.json \
  --nodes 100000,300000,1000000 \
  --repeats 5 \
  --output benchmarks/diagnostics/current-scaling-baseline.json
```

Use the same FEN, limits and repeat count for a challenger. Compare median NPS only after confirming
fixed-node equivalence; individual elapsed times are noisy.

1. Run the frozen V3 engine and save JSON output.
2. Make one isolated search change in a challenger directory.
3. Run the challenger at identical node limits and root depth.
4. Inspect score/PV changes, especially whether the reference defense or conversion appears.
5. Treat the positions as diagnostics only; require paired arena improvement before promotion.

This prevents four hand-selected positions from becoming an overfitted substitute for match
strength.
