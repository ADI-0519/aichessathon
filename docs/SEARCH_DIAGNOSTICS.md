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

For a realistic memory comparison, replay the actual rated-game histories. The tool searches each
earlier position on V3's turns using one persistent TT/history object, follows the historical moves,
and compares fresh and replayed memory at every critical position:

```bash
./.venv/Scripts/python.exe -m tools.search_memory_replay \
  --engine-root . \
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

1. Run the frozen V3 engine and save JSON output.
2. Make one isolated search change in a challenger directory.
3. Run the challenger at identical node limits and root depth.
4. Inspect score/PV changes, especially whether the reference defense or conversion appears.
5. Treat the positions as diagnostics only; require paired arena improvement before promotion.

This prevents four hand-selected positions from becoming an overfitted substitute for match
strength.
