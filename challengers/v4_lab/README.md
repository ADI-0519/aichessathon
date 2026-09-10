# V4 search ablation lab

This directory is a development copy of the frozen V3 Numba engine. Its baseline profile preserves
V3 search behaviour; three compile-time profiles change one search mechanism at a time:

| Profile | LMR | Qsearch SEE/delta pruning | Check extension |
|---|---:|---:|---:|
| `baseline` | on | on | none |
| `no-lmr` | off | on | none |
| `no-q-pruning` | on | off | none |
| `check-extension` | on | on | one per line |

Call `search.configure_experiment()` before `search.warmup()`. Numba treats these module globals as
compile-time constants, so the code rejects attempts to change profile after compilation. The
ablation runner starts a clean Python process for every profile.

This is not a submission candidate. A profile must first improve the critical-position evidence,
then clear paired games against frozen V3 and the independent validation split. Only the measured
winner should be copied into a standalone candidate and packaged.

`experiment.py` selects `check-extension` for harness games because it is the only search ablation
that improved a critical position. Changing this file changes the candidate fingerprint. The
diagnostic CLI bypasses `agent.py` and can still compile any named profile independently.

The `persistent` ablation is a sensitivity test: it retains memory across the ascending node
ladder, so its final probe has received earlier work and is not an equal-compute strength result.

Run all search ablations from Git Bash:

```bash
./.venv/Scripts/python.exe -m tools.search_ablations \
  --nodes 25000,100000,300000 \
  --root-depth 5 \
  --output benchmarks/diagnostics/v4-search-ablations.json
```

Run the first paired playing-strength gate against frozen V3:

```bash
./.venv/Scripts/python.exe -m tools.backtest \
  --candidate challengers/v4_lab \
  --opponent . \
  --suite benchmarks/suites/openings_8moves_v3_500.epd \
  --split development \
  --limit 50 \
  --base-ms 10000 \
  --increment-ms 100 \
  --output benchmarks/runs/v4-checkext-vs-v3-development-50
```

Do not run another CPU-heavy arena concurrently. The candidate advances only if the paired result
supports an improvement and contains no technical failures.
