# Qsearch lazy-accumulator profiler

This development-only challenger is copied from the cached V7 champion. It changes no search
decision and is not a submission candidate. Its only behavioral difference is recording how often
quiescence search updates the NNUE accumulator for a move that SEE/delta pruning immediately
rejects.

The profile reports:

- qsearch moves considered;
- accumulator updates performed;
- moves pruned after an update;
- moves retained because they give check despite otherwise satisfying a pruning condition; and
- recursive qsearch children.

For the current eager implementation, every considered move must receive one accumulator update,
and every considered move must be either pruned or searched. The scaling tool enforces those
accounting identities before writing a report.

Run the profiler from the repository root:

```bash
./.venv/Scripts/python.exe -m tools.numba_search_scaling \
  --engine-root challengers/exp_qs_lazy_profile_v7q \
  --suite benchmarks/suites/v5_priority_losses.json \
  --nodes 100000,300000,1000000 \
  --repeats 3 \
  --output benchmarks/diagnostics/qs-lazy-accumulator-profile.json
```

The principal decision metric is `q_accumulator_waste_rate`, calculated as accumulator updates for
actually pruned moves divided by all qsearch accumulator updates. Fixed-node moves, scores, depths,
node counts, and existing search statistics must remain identical to `current/`.
