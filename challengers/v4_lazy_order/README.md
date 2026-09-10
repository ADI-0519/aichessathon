# V4 lazy move ordering

This isolated challenger changes only how the main alpha-beta search consumes
ordered moves. V3 computes every move score and fully sorts the list before it
searches the first move. This candidate computes the same immutable score
snapshot, then extracts only the best remaining move immediately before that
move is searched. A beta cutoff therefore avoids sorting the unused suffix.

Tie ordering is stable and deliberately matches V3's insertion sort. Quiescence
keeps the full ordering because it filters the resulting list to forcing moves.
The root and recursive main search use lazy selection.

This is a performance hypothesis, not a promoted engine. It must remain exactly
equivalent to V3 at deterministic fixed-node limits and then beat the frozen
root agent in paired games before it can be considered for promotion.

Run its focused tests from the repository root:

```bash
./.venv/Scripts/python.exe -m unittest tests.test_v4_lazy_order -v
```

Run the first paired screen:

```bash
./.venv/Scripts/python.exe -m tools.backtest \
  --candidate challengers/v4_lazy_order \
  --opponent . \
  --suite benchmarks/suites/openings_8moves_v3_500.epd \
  --split development \
  --limit 30 \
  --base-ms 10000 \
  --increment-ms 100 \
  --output benchmarks/runs/v4-lazy-order-vs-v3-development-30
```
