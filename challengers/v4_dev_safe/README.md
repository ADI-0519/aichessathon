# V4 Dev safe

This challenger is the Dev search stack immediately before null-move pruning.
Its engine code is preserved from commit `01f0205` so that the safer changes can
be measured independently from the higher-risk null-move experiment.

Compared with frozen V3, it contains:

- capture-only quiescence outside check, while retaining every legal check evasion;
- a dedicated legal-capture generator with promotion and stalemate handling;
- a legality fast path when moving a piece cannot expose its king;
- iterative deepening that continues until the allocated deadline;
- a legal, proven root move when an iteration is interrupted.

It deliberately contains no null-move make/unmake or null-move pruning. It is a
challenger, not a submission build, until it clears correctness and paired-play
gates against frozen V3.

Run its focused correctness tests from the repository root:

```bash
./.venv/Scripts/python.exe -m unittest tests.test_v4_dev_safe -v
```

Run it through the platform-style harness:

```bash
./.venv/Scripts/python.exe -m harness.play \
  --white challengers/v4_dev_safe --black baselines/minimax
```

