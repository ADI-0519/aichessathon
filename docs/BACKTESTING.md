# Reproducible backtesting and pentanomial SPRT

`tools.backtest` is the reproducible strength-testing layer above the unmodified competition
referee. It is development-only and is never packaged with the chess agent.

## Guarantees

- Every selected position is played once with each colour.
- `--workers N` runs independent position pairs concurrently while keeping both colours of each
  pair sequential and committing results in deterministic suite order. The default is one worker.
- Candidate and opponent package inputs are SHA-256 fingerprinted in `manifest.json`.
- Those fingerprints are checked before every new game. Editing an engine stops the run instead of
  silently mixing versions.
- Every completed game is durably appended to `games.jsonl`; repeating the same command skips it.
- PGNs are written atomically under `games/`, and `summary.json` is refreshed after every game.
- Worker tasks keep records, PGNs, and stderr tails in memory; only the coordinator writes run
  artifacts.
- A pre-existing output directory can only resume an identical experiment configuration.
- An atomic writer lock prevents two processes from corrupting the same experiment directory.
- Voids are not misreported as draws, and technical failures are attributed to the candidate or
  opponent.
- Reports include W/D/L, colour splits, terminations, Elo from score, paired pentanomial counts,
  and an approximate 95% interval derived from the complete-pair sample variance.
- Optional SPRT state is stored in both the immutable manifest and the refreshed summary. Boundary
  checks occur only after both colours of a position have finished.

The confidence interval is descriptive; it is not the sequential decision rule. With `--sprt`, the
runner uses all five pair frequencies in a constrained-multinomial generalized likelihood ratio,
following fishtest's logistic-Elo GSPRT construction. Empty bins receive the same small `1e-3`
regularizer used by fishtest.

## Suites and leakage control

The suite can be `builtin` or a UTF-8 `.epd`, `.fen`, or `.pgn` file. EPD files contain one EPD
record per non-comment line; FEN files contain one complete six-field FEN per non-comment line. For
a PGN opening book, the position after each main line is used. Duplicate normalized FENs are
removed.

Positions receive a stable SHA-256 assignment:

- 60% development;
- 20% validation;
- 20% holdout.

The assignment depends only on normalized FEN and `--split-seed`, so reordering or extending a
suite cannot move an existing position between splits. Accessing `holdout` or `all` requires the
deliberate `--unlock-holdout` flag. Do not use the holdout to choose features or weights.

The repository includes a pinned 500-position suite at
`benchmarks/suites/openings_8moves_v3_500.epd`. Its manifest and provenance are documented in
`benchmarks/suites/README.md`. Use this suite for comparable experiments; keep the 15-position
`builtin` set for quick regressions rather than strength claims.

`tools.build_suite` can create another deterministic subset from an EPD, FEN, or PGN source. It
selects positions by a seeded SHA-256 rank, so source reordering cannot change the chosen set, and
writes a provenance manifest next to the generated EPD.

## Canonical champion example

```bash
./.venv/Scripts/python.exe -m tools.backtest \
  --candidate current \
  --opponent challengers/v6_stable_timeout \
  --suite benchmarks/suites/openings_8moves_v3_500.epd \
  --split validation \
  --base-ms 10000 \
  --increment-ms 100 \
  --output benchmarks/runs/v7-vs-v6-validation-8moves
```

## Sequential strength test

Use SPRT for changes that alter chess behaviour, such as pruning, evaluation, extensions, move
ordering, a new network, or time management:

```bash
./.venv/Scripts/python.exe -m tools.backtest \
  --candidate challengers/exp_candidate \
  --opponent current \
  --suite benchmarks/suites/openings_8moves_v3_500.epd \
  --split development \
  --limit 200 \
  --base-ms 10000 \
  --increment-ms 100 \
  --sprt \
  --sprt-elo0 0 \
  --sprt-elo1 20 \
  --sprt-alpha 0.05 \
  --sprt-beta 0.05 \
  --sprt-min-pairs 25 \
  --output benchmarks/runs/exp-candidate-sprt-0-20
```

For `SPRT[0,+20]`, crossing the upper boundary favours the `+20 Elo` hypothesis. Crossing the
lower boundary favours `0 Elo` over `+20 Elo`; it does not prove that the candidate is weaker.
When neither boundary is reached before the selected positions are exhausted, the result remains
inconclusive. Do not add an extra point-estimate stopping rule: doing so invalidates the configured
sequential error rates.

Do not run one logical SPRT as independent concurrent shards. Pair order and the cumulative stopping
state belong to one journal. Concurrent fixed-length screens remain supported with disjoint offsets
and output directories.

Correctness-preserving speed changes should not be forced through `SPRT[0,+20]`. Prove identical
fixed-node moves, scores and node counts, require repeatable throughput improvement, then run
technical games. Use SPRT when the change can alter played moves.

## Stockfish example

Stockfish is an offline opponent only and is never included in the submission.

```bash
./.venv/Scripts/python.exe -m tools.backtest \
  --candidate current \
  --stockfish "/c/Users/adirj/AppData/Local/Microsoft/WinGet/Packages/Stockfish.Stockfish_Microsoft.Winget.Source_8wekyb3d8bbwe/stockfish/stockfish-windows-x86-64-avx2.exe" \
  --stockfish-nodes 2000 \
  --suite benchmarks/suites/openings_8moves_v3_500.epd \
  --split validation \
  --base-ms 10000 \
  --increment-ms 100 \
  --output benchmarks/runs/v7-vs-sf2k-validation-8moves
```

Run the exact same command after an interruption to resume. If any hashed source changed, choose a
new output directory; old and new builds must never share a report.

Only one process may use an output directory at a time. A hard process kill can leave `.run.lock`
behind; after confirming no matching backtest is active, delete that one lock file and resume.

Use `--workers N` to run up to `N` position pairs at once. Every individual game still receives
fresh harness agent processes. The scheduler keeps at most `2 * N` pair tasks in flight or queued
to reduce head-of-line idling while bounding speculative work. Changing the worker count changes
the immutable experiment configuration, so resume with the same value. Because concurrent games
share the host's CPU, choose a worker count appropriate for the machine and time-control
sensitivity.

`summary.json["elapsed_s"]` is the sum of individual game durations, not end-to-end wall time. It
will exceed actual runtime when games overlap. For worker-scaling measurements, wrap the complete
Git Bash command in `time` and compare that wall-clock result instead.

Use `--limit N` for a short screen and `--offset N` for an explicitly selected later slice. The
position order is deterministically shuffled by `--order-seed` before those options apply.

## Reading the output

The pentanomial keys are the candidate's total points over a colour-swapped pair:

| Key | Pair outcome |
|---:|---|
| 0.0 | two losses |
| 0.5 | loss and draw |
| 1.0 | two draws, or one win and one loss |
| 1.5 | win and draw |
| 2.0 | two wins |

The candidate must have zero technical failures. A positive score on development positions is not
enough for promotion: require confirmation on validation, external opponents, and finally the
locked holdout.
